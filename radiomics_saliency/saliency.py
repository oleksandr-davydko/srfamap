import os
import sys
import time
from functools import reduce

import numpy as np
import quantus
import SimpleITK as sitk
import torch
from captum.attr import IntegratedGradients
from radiomics.featureextractor import RadiomicsFeatureExtractor
from tqdm import tqdm

from radiomics_saliency import config
from radiomics_saliency.feature_extraction import (
    get_feature_number,
    get_high_order_feature_number,
    get_mapping_method,
    get_method_features_list,
    get_texture_matrices,
    prepare_radiomics_rs_attributions,
    prepare_radiomics_rs_image,
)
from radiomics_saliency.models import MultiClassIGTarget, RadiomicMatricesVectorizer, RadiomicsMLPExplanationWrapper


class SaliencyMapGenerator:
    def __init__(self, model: torch.nn.Module, explanation_wrapper, feature_configs, device, mean, std,
                 attribution_mode: str = 'pairwise', eg_samples: int = 24, n_steps: int = 32,
                 eg_baseline_chunk: int = 4):
        self.model = model
        self.device = device
        self.radiomics_features_extractor = RadiomicsFeatureExtractor()
        self.wrapped_model = explanation_wrapper
        self.feature_configs = feature_configs
        self.wrapped_mlp = RadiomicsMLPExplanationWrapper(model, feature_configs, mean, std).to(device)
        self.integrated_gradients = IntegratedGradients(self.wrapped_mlp)
        self.class_model_integrated_gradients = IntegratedGradients(self.model)
        # Multi-class attribution: attribute a per-example logit margin instead of a
        # raw class probability, and use an Expected-Gradients baseline (averaged IG
        # over a pool of real training matrices) instead of the degenerate zero matrix.
        self.attribution_mode = attribution_mode
        self.eg_samples = eg_samples
        self.default_n_steps = n_steps
        # Number of EG baselines whose Integrated Gradients are computed in a single
        # batched captum call. Larger values trade GPU memory for throughput; 1 falls
        # back to the original one-call-per-baseline loop.
        self.eg_baseline_chunk = max(1, int(eg_baseline_chunk))
        self.baseline_pool = None  # [P, D] vectorized training texture matrices
        self.margin_target = MultiClassIGTarget(self.wrapped_mlp, mode=attribution_mode).to(device)
        self.margin_integrated_gradients = IntegratedGradients(self.margin_target)
        self.vectorizer = RadiomicMatricesVectorizer(self.feature_configs)
        self.activation_extractors = {}
        self.profile_enabled = False
        self.profile_stats = {
            'texture_matrix_seconds': 0.0,
            'vectorize_seconds': 0.0,
            'integrated_gradients_seconds': 0.0,
            'map_projection_seconds': 0.0,
            'activation_ig_seconds': 0.0,
            'activation_extractor_seconds': 0.0,
            'saliency_images_processed': 0,
            'activation_images_processed': 0,
        }

    def set_baseline_pool(self, images, sample_size: int = 64, seed: int = 0):
        """Build the Expected-Gradients baseline pool from a sample of (training)
        images. Each baseline is a real, on-manifold texture-matrix vector, which
        avoids the singular ``M=0`` baseline (undefined normalized features) and
        removes single-baseline arbitrariness by averaging over many references."""
        rng = np.random.default_rng(seed)
        n = len(images)
        sample_size = min(sample_size, n)
        idx = rng.choice(n, size=sample_size, replace=False)
        mats = [get_texture_matrices(images[i], self.feature_configs) for i in idx]
        self.baseline_pool = self.vectorizer.transform_batch(mats).to(torch.float).to(self.device)
        return self.baseline_pool

    def _set_multiclass_targets(self, inputs, c_idx):
        """Configure the margin target with the predicted class ``c`` and, for
        pairwise mode, the runner-up competitor ``r = argmax_{j!=c} f_j`` computed
        from the actual input's logits."""
        device = inputs.device
        c_idx = torch.as_tensor(c_idx, dtype=torch.long, device=device).reshape(-1)
        r_idx = None
        if self.attribution_mode == 'pairwise':
            with torch.no_grad():
                logits = self.wrapped_mlp(inputs)
            masked = logits.clone()
            masked[torch.arange(masked.shape[0], device=device), c_idx] = float('-inf')
            r_idx = masked.argmax(dim=1)
        self.margin_target.set_targets(c_idx, r_idx)
        return c_idx, r_idx

    def _sample_baseline_indices(self, k):
        # Deterministic fixed reference set (the pool was already a seeded random
        # sample). Keeping it fixed makes attributions reproducible and avoids
        # injecting sampling noise into the relativeInputStability metric.
        k = min(k, self.baseline_pool.shape[0])
        return torch.arange(k, device=self.baseline_pool.device)

    def _attribute_with_baselines(self, inputs, n_steps, baseline_indices):
        """Expected Gradients: average IG over the given pool baselines (targets
        must already be set on ``self.margin_target``).

        Baselines are processed in chunks of ``self.eg_baseline_chunk`` so several
        ``(input, baseline)`` pairs share one Integrated Gradients call. This is
        mathematically identical to attributing each baseline separately — the model
        is strictly per-example, so a row's gradient is unaffected by the other rows
        in the batch — but keeps the GPU far better utilised. ``chunk == 1`` reproduces
        the original per-baseline loop exactly. On CUDA OOM it retries with ``chunk=1``."""
        chunk = max(1, int(self.eg_baseline_chunk))
        try:
            return self._attribute_with_baselines_chunked(inputs, n_steps, baseline_indices, chunk)
        except RuntimeError as error:
            if chunk > 1 and 'out of memory' in str(error).lower():
                if self.device.type == 'cuda':
                    torch.cuda.empty_cache()
                print(f'EG baseline chunk={chunk} ran out of memory; retrying with chunk=1.')
                return self._attribute_with_baselines_chunked(inputs, n_steps, baseline_indices, 1)
            raise

    def _attribute_with_baselines_chunked(self, inputs, n_steps, baseline_indices, chunk):
        num_examples = inputs.shape[0]
        # The predicted class / competitor indices depend only on the input example,
        # so tile the per-example targets to match the [chunk * num_examples] batch.
        base_c = self.margin_target._c_idx
        base_r = self.margin_target._r_idx
        total = torch.zeros_like(inputs)
        num_baselines = len(baseline_indices)
        try:
            for start in range(0, num_baselines, chunk):
                chunk_indices = baseline_indices[start:start + chunk]
                k = len(chunk_indices)
                # Row p of the tiled batch pairs input (p % num_examples) with
                # baseline chunk_indices[p // num_examples].
                tiled_inputs = inputs.repeat(k, 1)
                tiled_baselines = self.baseline_pool[chunk_indices].repeat_interleave(num_examples, dim=0)
                self.margin_target.set_targets(
                    base_c.repeat(k),
                    None if base_r is None else base_r.repeat(k))
                attributions = self.margin_integrated_gradients.attribute(
                    tiled_inputs, baselines=tiled_baselines, target=0, n_steps=n_steps)
                total = total + attributions.reshape(k, num_examples, -1).sum(dim=0)
        finally:
            # Leave the caller's margin_target holding the untiled per-example targets.
            self.margin_target.set_targets(base_c, base_r)
        return total / float(num_baselines)

    def _attribute_batch(self, inputs, c_idx, n_steps):
        """Per-cell attributions for a batch using the configured multi-class target
        and an Expected-Gradients baseline. Falls back to a zero baseline if no
        pool was set (legacy behaviour)."""
        self._set_multiclass_targets(inputs, c_idx)
        if self.baseline_pool is None:
            return self.margin_integrated_gradients.attribute(inputs, target=0, n_steps=n_steps)
        return self._attribute_with_baselines(inputs, n_steps, self._sample_baseline_indices(self.eg_samples))

    def compute_completeness(self, images, targets, n_steps=None):
        """Mean relative completeness error of the attributions:
        ``|sum(A) - (g(x) - mean_b g(baseline_b))| / (|...| + eps)`` where ``g`` is
        the configured target scalar. A large value flags too-few IG steps or the
        path crossing singular/epsilon-clamped feature regions."""
        n_steps = self.default_n_steps if n_steps is None else n_steps
        tvals = np.asarray(targets).reshape(-1)
        errors = []
        for i, image in enumerate(images):
            mats = get_texture_matrices(image, self.feature_configs)
            x = self.vectorizer.transform(mats).to(torch.float).to(self.device)
            self._set_multiclass_targets(x, [int(tvals[i])])
            g_x = float(self.margin_target(x).item())
            if self.baseline_pool is None:
                attr = self.margin_integrated_gradients.attribute(x, target=0, n_steps=n_steps)
                g_base = float(self.margin_target(torch.zeros_like(x)).item())
            else:
                idx = self._sample_baseline_indices(self.eg_samples)
                attr = self._attribute_with_baselines(x, n_steps, idx)
                g_base = float(np.mean([self.margin_target(self.baseline_pool[b:b + 1]).item() for b in idx]))
            reference = g_x - g_base
            errors.append(abs(float(attr.sum().item()) - reference) / (abs(reference) + 1e-8))
        return float(np.mean(errors)) if errors else float('nan')

    def set_profiling(self, enabled: bool):
        self.profile_enabled = enabled

    def reset_profile_stats(self):
        for key in self.profile_stats:
            self.profile_stats[key] = 0.0 if 'seconds' in key else 0

    def get_profile_stats(self):
        stats = dict(self.profile_stats)
        saliency_images = max(int(stats['saliency_images_processed']), 1)
        activation_images = max(int(stats['activation_images_processed']), 1)
        stats['saliency_seconds_per_image'] = (
            stats['texture_matrix_seconds'] +
            stats['vectorize_seconds'] +
            stats['integrated_gradients_seconds'] +
            stats['map_projection_seconds']
        ) / saliency_images
        stats['activation_seconds_per_image'] = (
            stats['activation_ig_seconds'] +
            stats['activation_extractor_seconds']
        ) / activation_images
        return stats

    def _sync_device(self):
        if self.device.type == 'cuda':
            torch.cuda.synchronize(self.device)

    def _record_time(self, key: str, started_at: float):
        if not self.profile_enabled:
            return
        self._sync_device()
        self.profile_stats[key] += time.perf_counter() - started_at

    def _build_activation_extractor(self, method: str, feature_name: str, feature_settings):
        distance = feature_settings.get('distance', None)
        delta = feature_settings.get('delta', None)
        alpha = feature_settings.get('alpha', None)
        cache_key = (method, feature_name, distance, delta, alpha)
        if cache_key in self.activation_extractors:
            return self.activation_extractors[cache_key]

        extractor = RadiomicsFeatureExtractor()
        extractor.disableAllFeatures()
        extractor.settings = {
            'symmetricalGLCM': False
        }
        if distance is not None:
            extractor.settings['distances'] = [distance]
        if delta is not None:
            extractor.settings['distances'] = [delta]
        if alpha is not None:
            extractor.settings['gldm_a'] = alpha
        extractor.enableFeaturesByName(**{method: [feature_name]})
        self.activation_extractors[cache_key] = extractor
        return extractor

    def generate_saliency_maps(self, images: np.ndarray, targets, n_steps: int = None, batch_size: int = 4, **kwargs):
        results = []
        target_values = np.asarray(targets).reshape(-1)
        batch_size = max(1, int(batch_size))
        n_steps = self.default_n_steps if n_steps is None else n_steps

        for batch_start in range(0, len(images), batch_size):
            batch_end = min(batch_start + batch_size, len(images))
            batch_images = images[batch_start:batch_end]
            batch_targets = target_values[batch_start:batch_end].astype(int).tolist()

            texture_started_at = time.perf_counter()
            batch_texture_matrices = [
                get_texture_matrices(image, self.feature_configs)
                for image in batch_images
            ]
            self._record_time('texture_matrix_seconds', texture_started_at)

            vectorize_started_at = time.perf_counter()
            texture_matrices_vectorized = self.vectorizer.transform_batch(batch_texture_matrices).to(torch.float).to(self.device)
            self._record_time('vectorize_seconds', vectorize_started_at)

            ig_started_at = time.perf_counter()
            texture_properties_attributions = self._attribute_batch(
                texture_matrices_vectorized, batch_targets, n_steps)
            self._record_time('integrated_gradients_seconds', ig_started_at)

            mapping_started_at = time.perf_counter()
            attributed_matrices = self.vectorizer.inverse_transform(texture_properties_attributions)
            for local_index, image in enumerate(batch_images):
                result = []
                prepared_image = prepare_radiomics_rs_image(image)
                for cfg, attributions_matrix in zip(self.feature_configs, attributed_matrices):
                    try:
                        method, parameters = cfg
                        attribution_method = get_mapping_method(method, **parameters)
                        att = attribution_method(
                            image=prepared_image,
                            attributions=prepare_radiomics_rs_attributions(
                                attributions_matrix[local_index].detach().cpu().numpy()))
                        att = att / (np.max(att) + sys.float_info.epsilon)
                        result.append(att)
                    except Exception as e:
                        print(
                            f'Error occured while processing attributions for {cfg[0]} matrix '
                            f'(image_shape={prepared_image.shape}, image_dtype={prepared_image.dtype}): {e}')
                if result:
                    results.append(np.sum(np.asarray(result), axis=0).astype('float32'))
                else:
                    results.append(np.zeros_like(prepared_image, dtype='float32'))
            self._record_time('map_projection_seconds', mapping_started_at)

            if self.profile_enabled:
                self.profile_stats['saliency_images_processed'] += len(batch_images)

        return np.asarray(results, dtype='float32')

    def generate_saliency_map(self, image: np.ndarray, target: int, **kwargs):
        try:
          return self.generate_saliency_maps(
              np.asarray([image]),
              np.asarray([target]),
              n_steps=kwargs.get('n_steps', None),
              batch_size=1)[0]
        except Exception as e:
          print(e)
          return np.zeros_like(image).astype('float32')

    def generate_activation_map(self, image: np.ndarray, image_global_features, medians, target: int, top_features: int = 1):
        assert len(image.shape) == 2
        mask = (image > 0).astype('uint8')
        if np.unique(mask).shape[0] == 1:
            mask[0, 0] = 0
        activation_ig_started_at = time.perf_counter()
        feature_attr = self.class_model_integrated_gradients.attribute(
            image_global_features.reshape(1, -1).to(config.device),
            target=[target],
            n_steps=5)
        self._record_time('activation_ig_seconds', activation_ig_started_at)
        feature_indices = torch.argsort(feature_attr, descending=True).cpu().detach().numpy()[0][:top_features]
        feature_name_map = reduce(lambda x, y: x + y,
                                  [list(get_method_features_list(x[0]).values()) for x in self.feature_configs], [])
        feature_method_map = reduce(lambda x, y: x + y,
                                    [[x[0]] * get_high_order_feature_number(x[0]) for x in self.feature_configs], [])
        feature_extraction_settings_map = reduce(lambda x, y: x + y,
                                                 [[x[1]] * get_feature_number(x[0]) for x in self.feature_configs], [])
        sitk_image = sitk.GetImageFromArray(image)
        sitk_mask = sitk.GetImageFromArray(mask)
        placeholder = []  # np.empty((top_features, image.shape[0], image.shape[1]))
        mask_begin_x = 0
        mask_begin_y = 0
        for i in range(mask.shape[0]):
            if mask[i].any():
                mask_begin_x = max(i - 1, 0)
                break
        for j in range(mask.shape[1]):
            if mask[:, j].any():
                mask_begin_y = max(j - 1, 0)
                break
        feature_names = []
        for c_index, f_index in enumerate(feature_indices):
            int_idx = int(f_index)
            feature_name = feature_name_map[int_idx]
            method = feature_method_map[int_idx]
            feature_settings = feature_extraction_settings_map[int_idx]
            complex_name = f'original_{method}_{feature_name}'
            extractor_started_at = time.perf_counter()
            extractor = self._build_activation_extractor(method, feature_name, feature_settings)
            features = extractor.execute(sitk_image, sitk_mask, voxelBased=True)
            self._record_time('activation_extractor_seconds', extractor_started_at)
            feature_map = sitk.GetArrayFromImage(features[complex_name])
            if feature_map.shape == image.shape:
                placeholder.append(feature_map)
            else:
                padded = np.zeros_like(image, dtype='float32')
                padded[mask_begin_x:mask_begin_x + feature_map.shape[0], mask_begin_y:mask_begin_y + feature_map.shape[1]] = feature_map
                placeholder.append(padded)
            feature_names.append(f'{method} {feature_name}')
        if self.profile_enabled:
            self.profile_stats['activation_images_processed'] += 1
        return placeholder, feature_names

    def compute_map_metrics(self, images: np.ndarray, maps: np.ndarray, labels: np.ndarray, explain_func):
        metrics = {
            'faithfulnessCorrelation': [],
            'selectivity': [],
            'relativeInputStability': [],
            'sparseness': []
        }
        relative_input_stability = quantus.RelativeInputStability(
            nr_samples=2
        )
        sparseness = quantus.Sparseness()
        self.wrapped_model.eval()
        width, height = images.shape[1:]

        for i in tqdm(range(images.shape[0])):
            # Scale the faithfulness subset to the map's own support: 1/20 of the non-zero
            # attribution pixels, over 20 runs. The previous nr_runs=10 / subset_size=1
            # setting estimated a Pearson correlation from only 10 single-pixel black-outs,
            # which for a global-texture model produces noise-level logit deltas and a very
            # high-variance (often spuriously negative) correlation. Perturbing a patch
            # drawn from the attributed region makes each logit delta measurable and the
            # estimate far more stable.
            # The subset is capped at height-1 because quantus asserts subset_size < the
            # last input dimension for our (1, width, height) layout (it raises ValueError
            # otherwise); dense maps therefore saturate at that cap.
            nonzero = int(np.count_nonzero(maps[i]))
            subset_size = min(max(1, nonzero // 20), height - 1)
            faithfulness_correlation = quantus.FaithfulnessCorrelation(
                nr_runs=20,
                subset_size=subset_size,
                perturb_baseline="black",
                similarity_func=quantus.similarity_func.correlation_pearson,
                abs=False,
                normalise=False,
                aggregate_func=np.mean,
                return_aggregate=True,
                disable_warnings=True,
            )
            metrics_list = [
                ('faithfulnessCorrelation', faithfulness_correlation),
                ('relativeInputStability', relative_input_stability),
                ('sparseness', sparseness)
            ]
            for metric_name, executor in metrics_list:
                try:
                    metric_value = executor(
                        self.wrapped_model,
                        x_batch=images[i].reshape(1, width, height).astype('float64'),
                        y_batch=labels[i],
                        a_batch=maps[i].reshape(1, width, height),
                        channel_first=True,
                        explain_func=explain_func if metric_name == 'relativeInputStability' else None)
                    metrics[metric_name].append(np.mean(metric_value))
                except AssertionError as error:
                    print(error)
                    metrics[metric_name].append(None)

        return metrics


def _saliency_selection_mask(saliency_map, percentile=90, sign='abs'):
    """
    Build a boolean mask selecting the most-attributed pixels of a single saliency map.

    Pixels with exactly zero attribution (typically background outside the region of
    interest) never carry attribution, so they are excluded from the ranking; the
    percentile is taken over the non-zero attributions only.

    Args:
        saliency_map: Signed attribution map for one image
        percentile: Percentile threshold, e.g. 90 selects the top 10% (default: 90)
        sign: Which attributions to rank.
            'abs'      - largest magnitude regardless of direction (original behaviour)
            'positive' - largest positive attributions (evidence for the prediction)
            'negative' - largest negative attributions (evidence against the prediction)

    Returns:
        Boolean mask with True at the selected pixels
    """
    saliency_map = np.asarray(saliency_map)
    valid = saliency_map != 0
    if not np.any(valid):
        return np.zeros(saliency_map.shape, dtype=bool)

    if sign == 'abs':
        threshold = np.percentile(np.abs(saliency_map[valid]), percentile)
        return valid & (np.abs(saliency_map) >= threshold)

    if sign == 'positive':
        # Rank over every non-zero attribution so the positive and negative runs remove
        # the same number of pixels; clamp to strictly positive pixels in case fewer than
        # (100 - percentile)% of the attributions are positive.
        threshold = np.percentile(saliency_map[valid], percentile)
        return valid & (saliency_map > 0) & (saliency_map >= threshold)

    if sign == 'negative':
        threshold = np.percentile(saliency_map[valid], 100 - percentile)
        return valid & (saliency_map < 0) & (saliency_map <= threshold)

    raise ValueError(f"Unknown saliency sign '{sign}'; expected 'abs', 'positive' or 'negative'.")


def remove_high_saliency_pixels(images, saliency_maps, percentile=90, sign='abs'):
    """
    Remove pixels with highest saliency values from images.

    Args:
        images: Original images array
        saliency_maps: Saliency maps aligned with the given images array
        percentile: Percentile threshold for pixel removal (default: 90 for top 10%)
        sign: 'abs', 'positive' or 'negative'; see _saliency_selection_mask

    Returns:
        Modified images array with high-saliency pixels set to 0
    """
    images_modified = images.copy()

    for idx, saliency_map in enumerate(saliency_maps):
        images_modified[idx][_saliency_selection_mask(saliency_map, percentile, sign)] = 0

    return images_modified


def build_high_saliency_exclusion_masks(images, saliency_maps, percentile=90, sign='abs'):
    exclusion_masks = np.zeros_like(images, dtype='uint8')
    for idx, saliency_map in enumerate(saliency_maps):
        exclusion_masks[idx][_saliency_selection_mask(saliency_map, percentile, sign)] = 1
    return exclusion_masks


def generate_and_cache_saliency_maps(images, predicted_labels, saliency_map_generator: SaliencyMapGenerator,
                                      cache_path, description="Saliency maps", **kwargs):
    """
    Generate saliency maps for given indices with caching support.

    Args:
        images: Images to generate saliency maps for
        predicted_labels: Predicted labels for the images
        saliency_map_generator: SaliencyMapGenerator instance
        cache_path: Path to cache file (.npy)
        description: Description for progress bar
        **kwargs: Additional arguments to pass to generate_saliency_map

    Returns:
        Array of saliency maps
    """
    if os.path.exists(cache_path):
        return np.load(cache_path)

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    batch_size = max(1, int(kwargs.get('saliency_batch_size', 4)))
    n_steps = int(kwargs.get('saliency_n_steps', 32))
    saliency_maps = []
    for batch_start in tqdm(range(0, len(images), batch_size), desc=description):
        batch_end = min(batch_start + batch_size, len(images))
        saliency_maps.append(saliency_map_generator.generate_saliency_maps(
            images[batch_start:batch_end],
            predicted_labels[batch_start:batch_end],
            n_steps=n_steps,
            batch_size=batch_size))
    saliency_maps = np.concatenate(saliency_maps, axis=0) if saliency_maps else np.empty(images.shape, dtype='float32')
    np.save(cache_path, saliency_maps)
    return saliency_maps
