import json
import os

import numpy as np
import pandas as pd
import skimage.io as skio
import torch
from skimage.color import gray2rgba
from tqdm import tqdm

from radiomics_saliency.feature_extraction import generate_features
from radiomics_saliency.models import ExplanationWrapper
from radiomics_saliency.saliency import (
    SaliencyMapGenerator,
    build_high_saliency_exclusion_masks,
    generate_and_cache_saliency_maps,
)
from radiomics_saliency.training import train_and_evaluate_model
from radiomics_saliency.visualization import blend_image_with_mask, prepare_images_for_serialization


def _load_cached_saliency_maps(cache_path: str, description: str):
    if not os.path.exists(cache_path):
        raise FileNotFoundError(
            f'{description} cache not found at {cache_path}. Run with run_saliency=True '
            f'first to generate it, or point model_path at a directory that already has it.'
        )
    return np.load(cache_path)


def run_experiment(
        images: np.ndarray,
        labels: np.ndarray,
        indices_train: np.ndarray,
        indices_dev: np.ndarray,
        test_images: np.ndarray,
        test_labels: np.ndarray,
        device: torch.device,
        model_path: str, **kwargs):
    # images: (N, H, W), labels: (N, 1)
    # The texture-matrix and saliency-map routines key every column dimension off a single
    # width and several Rust routines assume height == width; non-square images would
    # silently mis-shape or panic deep in the extension. Fail loudly and early instead.
    for name, batch in (('train/dev', images), ('test', test_images)):
        if batch.shape[1] != batch.shape[2]:
            raise ValueError(
                f'{name} images must be square (height == width); got '
                f'{batch.shape[1]}x{batch.shape[2]}. The radiomics pipeline is square-only.')
    image_max_width = images.shape[1]
    test_ids = kwargs.pop('test_ids', None)
    run_saliency = kwargs.pop('run_saliency', True)
    run_roar = kwargs.pop('run_roar', True)
    run_metrics = kwargs.pop('run_metrics', True)
    extraction_parameters = [
        ('glcm', {'angle': 0, 'distance': 1, 'omit_zeros': True }),
        ('glcm', {'angle': 0, 'distance': 2, 'omit_zeros': True }),
        ('glcm', {'angle': 0, 'distance': 3, 'omit_zeros': True }),
        ('glcm', { 'angle': 90, 'distance': 1, 'omit_zeros': True }),
        ('glcm', { 'angle': 135, 'distance': 1, 'omit_zeros': True }),
        ('glrlm', {'omit_zeros': True}),
        ('glszm', {'omit_zeros': True, 'max_size': image_max_width}),
        ('gldm', {'alpha': 1, 'delta': 3, 'omit_zeros': True}),
        ('gldm', { 'alpha': 1, 'delta': 5, 'omit_zeros': True }),
        ('gldm', {'alpha': 2, 'delta': 3, 'omit_zeros': True}),
        ('gldm', { 'alpha': 2, 'delta': 5, 'omit_zeros': True }),
        ('ngtdm', {'delta': 1, 'omit_zeros': True}),
        ('ngtdm', {'delta': 2, 'omit_zeros': True}),
        ('ngtdm', { 'delta': 3, 'omit_zeros': True }),
    ]

    unique_labels = np.unique(np.concatenate([
        np.asarray(labels).reshape(-1),
        np.asarray(test_labels).reshape(-1)
    ]))

    # Step 1: Train initial model
    print('Training initial model...')
    model, model_trainer, features, test_features, m, s = train_and_evaluate_model(
        images, labels, indices_train, indices_dev, test_images, test_labels,
        extraction_parameters, device, model_path, None, unique_labels,
        kwargs['skip_training'], kwargs['model_type_name']
    )
    feature_count = features.shape[1]

    if not (run_saliency or run_roar or run_metrics):
        print('Skipping saliency generation, ROAR, and metrics steps as requested.')
        return

    # Step 2: Generate saliency maps for train and dev sets
    explanation_wrapper = ExplanationWrapper(model, extraction_parameters, m, s)
    saliency_map_generator = SaliencyMapGenerator(
        model, explanation_wrapper, extraction_parameters, device, m, s,
        attribution_mode=kwargs.get('attribution_mode', 'pairwise'),
        eg_samples=int(kwargs.get('eg_samples', 10)),
        n_steps=int(kwargs.get('saliency_n_steps', 20)),
        eg_baseline_chunk=int(kwargs.get('eg_baseline_chunk', 4)))
    # Expected-Gradients baseline pool built from real training images.
    saliency_map_generator.set_baseline_pool(
        images[indices_train], sample_size=int(kwargs.get('eg_baseline_pool', 64)))
    saliency_map_generator.set_profiling(kwargs.get('profile_saliency', True))
    saliency_map_generator.reset_profile_stats()

    predicted_labels_train, _, _ = model_trainer.eval_model_by_indices(indices_train)
    predicted_labels_dev, _, _ = model_trainer.eval_model_by_indices(indices_dev)
    predicted_labels_test, _, _ = model_trainer.eval_test_model()

    if run_saliency:
        print('Generating saliency maps for train and dev sets...')
        train_saliency_maps = generate_and_cache_saliency_maps(
            images[indices_train], predicted_labels_train, saliency_map_generator,
            f'{model_path}/maps/train_saliency_raw.npy', 'Train saliency maps', **kwargs
        )

        dev_saliency_maps = generate_and_cache_saliency_maps(
            images[indices_dev], predicted_labels_dev, saliency_map_generator,
            f'{model_path}/maps/dev_saliency_raw.npy', 'Dev saliency maps', **kwargs
        )

        test_saliency_maps = generate_and_cache_saliency_maps(
            test_images, predicted_labels_test, saliency_map_generator,
            f'{model_path}/maps/test_saliency_raw.npy', 'Test saliency maps', **kwargs
        )
        if kwargs.get('profile_saliency', True):
            with open(f'{model_path}/saliency_profile.json', 'w') as f:
                json.dump(saliency_map_generator.get_profile_stats(), f, indent=2)

        # Completeness diagnostic: how well the attributions satisfy the IG identity
        # sum(A) == g(x) - mean_baseline g(baseline) for the configured target scalar.
        completeness_sample = min(int(kwargs.get('completeness_sample', 16)), test_images.shape[0])
        if completeness_sample > 0:
            completeness_error = saliency_map_generator.compute_completeness(
                test_images[:completeness_sample], predicted_labels_test[:completeness_sample])
            print(f'Mean relative completeness error: {completeness_error:.4f} '
                  f'(mode={saliency_map_generator.attribution_mode}, n_steps={saliency_map_generator.default_n_steps}, '
                  f'eg_samples={saliency_map_generator.eg_samples})')
            with open(f'{model_path}/completeness.json', 'w') as f:
                json.dump({
                    'mean_relative_completeness_error': completeness_error,
                    'attribution_mode': saliency_map_generator.attribution_mode,
                    'n_steps': saliency_map_generator.default_n_steps,
                    'eg_samples': saliency_map_generator.eg_samples,
                    'sample_size': completeness_sample,
                }, f, indent=2)
    else:
        print('Skipping saliency map generation; loading cached saliency maps...')
        train_saliency_maps = _load_cached_saliency_maps(
            f'{model_path}/maps/train_saliency_raw.npy', 'Train saliency maps')
        dev_saliency_maps = _load_cached_saliency_maps(
            f'{model_path}/maps/dev_saliency_raw.npy', 'Dev saliency maps')
        test_saliency_maps = _load_cached_saliency_maps(
            f'{model_path}/maps/test_saliency_raw.npy', 'Test saliency maps')

    if not (run_roar or run_metrics):
        print('Skipping ROAR and metrics steps as requested.')
        return

    if not run_roar:
        print('Skipping ROAR (remove-and-retrain) step.')
    else:
        # Step 3: Remove top 10% pixels and retrain to assess accuracy change
        print('Removing top 10% pixels based on saliency maps...')

        exclusion_masks = np.zeros_like(images, dtype='uint8')
        exclusion_masks[indices_train] = build_high_saliency_exclusion_masks(
            images[indices_train], train_saliency_maps, percentile=90)
        exclusion_masks[indices_dev] = build_high_saliency_exclusion_masks(
            images[indices_dev], dev_saliency_maps, percentile=90)
        test_exclusion_masks = build_high_saliency_exclusion_masks(
            test_images, test_saliency_maps, percentile=90)
        images_modified = images.copy()
        images_modified[exclusion_masks == 1] = 0
        test_images_modified = test_images.copy()
        test_images_modified[test_exclusion_masks == 1] = 0
        images_modified_to_save = prepare_images_for_serialization(images_modified)
        test_images_modified_to_save = prepare_images_for_serialization(test_images_modified)

        # Save modified images to disk for inspection and reproducibility
        os.makedirs(f'{model_path}/retrained/modified_train_dev_images', exist_ok=True)
        os.makedirs(f'{model_path}/retrained/modified_test_images', exist_ok=True)
        np.save(f'{model_path}/retrained/train_dev_exclusion_masks.npy', exclusion_masks)
        np.save(f'{model_path}/retrained/test_exclusion_masks.npy', test_exclusion_masks)
        np.save(f'{model_path}/retrained/modified_train_dev_images.npy', images_modified_to_save)
        np.save(f'{model_path}/retrained/modified_test_images.npy', test_images_modified_to_save)
        # Also save individual PNG visualizations
        for ii in range(images_modified_to_save.shape[0]):
            img = images_modified_to_save[ii]
            img_rgba = gray2rgba(img)
            skio.imsave(f'{model_path}/retrained/modified_train_dev_images/{ii}_modified.png', img_rgba)
        for ii in range(test_images_modified_to_save.shape[0]):
            img = test_images_modified_to_save[ii]
            img_rgba = gray2rgba(img)
            skio.imsave(f'{model_path}/retrained/modified_test_images/{ii}_modified.png', img_rgba)

        print('Retraining model with modified images to assess accuracy change...')
        os.makedirs(f'{model_path}/retrained', exist_ok=True)
        model_retrained, model_trainer_retrained, _, _, _, _ = train_and_evaluate_model(
            images, labels, indices_train, indices_dev, test_images, test_labels,
            extraction_parameters, device, f'{model_path}/retrained', feature_count,
            unique_labels, kwargs['skip_training'], kwargs['model_type_name'],
            masks=exclusion_masks, test_masks=test_exclusion_masks
        )

        # Save retrained model metrics for comparison
        print('Retrained model evaluation completed. Metrics saved to retrained/ directory.')

    if not run_metrics:
        print('Skipping activation map, visualization, and metrics steps.')
        return

    # Step 4: Continue with original model for saliency map generation
    print('Generating test set saliency and activation maps using original model...')
    medians, _ = features[indices_train].median(dim=0)

    os.makedirs(f'{model_path}/maps/saliency', exist_ok=True)
    os.makedirs(f'{model_path}/maps/activation_1', exist_ok=True)
    os.makedirs(f'{model_path}/maps/activation_2', exist_ok=True)
    os.makedirs(f'{model_path}/maps/activation_3', exist_ok=True)

    if os.path.exists(f'{model_path}/maps/activation_raw.npy'):
        test_activation_maps = np.load(f'{model_path}/maps/activation_raw.npy')
    else:
        test_activation_maps = []
        features_data = []
        for i, image in tqdm(enumerate(test_images), desc='Activation maps', total=len(test_images)):
            activation_maps, top_features = saliency_map_generator.generate_activation_map(
                image, test_features[i], medians, predicted_labels_test[i], 3
            )
            test_activation_maps.append(activation_maps)
            image_id = test_ids[i] if test_ids is not None else i
            features_data.append([image_id] + top_features)
        test_activation_maps = np.asarray(test_activation_maps)
        features_data_df = pd.DataFrame(data=features_data, columns=['ImageId', 'Top1', 'Top2', 'Top3'])
        features_data_df.to_csv(f'{model_path}/activation_map_features_data.csv', sep=';')
        np.save(f'{model_path}/maps/activation_raw.npy', test_activation_maps)

    # Step 5: Save visualizations
    print('Saving saliency map visualizations...')

    for i in range(test_images.shape[0]):
        image = gray2rgba(test_images[i])
        pred_label = int(predicted_labels_test[i])
        true_label = int(np.asarray(test_labels[i]).reshape(-1)[0])

        blend_image_with_mask(
            f'{model_path}/maps/saliency/{i}_label_{pred_label}_true_{true_label}.png',
            image, test_saliency_maps[i])
        blend_image_with_mask(
            f'{model_path}/maps/activation_1/{i}_label_{pred_label}_true_{true_label}.png',
            image, test_activation_maps[i][0])
        blend_image_with_mask(
            f'{model_path}/maps/activation_2/{i}_label_{pred_label}_true_{true_label}.png',
            image, test_activation_maps[i][1])
        blend_image_with_mask(
            f'{model_path}/maps/activation_3/{i}_label_{pred_label}_true_{true_label}.png',
            image, test_activation_maps[i][2])

    def generate_activation_map_for_pertrubation(model, inputs, targets):
        features = generate_features(inputs, extraction_parameters, False)
        activation_map, _ = saliency_map_generator.generate_activation_map(inputs[0], features[0], medians, int(targets[0]))
        activation_map = np.asarray(activation_map)[0]
        activation_map = activation_map.reshape((-1,) + activation_map.shape).astype('float64')
        return activation_map

    def generate_saliency_map_for_pertrubation(model, inputs, targets):
        saliency_map = saliency_map_generator.generate_saliency_map(inputs[0].astype('uint8'), int(targets[0]))
        return saliency_map.reshape((1,) + saliency_map.shape)

    # Step 6: Compute faithfulness metrics
    print('Computing faithfulness metrics...')
    smm_metrics = saliency_map_generator.compute_map_metrics(
        test_images, test_saliency_maps, test_labels,
        generate_saliency_map_for_pertrubation)
    amm_metrics = saliency_map_generator.compute_map_metrics(
        test_images, test_activation_maps[:, 0], test_labels,
        generate_activation_map_for_pertrubation)
    amm_metrics2 = saliency_map_generator.compute_map_metrics(
        test_images, test_activation_maps[:, 1], test_labels,
        generate_activation_map_for_pertrubation)
    amm_metrics3 = saliency_map_generator.compute_map_metrics(
        test_images, test_activation_maps[:, 2], test_labels,
        generate_activation_map_for_pertrubation)

    columns = [
        'ImageId',
        'SMM_FaithfulnessCorrelation',
        'SMM_Sparseness',
        'SMM_RelativeInputStability',
        'AMM1_FaithfulnessCorrelation',
        'AMM1_Sparseness',
        'AMM1_RelativeInputStability',
        'AMM2_FaithfulnessCorrelation',
        'AMM2_Sparseness',
        'AMM2_RelativeInputStability',
        'AMM3_FaithfulnessCorrelation',
        'AMM3_Sparseness',
        'AMM3_RelativeInputStability',
    ]

    faithfulness_metrics_data = np.column_stack([
            np.asarray(test_ids if test_ids is not None else np.arange(test_images.shape[0])),
        np.asarray(smm_metrics['faithfulnessCorrelation']),
        np.asarray(smm_metrics['sparseness']),
        np.asarray(smm_metrics['relativeInputStability']),
        np.asarray(amm_metrics['faithfulnessCorrelation']),
        np.asarray(amm_metrics['sparseness']),
        np.asarray(amm_metrics['relativeInputStability']),
        np.asarray(amm_metrics2['faithfulnessCorrelation']),
        np.asarray(amm_metrics2['sparseness']),
        np.asarray(amm_metrics2['relativeInputStability']),
        np.asarray(amm_metrics3['faithfulnessCorrelation']),
        np.asarray(amm_metrics3['sparseness']),
        np.asarray(amm_metrics3['relativeInputStability']),
    ])

    faithfulness_metrics_df = pd.DataFrame(columns=columns, data=faithfulness_metrics_data)
    faithfulness_metrics_df.to_csv(f'{model_path}/faithfulness.csv', sep=';')
