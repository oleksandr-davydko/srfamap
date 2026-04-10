import os
import time

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["RUST_BACKTRACE"] = "full"
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

import sys

import torch
import numpy as np
import radiomics_rs as r
import argparse
import SimpleITK as sitk
import json
import torch.nn.functional as F
from skimage.util import img_as_ubyte
from skimage import img_as_float
import skimage.io as skio
from skimage.color import gray2rgba
from captum.attr import IntegratedGradients
from radiomics.featureextractor import RadiomicsFeatureExtractor
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from sklearn.metrics import matthews_corrcoef, f1_score, accuracy_score, classification_report
from functools import partial, reduce
import quantus
import pandas as pd

from nn.pytorch.radiomics.glcm import GlcmFeatures
from nn.pytorch.radiomics.gldm import GldmFeatures
from nn.pytorch.radiomics.glrlm import GlrlmFeatures
from nn.pytorch.radiomics.glszm import GlszmFeatures
from nn.pytorch.radiomics.ngtdm import NgtdmFeatures
from nn.pytorch.radiomics.radiomic_mlp import StatisticalRadiomicMLP, StatisticalRadiomicTransformer

from data.datasets import load_dataset_with_heldout_test


image_max_width = 64

method_to_layer_dict = {
    'glcm': GlcmFeatures,
    'glrlm': lambda: GlrlmFeatures(64 * 64),
    'glszm': GlszmFeatures,
    'gldm': GldmFeatures,
    'ngtdm': NgtdmFeatures,
}


def get_mapping_method(method: str, **kwargs):
    if method == 'glcm':
        return partial(r.glcm_saliency_map, angle=kwargs['angle'], distance=kwargs['distance'],
                       omit_zeros=kwargs['omit_zeros'])
    elif method == 'glrlm':
        return partial(r.glrlm_saliency_map, omit_zeros=kwargs['omit_zeros'])
    elif method == 'glszm':
        return partial(r.glszm_saliency_map, omit_zeros=kwargs['omit_zeros'])
    elif method == 'gldm':
        return partial(r.gldm_saliency_map, alpha=kwargs['alpha'], delta=kwargs['delta'],
                       omit_zeros=kwargs['omit_zeros'])
    elif method == 'ngtdm':
        return partial(r.ngtdm_saliency_map, delta=kwargs['delta'], omit_zeros=kwargs['omit_zeros'])
    else:
        raise Exception('Method {} not implemented'.format(method))


def get_matrix_func_by_method(method: str, **kwargs):
    if method == 'glcm':
        return partial(r.glcm, angle=kwargs['angle'], distance=kwargs['distance'], omit_zeros=kwargs['omit_zeros'])
    elif method == 'glrlm':
        return partial(r.glrlm, omit_zeros=kwargs['omit_zeros'])
    elif method == 'glszm':
        return partial(r.glszm, omit_zeros=kwargs['omit_zeros'], max_size=kwargs['max_size'])
    elif method == 'gldm':
        return partial(r.gldm, alpha=kwargs['alpha'], delta=kwargs['delta'], omit_zeros=kwargs['omit_zeros'])
    elif method == 'ngtdm':
        return partial(r.ngtdm, delta=kwargs['delta'], omit_zeros=kwargs['omit_zeros'])
    else:
        raise Exception('Method {} not implemented'.format(method))


def prepare_radiomics_rs_image(image: np.ndarray) -> np.ndarray:
    image_array = np.asarray(image)
    image_array = np.squeeze(image_array)
    if image_array.ndim != 2:
        raise ValueError(f'Expected a 2D image for radiomics_rs, got shape {image_array.shape}')
    return np.ascontiguousarray(image_array.astype(np.uint8, copy=False))


def prepare_radiomics_rs_attributions(attributions: np.ndarray) -> np.ndarray:
    attribution_array = np.asarray(attributions)
    return np.ascontiguousarray(attribution_array.astype(np.float32, copy=False))


def index_to_feature_name(method: str, index: int):
    return ({
        'glcm': {
            0: 'Autocorrelation',
            1: 'ClusterProminence',
            2: 'ClusterShade',
            3: 'ClusterTendency',
            4: 'Contrast',
            5: 'JointAverage',
            6: 'Correlation',
            7: 'DifferenceAverage',  # 7
            8: 'DifferenceEntropy',
            9: 'DifferenceVariance',
            10: 'JointEnergy',
            11: 'JointEntropy',
            12: 'Imc1',  # 12
            13: 'Imc2',
            14: 'Idm',
            15: 'MCC',
            16: 'Idmn',
            17: 'Id',
            18: 'Idn',
            19: 'InverseVariance',  # 18
            20: 'MaximumProbability',
            21: 'SumAverage',
            22: 'SumEntropy',
            23: 'SumSquares'
        },
        'glrlm': {
            0: 'ShortRunEmphasis',
            1: 'LongRunEmphasis',
            2: 'GrayLevelNonUniformity',
            3: 'GrayLevelNonUniformityNormalized',
            4: 'RunLengthNonUniformity',
            5: 'RunLengthNonUniformityNormalized',
            6: 'RunPercentage',
            7: 'GrayLevelVariance',
            8: 'RunVariance',
            9: 'RunEntropy',
            10: 'LowGrayLevelRunEmphasis',
            11: 'HighGrayLevelRunEmphasis',
            12: 'ShortRunLowGrayLevelEmphasis',
            13: 'ShortRunHighGrayLevelEmphasis',
            14: 'LongRunLowGrayLevelEmphasis',
            15: 'LongRunHighGrayLevelEmphasis'
        },
        'glszm': {
            0: 'SmallAreaEmphasis',
            1: 'LargeAreaEmphasis',
            2: 'GrayLevelNonUniformity',
            3: 'GrayLevelNonUniformityNormalized',
            4: 'SizeZoneNonUniformity',
            5: 'SizeZoneNonUniformityNormalized',
            6: 'ZonePercentage',
            7: 'GrayLevelVariance',
            8: 'ZoneVariance',
            9: 'ZoneEntropy',
            10: 'LowGrayLevelZoneEmphasis',
            11: 'HighGrayLevelZoneEmphasis',
            12: 'SmallAreaLowGrayLevelEmphasis',
            13: 'SmallAreaHighGrayLevelEmphasis',
            14: 'LargeAreaLowGrayLevelEmphasis',
            15: 'LargeAreaHighGrayLevelEmphasis'
        },
        'gldm': {
            0: 'SmallDependenceEmphasis',
            1: 'LargeDependenceEmphasis',
            2: 'GrayLevelNonUniformity',
            3: 'DependenceNonUniformity',
            4: 'DependenceNonUniformityNormalized',
            5: 'GrayLevelVariance',
            6: 'DependenceVariance',
            7: 'DependenceEntropy',
            8: 'LowGrayLevelEmphasis',
            9: 'HighGrayLevelEmphasis',
            10: 'SmallDependenceLowGrayLevelEmphasis',
            11: 'SmallDependenceHighGrayLevelEmphasis',
            12: 'LargeDependenceLowGrayLevelEmphasis',
            13: 'LargeDependenceHighGrayLevelEmphasis'
        },
        'ngtdm': {
            0: 'Coarseness',
            1: 'Contrast',
            2: 'Busyness',
            3: 'Complexity',
            4: 'Strength'
        }
    }[method][index])


def get_method_features_list(method: str):
    return ({
        'glcm': {
            0: 'Autocorrelation',
            1: 'ClusterProminence',
            2: 'ClusterShade',
            3: 'ClusterTendency',
            4: 'Contrast',
            5: 'JointAverage',
            6: 'Correlation',
            7: 'DifferenceAverage',  # 7
            8: 'DifferenceEntropy',
            9: 'DifferenceVariance',
            10: 'JointEnergy',
            11: 'JointEntropy',
            12: 'Imc1',  # 12
            13: 'Imc2',
            14: 'Idm',
            15: 'MCC',
            16: 'Idmn',
            17: 'Id',
            18: 'Idn',
            19: 'InverseVariance',  # 18
            20: 'MaximumProbability',
            21: 'SumAverage',
            22: 'SumEntropy',
            23: 'SumSquares'
        },
        'glrlm': {
            0: 'ShortRunEmphasis',
            1: 'LongRunEmphasis',
            2: 'GrayLevelNonUniformity',
            3: 'GrayLevelNonUniformityNormalized',
            4: 'RunLengthNonUniformity',
            5: 'RunLengthNonUniformityNormalized',
            6: 'RunPercentage',
            7: 'GrayLevelVariance',
            8: 'RunVariance',
            9: 'RunEntropy',
            10: 'LowGrayLevelRunEmphasis',
            11: 'HighGrayLevelRunEmphasis',
            12: 'ShortRunLowGrayLevelEmphasis',
            13: 'ShortRunHighGrayLevelEmphasis',
            14: 'LongRunLowGrayLevelEmphasis',
            15: 'LongRunHighGrayLevelEmphasis'
        },
        'glszm': {
            0: 'SmallAreaEmphasis',
            1: 'LargeAreaEmphasis',
            2: 'GrayLevelNonUniformity',
            3: 'GrayLevelNonUniformityNormalized',
            4: 'SizeZoneNonUniformity',
            5: 'SizeZoneNonUniformityNormalized',
            6: 'ZonePercentage',
            7: 'GrayLevelVariance',
            8: 'ZoneVariance',
            9: 'ZoneEntropy',
            10: 'LowGrayLevelZoneEmphasis',
            11: 'HighGrayLevelZoneEmphasis',
            12: 'SmallAreaLowGrayLevelEmphasis',
            13: 'SmallAreaHighGrayLevelEmphasis',
            14: 'LargeAreaLowGrayLevelEmphasis',
            15: 'LargeAreaHighGrayLevelEmphasis'
        },
        'gldm': {
            0: 'SmallDependenceEmphasis',
            1: 'LargeDependenceEmphasis',
            2: 'GrayLevelNonUniformity',
            3: 'DependenceNonUniformity',
            4: 'DependenceNonUniformityNormalized',
            5: 'GrayLevelVariance',
            6: 'DependenceVariance',
            7: 'DependenceEntropy',
            8: 'LowGrayLevelEmphasis',
            9: 'HighGrayLevelEmphasis',
            10: 'SmallDependenceLowGrayLevelEmphasis',
            11: 'SmallDependenceHighGrayLevelEmphasis',
            12: 'LargeDependenceLowGrayLevelEmphasis',
            13: 'LargeDependenceHighGrayLevelEmphasis'
        },
        'ngtdm': {
            0: 'Coarseness',
            1: 'Contrast',
            2: 'Busyness',
            3: 'Complexity',
            4: 'Strength'
        }
    }[method])






def get_feature_vector(method, module, image, mask=None):
    mats = []
    for idx, im in enumerate(image):
        current_mask = None if mask is None else mask[idx].astype('uint8')
        if current_mask is None:
            mat = method(im.astype('uint8')).astype('float32')
        else:
            mat = method(im.astype('uint8'), mask=current_mask).astype('float32')
        mats.append(mat)
    try:
        mats = torch.from_numpy(np.asarray(mats))
        return module(mats.to(device))
    except ValueError:
        mats = torch.concatenate(
            [module(torch.from_numpy(mat.reshape((1,) + mat.shape)).to(device)).unsqueeze(0) for mat in mats])
    return mats


def get_texture_matrices(image, feature_configs, mask=None):
    matrices = []
    prepared_image = prepare_radiomics_rs_image(image)
    prepared_mask = None if mask is None else np.ascontiguousarray(np.asarray(mask).astype(np.uint8, copy=False))
    for method, parameters in feature_configs:
        first_order_features_method = get_matrix_func_by_method(method, **parameters)
        try:
            if prepared_mask is None:
                matrices.append(first_order_features_method(prepared_image).astype('float32'))
            else:
                matrices.append(first_order_features_method(prepared_image, mask=prepared_mask).astype('float32'))
        except:
            print(method)
    return matrices


def generate_features(images, feature_configs, use_progress=False, masks=None):
    feature_blocks = []
    with torch.no_grad():
        for method, parameters in feature_configs:
            first_order_features_method = get_matrix_func_by_method(method, **parameters)
            high_order_features_module = method_to_layer_dict[method]().to(device)
            features = []
            image_iterator = enumerate(tqdm(images) if use_progress else images)
            for idx, image in image_iterator:
                image = image.reshape((1,) + image.shape)
                current_mask = None if masks is None else masks[idx].reshape((1,) + masks[idx].shape)
                if image.max() == 0:
                    features.append(torch.zeros((image.shape[0], get_high_order_feature_number(method))).to(device))
                else:
                    fv = get_feature_vector(first_order_features_method, high_order_features_module, image, mask=current_mask).to(device)
                    features.append(fv)
            features = torch.concatenate(features, dim=0)
            feature_blocks.append(features)
        features_together = torch.cat(feature_blocks, dim=1)
    return features_together


class ExplanationWrapper(torch.nn.Module):
    def __init__(self, model, feature_configs):
        super(ExplanationWrapper, self).__init__()
        self.model = model
        self.feature_configs = feature_configs
        self.texture_layers = []
        self.matrix_methods = []
        for method, parameters in feature_configs:
            first_order_features_method = get_matrix_func_by_method(method, **parameters)
            high_order_features_module = method_to_layer_dict[method]()
            self.texture_layers.append(high_order_features_module)
            self.matrix_methods.append(first_order_features_method)

    # def __call__(self, *args, **kwargs):
    #     inp = args[0]
    #     features_together = generate_features(inp[:, 0].cpu().detach().numpy(), self.feature_configs, False)
    #     return F.softmax(self.model(features_together.float().to(device)), dim=1)
    
    def forward(self, inp):
        #inp = args[0]
        features_together = generate_features(inp.cpu().detach().numpy(), self.feature_configs, False)
        probs = self.model(features_together.float().to(device))
        return probs#F.softmax(probs, dim=1)
    


def get_feature_number(method):
    if method == 'glcm':
        return 256 * 256
    elif method == 'glrlm':
        return 256 * image_max_width
    elif method == 'glszm':
        return 256 * image_max_width
    elif method == 'gldm':
        return 256 * 256
    elif method == 'ngtdm':
        return 256 * 3
    else:
        raise Exception('Method {} not implemented'.format(method))


def get_high_order_feature_number(method):
    if method == 'glcm':
        return 24
    elif method == 'glrlm':
        return 16
    elif method == 'glszm':
        return 16
    elif method == 'gldm':
        return 14
    elif method == 'ngtdm':
        return 5
    else:
        raise Exception('Method {} not implemented'.format(method))


def get_matrix_shape(method):
    if method == 'glcm':
        return (256, 256)
    elif method == 'glrlm':
        return (256, image_max_width)
    elif method == 'glszm':
        return (256, image_max_width)
    elif method == 'gldm':
        return (256, 256)
    elif method == 'ngtdm':
        return (256, 3)
    else:
        raise Exception('Method {} not implemented'.format(method))


class RadiomicsMLPExplanationWrapper(torch.nn.Module):
    def __init__(self, model, feature_configs, mean, std, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = model
        self.feature_configs = feature_configs
        self.texture_layers = torch.nn.ModuleList()
        self.layer_offsets = []
        self.register_buffer('mean', mean.to(device))
        self.register_buffer('std', std.to(device))
        offset = 0
        layer_id = 0
        for method, parameters in feature_configs:
            high_order_features_module = method_to_layer_dict[method]()
            self.texture_layers.append(high_order_features_module)
            self.layer_offsets.append(offset)
            offset += get_feature_number(method)
            layer_id += 1

    def forward(self, texture_matrices):
        if texture_matrices.dim() == 1:
            texture_matrices = texture_matrices.unsqueeze(0)
        features = []
        for layer, offset, config in zip(self.texture_layers, self.layer_offsets, self.feature_configs):
            ft_num = get_feature_number(config[0])
            ms = (-1, 1) + get_matrix_shape(config[0])
            ft_inp = texture_matrices[:, offset:offset + ft_num].reshape(ms)
            features.append(layer(ft_inp))
        features = torch.cat(features, dim=1)
        features -= self.mean
        features /= self.std
        return self.model(features)


class ModelTrainer:
    def __init__(self,
                 model: torch.nn.Module,
                 device: torch.device,
                 texture_statistics_matrices,
                 labels,
                 train_indices,
                 val_indices,
                 test_texture_statistics_matrices,
                 test_labels,
                 storage_path: str,
                 batch_size: int):
        self.model = model
        self.device = device
        self.data = texture_statistics_matrices
        self.labels = np.asarray(labels).reshape(-1)
        self.train_indices = train_indices
        self.val_indices = val_indices
        self.test_data = test_texture_statistics_matrices
        self.test_labels = np.asarray(test_labels).reshape(-1)
        self.current_epoch = 1
        self.best_metric = 0
        self.patience_value = 40
        self.storage_path = storage_path
        self.model_metrics = {}
        self.batch_size = batch_size

    def eval_model(self, eval_data, eval_labels):
        predicted_labels = []
        self.model.eval()
        with torch.no_grad():
            for example in tqdm(eval_data):
                texture_matrices = example.reshape(1, -1).to(self.device)
                logits = F.softmax(self.model.forward(texture_matrices), dim=1)
                predicted_labels += torch.argmax(logits, dim=1).cpu().detach().numpy().tolist()
        eval_labels = np.asarray(eval_labels).reshape(-1)
        mcc = matthews_corrcoef(predicted_labels, eval_labels)
        f1_scores = f1_score(predicted_labels, eval_labels, average=None)
        f1_avg = f1_score(predicted_labels, eval_labels, average='weighted')
        epoch_accuracy = accuracy_score(predicted_labels, eval_labels)
        metrics_dict = {
            "mcc": mcc,
            "f1": {k: v for (k, v) in zip(range(1, len(f1_scores) + 1), f1_scores)},
            "f1_avg": f1_avg,
            "accuracy": epoch_accuracy,
        }
        return predicted_labels, metrics_dict, classification_report(predicted_labels, eval_labels)

    def eval_model_by_indices(self, eval_indices: np.ndarray):
        return self.eval_model(self.data[eval_indices], self.labels[eval_indices])

    def eval_test_model(self):
        return self.eval_model(self.test_data, self.test_labels)

    def train_step(self, optimizer, criterion):
        self.model.train()
        print(f'Train epoch {self.current_epoch + 1}')
        epoch_losses = []
        train_preds = []
        with tqdm(total=len(self.train_indices)) as progress:
            batches = np.array_split(self.train_indices, len(self.train_indices) // self.batch_size)
            for i in batches:
                data = self.data[i]
                texture_matrices = data.to(self.device).reshape(-1, data.shape[1])
                label = torch.as_tensor(self.labels[i], dtype=torch.long, device=self.device).reshape(-1)
                optimizer.zero_grad()
                logits = F.softmax(self.model.forward(texture_matrices), dim=1)
                prediction = torch.argmax(logits, dim=1)
                train_preds += prediction.type(torch.LongTensor).cpu().detach().numpy().tolist()
                loss = criterion(logits, label)
                loss.backward()
                optimizer.step()
                epoch_losses.append(loss.item())
                progress.set_postfix(loss=np.mean(epoch_losses), epoch=self.current_epoch, refresh=False)
                progress.update(len(i))

    def train_model(self):
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)
        epochs_without_improvement = 0
        while True:
            self.train_step(optimizer, criterion)
            print(f'Development set evaluation epoch {self.current_epoch}')
            _, val_metrics, val_report = self.eval_model_by_indices(self.val_indices)
            print(val_report)
            epoch_matthews = val_metrics['mcc']
            print(f'{epoch_matthews}')
            if epoch_matthews > self.best_metric:
                print(f'Epoch {self.current_epoch} has MCC improvement.')
                torch.save(self.model.state_dict(), f'{self.storage_path}/weights.pth')
                self.best_metric = epoch_matthews
                self.model_metrics['dev'] = val_metrics
                with open(f'{self.storage_path}/validation.json', 'w') as f:
                    f.write(json.dumps(val_metrics))
                epochs_without_improvement = 0
            else:
                epochs_without_improvement = epochs_without_improvement + 1
            if epochs_without_improvement >= self.patience_value:
                break
            self.current_epoch = self.current_epoch + 1
        self.model.load_state_dict(torch.load(f'{self.storage_path}/weights.pth'))
        _, test_metrics, test_report = self.eval_test_model()
        self.model_metrics['test'] = test_metrics
        with open(f'{self.storage_path}/test.json', 'w') as f:
            f.write(json.dumps(test_metrics))
        print(f'Test set evaluation')
        print(test_report)


class RadiomicMatricesVectorizer:
    def __init__(self, feature_configs):
        self.feature_configs = feature_configs
        self.layer_offsets = []
        offset = 0
        for method, parameters in feature_configs:
            self.layer_offsets.append(offset)
            offset += get_feature_number(method)

    def transform(self, data):
        return torch.cat([torch.from_numpy(x.astype('float32')).reshape(-1) for x in data]).reshape(1, -1)

    def transform_batch(self, batch_data):
        return torch.cat([
            self.transform(data)
            for data in batch_data
        ], dim=0)

    def inverse_transform(self, data):
        assert len(data.shape) == 2
        matrices = []
        for offset, cfg in zip(self.layer_offsets, self.feature_configs):
            method, _ = cfg
            ft_num = get_feature_number(method)
            ms = (-1,) + get_matrix_shape(method)
            ft_inp = data[:, offset:offset + ft_num].reshape(ms)
            matrices.append(ft_inp)
        return matrices


class SaliencyMapGenerator:
    def __init__(self, model: torch.nn.Module, explanation_wrapper, feature_configs, device, mean, std):
        self.model = model
        self.device = device
        self.radiomics_features_extractor = RadiomicsFeatureExtractor()
        self.wrapped_model = explanation_wrapper
        self.feature_configs = feature_configs
        self.wrapped_mlp = RadiomicsMLPExplanationWrapper(model, feature_configs, mean, std).to(device)
        self.integrated_gradients = IntegratedGradients(self.wrapped_mlp)
        self.class_model_integrated_gradients = IntegratedGradients(self.model)
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

    def generate_saliency_maps(self, images: np.ndarray, targets, n_steps: int = 5, batch_size: int = 4, **kwargs):
        results = []
        target_values = np.asarray(targets).reshape(-1)
        batch_size = max(1, int(batch_size))

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
            texture_properties_attributions = self.integrated_gradients.attribute(
                texture_matrices_vectorized,
                target=batch_targets,
                n_steps=n_steps)
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
              n_steps=kwargs.get('n_steps', 5),
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
            image_global_features.reshape(1, -1).to(device),
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
        faithfulness_correlation = quantus.FaithfulnessCorrelation(
            nr_runs=10,
            subset_size=1,
            perturb_baseline="black",
            similarity_func=quantus.similarity_func.correlation_pearson,
            abs=False,
            normalise=False,
            aggregate_func=np.mean,
            return_aggregate=True,
        )
        relative_input_stability = quantus.RelativeInputStability(
            nr_samples=2
        )
        sparseness = quantus.Sparseness()
        self.wrapped_model.eval()
        width, height = images.shape[1:]

        metrics_list = [
            ('faithfulnessCorrelation', faithfulness_correlation),
            ('relativeInputStability', relative_input_stability),
            ('sparseness', sparseness)
        ]

        for i in tqdm(range(images.shape[0])):
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


def blend_image_with_mask(filename: str, image: np.ndarray, mask: np.ndarray):
    # Ensure image is RGB (drop alpha if present)
    img = image.copy()
    if img.ndim == 2:
        # grayscale -> convert to RGB
        img = gray2rgba(img)
    if img.shape[2] == 4:
        img = img[:, :, :3]

    # convert to float [0,1]
    img_f = img_as_float(img)
    # treat positive values as red, negative as blue; zeros transparent
    m = mask.astype('float32')
    eps = np.finfo(float).eps
    pos = np.clip(m, 0.0, None)
    neg = np.clip(-m, 0.0, None)
    max_pos = pos.max() if pos.size else 0.0
    max_neg = neg.max() if neg.size else 0.0
    pos_n = pos / (max_pos + eps)
    neg_n = neg / (max_neg + eps)

    # per-pixel magnitude (0 for zeros -> fully transparent)
    mask_mag = np.clip(pos_n + neg_n, 0.0, 1.0)

    # overlay colors
    overlay_pos = np.stack([pos_n, np.zeros_like(pos_n), np.zeros_like(pos_n)], axis=-1)
    overlay_neg = np.stack([np.zeros_like(neg_n), np.zeros_like(neg_n), neg_n], axis=-1)
    overlay = overlay_pos + overlay_neg

    alpha = 0.7
    # alpha blending where alpha is scaled by mask magnitude; zeros remain original
    blended = (1.0 - (alpha * mask_mag)[..., None]) * img_f + (alpha * mask_mag)[..., None] * overlay

    # save as uint8 image
    skio.imsave(filename, img_as_ubyte(np.clip(blended, 0.0, 1.0)))


def get_model(model_type_name: str, device, input_size, n_classes=2):
    if model_type_name == 'mlp':
        return StatisticalRadiomicMLP(device, input_size, n_classes)
    elif model_type_name == 'transformer':
        return StatisticalRadiomicTransformer(device, 175)


def train_and_evaluate_model(images, labels, indices_train, indices_dev, test_images,
                              test_labels, extraction_parameters, device, model_path,
                              feature_count, unique_labels, skip_training,
                              model_type_name, masks=None, test_masks=None):
    """
    Train a model on the given images and return the model, trainer, features, and statistics.
    """
    # Generate or load features
    if (os.path.exists(f'{model_path}/features_cache.pt') and
            os.path.exists(f'{model_path}/test_features_cache.pt') and
            os.path.exists(f'{model_path}/mean_cache.pt') and
            os.path.exists(f'{model_path}/std_cache.pt')):
        features = torch.load(f'{model_path}/features_cache.pt')
        test_features = torch.load(f'{model_path}/test_features_cache.pt')
        m = torch.load(f'{model_path}/mean_cache.pt')
        s = torch.load(f'{model_path}/std_cache.pt')
    else:
        with torch.no_grad():
            features = generate_features(images, extraction_parameters, True, masks=masks)
            test_features = generate_features(test_images, extraction_parameters, True, masks=test_masks)
            m = features[indices_train].mean(0, keepdim=True)
            s = features[indices_train].std(0, unbiased=False, keepdim=True)
            features -= m
            features /= s
            test_features -= m
            test_features /= s
            torch.save(features, f'{model_path}/features_cache.pt')
            torch.save(test_features, f'{model_path}/test_features_cache.pt')
            torch.save(m, f'{model_path}/mean_cache.pt')
            torch.save(s, f'{model_path}/std_cache.pt')
    feature_count = features.shape[1]
    # Create and train model
    model = get_model(model_type_name, device, feature_count, len(unique_labels))
    model.to(device).to(torch.float)
    model_trainer = ModelTrainer(model, device, features, labels, indices_train, indices_dev, 
                                 test_features, test_labels, model_path, 60)
    
    if not skip_training:
        model_trainer.train_model()
    else:
        model_trainer.model.load_state_dict(torch.load(f'{model_path}/weights.pth'))
    
    return model, model_trainer, features, test_features, m, s


def remove_high_saliency_pixels(images, saliency_maps, percentile=90):
    """
    Remove pixels with highest saliency values from images.
    
    Args:
        images: Original images array
        saliency_maps: Saliency maps aligned with the given images array
        percentile: Percentile threshold for pixel removal (default: 90 for top 10%)
    
    Returns:
        Modified images array with high-saliency pixels set to 0
    """
    images_modified = images.copy()
    
    for idx, saliency_map in enumerate(saliency_maps):
        saliency_abs = np.abs(saliency_map)
        if np.any(saliency_abs > 0):
            threshold = np.percentile(saliency_abs[saliency_abs > 0], percentile)
            images_modified[idx][saliency_abs >= threshold] = 0
    
    return images_modified


def prepare_image_for_serialization(image):
    if image.dtype == np.uint8:
        return image
    if np.issubdtype(image.dtype, np.floating):
        image = np.nan_to_num(image, nan=0.0, posinf=255.0, neginf=0.0)
        if image.size == 0:
            return image.astype('uint8')
        if image.min() >= 0.0 and image.max() <= 1.0:
            return img_as_ubyte(image)
    return np.clip(image, 0, 255).astype('uint8')


def prepare_images_for_serialization(images):
    return np.asarray([prepare_image_for_serialization(image) for image in images], dtype='uint8')


def build_high_saliency_exclusion_masks(images, saliency_maps, percentile=90):
    exclusion_masks = np.zeros_like(images, dtype='uint8')
    for idx, saliency_map in enumerate(saliency_maps):
        saliency_abs = np.abs(saliency_map)
        if np.any(saliency_abs > 0):
            threshold = np.percentile(saliency_abs[saliency_abs > 0], percentile)
            exclusion_masks[idx][saliency_abs >= threshold] = 1
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
    n_steps = int(kwargs.get('saliency_n_steps', 5))
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
    image_max_width = images.shape[1]
    test_ids = kwargs.pop('test_ids', None)
    extraction_parameters = [
        ('glcm', {'angle': 0, 'distance': 1, 'omit_zeros': True }),
        ('glcm', {'angle': 0, 'distance': 2, 'omit_zeros': True }),
        ('glcm', {'angle': 0, 'distance': 3, 'omit_zeros': True }),
        ('glcm', { 'angle': 90, 'distance': 1, 'omit_zeros': True }),
        ('glcm', { 'angle': 135, 'distance': 1, 'omit_zeros': True }),
        ('glrlm', {'omit_zeros': True}),
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
    
    # Step 2: Generate saliency maps for train and dev sets
    print('Generating saliency maps for train and dev sets...')
    explanation_wrapper = ExplanationWrapper(model, extraction_parameters)
    saliency_map_generator = SaliencyMapGenerator(model, explanation_wrapper, extraction_parameters, device, m, s)
    saliency_map_generator.set_profiling(kwargs.get('profile_saliency', False))
    saliency_map_generator.reset_profile_stats()
    
    predicted_labels_train, _, _ = model_trainer.eval_model_by_indices(indices_train)
    predicted_labels_dev, _, _ = model_trainer.eval_model_by_indices(indices_dev)
    predicted_labels_test, _, _ = model_trainer.eval_test_model()
    
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
    if kwargs.get('profile_saliency', False):
        with open(f'{model_path}/saliency_profile.json', 'w') as f:
            json.dump(saliency_map_generator.get_profile_stats(), f, indent=2)
    
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='Second-order statistical radiomic features saliency maps benchmark')
    parser.add_argument('-b', '--begin', type=int)
    parser.add_argument('-e', '--end', type=int)
    parser.add_argument('-d', '--device', type=int)
    parser.add_argument('-dt', '--datapath', type=str)
    parser.add_argument('-dn', '--datasetname', type=str)
    parser.add_argument('-f', '--features', type=str)
    parser.add_argument('-a', '--alpha', type=int, default=0)
    parser.add_argument('-dl', '--delta', type=int, default=1)
    parser.add_argument('-z', '--omitzeros', type=bool, default=True)
    parser.add_argument('-r', '--resultdir', type=str)
    parser.add_argument('-st', '--skiptraining', type=bool, default=False)
    parser.add_argument('-m', '--model', type=str)
    parser.add_argument('--saliency-batch-size', type=int, default=4)
    parser.add_argument('--saliency-n-steps', type=int, default=5)
    parser.add_argument('--profile-saliency', action='store_true')
    args = parser.parse_args()
    model_name = f'{args.features}'
    device = torch.device(f'cuda:{args.device}')
    train_dev_images, train_dev_labels, heldout_test_images, heldout_test_labels, heldout_test_ids = \
        load_dataset_with_heldout_test(args.datasetname, args.datapath)
    print(train_dev_images.shape)
    print(train_dev_labels.shape)
    print(heldout_test_images.shape)
    print(heldout_test_labels.shape)
    image_max_width = train_dev_images.shape[1]
    dev_size_within_train_dev = 15 / 85

    # Create one held-out test set, then resplit only the train/dev pool per iteration.
    for iteration in range(args.begin, args.end):
        train_dev_indices = np.arange(train_dev_images.shape[0])
        indices_train, indices_dev, _, _ = train_test_split(
            train_dev_indices, train_dev_labels,
            test_size=dev_size_within_train_dev,
            random_state=iteration,
            stratify=train_dev_labels.reshape(-1)
        )

        result_dir = f'{args.resultdir}{model_name}/{iteration}'
        os.makedirs(result_dir, exist_ok=True)

        run_experiment(train_dev_images, train_dev_labels, indices_train, indices_dev,
                       heldout_test_images, heldout_test_labels, device,
                       result_dir, test_ids=heldout_test_ids,
                       alpha=args.alpha, delta=args.delta, omit_zeros=args.omitzeros,
                       skip_training=args.skiptraining, model_type_name=args.model,
                       saliency_batch_size=args.saliency_batch_size,
                       saliency_n_steps=args.saliency_n_steps,
                       profile_saliency=args.profile_saliency)


