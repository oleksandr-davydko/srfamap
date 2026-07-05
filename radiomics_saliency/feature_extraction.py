import numpy as np
import radiomics_rs as r
import torch
from functools import partial
from tqdm import tqdm

from nn.pytorch.radiomics.glcm import GlcmFeatures
from nn.pytorch.radiomics.gldm import GldmFeatures
from nn.pytorch.radiomics.glrlm import GlrlmFeatures
from nn.pytorch.radiomics.glszm import GlszmFeatures
from nn.pytorch.radiomics.ngtdm import NgtdmFeatures

from radiomics_saliency import config

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


def get_feature_number(method):
    if method == 'glcm':
        return 256 * 256
    elif method == 'glrlm':
        return 256 * config.image_max_width
    elif method == 'glszm':
        return 256 * config.image_max_width
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
        return (256, config.image_max_width)
    elif method == 'glszm':
        return (256, config.image_max_width)
    elif method == 'gldm':
        return (256, 256)
    elif method == 'ngtdm':
        return (256, 3)
    else:
        raise Exception('Method {} not implemented'.format(method))


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
        return module(mats.to(config.device))
    except ValueError:
        mats = torch.concatenate(
            [module(torch.from_numpy(mat.reshape((1,) + mat.shape)).to(config.device)).unsqueeze(0) for mat in mats])
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
            high_order_features_module = method_to_layer_dict[method]().to(config.device)
            features = []
            image_iterator = enumerate(tqdm(images) if use_progress else images)
            for idx, image in image_iterator:
                image = image.reshape((1,) + image.shape)
                current_mask = None if masks is None else masks[idx].reshape((1,) + masks[idx].shape)
                if image.max() == 0:
                    features.append(torch.zeros((image.shape[0], get_high_order_feature_number(method))).to(config.device))
                else:
                    fv = get_feature_vector(first_order_features_method, high_order_features_module, image, mask=current_mask).to(config.device)
                    features.append(fv)
            features = torch.concatenate(features, dim=0)
            feature_blocks.append(features)
        features_together = torch.cat(feature_blocks, dim=1)
    return features_together
