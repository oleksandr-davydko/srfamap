import torch

from nn.pytorch.radiomics.radiomic_mlp import StatisticalRadiomicMLP, StatisticalRadiomicTransformer

from radiomics_saliency import config
from radiomics_saliency.feature_extraction import (
    generate_features,
    get_feature_number,
    get_matrix_func_by_method,
    get_matrix_shape,
    method_to_layer_dict,
)


class ExplanationWrapper(torch.nn.Module):
    def __init__(self, model, feature_configs, mean=None, std=None):
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
        # Train-set normalisation statistics. The model is trained on standardised
        # features, so they must be applied here for predictions to match the classifier.
        self.register_buffer('mean', None if mean is None else mean.to(config.device))
        self.register_buffer('std', None if std is None else std.to(config.device))

    def forward(self, inp):
        features_together = generate_features(inp.cpu().detach().numpy(), self.feature_configs, False)
        features_together = features_together.float().to(config.device)
        if self.mean is not None:
            features_together = features_together - self.mean
        if self.std is not None:
            features_together = features_together / self.std
        probs = self.model(features_together)
        return probs


class RadiomicsMLPExplanationWrapper(torch.nn.Module):
    def __init__(self, model, feature_configs, mean, std, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = model
        self.feature_configs = feature_configs
        self.texture_layers = torch.nn.ModuleList()
        self.layer_offsets = []
        self.register_buffer('mean', mean.to(config.device))
        self.register_buffer('std', std.to(config.device))
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
        for layer, offset, config_entry in zip(self.texture_layers, self.layer_offsets, self.feature_configs):
            ft_num = get_feature_number(config_entry[0])
            ms = (-1, 1) + get_matrix_shape(config_entry[0])
            ft_inp = texture_matrices[:, offset:offset + ft_num].reshape(ms)
            features.append(layer(ft_inp))
        features = torch.cat(features, dim=1)
        features -= self.mean
        features /= self.std
        return self.model(features)


def get_model(model_type_name: str, device, input_size, n_classes=2):
    if model_type_name == 'mlp':
        return StatisticalRadiomicMLP(device, input_size, n_classes)
    elif model_type_name == 'transformer':
        return StatisticalRadiomicTransformer(device, 175)


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


class MultiClassIGTarget(torch.nn.Module):
    """Wrap a ``texture matrices -> logits`` function so Integrated Gradients can
    attribute a single per-example scalar suitable for multi-class explanations.

    Modes:
      - ``'logit'``      : ``f_c``                         (raw target-class logit)
      - ``'ovr_margin'`` : ``f_c - logsumexp_{j!=c} f_j``  ("why class c at all")
      - ``'pairwise'``   : ``f_c - f_r``                   ("why c and not competitor r")

    Target indices ``c`` (predicted class) and reference indices ``r`` (competitor)
    are set per batch with :meth:`set_targets` and must have shape ``[N]``.
    Because the wrapper collapses the output to a single column, call
    ``IntegratedGradients.attribute(..., target=0)``.
    Logits are used deliberately: with many classes a confident softmax saturates
    and its gradients vanish, whereas logit margins stay informative.
    """

    def __init__(self, matrices_to_logits: torch.nn.Module, mode: str = 'pairwise'):
        super().__init__()
        self.f = matrices_to_logits
        self.mode = mode
        self._c_idx = None
        self._r_idx = None

    def set_targets(self, c_idx: torch.Tensor, r_idx=None):
        self._c_idx = c_idx
        self._r_idx = r_idx

    def forward(self, x):
        logits = self.f(x)
        n = logits.shape[0]
        # Integrated Gradients feeds the forward an interpolation-expanded batch of
        # shape (n_steps * N, ...); tile the per-example target indices to match.
        c = self._c_idx.to(logits.device)
        if c.shape[0] != n:
            c = c.repeat(n // c.shape[0])
        rows = torch.arange(n, device=logits.device)
        fc = logits[rows, c]
        if self.mode == 'logit':
            return fc.unsqueeze(1)
        if self.mode == 'pairwise':
            r = self._r_idx.to(logits.device)
            if r.shape[0] != n:
                r = r.repeat(n // r.shape[0])
            return (fc - logits[rows, r]).unsqueeze(1)
        if self.mode == 'ovr_margin':
            mask = torch.ones_like(logits, dtype=torch.bool)
            mask[rows, c] = False
            rest = logits.masked_fill(~mask, float('-inf')).logsumexp(dim=1)
            return (fc - rest).unsqueeze(1)
        raise ValueError(f'Unknown attribution mode: {self.mode}')
