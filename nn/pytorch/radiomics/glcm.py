import sys
import torch


class GlcmFeatures(torch.nn.Module):
    def __init__(self, gray_levels: int = 256):
        super().__init__()
        self.output_size = 24
        self.gray_levels = gray_levels
        self.dummy_param = torch.nn.Parameter(torch.empty(0))
        self.register_buffer('difference_index', torch.arange(0, gray_levels - 1, dtype=torch.float32))
        self.register_buffer('sum_index', torch.arange(2, gray_levels * 2, dtype=torch.float32))

    def get_difference_average(self, p_x_minus_y) -> torch.Tensor:
        return torch.sum(p_x_minus_y * torch.arange(0, 255).to(self.dummy_param.device), dim=1)

    def get_difference_variance(self, difference_average: float, p_x_minus_y) -> torch.Tensor:
        kda = (torch.arange(255).repeat(difference_average.size(0)).reshape(difference_average.size(0), 255).to(self.dummy_param.device) - difference_average.reshape(-1, 1)) ** 2
        return torch.sum(kda * p_x_minus_y, dim=1)

    def get_difference_entropy(self, p_x_minus_y) -> torch.Tensor:
        return -torch.sum(p_x_minus_y * torch.log2_(p_x_minus_y + sys.float_info.epsilon), dim=1)

    def forward(self, inp):
        if inp.dim() == 4:
            inp = inp[:, 0]
        if inp.dim() != 3:
            inp = inp.unsqueeze(0)
        x = inp.to(self.dummy_param.device)
        x_sum = torch.sum(x, dim=(1, 2))
        x_sum[x_sum == 0] = 1
        x_normed = x / x_sum.reshape(-1, 1, 1)
        inx_x = (torch.arange(x.shape[1]) + 1).to(self.dummy_param.device)
        inx_y = (torch.arange(x.shape[2]) + 1).to(self.dummy_param.device)
        i = (torch.arange(x.shape[1]) + 1).repeat(x.shape[2], 1).transpose(1, 0).to(self.dummy_param.device)
        j = (torch.arange(x.shape[2]) + 1).repeat(x.shape[1], 1).to(self.dummy_param.device)
        px = torch.sum(x_normed, dim=1)
        py = torch.sum(x_normed, dim=2)
        mu_x = torch.sum(px * inx_x.reshape(1, -1), dim=1)
        mu_y = torch.sum(py * inx_y.reshape(1, -1), dim=1)
        mu_x_matrix = mu_x.reshape(-1, 1, 1)
        mu_y_matrix = mu_y.reshape(-1, 1, 1)
        cluster_prominence = torch.sum(((i + j - mu_x_matrix - mu_y_matrix) ** 4) * x_normed, dim=(1, 2))
        cluster_shade = torch.sum(((i + j - mu_x_matrix - mu_y_matrix) ** 3) * x_normed, dim=(1, 2))
        cluster_tendency = torch.sum(((i + j - mu_x_matrix - mu_y_matrix) ** 2) * x, dim=(1, 2))
        autocorr = torch.sum(i * j * x_normed, dim=(1, 2))
        contrast = torch.sum(((i - j) ** 2) * x, dim=(1, 2))
        joint_avg = torch.sum(i * x, dim=(1, 2))
        flat_x_normed = x_normed.reshape(x.shape[0], -1)
        diff_indices = torch.abs(i - j).reshape(-1).long()
        valid_diff = diff_indices < self.difference_index.numel()
        p_x_minus_y = torch.zeros(
            (x.shape[0], self.difference_index.numel()),
            dtype=x_normed.dtype,
            device=x_normed.device)
        if valid_diff.any():
            p_x_minus_y.scatter_add_(
                1,
                diff_indices[valid_diff].unsqueeze(0).expand(x.shape[0], -1),
                flat_x_normed[:, valid_diff])
        difference_average = self.get_difference_average(p_x_minus_y)
        difference_entropy = self.get_difference_entropy(p_x_minus_y)
        difference_variance = self.get_difference_variance(difference_average, p_x_minus_y)
        correlation_denom = (
            torch.sum(x_normed * (i - mu_x_matrix) ** 2, dim=(1, 2)) ** 0.5
        ) * (
            torch.sum(x_normed * (j - mu_y_matrix) ** 2, dim=(1, 2)) ** 0.5
        )
        correlation_denom[correlation_denom == 0] = 1
        correlation = (torch.sum(x_normed * (i - mu_x_matrix) * (j - mu_y_matrix), dim=(1, 2))) / correlation_denom
        joint_energy = torch.sum(x_normed ** 2, dim=(1, 2))
        joint_entropy = -torch.sum(x_normed * torch.log2_(x_normed + sys.float_info.epsilon), dim=(1, 2))
        hx = -torch.sum(px * torch.log2_(px + sys.float_info.epsilon), dim=1)
        hy = -torch.sum(py * torch.log2_(py + sys.float_info.epsilon), dim=1)
        hxy = -torch.sum(x_normed *torch.log2_(x_normed + sys.float_info.epsilon), dim=(1, 2))
        margin_probs_cartesian_prod = px.unsqueeze(2) * py.unsqueeze(1)
        hxy1 = - torch.sum(x_normed * torch.log2_(margin_probs_cartesian_prod + sys.float_info.epsilon), dim=(1, 2))
        hxy2 = - torch.sum(margin_probs_cartesian_prod * torch.log2_(margin_probs_cartesian_prod + sys.float_info.epsilon), dim=(1, 2))
        imc1_denom = torch.max(hx, hy)
        imc1_denom[x_sum == 0] = 1
        imc1 = (hxy - hxy1) / imc1_denom
        imc2 = torch.sqrt_(1 - torch.exp_(-2 * (hxy2 - hxy)))
        k_indexes = self.difference_index.to(x_normed.dtype)
        idm = torch.sum(p_x_minus_y / (1 + k_indexes ** 2), dim=1)
        # Maximal Correlation Coefficient: eigenvalues of Q = R @ C^T per example, where
        # R normalizes each row by px and C each column by py. This is batched over the
        # whole batch dimension so the (expensive, autograd-heavy) eigendecomposition runs
        # as a single fused call instead of a Python loop of per-example eigvals — the
        # dominant cost when Integrated Gradients backpropagates through this layer.
        # Numerically identical to the per-example formulation.
        normalized_rows = x_normed / (px.unsqueeze(2) + sys.float_info.epsilon)
        normalized_cols = x_normed / (py.unsqueeze(1) + sys.float_info.epsilon)
        q = torch.bmm(normalized_rows, normalized_cols.transpose(1, 2))
        eig = torch.linalg.eigvals(q).real.to(torch.float32)
        # Second largest eigenvalue per example (argsort ascending, take the runner-up).
        second_largest = torch.sort(eig, dim=1).values[:, -2]
        maximal_correlation_coefficient = torch.sqrt(second_largest.clamp_min(0))
        idmn = torch.sum(p_x_minus_y / (1 + ((k_indexes ** 2) / (self.gray_levels ** 2))), dim=1)
        id = torch.sum(p_x_minus_y / (1 + k_indexes), dim=1)
        idn = torch.sum(p_x_minus_y / (1 + (k_indexes / 9)), dim=1)
        inverse_variance = torch.sum(p_x_minus_y[:] / (1 + k_indexes[:]) ** 2, dim=1)
        maximum_probability = torch.max(x.reshape(x.shape[0], -1), dim=1).values
        sum_indices = (i + j).reshape(-1).long()
        valid_sum = (sum_indices >= 2) & (sum_indices < (self.gray_levels * 2))
        p_x_plus_y = torch.zeros(
            (x.shape[0], self.sum_index.numel()),
            dtype=x_normed.dtype,
            device=x_normed.device)
        if valid_sum.any():
            p_x_plus_y.scatter_add_(
                1,
                (sum_indices[valid_sum] - 2).unsqueeze(0).expand(x.shape[0], -1),
                flat_x_normed[:, valid_sum])
        k_plus_index = self.sum_index.to(x_normed.dtype)
        sum_average = torch.sum(p_x_plus_y * k_plus_index, dim=1)
        sum_entropy = -torch.sum(p_x_plus_y * torch.log2(p_x_plus_y + sys.float_info.epsilon), dim=1)
        s_of_squares = torch.sum(x_normed * ((i - mu_x_matrix) ** 2), dim=(1, 2))
        ft = torch.concatenate([
            autocorr,
            cluster_prominence,
            cluster_shade,
            cluster_tendency,
            contrast,
            joint_avg,
            correlation,
            difference_average, #7
            difference_entropy,
            difference_variance,
            joint_energy,
            joint_entropy,
            imc1, #12
            imc2,
            idm,
            maximal_correlation_coefficient,
            idmn,
            id,
            idn,
            inverse_variance, #18
            maximum_probability,
            sum_average,
            sum_entropy,
            s_of_squares,
        ]).reshape(-1, 24)
        return ft