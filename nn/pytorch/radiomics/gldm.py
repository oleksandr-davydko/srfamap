import torch
import sys

class GldmFeatures(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dummy_param = torch.nn.Parameter(torch.empty(0))
        self.output_size = 14

    def forward(self, inp):
        if inp.dim() == 4:
            inp = inp[:, 0]
        if inp.dim() != 3:
            inp = inp.unsqueeze(0)
        x = inp.to(dtype=torch.float, device=inp.device)
        i = (torch.arange(x.shape[1]) + 1).repeat(x.shape[2], 1).transpose(1, 0).to(inp.device)
        j = (torch.arange(x.shape[2]) + 1).repeat(x.shape[1], 1).to(inp.device)
        nz = torch.sum(x, dim=(1, 2))
        nz[nz == 0] = 1
        x_normed = x / nz.reshape(-1, 1, 1)
        small_dependence_emphasis = torch.sum(x / (j ** 2), dim=(1, 2)) / nz
        large_dependence_emphasis = torch.sum(x * (j ** 2), dim=(1, 2)) / nz
        gray_level_non_uniformity = torch.sum(torch.sum(x, dim=2) ** 2, dim=1) / nz
        dependence_non_uniformity = torch.sum(torch.sum(x, dim=1) ** 2, dim=1) / nz
        dependence_non_uniformity_normalized = torch.sum(torch.sum(x, dim=1) ** 2, dim=1) / nz ** 2
        mu_i = torch.sum(x_normed * i, dim=(1, 2)).reshape(-1, 1, 1)
        grey_level_variance = torch.sum(x_normed * ((i - mu_i) ** 2), dim=(1, 2))
        mu_j = torch.sum(x_normed * j, dim=(1, 2)).reshape(-1, 1, 1)
        dependence_variance = torch.sum(x_normed * (j - mu_j) ** 2, dim=(1, 2))
        x_log = torch.log2(x_normed + sys.float_info.epsilon)
        dependence_entropy = -torch.sum(x_normed * x_log, dim=(1, 2))
        low_grey_level_emphasis = torch.sum(x / ((i ** 2) * torch.ones_like(j)), dim=(1, 2)) / nz
        high_grey_level_emphasis = torch.sum(x * (i ** 2), dim=(1, 2)) / nz
        small_dependence_low_grey_level_run_emphasis = torch.sum(x / ((i ** 2) * (j ** 2)), dim=(1, 2)) / nz
        small_dependence_high_grey_level_run_emphasis = torch.sum(x * (i ** 2) / (j ** 2), dim=(1, 2)) / nz
        large_dependence_low_grey_level_run_emphasis = torch.sum(x * (j ** 2) / (i ** 2), dim=(1, 2)) / nz
        large_dependence_high_grey_level_run_emphasis = torch.sum(x * ((j ** 2) * (i ** 2)), dim=(1, 2)) / nz
        ft = torch.concatenate([
            small_dependence_emphasis.reshape(-1, 1),
            large_dependence_emphasis.reshape(-1, 1),
            gray_level_non_uniformity.reshape(-1, 1),
            dependence_non_uniformity.reshape(-1, 1),
            dependence_non_uniformity_normalized.reshape(-1, 1),
            grey_level_variance.reshape(-1, 1),
            dependence_variance.reshape(-1, 1),
            dependence_entropy.reshape(-1, 1),
            low_grey_level_emphasis.reshape(-1, 1),
            high_grey_level_emphasis.reshape(-1, 1),
            small_dependence_low_grey_level_run_emphasis.reshape(-1, 1),
            small_dependence_high_grey_level_run_emphasis.reshape(-1, 1),
            large_dependence_low_grey_level_run_emphasis.reshape(-1, 1),
            large_dependence_high_grey_level_run_emphasis.reshape(-1, 1)
        ], dim=1).reshape(inp.size(0), 14)
        return ft