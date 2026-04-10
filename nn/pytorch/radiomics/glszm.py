import torch
import sys

class GlszmFeatures(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dummy_param = torch.nn.Parameter(torch.empty(0))
        self.output_size = 16

    def forward(self, inp):
        if inp.dim() == 4:
            inp = inp[:, 0, :, :]
        if inp.dim() == 2:
            inp = inp.unsqueeze(0)
        x = inp
        i = (torch.arange(x.shape[1]) + 1).repeat(x.shape[2], 1).transpose(1, 0).to(inp.device)
        j = (torch.arange(x.shape[2]) + 1).repeat(x.shape[1], 1).to(inp.device)
        nz = torch.sum(x, dim=(1, 2))
        nz[nz == 0] = 1
        np = x.size(1) * x.size(2)
        small_area_emphasis = torch.sum(x / (j ** 2), dim=(1, 2)) / nz
        large_area_emphasis = torch.sum(x * (j ** 2), dim=(1, 2)) / nz
        gray_level_non_uniformity = torch.sum(torch.sum(x, dim=2) ** 2, dim=1) / nz
        gray_level_non_uniformity_normalized = torch.sum(torch.sum(x, dim=2) ** 2, dim=1) / nz ** 2
        size_zone_non_uniformity = torch.sum(torch.sum(x, dim=1) ** 2, dim=1) / nz
        size_zone_non_uniformity_normalized = torch.sum(torch.sum(x, dim=1) ** 2, dim=1) / nz ** 2
        zone_percentage = nz / np
        x_normed = x / nz.reshape(-1, 1, 1)
        mu_i = torch.sum(x_normed * i, dim=(1, 2)).reshape(-1, 1, 1)
        grey_level_variance = torch.sum(x_normed * ((i - mu_i) ** 2), dim=(1, 2))
        mu_j = torch.sum(x_normed * j, dim=(1, 2)).reshape(-1, 1, 1)
        zone_variance = torch.sum(x_normed * (j - mu_j) ** 2, dim=(1, 2))
        x_log = torch.log2(x_normed + sys.float_info.epsilon)
        zone_entropy = - torch.sum(x_normed * x_log, dim=(1, 2))
        low_grey_level_zone_emphasis = torch.sum(x / ((i ** 2) * torch.ones_like(j).to(inp.device)), dim=(1, 2)) / nz
        high_grey_level_zone_emphasis = torch.sum(x * (i ** 2), dim=(1, 2)) / nz
        short_run_low_grey_level_zone_emphasis = torch.sum(x / ((i ** 2) * (j ** 2)), dim=(1, 2)) / nz
        short_run_high_grey_level_zone_emphasis = torch.sum(x * (i ** 2) / (torch.ones_like(i).to(inp.device) * (j ** 2)), dim=(1, 2)) / nz
        long_run_low_grey_level_zone_emphasis = torch.sum(x * (j ** 2) / ((i ** 2) * torch.ones_like(j).to(inp.device)), dim=(1, 2)) / nz
        long_run_high_grey_level_zone_emphasis = torch.sum(x * ((j ** 2) * (i ** 2)), dim=(1, 2)) / nz
        ft = torch.concatenate([
            small_area_emphasis.reshape(-1, 1),
            large_area_emphasis.reshape(-1, 1),
            gray_level_non_uniformity.reshape(-1, 1),
            gray_level_non_uniformity_normalized.reshape(-1, 1),
            size_zone_non_uniformity.reshape(-1, 1),
            size_zone_non_uniformity_normalized.reshape(-1, 1),
            zone_percentage.reshape(-1, 1),
            grey_level_variance.reshape(-1, 1),
            zone_variance.reshape(-1, 1),
            zone_entropy.reshape(-1, 1),
            low_grey_level_zone_emphasis.reshape(-1, 1),
            high_grey_level_zone_emphasis.reshape(-1, 1),
            short_run_low_grey_level_zone_emphasis.reshape(-1, 1),
            short_run_high_grey_level_zone_emphasis.reshape(-1, 1),
            long_run_low_grey_level_zone_emphasis.reshape(-1, 1),
            long_run_high_grey_level_zone_emphasis.reshape(-1, 1)
        ], dim=1).reshape(-1, 16)
        return ft