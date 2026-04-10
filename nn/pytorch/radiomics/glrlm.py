import torch
import sys

class GlrlmFeatures(torch.nn.Module):
    def __init__(self, image_pixels_count: int):
        super().__init__()
        self.dummy_param = torch.nn.Parameter(torch.empty(0))
        self.output_size = 16
        self.image_pixels_count = image_pixels_count

    def forward(self, inp):
        if inp.dim() == 4:
            inp = inp[:, 0, :, :]
        if inp.dim() != 3:
            inp = inp.unsqueeze(0)
        x = inp
        i = (torch.arange(x.shape[1]) + 1).repeat(x.shape[2], 1).transpose(1, 0).to(inp.device)
        j = (torch.arange(x.shape[2]) + 1).repeat(x.shape[1], 1).to(inp.device)
        nr = torch.sum(x, dim=(1, 2))
        # when image contains 0 only
        nr[nr == 0] = 1
        x_normed = x / nr.reshape(-1, 1, 1)
        np = self.image_pixels_count
        short_run_emphasis = torch.sum(x / (j ** 2), dim=(1, 2)) / nr
        long_run_emphasis = torch.sum(x * (j ** 2), dim=(1, 2)) / nr
        gray_level_non_uniformity = torch.sum(torch.sum(x, dim=2) ** 2, dim=1) / nr
        gray_level_non_uniformity_normalized = torch.sum(torch.sum(x, dim=2) ** 2, dim=1) / nr ** 2
        run_length_non_uniformity = torch.sum(torch.sum(x, dim=1) ** 2, dim=1) / nr
        run_length_non_uniformity_normalized = torch.sum(torch.sum(x, dim=1) ** 2, dim=1) / nr ** 2
        run_percentage = nr / np
        mu_i = torch.sum(x_normed * i, dim=(1, 2)).reshape(-1, 1, 1)
        grey_level_variance = torch.sum(x_normed * ((i - mu_i) ** 2), dim=(1, 2))
        mu_j = torch.sum(x_normed * j, dim=(1, 2)).reshape(-1, 1, 1)
        run_length_variance = torch.sum(x_normed * (j - mu_j) ** 2, dim=(1, 2))
        x_log = torch.log2(x_normed + sys.float_info.epsilon)
        run_entropy = - torch.sum(x_normed * x_log, dim=(1, 2))
        low_grey_level_run_emphasis = torch.sum(x / ((i ** 2) * torch.ones_like(j).to(inp.device)), dim=(1, 2)) / nr
        high_grey_level_run_emphasis = torch.sum(x * (i ** 2), dim=(1, 2)) / nr
        short_run_low_grey_level_run_emphasis = torch.sum(x / ((i ** 2) * (j ** 2)), dim=(1, 2)) / nr
        short_run_high_grey_level_run_emphasis = torch.sum(x * (i ** 2) / (torch.ones_like(i).to(inp.device) * (j ** 2)), dim=(1, 2)) / nr
        long_run_low_grey_level_run_emphasis = torch.sum(x * (j ** 2) / ((i ** 2) * torch.ones_like(j).to(inp.device)), dim=(1, 2)) / nr
        long_run_high_grey_level_run_emphasis = torch.sum(x * ((j ** 2) * (i ** 2)), dim=(1, 2)) / nr
        ft = torch.concatenate([
            short_run_emphasis.reshape(-1, 1),
            long_run_emphasis.reshape(-1, 1),
            gray_level_non_uniformity.reshape(-1, 1),
            gray_level_non_uniformity_normalized.reshape(-1, 1),
            run_length_non_uniformity.reshape(-1, 1),
            run_length_non_uniformity_normalized.reshape(-1, 1),
            run_percentage.reshape(-1, 1),
            grey_level_variance.reshape(-1, 1),
            run_length_variance.reshape(-1, 1),
            run_entropy.reshape(-1, 1),
            low_grey_level_run_emphasis.reshape(-1, 1),
            high_grey_level_run_emphasis.reshape(-1, 1),
            short_run_low_grey_level_run_emphasis.reshape(-1, 1),
            short_run_high_grey_level_run_emphasis.reshape(-1, 1),
            long_run_low_grey_level_run_emphasis.reshape(-1, 1),
            long_run_high_grey_level_run_emphasis.reshape(-1, 1)
        ], dim=1).reshape(-1, 16)
        return ft