import sys

import torch


class NgtdmFeatures(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dummy_param = torch.nn.Parameter(torch.empty(0))
        self.output_size = 5

    def forward(self, inp):
        if inp.dim() == 4:
            inp = inp[:, 0]
        if inp.dim() != 3:
            inp = inp.unsqueeze(0)
        # Vectorized over the batch dimension: every per-example NGTDM computation below
        # is expressed as a batched tensor op instead of a Python loop, so Integrated
        # Gradients backpropagates through a single fused graph rather than one per row.
        # Numerically identical to the per-example formulation (verified against it).
        x = inp.to(dtype=torch.float, device=inp.device)
        batch_size, gray_levels, _ = x.shape
        col0 = x[:, :, 0]                       # voxel counts per gray level  [B, G]
        pi = x[:, :, 1]                          # p_i                          [B, G]
        si = x[:, :, 2]                          # s_i                          [B, G]
        i = (torch.arange(gray_levels, device=inp.device) + 1).to(torch.float)   # [G]
        diff_sq = (i[:, None] - i[None, :]) ** 2         # (i - j)^2  [G, G]
        absdiff = torch.abs(i[:, None] - i[None, :])     # |i - j|    [G, G]

        # All-zero (empty) matrices were short-circuited to a zero row in the original
        # (before nan_to_num, so no inf ever reached autograd). Reproduce that here by
        # adding the empty-row mask to every denominator that is zero *only* for empty
        # rows: it adds 0.0 to non-empty rows (leaving them bit-for-bit identical) and
        # keeps empty rows finite so 0 * inf = nan can't corrupt their gradient. The
        # empty rows are then overwritten with zeros to match the original value.
        empty = ~(x.reshape(batch_size, -1).any(dim=1))
        empty_f = empty.to(torch.float)                  # [B]

        nvp = torch.sum(col0, dim=1)                     # [B]
        # ngp = sum over nonzero entries of (v / v) == number of gray levels present.
        # (constant w.r.t. the input, exactly as in the original per-example code).
        ngp = torch.sum((col0 > 0).to(torch.float), dim=1)   # [B]
        p_s_prod = pi * si                               # [B, G]

        coarseness = 1 / (torch.sum(p_s_prod, dim=1) + empty_f)
        contrast = (torch.sum(pi[:, :, None] * pi[:, None, :] * diff_sq[None], dim=(1, 2))
                    / (ngp * (ngp - 1) + empty_f)) * (torch.sum(si, dim=1) / (nvp + empty_f))

        ip = i[None, :] * pi                             # (i * p_i)  [B, G]
        busyness_divisor = torch.abs(ip[:, :, None] - ip[:, None, :])
        busyness_divisor = torch.where(
            busyness_divisor == 0, torch.ones_like(busyness_divisor), busyness_divisor)
        busyness = torch.sum(p_s_prod, dim=1) / torch.sum(busyness_divisor, dim=(1, 2))

        divisor = pi[:, :, None] + pi[:, None, :]
        divisor = torch.where(divisor == 0, torch.ones_like(divisor), divisor)
        complexity = torch.sum(
            absdiff[None] * (p_s_prod[:, :, None] + p_s_prod[:, None, :]) / divisor,
            dim=(1, 2)) / (nvp + empty_f)

        strength = torch.sum((pi[:, :, None] + pi[:, None, :]) * diff_sq[None],
                             dim=(1, 2)) / (torch.sum(si, dim=1) + empty_f)

        result = torch.stack([coarseness, contrast, busyness, complexity, strength], dim=1)
        result = torch.nan_to_num(result)
        if empty.any():
            result = result.clone()
            result[empty] = 0.0
        return result