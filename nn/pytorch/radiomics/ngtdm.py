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
        result = torch.zeros((inp.size(0), self.output_size), dtype=torch.float, device=inp.device)
        for indexer in range(inp.size(0)):
            x = inp[indexer]
            if not x.any():
                result[indexer] = torch.zeros((5,))
                continue
            nvp = torch.sum(x[:, 0])
            i = (torch.arange(x.shape[0]) + 1).to(inp.device)
            non_zero_indices = x[:, 0] > 0
            ngp = torch.sum(x[:, 0][non_zero_indices] / x[:, 0][non_zero_indices])
            pi = x[:, 1]
            si = x[:, 2]
            coarseness = 1 / torch.sum(pi * si)
            contrast =  (torch.sum(pi[:, None] * pi[None, :] * ((i[:, None] - i[None, :]) ** 2)) / (ngp * (ngp - 1))) * (torch.sum(si) / nvp)
            busyness_divisor = torch.abs((i * pi)[:, None] - (i * pi)[None, :])
            busyness_divisor[busyness_divisor == 0] = 1
            busyness = torch.sum(pi * si) / (torch.sum(busyness_divisor))
            p_s_prod = pi * si
            divisor = pi[:, None] + pi[None, :]
            divisor[divisor == 0] = 1
            complexity = torch.sum(torch.abs(i[:, None] - i[None, :]) * (p_s_prod[:, None] + p_s_prod[None, :]) / divisor) / nvp
            strength = torch.sum((pi[:, None] + pi[None, :]) * ((i[:, None] - i[None, :]) ** 2)) / torch.sum(si)
            ft = torch.concatenate([
                coarseness.reshape((1,)),
                contrast.reshape((1,)),
                busyness.reshape((1,)),
                complexity.reshape((1,)),
                strength.reshape((1,))
            ]).reshape(1, -1)
            result[indexer] = ft.reshape((-1,))
            result = torch.nan_to_num(result)
        return result