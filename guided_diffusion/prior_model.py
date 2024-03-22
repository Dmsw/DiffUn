import torch as th
import torch.nn as nn
import numpy as np


class PriorModel(nn.Module):
    def __init__(self, A, bar_alpha, mask=None, **kwargs):
        super().__init__()
        if mask is not None:
            A = A[:, ~mask]
        self.A = nn.Parameter(A * 2 -1)
        self.bar_alpha = th.from_numpy(bar_alpha).float().to(A.device)

    def forward(self, x, t, **kwargs):
        t = t.int()
        R, _, C = x.shape
        res = th.zeros_like(x, device=x.device)
        x = th.squeeze(x)
        for r in range(R):
            distance = (-th.norm(th.sqrt(self.bar_alpha[t[r]]) * self.A - x[r], dim=1)**2 / (2*(1-self.bar_alpha[t[r]])))
            distance = distance - distance.max()
            exp = th.exp(distance)
            res[r, 0, :] = exp @ self.A / th.sum(exp)

        return res

    def convert_to_fp16(self):
        self.A.data = self.A.data.half()
        