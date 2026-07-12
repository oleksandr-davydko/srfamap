import sys
import torch


class GlcmFeatures(torch.nn.Module):
    def __init__(self, gray_levels: int = 256, mcc_power_iters: int = 30, mcc_subspace_block: int = 6):
        super().__init__()
        self.output_size = 24
        self.gray_levels = gray_levels
        # Orthogonal (block power) iteration settings for the Maximal Correlation
        # Coefficient. ``block`` vectors are iterated together and QR-reorthonormalised
        # each step, so the columns converge to the leading eigenvectors *in order*;
        # more iterations converge them more tightly (matters only when the spectrum
        # is clustered). block > 2 gives the 2nd eigenvector some margin.
        self.mcc_power_iters = mcc_power_iters
        self.mcc_subspace_block = mcc_subspace_block
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

    def maximal_correlation_coefficient(self, x_normed, px, py) -> torch.Tensor:
        """Maximal Correlation Coefficient: ``sqrt`` of the second-largest
        eigenvalue of ``q = diag(1/px) @ x_normed @ diag(1/py) @ x_normed^T``.

        ``q`` is not symmetric, but it is *similar* to the symmetric PSD matrix
        ``S = G @ G^T`` with ``G = x_normed / (sqrt(px) * sqrt(py))`` (because
        ``q = D1 (x_normed D2 x_normed^T)`` with positive diagonal ``D1`` factors
        out as ``D1^{1/2} S D1^{-1/2}``), so it has the *same*, real, non-negative
        eigenvalues. Instead of a full non-symmetric ``eigvals`` — whose forward is
        O(n^3) and whose backward, evaluated at every Integrated Gradients step,
        was the dominant cost of this layer — we recover only the leading
        eigenvectors of ``S`` with orthogonal (block power) iteration plus a
        Rayleigh-Ritz extraction:

          * a block of ``mcc_subspace_block`` vectors is iterated together and
            QR-reorthonormalised each step, converging to the top invariant
            subspace; a small ``block x block`` Rayleigh-Ritz solve then rotates
            that subspace onto its eigenvectors and reads off the 2nd. All of this
            runs under ``no_grad`` — the vectors only need to be accurate, not
            differentiable — and Rayleigh-Ritz makes it converge in ~20 iterations
            instead of the hundreds raw per-vector iteration would need;
          * the second eigenvalue is then returned as the Rayleigh quotient
            ``u2^T S u2 = ||G^T u2||^2`` of the *detached* 2nd eigenvector, whose
            gradient ``dλ = u2^T dS u2`` is exact at an eigenvector and flows back
            through nothing more expensive than a single matmul.

        The small symmetric solve is hardened (symmetrised + ``nan_to_num``) and, if
        it still fails to converge on a degenerate spectrum (empty matrices,
        repeated eigenvalues), falls back to the raw sorted 2nd subspace vector — so
        this never raises the LAPACK errors a bare ``eigh`` throws. Numerically
        equivalent to ``sort(eigvals(q).real)[-2]``.
        """
        eps = sys.float_info.epsilon
        g = x_normed / (torch.sqrt(px.unsqueeze(2) + eps) * torch.sqrt(py.unsqueeze(1) + eps))
        batch, n, _ = g.shape
        block = max(2, min(self.mcc_subspace_block, n))
        gt = g.transpose(1, 2)

        def apply_s(mat):
            # S @ mat = G @ (G^T @ mat), never forming the n*n matrix S.
            return torch.bmm(g, torch.bmm(gt, mat))

        with torch.no_grad():
            # Deterministic, batch-position-independent init so attributions stay
            # reproducible (relativeInputStability relies on it).
            generator = torch.Generator(device=g.device).manual_seed(0)
            v = torch.randn(1, n, block, generator=generator, device=g.device, dtype=g.dtype)
            v = v.expand(batch, -1, -1).contiguous()
            v, _ = torch.linalg.qr(v)
            for _ in range(self.mcc_power_iters):
                v, _ = torch.linalg.qr(apply_s(v))
            # Rayleigh-Ritz: eigenpairs of the small projection T = V^T S V =
            # (G^T V)^T (G^T V). Its eigenvalues are the top Ritz values (ascending),
            # so the second-largest Ritz vector, lifted back by V, is the 2nd
            # eigenvector of S. Symmetrise + sanitise T so the tiny eigh is stable.
            w = torch.bmm(gt, v)
            t = torch.bmm(w.transpose(1, 2), w)
            t = torch.nan_to_num(0.5 * (t + t.transpose(1, 2)))
            try:
                _, ritz_vectors = torch.linalg.eigh(t)
                u2 = torch.bmm(v, ritz_vectors[:, :, -2:-1])
            except RuntimeError:
                # Orthogonal iteration already sorts columns by eigenvalue; column 1
                # is a (less tightly converged) second eigenvector. Never raises.
                u2 = v[:, :, 1:2]

        gu = torch.bmm(gt, u2)
        # eps floor keeps the sqrt gradient finite where the second eigenvalue is 0
        # (degenerate/empty matrices); there the upstream gradient is 0, so the
        # feature and its gradient both stay 0 instead of becoming inf/NaN.
        second_largest = torch.sum(gu * gu, dim=(1, 2))
        return torch.sqrt(second_largest.clamp_min(0) + eps)

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
        # Maximal Correlation Coefficient: sqrt of the second-largest eigenvalue of
        # Q = diag(1/px) @ x_normed @ diag(1/py) @ x_normed^T. Recovered via matvec-only
        # block power iteration on the symmetric matrix similar to Q (see the method),
        # replacing the full non-symmetric eigvals whose backward dominated this layer.
        maximal_correlation_coefficient = self.maximal_correlation_coefficient(x_normed, px, py)
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