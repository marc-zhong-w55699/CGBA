import copy
import numpy as np
import torch
from utils import clip_image_values
from torch.autograd import Variable
import math


# ============================================================================
# CIFAR-10 explore (λ=0.2,0.2,0.6) + SurFree-style DCT direction sampling.
# Pure attack class.
#
# vs M2D_explore_dct_cifar10.py 的关键升级:
#   ① Mask:   square low-freq block  →  zigzag mask
#   ② Range:  freq_range (0, 1/4=25%)  →  (0, 0.5) = 50%
#             32×32 上保留 512 freq/channel（之前 r=1/4 只有 64）
#   ③ Direction generation:
#       之前: u = idct(low_freq_block × randn)              (pure random)
#       现在: u = idct(ternary{-1,0,1} × tanh(image_dct × zigzag_mask))
#             方向与 image content 相关（low-freq sign 跟图像一致）
#
# 等价于 SurFree 默认 Basis(basis_type='dct', dct_type='full',
#                           frequence_range=(0,0.5), function='tanh', tanh_gamma=1).
# ============================================================================


class Proposed_attack():
    def __init__(self, model, src_img, mean, std, lb, ub, dim_reduc_factor=4,
                 tar_img=None, iteration=93, tol=1e-5, attack_method='manifold_search_2d',
                 verbose_control='Yes',
                 freq_range=(0.0, 0.5),        
                 tanh_gamma=1.0):             
        self.model = model
        self.src_img = src_img
        self.src_lbl = torch.argmax(self.model.forward(Variable(self.src_img, requires_grad=True)).data).item()
        self.tar_img = tar_img
        if tar_img != None:
            self.tar_lbl = torch.argmax(self.model.forward(Variable(self.tar_img, requires_grad=True)).data).item()
        self.iteration = iteration
        self.mean = mean
        self.std = std
        self.lb = lb
        self.ub = ub
        self.tol = tol
        self.verbose_control = verbose_control
        self.attack_method = attack_method
        self.dim_reduc_factor = dim_reduc_factor
        self.freq_range = freq_range
        self.tanh_gamma = tanh_gamma

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.all_queries = 0

        # --- Precompute DCT matrices + zigzag mask + image-DCT cache ---
        _, C, H, W = self.src_img.shape
        self._C, self._H, self._W = C, H, W
        self._Dh = self._dct_matrix(H).to(self.device)        # (H, H)
        self._Dw = self._Dh if W == H else self._dct_matrix(W).to(self.device)

        # Zigzag mask (H, W) → expand to (C, H, W)
        mask_2d = self._zigzag_mask(H, W, freq_range).to(self.device)
        self._mask_3d = mask_2d.unsqueeze(0).expand(C, H, W).contiguous()

        # Cache tanh(image_dct × zigzag_mask) for the source image
        src_dct = self._dct2d(self.src_img[0])                # (C, H, W)
        masked  = src_dct * self._mask_3d
        self._image_dct_cache = torch.tanh(self.tanh_gamma * masked)



    # ---------- DCT helpers (pure torch) ----------
    def _dct_matrix(self, N):
        """DCT-II orthonormal matrix, row k = frequency k."""
        n = torch.arange(N, dtype=torch.float32).view(1, -1)
        k = torch.arange(N, dtype=torch.float32).view(-1, 1)
        D = torch.cos(math.pi * (2 * n + 1) * k / (2 * N))
        D = D * math.sqrt(2.0 / N)
        D[0] = D[0] / math.sqrt(2.0)
        return D

    def _dct2d(self, x):
        """Forward 2D DCT per channel. x: (C, H, W) -> F: (C, H, W).
           F = Dh @ x @ Dw^T."""
        tmp = torch.einsum('ij,cjk->cik', self._Dh, x)
        F   = torch.einsum('cik,lk->cil', tmp, self._Dw)
        return F

    def _idct2d(self, F):
        """Inverse 2D DCT per channel. F: (C, H, W) -> x: (C, H, W).
           x = Dh^T @ F @ Dw."""
        tmp = torch.einsum('ji,cjk->cik', self._Dh, F)
        x   = torch.einsum('cik,kl->cil', tmp, self._Dw)
        return x

    def _zigzag_mask(self, H, W, freq_range):
        """SurFree-style zigzag mask. Keep `freq_range[1]*H*W` lowest frequencies
           in anti-diagonal (zigzag) order, skipping `freq_range[0]*H*W` from top."""
        total = H * W
        n_keep  = int(total * min(1.0, freq_range[1]))
        n_start = int(total * max(0.0, freq_range[0]))
        mask = torch.zeros(H, W, dtype=torch.float32)
        s = 0
        while n_keep > 0:
            for i in range(min(s + 1, H)):
                for j in range(min(s + 1, W)):
                    if i + j != s:
                        continue
                    if n_start > 0:
                        n_start -= 1
                        continue
                    # Zigzag direction alternates with s parity
                    if s % 2:
                        mask[i, j] = 1.0
                    else:
                        mask[j, i] = 1.0
                    n_keep -= 1
                    if n_keep == 0:
                        return mask
            s += 1
            if s > H + W:
                break
        return mask

    def _surfree_direction(self, shape):
        """SurFree-style image-conditioned direction.
           shape: (1, C, H, W). Returns spatial direction of same shape."""
        C, H, W = self._C, self._H, self._W
        # Ternary random in {-1, 0, 1}
        ternary = torch.randint(0, 3, (C, H, W), device=self.device).float() - 1.0
        direction_freq    = self._image_dct_cache * ternary    # (C, H, W)
        direction_spatial = self._idct2d(direction_freq)       # (C, H, W)
        return direction_spatial.unsqueeze(0)                  # (1, C, H, W)



    def is_adversarial(self, image):
        predict_label = torch.argmax(self.model.forward(Variable(image, requires_grad=True)).data).item()
        self.all_queries += 1
        if self.tar_img == None:
            is_adv = predict_label != self.src_lbl
        else:
            is_adv = predict_label == self.tar_lbl
        if is_adv:
            return 1
        else:
            return -1



    def find_random_adversarial(self, image, step=3.0, eps_max=15, n=60):
        num_calls = 0
        perturbed = image
        candidate = image
        max_calls=50
        for _ in range(n):
            u = self._surfree_direction(image.shape).to(self.device)
            u = u / torch.norm(u)

            eps = step
            candidate = clip_image_values(candidate + eps * u, self.lb, self.ub).to(self.device)
            is_adv = self.is_adversarial(candidate)
            num_calls += 1

            while is_adv == -1 and eps <= eps_max:
                eps += step
                candidate = clip_image_values(candidate + eps * u, self.lb, self.ub).to(self.device)
                is_adv = self.is_adversarial(candidate)
                num_calls += 1

            if is_adv == 1:
                perturbed = candidate
                x_b, bin_calls = self.bin_search(image, perturbed, max_calls)
                num_calls += bin_calls
                return x_b, num_calls

        print("Warning: find_random_adversarial failed to find an adversarial direction after {} trials, falling back to cumulative random walk.".format(n))
        num_calls = 1
        step_fb = 0.02
        perturbed = image
        while self.is_adversarial(perturbed) == -1:
            pert = self._surfree_direction(image.shape).to(self.device)
            perturbed = image + num_calls * step_fb * pert
            perturbed = clip_image_values(perturbed, self.lb, self.ub).to(self.device)
            num_calls += 1
        return perturbed, num_calls



    def bin_search(self, x_0, x_random, max_calls=100):
        num_calls = 0
        adv = x_random
        cln = x_0
        while True:
            mid = (cln + adv) / 2.0
            num_calls += 1
            if self.is_adversarial(mid) == 1:
                adv = mid
            else:
                cln = mid
            if torch.norm(adv-cln).cpu().numpy() < self.tol or num_calls >= max_calls:
                break
        return adv, num_calls



    def _proj_and_normalize(self, vec, v_ref, eps=1e-8):
        vec = vec - torch.dot(vec.reshape(-1), v_ref.reshape(-1)) * v_ref
        nrm = torch.norm(vec)
        if nrm < eps:
            return None
        return vec / nrm



    def manifold_search_2d(self, x_o, x_b,
                           alpha=0.99,
                           beta=math.pi / 12,
                           beta_min=math.pi / 1000,
                           n=None,
                           u=None):
        if n is None:
            n = self.iteration
        num_calls = 0

        diff = x_b - x_o
        r = torch.norm(diff)
        v = diff / r

        if u is None:
            u = self._surfree_direction(x_o.shape).to(self.device)
        u = u.to(self.device)
        u = u - torch.dot(u.reshape(-1), v.reshape(-1)) * v
        u_norm = torch.norm(u)
        if u_norm < 1e-8:
            u = self._surfree_direction(x_o.shape).to(self.device)
            u = u - torch.dot(u.reshape(-1), v.reshape(-1)) * v
            u_norm = torch.norm(u)
        u = u / u_norm

        s = 0
        cur_beta = beta
        while cur_beta > beta_min:
            cos_b, sin_b = math.cos(cur_beta), math.sin(cur_beta)

            cand_p = clip_image_values(x_o + r * (v * cos_b + u * sin_b), self.lb, self.ub).to(self.device)
            num_calls += 1
            if self.is_adversarial(cand_p) == 1:
                s = +1
                break

            cand_m = clip_image_values(x_o + r * (v * cos_b - u * sin_b), self.lb, self.ub).to(self.device)
            num_calls += 1
            if self.is_adversarial(cand_m) == 1:
                s = -1
                break

            cur_beta = cur_beta / 2

        if s == 0:
            return x_b, num_calls

        x_s = x_b
        x_e = x_b
        for i in range(1, n + 1):
            w = (x_s - x_o) / torch.norm(x_s - x_o)
            alpha_i = alpha ** (1 + (n - i) / n)

            while True:
                cand = clip_image_values(x_o + (alpha_i * r) * w, self.lb, self.ub).to(self.device)
                num_calls += 1
                if self.is_adversarial(cand) == 1:
                    r = alpha_i * r
                else:
                    break
                if r < 1e-6:
                    break

            x_r = x_o + r * w

            angle = i * beta
            cos_a, sin_a = math.cos(angle), math.sin(angle)
            x_s_cand = clip_image_values(x_o + r * (v * cos_a + s * u * sin_a), self.lb, self.ub).to(self.device)
            num_calls += 1
            if self.is_adversarial(x_s_cand) != 1:
                x_e = x_r
                break
            else:
                x_s = x_s_cand
                x_e = x_s

        x_e = clip_image_values(x_e, self.lb, self.ub).to(self.device)
        return x_e, num_calls



    def Attack(self):
        norms = []
        n_query = []
        grad = 0
        total_grad_queries     = 0
        total_boundary_queries = 0

        x_inv = self.inv_tf(copy.deepcopy(self.src_img.cpu()[0,:,:,:].squeeze()), self.mean, self.std)
        if self.tar_img == None:
            x_random, query_random = self.find_random_adversarial(self.src_img)
        if self.tar_img != None:
            x_random, query_random = self.tar_img, 0
        x_b, query_b = self.bin_search(self.src_img, x_random)
        x_b_inv = self.inv_tf(copy.deepcopy(x_b.cpu()[0,:,:,:].squeeze()), self.mean, self.std)
        norm_initial = torch.norm(x_b_inv - x_inv)
        norms.append(norm_initial)
        q_num = query_random + query_b
        print('Initial boundary norm', torch.norm(norm_initial).item())
        print('query_b', query_b)
        print('initial query', q_num)

        n_query.append(q_num)
        size = self.src_img.shape

        outer_iter = self.iteration
        inner_n = 15
        lam1, lam2, lam3 = 0.2, 0.2, 0.6      # explore

        u_prev = None
        x_e_prev = None
        x_b_prev = None
        x_adv = x_b

        for it in range(outer_iter):
            diff = x_b - self.src_img
            r_cur = torch.norm(diff)
            if r_cur < 1e-8:
                break
            v_new = diff / r_cur

            d1 = self._proj_and_normalize(x_e_prev - x_b_prev, v_new) \
                 if (x_e_prev is not None and x_b_prev is not None) else None
            d2 = self._proj_and_normalize(u_prev, v_new) if u_prev is not None else None
            d3 = self._proj_and_normalize(self._surfree_direction(x_b.shape).to(self.device), v_new)

            if d1 is None and d2 is None:
                u_new = d3
            else:
                combo = lam3 * d3
                if d1 is not None:
                    combo = combo + lam1 * d1
                if d2 is not None:
                    combo = combo + lam2 * d2
                u_new = self._proj_and_normalize(combo, v_new)
                if u_new is None:
                    u_new = d3

            x_adv, qs = self.manifold_search_2d(
                self.src_img, x_b, n=inner_n, u=u_new
            )

            x_e_prev = x_adv
            x_b_prev = x_b
            u_prev = u_new
            x_b = x_adv

            q_num = q_num + qs
            total_boundary_queries += qs

            x_adv_inv = self.inv_tf(copy.deepcopy(x_adv.cpu()[0,:,:,:].squeeze()), self.mean, self.std)
            norm = torch.norm(x_inv - x_adv_inv)

            if it % 4 == 0 or it == outer_iter - 1:
                if self.verbose_control == 'Yes':
                    print('Manifold2D iter -> ' + str(it) +
                          '   Queries ' + str(q_num) +
                          '   norm -> ' + f'{norm.item():.3f}' +
                          f'   inner_q={qs}')

            norms.append(norm)
            n_query.append(q_num)

        print(f'\n── Query num ──────────────────────────────────')
        print(f'Gradient estimation queries : {total_grad_queries}')
        print(f'Boundary search queries     : {total_boundary_queries}')
        print(f'Total queries               : {q_num}')
        print(f'────────────────────────────────────────────────')

        x_adv = clip_image_values(x_adv, self.lb, self.ub)
        return x_adv, n_query, norms



    def inv_tf(self, x, mean, std):
        for i in range(len(mean)):
            x[i] = np.multiply(x[i], std[i], dtype=np.float32)
            x[i] = np.add(x[i], mean[i], dtype=np.float32)
        x = np.swapaxes(x, 0, 2)
        x = np.swapaxes(x, 0, 1)
        return x
