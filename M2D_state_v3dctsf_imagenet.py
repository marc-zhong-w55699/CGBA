import copy
import numpy as np
import torch
from utils import clip_image_values
from torch.autograd import Variable
import math


# ============================================================================
# ImageNet M2D-state (EXPLOIT/EXPLORE/REFINE) + DCT-SF direction + v3 geometry.
# State machine same as pro_atk_tuned.py.
# ============================================================================


class Proposed_attack():
    def __init__(self, model, src_img, mean, std, lb, ub, dim_reduc_factor=4,
                 tar_img=None, iteration=700, tol=1e-5, attack_method='manifold_search_2d',
                 verbose_control='Yes',
                 freq_range=(0.0, 0.5),
                 tanh_gamma=1.0,
                 theta_max=math.pi / 3,
                 BS_iter=7):
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
        self.theta_max = theta_max
        self.BS_iter = BS_iter

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.all_queries = 0

        _, C, H, W = self.src_img.shape
        self._C, self._H, self._W = C, H, W
        self._Dh = self._dct_matrix(H).to(self.device)
        self._Dw = self._Dh if W == H else self._dct_matrix(W).to(self.device)
        mask_2d = self._zigzag_mask(H, W, freq_range).to(self.device)
        self._mask_3d = mask_2d.unsqueeze(0).expand(C, H, W).contiguous()
        src_dct = self._dct2d(self.src_img[0])
        masked = src_dct * self._mask_3d
        self._image_dct_cache = torch.tanh(self.tanh_gamma * masked)



    def _dct_matrix(self, N):
        n = torch.arange(N, dtype=torch.float32).view(1, -1)
        k = torch.arange(N, dtype=torch.float32).view(-1, 1)
        D = torch.cos(math.pi * (2 * n + 1) * k / (2 * N))
        D = D * math.sqrt(2.0 / N)
        D[0] = D[0] / math.sqrt(2.0)
        return D

    def _dct2d(self, x):
        tmp = torch.einsum('ij,cjk->cik', self._Dh, x)
        F   = torch.einsum('cik,lk->cil', tmp, self._Dw)
        return F

    def _idct2d(self, F):
        tmp = torch.einsum('ji,cjk->cik', self._Dh, F)
        x   = torch.einsum('cik,kl->cil', tmp, self._Dw)
        return x

    def _zigzag_mask(self, H, W, freq_range):
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
        C, H, W = self._C, self._H, self._W
        ternary = torch.randint(0, 3, (C, H, W), device=self.device).float() - 1.0
        direction_freq    = self._image_dct_cache * ternary
        direction_spatial = self._idct2d(direction_freq)
        return direction_spatial.unsqueeze(0)



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



    def _circ_x_at(self, x_o, r, v, u, s, theta):
        cos_t, sin_t = math.cos(theta), math.sin(theta)
        return clip_image_values(
            x_o + r * cos_t * (v * cos_t + s * u * sin_t),
            self.lb, self.ub
        ).to(self.device)



    def _circ_binary_search(self, x_o, r, v, u, s, theta_max):
        lower, upper = 0.0, theta_max
        best_angle = 0.0
        x_best = None
        num_q = 0
        for _ in range(self.BS_iter):
            mid = (lower + upper) / 2.0
            x_mid = self._circ_x_at(x_o, r, v, u, s, mid)
            num_q += 1
            if self.is_adversarial(x_mid) == 1:
                lower = mid
                best_angle = mid
                x_best = x_mid
            else:
                upper = mid
        return best_angle, x_best, num_q



    def manifold_search_2d(self, x_o, x_b,
                           beta=math.pi / 30,
                           beta_min=math.pi / 1000,
                           u=None,
                           **kwargs):
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

        probe_angle = self.theta_max / 4.0
        s = 0
        x_pos = self._circ_x_at(x_o, r, v, u, +1, probe_angle)
        num_calls += 1
        if self.is_adversarial(x_pos) == 1:
            s = +1
        else:
            x_neg = self._circ_x_at(x_o, r, v, u, -1, probe_angle)
            num_calls += 1
            if self.is_adversarial(x_neg) == 1:
                s = -1

        if s == 0:
            cur_beta = beta
            while cur_beta > beta_min:
                x_pos = self._circ_x_at(x_o, r, v, u, +1, cur_beta)
                num_calls += 1
                if self.is_adversarial(x_pos) == 1:
                    s = +1
                    break
                x_neg = self._circ_x_at(x_o, r, v, u, -1, cur_beta)
                num_calls += 1
                if self.is_adversarial(x_neg) == 1:
                    s = -1
                    break
                cur_beta = cur_beta / 2
            if s == 0:
                return x_b, num_calls

        best_angle, x_best, bs_q = self._circ_binary_search(x_o, r, v, u, s, self.theta_max)
        num_calls += bs_q

        if x_best is not None and best_angle > 0:
            return x_best, num_calls
        else:
            return x_b, num_calls



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

        # ── State machine constants ──
        STATE_EXPLOIT = 'EXPLOIT'
        STATE_EXPLORE = 'EXPLORE'
        STATE_REFINE  = 'REFINE'
        STATE_WEIGHTS = {
            STATE_EXPLOIT: (0.4, 0.3, 0.3),
            STATE_EXPLORE: (0.2, 0.2, 0.6),
            STATE_REFINE:  (0.3, 0.5, 0.2),
        }
        EPS_PROGRESS   = 0.005
        STAG_THRESH    = 5
        R_REFINE_RATIO = 0.15
        MIN_DWELL      = 2

        outer_iter = self.iteration

        state  = STATE_EXPLOIT
        stag_k = 0
        dwell  = 0
        r_init = float(torch.norm(x_b - self.src_img).item())
        r_prev = r_init

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

            r_now     = float(r_cur.item())
            delta_rel = (r_prev - r_now) / max(r_prev, 1e-12)
            r_ratio   = r_now / max(r_init, 1e-12)
            if delta_rel < EPS_PROGRESS:
                stag_k += 1
            else:
                stag_k = 0

            dwell += 1
            if dwell >= MIN_DWELL:
                if r_ratio < R_REFINE_RATIO:
                    new_state = STATE_REFINE
                elif stag_k >= STAG_THRESH:
                    new_state = STATE_EXPLORE
                elif delta_rel > EPS_PROGRESS:
                    new_state = STATE_EXPLOIT
                else:
                    new_state = state
                if new_state != state:
                    state  = new_state
                    dwell  = 0
                    stag_k = 0
            lam1, lam2, lam3 = STATE_WEIGHTS[state]

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
                self.src_img, x_b, u=u_new
            )

            x_e_prev = x_adv
            x_b_prev = x_b
            u_prev = u_new
            x_b = x_adv

            q_num = q_num + qs
            total_boundary_queries += qs

            x_adv_inv = self.inv_tf(copy.deepcopy(x_adv.cpu()[0,:,:,:].squeeze()), self.mean, self.std)
            norm = torch.norm(x_inv - x_adv_inv)

            if it % 20 == 0 or it == outer_iter - 1:
                if self.verbose_control == 'Yes':
                    print('Manifold2D-v3-state-dctsf iter -> ' + str(it) +
                          '   Queries ' + str(q_num) +
                          '   norm -> ' + f'{norm.item():.3f}' +
                          f'   inner_q={qs}' +
                          f'   state={state}')

            r_prev = r_now

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
