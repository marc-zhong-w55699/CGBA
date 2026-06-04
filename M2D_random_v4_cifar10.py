import copy
import numpy as np
import torch
from utils import clip_image_values
from torch.autograd import Variable
import math


# ============================================================================
# CIFAR-10 M2D-random (λ=0,0,1) + v3 geometry + v4 path A line search.
# Pure attack class. No DCT.
#
# v4 path A = v3 + line search AFTER each outer iter's circular search.
#
# (v4 path A) Motivation:
#   v3's circular search returns x_b_new = Φ(θ*, s) at distance r·cos(θ*).
#   The radial line (x_o → x_b_new) is a NEW direction NEVER searched by walk,
#   so it can contain unexplored boundary crossings. LS probes this line
#   and finds the CLOSEST adv crossing.
#
# Note: v4 path B (init LS) was tested and proven useless on CIFAR
#   (walk implicitly does line search → no improvement).
#
# Injection point:
#   After each outer iter's manifold_search_2d, call _line_search_closest
#   along (x_o → x_adv) direction (every LS_every iterations).
#
# Cost per LS call: ~5 probes + ~7 bin_search = ~12 queries
# Default LS_every=1 (every iter); iteration reduced to compensate cost.
# ============================================================================


class Proposed_attack():
    def __init__(self, model, src_img, mean, std, lb, ub, dim_reduc_factor=4,
                 tar_img=None, iteration=500, tol=1e-5, attack_method='manifold_search_2d',  # ★ iter 1000→500 (q/iter ~26 with LS)
                 verbose_control='Yes',
                 theta_max=math.pi / 3,
                 BS_iter=7,
                 LS_probes=5,                   # ★ line search probe count
                 LS_every=1):                 # ★ v4 path A: LS every N outer iters
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
        self.theta_max = theta_max
        self.BS_iter = BS_iter
        self.LS_probes = LS_probes
        self.LS_every = LS_every

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.all_queries = 0



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



    def _line_search_closest(self, x_0, x_random, max_calls=50):
        """★ v4 line search: handles non-convex (multi-crossing) boundary.
        Probes n_probes points along x_0 → x_random, finds the FIRST adv crossing,
        then bin_search refines within that interval.
        Returns (x_b, num_queries)."""
        num_calls = 0
        n_probes = self.LS_probes

        v_dir = x_random - x_0

        # --- Coarse probe: find smallest t such that x_0 + t·v is adversarial ---
        closest_t = 1.0
        closest_idx = n_probes
        for k in range(1, n_probes + 1):
            t = k / n_probes
            x_test = clip_image_values(x_0 + t * v_dir, self.lb, self.ub).to(self.device)
            num_calls += 1
            if self.is_adversarial(x_test) == 1:
                closest_t = t
                closest_idx = k
                break

        # If no probe was adv, fall back to original x_random
        if closest_idx > n_probes:
            adv = x_random
            cln = x_0
        else:
            # Boundary is in [(closest_idx-1)/n_probes, closest_idx/n_probes]
            cln_t = (closest_idx - 1) / n_probes
            cln = clip_image_values(x_0 + cln_t * v_dir, self.lb, self.ub).to(self.device)
            adv = clip_image_values(x_0 + closest_t * v_dir, self.lb, self.ub).to(self.device)

        # --- bin_search refine in [cln, adv] ---
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



    def find_random_adversarial(self, image, step=3.0, eps_max=15, n=60):
        num_calls = 0
        perturbed = image
        candidate = image
        max_calls=50
        for _ in range(n):
            u = torch.randn(image.shape).to(self.device)
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
                # ★ v4 path A: revert to plain bin_search (init LS proven useless)
                x_b, bin_calls = self.bin_search(image, perturbed, max_calls)
                num_calls += bin_calls
                return x_b, num_calls

        print("Warning: find_random_adversarial failed to find an adversarial direction after {} trials, falling back to cumulative random walk.".format(n))
        num_calls = 1
        step_fb = 0.02
        perturbed = image
        while self.is_adversarial(perturbed) == -1:
            pert = torch.randn(image.shape).to(self.device)
            perturbed = image + num_calls * step_fb * pert
            perturbed = clip_image_values(perturbed, self.lb, self.ub).to(self.device)
            num_calls += 1
        return perturbed, num_calls



    def bin_search(self, x_0, x_random, max_calls=100):
        """Standard bin_search (kept for compatibility; v4's Attack uses line_search instead)."""
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
            u = torch.randn(x_o.shape).to(self.device)
        u = u.to(self.device)
        u = u - torch.dot(u.reshape(-1), v.reshape(-1)) * v
        u_norm = torch.norm(u)
        if u_norm < 1e-8:
            u = torch.randn(x_o.shape).to(self.device)
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

        # ★ v4 path A: plain bin_search at init (LS moved to post-rotation)
        x_b, query_b = self.bin_search(self.src_img, x_random)

        x_b_inv = self.inv_tf(copy.deepcopy(x_b.cpu()[0,:,:,:].squeeze()), self.mean, self.std)
        norm_initial = torch.norm(x_b_inv - x_inv)
        norms.append(norm_initial)
        q_num = query_random + query_b
        print('Initial boundary norm', torch.norm(norm_initial).item())
        print('query_b (line_search)', query_b)
        print('initial query', q_num)

        n_query.append(q_num)
        size = self.src_img.shape

        outer_iter = self.iteration
        lam1, lam2, lam3 = 0, 0, 1            # ★ random (winning variant in v3)

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
            d3 = self._proj_and_normalize(torch.randn(x_b.shape).to(self.device), v_new)

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

            # ★ v4 path A: LS refinement along NEW radial direction (x_o → x_adv)
            # This direction was never searched by walk; non-convex multi-crossing
            # may have left a closer adv boundary point hidden along it.
            if (it + 1) % self.LS_every == 0:
                x_adv, ls_q = self._line_search_closest(self.src_img, x_adv, max_calls=15)
                qs += ls_q

            x_e_prev = x_adv
            x_b_prev = x_b
            u_prev = u_new
            x_b = x_adv

            q_num = q_num + qs
            total_boundary_queries += qs

            x_adv_inv = self.inv_tf(copy.deepcopy(x_adv.cpu()[0,:,:,:].squeeze()), self.mean, self.std)
            norm = torch.norm(x_inv - x_adv_inv)

            if it % 50 == 0 or it == outer_iter - 1:
                if self.verbose_control == 'Yes':
                    print('Manifold2D-v4 iter -> ' + str(it) +
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
