import copy
import numpy as np
import torch
from utils import clip_image_values
from torch.autograd import Variable
import math


# ============================================================================
# ImageNet M2D-random (λ=0,0,1)  v10 = v6 + v8 walk + v10 sign-decoupled
#   + DCT (square low-freq block, dct_ratio=1/8)
#   + v3 geometry (circular evolution)
#   + v5b adaptive theta_max
#   + v6.4 reverse floor-bump
#   + v8 increment-doubling walk (replaces BS)
#   + v10 decoupled sign probe: test BOTH sides at min(max(θ_max/4, 0.5°), 8°)
#     walk init = sign_found_angle, walk δ_init = θ_max/8 (independent)
#
# v6.4 KEY (vs v5): when adaptive θ_max stays at floor for ≥ N consecutive
# iters, FORCE θ_max to a value SMALLER than the floor (a "reverse bump").
# This gives the next 2D step a tiny probe angle that's more likely to fit
# inside the boundary's narrow adversarial sliver. Then adaptive recovers.
#
# Default config:
#   theta_min_bound = π/90   (= 2°,   adaptive floor; lowered from π/60)
#   bump_target     = π/360  (= 0.5°, reverse bump destination, below floor)
#   bump_floor_streak = 10   (trigger after 10 consec floor iters)
#
# Cost: ZERO extra queries. The bump iter spends fewer queries than usual
# (because tiny probe angle ⇒ sign probe usually succeeds first try).
#
# ImageNet-specific: keeps the simple DCT (square low-freq, dct_ratio=1/8)
# direction sampling from v5. DCT prior + reverse bump together.
# ============================================================================


class Proposed_attack():
    def __init__(self, model, src_img, mean, std, lb, ub, dim_reduc_factor=4,
                 tar_img=None, iteration=700, tol=1e-5, attack_method='manifold_search_2d',
                 verbose_control='Yes',
                 dct_ratio=1.0/8,
                 theta_max=math.pi / 3.6,         # v11.2: 50° init
                 theta_min_bound=math.pi / 90,    # 2°
                 theta_max_bound=math.pi / 3,     # v11.2: 60° cap
                 grow_factor=1.15,
                 shrink_factor=0.85,
                 shrink_thresh=0.15,
                 BS_iter=3,
                 # ★ v6.4 reverse bump params (ImageNet-tuned, with new mechanisms)
                 bump_best_theta_thresh=math.pi / 180,  # 1° (ImageNet best_θ P50 ≈ 1°)
                 bump_ratio_thresh=5.0,            # ★ v6.5 TEST: ratio trigger θ_max ≥ N×best_θ
                 bump_streak=2,                    # ImageNet: 2 (faster than CIFAR's 3)
                 bump_target=math.pi / 360,        # 0.5° (deeper bump than CIFAR's 1°)
                 bump_cooldown=50,                 # ImageNet: 20 (shorter than CIFAR's 50)
                 bump_warmup=100,                  # ImageNet: 100 (earlier than CIFAR's 500)
                 bump_max_per_image=0,             # ★ v11 test: bump disabled
                 bump_norm_gate=0.0,               # 0 = always on (ImageNet needs bump)
                 # ★ v10 decoupled sign probe (cap + floor)
                 sign_probe_cap=math.pi/22.5,      # max sign_probe angle (= 8°)
                 sign_probe_floor=math.pi/180,     # 1° (default v11)
                 # ★ v11 u-rejection + halving cap
                 max_u_attempts=4,                 # retry u up to N times
                 halving_min=math.pi/360,          # 0.5° sign fallback halving stop
                 # ★ v12 walk-halving
                 walk_halving_min=math.pi/360):    # ★ v12.2: 0.5° walk halving stop (was 0.25°)
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
        self.dct_ratio = dct_ratio
        self.theta_max = theta_max
        self.theta_min_bound = theta_min_bound
        self.theta_max_bound = theta_max_bound
        self.grow_factor = grow_factor
        self.shrink_factor = shrink_factor
        self.shrink_thresh = shrink_thresh
        self.BS_iter = BS_iter

        # ★ v6.4 reverse bump (best_θ-based)
        self.bump_best_theta_thresh = bump_best_theta_thresh
        self.bump_ratio_thresh      = bump_ratio_thresh
        self.bump_streak            = bump_streak
        self.bump_target            = bump_target
        self.bump_cooldown          = bump_cooldown
        self.bump_warmup            = bump_warmup
        self.bump_max_per_image     = bump_max_per_image
        self.bump_norm_gate         = bump_norm_gate

        # ★ v10 decoupled sign probe (cap + floor)
        self.sign_probe_cap   = sign_probe_cap
        self.sign_probe_floor = sign_probe_floor

        # ★ v11 u-rejection + halving cap
        self.max_u_attempts = max_u_attempts
        self.halving_min    = halving_min

        # ★ v12 walk-halving
        self.walk_halving_min = walk_halving_min

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.all_queries = 0

        _, _, H, W = self.src_img.shape
        self._H, self._W = H, W
        self._Dh = self._dct_matrix(H).to(self.device)
        self._Dw = self._Dh if W == H else self._dct_matrix(W).to(self.device)
        self._k_h = max(1, int(round(H * self.dct_ratio)))
        self._k_w = max(1, int(round(W * self.dct_ratio)))



    def _dct_matrix(self, N):
        n = torch.arange(N, dtype=torch.float32).view(1, -1)
        k = torch.arange(N, dtype=torch.float32).view(-1, 1)
        D = torch.cos(math.pi * (2 * n + 1) * k / (2 * N))
        D = D * math.sqrt(2.0 / N)
        D[0] = D[0] / math.sqrt(2.0)
        return D

    def _idct2d(self, coeff):
        tmp = torch.einsum('ij,cjk->cik', self._Dh.t(), coeff)
        x   = torch.einsum('cik,kl->cil', tmp, self._Dw)
        return x

    def _low_freq_random(self, shape):
        _, C, H, W = shape
        coeff = torch.zeros(C, H, W, device=self.device)
        coeff[:, :self._k_h, :self._k_w] = torch.randn(C, self._k_h, self._k_w, device=self.device)
        spatial = self._idct2d(coeff)
        return spatial.unsqueeze(0)



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
            u = self._low_freq_random(image.shape).to(self.device)
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
            pert = self._low_freq_random(image.shape).to(self.device)
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
        """v5/v6/v7 BS — kept here as reference / ablation. Not called by v10."""
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



    def _circ_inc_walk(self, x_o, r, v, u, s, theta_safety_cap,
                        init_angle, init_x, delta_init=None):
        """★ v8 increment-doubling walk (replaces BS).
           ★ v10: delta_init decoupled from init_angle (default 2× init_angle).
           ★ v12: HALVING-AFTER-FAIL — on failed test, halve δ and retry
                  until adv is found OR δ < walk_halving_min. This closes the
                  systematic 33-37% gap between walk waypoints (sweep evidence:
                  late gap P50 = 0.52°, 56% iter have gap ∈ [0.3°, 1°]).
        """
        best_angle = init_angle
        x_best     = init_x
        num_q      = 0

        θ_cur = init_angle
        δ     = delta_init if delta_init is not None else (init_angle * 2.0)

        while True:
            θ_next = θ_cur + δ
            if θ_next >= theta_safety_cap:
                break    # safety cap

            x_test = self._circ_x_at(x_o, r, v, u, s, θ_next)
            num_q += 1

            if self.is_adversarial(x_test) == 1:
                best_angle = θ_next
                x_best     = x_test
                θ_cur      = θ_next
                δ         *= 2.0    # increment doubles
            else:
                # ★ v12.1: halving retry after fail — STOP on first adv (no chain)
                # Trade-off: capture main gap [0.25°, 1°] without expensive continuation.
                while δ > self.walk_halving_min:
                    δ /= 2.0
                    θ_h = θ_cur + δ
                    if θ_h >= theta_safety_cap:
                        continue
                    x_h = self._circ_x_at(x_o, r, v, u, s, θ_h)
                    num_q += 1
                    if self.is_adversarial(x_h) == 1:
                        best_angle = θ_h
                        x_best     = x_h
                        break
                break    # ★ v12.1: always end walk after any fail (with or without halve)

        return best_angle, x_best, num_q



    def manifold_search_2d(self, x_o, x_b,
                           beta=math.pi / 30,
                           beta_min=math.pi / 1000,
                           u=None,
                           theta_max_cur=None,
                           reject_at_floor=False,     # ★ v11: return -1 if sign fails at floor
                           **kwargs):
        """Returns (x_e, num_calls, best_angle).
        ★ v11: reject_at_floor=True → early-return -1 on double-sign-fail (retry with new u)
        ★ v11.1: skip walk when sign_found_angle < floor (truly narrow u)
        """
        num_calls = 0
        theta_max = theta_max_cur if theta_max_cur is not None else self.theta_max

        diff = x_b - x_o
        r = torch.norm(diff)
        v = diff / r

        if u is None:
            u = self._low_freq_random(x_o.shape).to(self.device)
        u = u.to(self.device)
        u = u - torch.dot(u.reshape(-1), v.reshape(-1)) * v
        u_norm = torch.norm(u)
        if u_norm < 1e-8:
            u = self._low_freq_random(x_o.shape).to(self.device)
            u = u - torch.dot(u.reshape(-1), v.reshape(-1)) * v
            u_norm = torch.norm(u)
        u = u / u_norm

        # ★ v10: decoupled sign probe (larger) + walk (decoupled δ)
        sign_probe_angle = min(max(theta_max / 4.0, self.sign_probe_floor),
                                self.sign_probe_cap)
        s = 0
        sign_found_angle = None
        sign_found_x     = None

        # Test +s and -s at sign_probe (always 2 q)
        x_pos = self._circ_x_at(x_o, r, v, u, +1, sign_probe_angle)
        num_calls += 1
        pos_adv = (self.is_adversarial(x_pos) == 1)

        x_neg = self._circ_x_at(x_o, r, v, u, -1, sign_probe_angle)
        num_calls += 1
        neg_adv = (self.is_adversarial(x_neg) == 1)

        if pos_adv and not neg_adv:
            s = +1
            sign_found_angle = sign_probe_angle
            sign_found_x     = x_pos
        elif neg_adv and not pos_adv:
            s = -1
            sign_found_angle = sign_probe_angle
            sign_found_x     = x_neg
        elif pos_adv and neg_adv:
            s = +1
            sign_found_angle = sign_probe_angle
            sign_found_x     = x_pos

        # ★ v11: u-rejection early return
        if s == 0 and reject_at_floor:
            return x_b, num_calls, -1.0

        if s == 0:
            cur_beta = sign_probe_angle / 2.0
            # ★ v11.1: use self.halving_min (0.25° → halving only 1 test at floor/2)
            while cur_beta > self.halving_min:
                x_pos = self._circ_x_at(x_o, r, v, u, +1, cur_beta)
                num_calls += 1
                if self.is_adversarial(x_pos) == 1:
                    s = +1
                    sign_found_angle = cur_beta
                    sign_found_x     = x_pos
                    break
                x_neg = self._circ_x_at(x_o, r, v, u, -1, cur_beta)
                num_calls += 1
                if self.is_adversarial(x_neg) == 1:
                    s = -1
                    sign_found_angle = cur_beta
                    sign_found_x     = x_neg
                    break
                cur_beta = cur_beta / 2
            if s == 0:
                return x_b, num_calls, 0.0

        # ★ v11.1: skip walk when fallback halving found sign at < floor
        if sign_found_angle < self.sign_probe_floor:
            best_angle = sign_found_angle
            x_best     = sign_found_x
            walk_q     = 0
            num_calls += walk_q
            if x_best is not None and best_angle > 0:
                return x_best, num_calls, best_angle
            else:
                return x_b, num_calls, 0.0

        # ★ v11.2: walk from sign_found_angle,
        # δ_init = max(θ_max/8, sign_probe_angle) — walk step at least as big
        # as sign probe (ensures walk expands outward, not just inches around probe)
        best_angle, x_best, walk_q = self._circ_inc_walk(
            x_o, r, v, u, s,
            theta_safety_cap=self.theta_max_bound,
            init_angle=sign_found_angle,
            init_x=sign_found_x,
            delta_init=max(theta_max / 8.0, sign_probe_angle),
        )
        num_calls += walk_q

        if x_best is not None and best_angle > 0:
            return x_best, num_calls, best_angle
        else:
            return x_b, num_calls, 0.0



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
        lam1, lam2, lam3 = 0, 0, 1            # ★ random

        u_prev = None
        x_e_prev = None
        x_b_prev = None
        x_adv = x_b

        theta_max_cur = self.theta_max
        theta_history = []

        # ★ v6.4 reverse bump state (best_θ-based)
        small_best_streak = 0
        bump_count        = 0
        bump_cooldown_ctr = 0

        for it in range(outer_iter):
            diff = x_b - self.src_img
            r_cur = torch.norm(diff)
            if r_cur < 1e-8:
                break
            v_new = diff / r_cur

            d1 = self._proj_and_normalize(x_e_prev - x_b_prev, v_new) \
                 if (x_e_prev is not None and x_b_prev is not None) else None
            d2 = self._proj_and_normalize(u_prev, v_new) if u_prev is not None else None

            # ★ v11 u-rejection: retry up to max_u_attempts if boundary < floor
            qs_total = 0
            for u_attempt in range(self.max_u_attempts):
                # ★ ImageNet uses DCT low-freq sampling (not torch.randn)
                d3 = self._proj_and_normalize(self._low_freq_random(x_b.shape).to(self.device), v_new)
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

                is_last = (u_attempt == self.max_u_attempts - 1)
                x_adv, qs, best_angle = self.manifold_search_2d(
                    self.src_img, x_b, u=u_new, theta_max_cur=theta_max_cur,
                    reject_at_floor=(not is_last),   # only reject on non-last attempts
                )
                qs_total += qs
                if best_angle >= 0:
                    break
            qs = qs_total

            # v5b adaptive theta_max
            if best_angle > 0:
                if best_angle > 0.8 * theta_max_cur:
                    theta_max_cur = min(theta_max_cur * self.grow_factor, self.theta_max_bound)
                elif best_angle < self.shrink_thresh * theta_max_cur:
                    theta_max_cur = max(theta_max_cur * self.shrink_factor, self.theta_min_bound)

            theta_history.append(theta_max_cur)

            x_e_prev = x_adv
            x_b_prev = x_b
            u_prev = u_new
            x_b = x_adv

            q_num = q_num + qs
            total_boundary_queries += qs

            x_adv_inv = self.inv_tf(copy.deepcopy(x_adv.cpu()[0,:,:,:].squeeze()), self.mean, self.std)
            norm = torch.norm(x_inv - x_adv_inv)

            # ============================================================
            # ★ v6.4: reverse bump triggered by EITHER
            #   (a) best_θ ∈ (0, bump_best_theta_thresh]      "absolute small"
            #   (b) θ_max_cur ≥ bump_ratio_thresh × best_θ    "relative loose"
            # (both clauses exclude sign-fail; gated by norm_cur/norm_init)
            # ============================================================
            bump_log = ''
            is_abs_small  = (0 < best_angle <= self.bump_best_theta_thresh)
            is_rel_loose  = (best_angle > 0
                             and theta_max_cur >= self.bump_ratio_thresh * best_angle)
            is_small_best = is_abs_small or is_rel_loose
            if is_small_best:
                small_best_streak += 1
            else:
                small_best_streak = 0

            if bump_cooldown_ctr > 0:
                bump_cooldown_ctr -= 1

            # ★ A: image-aware gate — don't bump if image already mostly converged
            norm_ratio = float(norm.item()) / max(float(norm_initial.item()), 1e-12)
            gate_ok = (norm_ratio >= self.bump_norm_gate)

            if (it >= self.bump_warmup
                and small_best_streak >= self.bump_streak
                and bump_count < self.bump_max_per_image
                and bump_cooldown_ctr == 0
                and gate_ok):

                # BUMP: halve current θ_max (clamped at bump_target as floor)
                bump_count += 1
                new_theta = max(theta_max_cur / 2.0, self.bump_target)
                theta_max_cur = new_theta
                u_prev = None
                x_e_prev = None
                x_b_prev = None
                small_best_streak = 0
                bump_cooldown_ctr = self.bump_cooldown
                bump_log = (f'  [BUMP #{bump_count} θ_max→{math.degrees(new_theta):.2f}° r/r₀={norm_ratio:.2f}]')

            if it % 50 == 0 or it == outer_iter - 1 or bump_log:
                if self.verbose_control == 'Yes':
                    print('Manifold2D-v12.2-random-halve05 iter -> ' + str(it) +
                          '   Queries ' + str(q_num) +
                          '   norm -> ' + f'{norm.item():.3f}' +
                          f'   inner_q={qs}' +
                          f'   θ_max={math.degrees(theta_max_cur):.1f}°' +
                          f'   best_θ={math.degrees(best_angle):.1f}°' +
                          f'   sml_streak={small_best_streak}' +
                          bump_log)

            norms.append(norm)
            n_query.append(q_num)

        print(f'\n── Query num ──────────────────────────────────')
        print(f'Gradient estimation queries : {total_grad_queries}')
        print(f'Boundary search queries     : {total_boundary_queries}')
        print(f'Total queries               : {q_num}')
        if theta_history:
            print(f'θ_max trajectory: init={math.degrees(theta_history[0]):.1f}°, '
                  f'final={math.degrees(theta_history[-1]):.1f}°, '
                  f'mean={math.degrees(np.mean(theta_history)):.1f}°, '
                  f'min={math.degrees(min(theta_history)):.1f}°, '
                  f'max={math.degrees(max(theta_history)):.1f}°')
        print(f'Bump count: {bump_count}/{self.bump_max_per_image}')
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
