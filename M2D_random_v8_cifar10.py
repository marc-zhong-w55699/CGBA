import copy
import numpy as np
import torch
from utils import clip_image_values
from torch.autograd import Variable
import math


# ============================================================================
# CIFAR-10 M2D-random (λ=0,0,1) + v5b adaptive theta_max
#                                + v6.4 reverse bump (CIFAR-tuned)
#                                + v7 optimized sign probe
# Pure attack class. No DCT.
#
# ★ v7 inherited (probe=θ_max/16, fallback continuation, Smart-BS init).
#
# ★ v8 KEY CHANGE: replace binary search with increment-doubling walk.
#
# Motivation:
#   Adaptive θ_max is only a *weak prior* on the current iter's true best_θ
#   (boundary curvature varies per iter as u is resampled). Constraining BS
#   to [0, theta_max_cur] artificially caps best_θ — BS often misses large
#   swings allowed by current geometry. Walk decouples search from adaptive:
#
#   - Walk starts at sign-verified probe_angle (init_lower equivalent)
#   - Each step, advance by δ; on success, δ *= 2 (geometric expansion)
#   - Stop on first non-adv OR hit theta_max_bound (60° hard cap)
#   - Cost = log2(best_θ / probe), variable per-iter
#
# Trade-off vs v7 BS:
#   - Walk in late-stuck (best≈probe): 1-2 walk q + 1 sign = 2-3q (vs 4q)
#   - Walk in mid (best~2°):           2-3 walk q + 1 sign = 3-4q (vs 4q)
#   - Walk in early (best~5°):         3-4 walk q + 1 sign = 4-5q (vs 4q)
#   Average ~25% query reduction → ~33% more outer iters at same budget.
#
# Precision: each walk step found within ×2 factor (no refine — boundary
#   varies across iters anyway, so single-iter precision is less critical).
#
# v6.4 KEY: when best_θ stays small (≤ thresh, excluding sign-fail) for K
# consecutive iters, halve current θ_max (clamped at bump_target as floor)
# and clear momentum state. This gives the next 2D step a smaller probe
# angle that fits inside the boundary's narrow adversarial sliver and
# refreshes the direction proposal.
#
# Default config (CIFAR-tuned, conservative):
#   theta_min_bound        = π/90   (= 2°,   adaptive floor)
#   bump_best_theta_thresh = π/360  (= 0.5°, strict trigger)
#   bump_streak            = 3      (consecutive small best_θ needed)
#   bump_target            = π/180  (= 1°,   halving floor — 1 step below adaptive)
#   bump_cooldown          = 20     (min iters between bumps)
#   bump_warmup            = 500    (skip exploration phase)
#   bump_max_per_image     = 50     (safety cap)
#
# Mechanism (per outer iter):
#   is_small_best = (
#       (0 < best_angle ≤ bump_best_theta_thresh)           # (a) absolute small
#       or (best_angle > 0 and theta_max_cur ≥ R × best_angle)  # (b) relative loose
#   )                                                        # both exclude sign-fail
#   if is_small_best:                small_best_streak += 1
#   else:                            small_best_streak = 0
#
#   if (it ≥ warmup
#       and small_best_streak ≥ bump_streak
#       and bump_count < cap
#       and cooldown_ctr == 0
#       and norm_cur / norm_init ≥ bump_norm_gate):   # ★ A: image-aware gate
#       new_theta = max(theta_max_cur / 2, bump_target)
#       theta_max_cur = new_theta
#       u_prev = x_e_prev = x_b_prev = None    # fresh momentum
#       small_best_streak = 0
#       cooldown_ctr = bump_cooldown
#
# The norm_gate prevents bump on already-converged images where v5b's
# adaptive trajectory is already near-optimal.
#
# Cost: ZERO extra queries. The bump iter actually saves queries (smaller
# probe angle → higher sign-success rate, less fallback halving).
#
# Why earlier mechanisms (v6.0-v6.3) failed:
#   * v6.0-v6.2 (forced radial shrink γr): jumps into narrow adv pocket → trap
#   * v6.3 (1D ray, random direction in R^n): 0% success
#
# CIFAR vs ImageNet tuning rationale:
#   * CIFAR ViT v5b is already near boundary-geometry limit → bumps too eager
#     hurt convergence. We require stricter threshold (0.5° vs 1°), longer
#     streak (3 vs 2), later warmup (500 vs 100), and shallower bump (1° vs 0.5°).
# ============================================================================


class Proposed_attack():
    def __init__(self, model, src_img, mean, std, lb, ub, dim_reduc_factor=4,
                 tar_img=None, iteration=1600, tol=1e-5, attack_method='manifold_search_2d',
                 verbose_control='Yes',
                 theta_max=math.pi / 3.6,
                 theta_min_bound=math.pi / 90,    # ★ v6.4: lowered from π/60 (3°) to π/90 (2°)
                 theta_max_bound=math.pi / 3,
                 grow_factor=1.15,
                 shrink_factor=0.85,
                 shrink_thresh=0.15,
                 BS_iter=3,
                 # ★ v6.4 reverse bump params (CIFAR-tuned: conservative)
                 bump_best_theta_thresh=math.pi / 360,  # absolute trigger: best_θ ≤ this (= 1°)
                 bump_ratio_thresh=5.0,        # ★ NEW: ratio trigger — θ_max ≥ N × best_θ
                 bump_streak=3,                # consecutive small/loose iters needed
                 bump_target=math.pi / 180,    # ★ B': 1° (halving floor)
                 bump_cooldown=50,             # min iters between bumps
                 bump_warmup=500,              # earliest iter to allow bump
                 bump_max_per_image=50,        # safety cap
                 bump_norm_gate=0.5):          # ★ A: only bump when norm_cur ≥ norm_init × this
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



    def _circ_binary_search(self, x_o, r, v, u, s, theta_max,
                             init_lower=0.0, init_best=0.0, init_x_best=None):
        """v7 Smart-BS — kept here as reference / ablation. Not called by v8."""
        lower, upper = init_lower, theta_max
        best_angle = init_best
        x_best = init_x_best
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
                        init_angle, init_x, theta_max_cur=None):
        """★ v8 INCREMENT-DOUBLING WALK (replaces BS).

           Start from sign-verified (init_angle, init_x). Each step, advance
           by an increment δ that doubles after each successful test. Stop
           when next angle is non-adv or hits theta_safety_cap.

           ★ v8b: δ_init = max(2 × probe, θ_max_cur / 4)
              Probe was kept small (θ_max/16) for sign-success rate, but
              empirically best_θ / probe ≈ 6× (median), so walk wasted ~2
              steps climbing. Boosting δ_init to ~θ_max/4 lets walk reach
              the typical best_θ range in 1-2 steps.

           Sequence (initial increment = δ_init):
              θ_0 = init_angle             ← probe (sign-verified)
              θ_1 = θ_0 + δ_init
              θ_2 = θ_1 + 2·δ_init
              θ_k = θ_0 + δ_init × (2^k - 1)

           No refine — single-iter precision matters less than per-iter cost
           savings (boundary varies across iters anyway).

           Returns (best_angle, x_best, num_queries).
        """
        best_angle = init_angle
        x_best     = init_x
        num_q      = 0

        θ_cur = init_angle
        # ★ v8b: floor δ_init at θ_max_cur/4 to avoid tiny first step
        δ_min = (theta_max_cur / 4.0) if theta_max_cur is not None else 0.0
        δ     = max(init_angle * 2.0, δ_min)

        while True:
            θ_next = θ_cur + δ
            if θ_next >= theta_safety_cap:
                break    # 撞 safety cap

            x_test = self._circ_x_at(x_o, r, v, u, s, θ_next)
            num_q += 1

            if self.is_adversarial(x_test) == 1:
                best_angle = θ_next
                x_best     = x_test
                θ_cur      = θ_next
                δ         *= 2.0    # increment doubles
            else:
                break    # 撞 boundary

        return best_angle, x_best, num_q



    def manifold_search_2d(self, x_o, x_b,
                           beta=math.pi / 30,
                           beta_min=math.pi / 1000,
                           u=None,
                           theta_max_cur=None,
                           **kwargs):
        """Returns (x_e, num_calls, best_angle)."""
        num_calls = 0
        theta_max = theta_max_cur if theta_max_cur is not None else self.theta_max

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

        probe_angle = theta_max / 16.0   # ★ v7: /4 → /16 (higher 1-shot success)
        s = 0
        sign_found_angle = None    # ★ v7 Smart-BS: track where sign was confirmed
        sign_found_x     = None    # ★ v7 Smart-BS: keep the adv point for BS init

        x_pos = self._circ_x_at(x_o, r, v, u, +1, probe_angle)
        num_calls += 1
        if self.is_adversarial(x_pos) == 1:
            s = +1
            sign_found_angle = probe_angle
            sign_found_x     = x_pos
        else:
            x_neg = self._circ_x_at(x_o, r, v, u, -1, probe_angle)
            num_calls += 1
            if self.is_adversarial(x_neg) == 1:
                s = -1
                sign_found_angle = probe_angle
                sign_found_x     = x_neg

        if s == 0:
            cur_beta = probe_angle / 2.0   # ★ v7: continue from probe (was fixed π/30)
            while cur_beta > beta_min:
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

        # ★ v8 INCREMENT-DOUBLING WALK (replaces BS)
        # Walk uses theta_max_bound as hard safety cap (= π/3 = 60°),
        # NOT adaptive theta_max (which is a weak prior from previous iter).
        # This decouples search range from adaptive — walk finds large best_θ
        # when boundary allows, regardless of recent best_θ history.
        best_angle, x_best, walk_q = self._circ_inc_walk(
            x_o, r, v, u, s,
            theta_safety_cap=self.theta_max_bound,
            init_angle=sign_found_angle,
            init_x=sign_found_x,
            theta_max_cur=theta_max,         # ★ v8b: δ_init floor = θ_max/4
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

        # v5 adaptive theta_max
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

            x_adv, qs, best_angle = self.manifold_search_2d(
                self.src_img, x_b, u=u_new, theta_max_cur=theta_max_cur
            )

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
            # (both clauses require best_θ > 0 to exclude sign-fail noise)
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

            # cooldown countdown
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

                # BUMP: halve current θ_max (clamped at bump_target as floor), clear momentum
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
                    print('Manifold2D-v8b-random-walk iter -> ' + str(it) +
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
