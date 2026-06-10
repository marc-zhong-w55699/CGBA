import copy
import numpy as np
import torch
from utils import clip_image_values
from torch.autograd import Variable
import math


# ============================================================================
# CIFAR-10 M2D-explore (λ=0.2,0.2,0.6) + v5b adaptive theta_max
#                                       + v6.4 floor-bounce
# Pure attack class. No DCT.
#
# v6.4 KEY: replaces all "escape mechanism" attempts (v6.0-v6.3) with a
# much simpler mechanism — when adaptive θ_max stays at floor for ≥ N
# consecutive iterations, FORCE θ_max back to the initial value (π/3) and
# clear momentum state. This gives 2D a chance to find a larger swing in
# the next iter without spending any escape queries.
#
# Mechanism:
#   if floor_streak >= bump_streak:
#       theta_max_cur = bump_target        # bump to π/6 = 30° (conservative)
#       u_prev, x_e_prev, x_b_prev = None, None, None   # fresh u
#       floor_streak = 0
#
# Cost: ZERO escape queries. The bump just lets the next 2D step probe a
# bigger angle (mostly wasting ~5 sign-fail queries that iter, but no
# external machinery).
#
# Why v6.0-v6.3 failed (background):
#   * v6.0-v6.2 (forced shrink γr): jumps into narrow adv pocket → trap
#   * v6.3 (1D ray): random direction in R^n almost never adv at radius r
#     on ViT — 0% success rate observed.
#
# v6.4 doesn't assume any "exploitable pocket". It only relies on the
# stochastic chance that fresh u + bigger θ_max range finds a swing.
# ============================================================================


class Proposed_attack():
    def __init__(self, model, src_img, mean, std, lb, ub, dim_reduc_factor=4,
                 tar_img=None, iteration=1600, tol=1e-5, attack_method='manifold_search_2d',
                 verbose_control='Yes',
                 theta_max=math.pi / 3,
                 theta_min_bound=math.pi / 60,
                 theta_max_bound=math.pi / 2,
                 grow_factor=1.15,
                 shrink_factor=0.85,
                 shrink_thresh=0.15,
                 BS_iter=3,
                 # ★ v6.4 floor bounce params
                 bump_floor_streak=10,        # bump θ_max after this many consecutive floor iters
                 bump_target=math.pi / 36,      # value to bump to; π/6 = 30° (conservative, vs initial π/3)
                 bump_warmup=100,              # earliest iter to allow bump
                 bump_max_per_image=50):       # safety cap on number of bumps per image
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

        # ★ v6.4 floor bounce
        self.bump_floor_streak   = bump_floor_streak
        self.bump_target         = bump_target
        self.bump_warmup         = bump_warmup
        self.bump_max_per_image  = bump_max_per_image

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

        probe_angle = theta_max / 4.0
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
                return x_b, num_calls, 0.0

        best_angle, x_best, bs_q = self._circ_binary_search(x_o, r, v, u, s, theta_max)
        num_calls += bs_q

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
        lam1, lam2, lam3 = 0.2, 0.2, 0.6      # ★ explore

        u_prev = None
        x_e_prev = None
        x_b_prev = None
        x_adv = x_b

        # v5 adaptive theta_max
        theta_max_cur = self.theta_max
        theta_history = []

        # ★ v6.4 floor bounce state
        floor_streak  = 0
        bump_count    = 0

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
            # ★ v6.4: floor bounce
            # ============================================================
            bump_log = ''
            is_at_floor = theta_max_cur <= 1.1 * self.theta_min_bound
            if is_at_floor:
                floor_streak += 1
            else:
                floor_streak = 0

            if (it >= self.bump_warmup
                and floor_streak >= self.bump_floor_streak
                and bump_count < self.bump_max_per_image):

                # BUMP: force θ_max back to target, clear momentum
                bump_count += 1
                theta_max_cur = self.bump_target
                u_prev = None
                x_e_prev = None
                x_b_prev = None
                floor_streak = 0
                bump_log = (f'  [BUMP #{bump_count} θ_max→{math.degrees(self.bump_target):.0f}°]')

            if it % 50 == 0 or it == outer_iter - 1 or bump_log:
                if self.verbose_control == 'Yes':
                    print('Manifold2D-v6.4-explore-bump iter -> ' + str(it) +
                          '   Queries ' + str(q_num) +
                          '   norm -> ' + f'{norm.item():.3f}' +
                          f'   inner_q={qs}' +
                          f'   θ_max={math.degrees(theta_max_cur):.1f}°' +
                          f'   best_θ={math.degrees(best_angle):.1f}°' +
                          f'   flr_streak={floor_streak}' +
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
