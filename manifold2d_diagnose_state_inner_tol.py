"""
Manifold2D state + inner_n + tol 联合 ablation（β 保持原版）
-----------------------------------------------------------
**改两个参数**：inner_n=15, tol=1e-5。β 保持原版 π/30。

    ┌─────────────┬─────────────┬─────────────┐
    │ 参数        │ pro_atk 原版│ 本脚本      │
    ├─────────────┼─────────────┼─────────────┤
    │ beta        │ π/30 ≈ 6°   │ π/30 (保持) │
    │ inner_n     │ 10          │ 15          │  ★
    │ tol         │ 1e-4        │ 1e-5        │  ★
    └─────────────┴─────────────┴─────────────┘

假说：β 才是状态机的真正破坏者，inner_n 和 tol 各自单独不会破坏 EXPLORE 触发。
跳过 β 但保留另外两个 tuning，理论上能拿到"两边好处"。

预期对照（重点看 Image #2）：
  - state 原版:                        9.34  （基线）
  - state + β only:                    13.27 （β 是大破坏）
  - state + β + inner_n:               10.49 （inner_n 补偿大半）
  - state + tuned (β+inner_n+tol):     11.26 （tol 再添小破坏）
  - **state + inner_n + tol (无 β):    ??**  本脚本

最理想结果：Image #2 ≈ 9 ~ 9.5（保留 state 原版的救场），
            Image #1, #3 接近 tuned 全部（拿到 inner_n 和 tol 的几何收益）

用法:
    python manifold2d_diagnose_state_inner_tol.py --num_img 3 --iter 100
"""

import argparse
import copy
import math

import numpy as np
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from PIL import Image

from models_cifar10 import load_model
from pro_atk import Proposed_attack
from utils import clip_image_values, valid_bounds


# ── 状态机参数（不动） ─────────────────────────────────────────────────────────
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


# ── ★ 改两个参数，β 保持原版 ─────────────────────────────────────────────────
ORIG_BETA     = math.pi / 30   # 保持原版（重要！）
TUNED_INNER_N = 15             # ★
TUNED_TOL     = 1e-5           # ★


class DiagnosticStateInnerTolAttack(Proposed_attack):
    """State + inner_n=15 + tol=1e-5; β stays original."""

    def __init__(self, *args, **kwargs):
        kwargs.setdefault('tol', TUNED_TOL)   # ★
        super().__init__(*args, **kwargs)
        self.last_actual_inner = 0
        self.state_transitions = []

    def manifold_search_2d(self, x_o, x_b,
                           alpha=0.99,
                           beta=ORIG_BETA,              # 保持原版 π/30
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
            u = torch.randn(x_o.shape).to(self.device)
        u = u.to(self.device)
        u = u - torch.dot(u.reshape(-1), v.reshape(-1)) * v
        u_norm = torch.norm(u)
        if u_norm < 1e-8:
            u = torch.randn(x_o.shape).to(self.device)
            u = u - torch.dot(u.reshape(-1), v.reshape(-1)) * v
            u_norm = torch.norm(u)
        u = u / u_norm

        s = 0
        cur_beta = beta
        while cur_beta > beta_min:
            cos_b, sin_b = math.cos(cur_beta), math.sin(cur_beta)
            cand_p = clip_image_values(x_o + r * (v * cos_b + u * sin_b),
                                       self.lb, self.ub).to(self.device)
            num_calls += 1
            if self.is_adversarial(cand_p) == 1:
                s = +1
                break
            cand_m = clip_image_values(x_o + r * (v * cos_b - u * sin_b),
                                       self.lb, self.ub).to(self.device)
            num_calls += 1
            if self.is_adversarial(cand_m) == 1:
                s = -1
                break
            cur_beta = cur_beta / 2

        if s == 0:
            self.last_actual_inner = 0
            return x_b, num_calls

        x_s = x_b
        x_e = x_b
        last_i = 0
        for i in range(1, n + 1):
            last_i = i
            w = (x_s - x_o) / torch.norm(x_s - x_o)
            alpha_i = alpha ** (1 + (n - i) / n)

            while True:
                cand = clip_image_values(x_o + (alpha_i * r) * w,
                                         self.lb, self.ub).to(self.device)
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
            x_s_cand = clip_image_values(x_o + r * (v * cos_a + s * u * sin_a),
                                         self.lb, self.ub).to(self.device)
            num_calls += 1
            if self.is_adversarial(x_s_cand) != 1:
                x_e = x_r
                break
            else:
                x_s = x_s_cand
                x_e = x_s

        x_e = clip_image_values(x_e, self.lb, self.ub).to(self.device)
        self.last_actual_inner = last_i
        return x_e, num_calls

    def Attack(self):
        norms = []
        n_query = []

        x_inv = self.inv_tf(copy.deepcopy(self.src_img.cpu()[0, :, :, :].squeeze()),
                            self.mean, self.std)
        if self.tar_img is None:
            x_random, query_random = self.find_random_adversarial(self.src_img)
        else:
            x_random, query_random = self.tar_img, 0
        x_b, query_b = self.bin_search(self.src_img, x_random)
        x_b_inv = self.inv_tf(copy.deepcopy(x_b.cpu()[0, :, :, :].squeeze()),
                              self.mean, self.std)
        norm_initial = torch.norm(x_b_inv - x_inv)
        norms.append(norm_initial)
        q_num = query_random + query_b
        n_query.append(q_num)

        print(f'  [Init] r_cur={float(torch.norm(x_b - self.src_img).item()):.4f}  '
              f'norm_pix={norm_initial.item():.4f}  query={q_num}')

        outer_iter = self.iteration
        inner_n = TUNED_INNER_N         # ★

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
            prev_state = state
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
                    self.state_transitions.append((it, state, new_state))
                    state  = new_state
                    dwell  = 0
                    stag_k = 0
            lam1, lam2, lam3 = STATE_WEIGHTS[state]

            d1 = (self._proj_and_normalize(x_e_prev - x_b_prev, v_new)
                  if (x_e_prev is not None and x_b_prev is not None) else None)
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

            # 显式传 ORIG_BETA（虽然默认值也是它，明确一下）
            x_adv, qs = self.manifold_search_2d(
                self.src_img, x_b, n=inner_n, u=u_new, beta=ORIG_BETA
            )
            actual_inner = self.last_actual_inner

            r_after = float(torch.norm(x_adv - self.src_img).item())
            delta_r_this = r_now - r_after

            tag = f'[{state:7s}]'
            if prev_state != state:
                tag = f'[{state:7s}*]'

            print(f'  iter {it:3d}  {tag}  '
                  f'r_cur={r_after:.4f}  Δr={delta_r_this:+.5f}  '
                  f'inner={actual_inner}/{inner_n}  q+={qs}  '
                  f'stag={stag_k}  Δrel={delta_rel*100:+.2f}%  ratio={r_ratio:.3f}')

            x_e_prev = x_adv
            x_b_prev = x_b
            u_prev = u_new
            x_b = x_adv
            q_num = q_num + qs
            r_prev = r_now

            x_adv_inv = self.inv_tf(copy.deepcopy(x_adv.cpu()[0, :, :, :].squeeze()),
                                    self.mean, self.std)
            norm = torch.norm(x_inv - x_adv_inv)
            norms.append(norm)
            n_query.append(q_num)

        x_adv = clip_image_values(x_adv, self.lb, self.ub)
        return x_adv, n_query, norms


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_img', type=int, default=3)
    parser.add_argument('--iter', dest='iteration', type=int, default=100)
    parser.add_argument('--model', default='wideresnet40_2')
    args = parser.parse_args()

    torch.manual_seed(992); np.random.seed(992)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f'\n{"="*90}')
    print(f'  Manifold2D State + ★ inner_n + tol (β unchanged)')
    print(f'  num_img = {args.num_img}   iteration = {args.iteration}')
    print(f'  UNCHANGED: beta=π/{int(math.pi/ORIG_BETA)}')
    print(f'  ★ Changed: inner_n={TUNED_INNER_N}  tol={TUNED_TOL}')
    print(f'  (state thresholds all original)')
    print(f'  model = {args.model}    device = {device}')
    print(f'{"="*90}')

    mean = [0.4914, 0.4822, 0.4465]
    std  = [0.2023, 0.1994, 0.2010]

    cifar10 = datasets.CIFAR10(root='./data', train=False, download=True, transform=None)
    tf_normalize = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    def to_tensor(arr): return tf_normalize(Image.fromarray(arr))[None].to(device)

    net = load_model(args.model, ckpt_dir='checkpoints', device=device)

    done = 0
    all_transitions = []
    for idx in range(len(cifar10)):
        if done >= args.num_img:
            break
        im_pil, gt = cifar10[idx]
        x_0 = tf_normalize(im_pil)[None].to(device)
        with torch.no_grad():
            pred = torch.argmax(net(x_0)).item()
        if pred != gt:
            continue
        done += 1

        lb_np, ub_np = valid_bounds(im_pil, delta=255)
        lb = to_tensor(lb_np); ub = to_tensor(ub_np)

        print(f'\n── Image #{done}  (cifar idx={idx}, gt={gt}) ─────────────')
        attacker = DiagnosticStateInnerTolAttack(
            net, x_0, mean, std, lb, ub,
            iteration=args.iteration,
            verbose_control='No',
        )
        with torch.no_grad():
            x_adv, n_query, norms = attacker.Attack()
        with torch.no_grad():
            adv_pred = torch.argmax(net(x_adv)).item()

        trans = attacker.state_transitions
        all_transitions.append((done, idx, gt, trans, float(norms[-1])))

        print(f'  [Done] final_r={float(norms[-1]):.4f}  total_q={n_query[-1]}  '
              f'pred {gt}→{adv_pred}  {"SUCCESS" if adv_pred != gt else "FAIL"}')
        print(f'  [State transitions] {len(trans)} times')

    print(f'\n{"="*90}')
    print('  Summary:')
    for i, idx, gt, trans, final_r in all_transitions:
        print(f'  Image #{i} (gt={gt}, final_r={final_r:.3f}): '
              f'{len(trans)} state transitions')
    print(f'{"="*90}')


if __name__ == '__main__':
    main()
