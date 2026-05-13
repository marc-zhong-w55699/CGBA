"""
Manifold2D 调参诊断脚本（geometric / precision tuning）
-------------------------------------------------------
**不改方向参数**（λ、状态机阈值都不动），只调几何/精度参数：

    ┌─────────────┬─────────────┬─────────────┐
    │ 参数        │ 原版        │ 本脚本      │
    ├─────────────┼─────────────┼─────────────┤
    │ beta        │ π/30 ≈ 6°   │ π/60 ≈ 3°   │  ★ 减小旋转角，提高 inner 成功率
    │ inner_n     │ 10          │ 15          │  ★ 给"好链"更多发挥空间
    │ tol         │ 1e-4        │ 1e-5        │  ★ 让 bin_search 收得更紧
    └─────────────┴─────────────┴─────────────┘

变体选择：默认 exploit (0.4, 0.3, 0.3)，可通过 --variant 切换。

用法:
    python manifold2d_diagnose_tuned.py                   # exploit, 3 张图, iter=50
    python manifold2d_diagnose_tuned.py --variant random  # 切换 λ
    python manifold2d_diagnose_tuned.py --num_img 1       # 单张图
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
from pro_attack import Proposed_attack
from utils import clip_image_values, valid_bounds


# ── 5 个 Manifold2D 变体的 λ 三元组（未动） ───────────────────────────────────
VARIANTS = {
    'exploit': (0.4, 0.3, 0.3),
    'explore': (0.2, 0.2, 0.6),
    'refine':  (0.3, 0.5, 0.2),
    'random':  (0.0, 0.0, 1.0),
}

# ── ★ 调参常量 ───────────────────────────────────────────────────────────────
TUNED_BETA    = math.pi / 60   # ★ 原本 π/30
TUNED_INNER_N = 15             # ★ 原本 10
TUNED_TOL     = 1e-5           # ★ 原本 1e-4


class DiagnosticAttack(Proposed_attack):
    """Subclass of Proposed_attack with diagnostic instrumentation + tuned params."""

    def __init__(self, *args, lam=(0.4, 0.3, 0.3), **kwargs):
        # ★ 把 tol 注入到父类 __init__
        kwargs.setdefault('tol', TUNED_TOL)
        super().__init__(*args, **kwargs)
        self.lam1, self.lam2, self.lam3 = lam
        self.last_actual_inner = 0

    # ────────────────────────────────────────────────────────────────────────
    # 完全复制父类逻辑，唯一改动：记录 actual_inner；接受调过的 beta default
    # ────────────────────────────────────────────────────────────────────────
    def manifold_search_2d(self, x_o, x_b,
                           alpha=0.99,
                           beta=TUNED_BETA,             # ★ 默认值改成 π/60
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

    # ────────────────────────────────────────────────────────────────────────
    # Attack：和 manifold2d_diagnose.py 一样，但 inner_n 改成 15
    # ────────────────────────────────────────────────────────────────────────
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
        inner_n = TUNED_INNER_N    # ★ 原本是 10
        lam1, lam2, lam3 = self.lam1, self.lam2, self.lam3

        u_prev = None
        x_e_prev = None
        x_b_prev = None
        x_adv = x_b

        r_prev = float(torch.norm(x_b - self.src_img).item())

        for it in range(outer_iter):
            diff = x_b - self.src_img
            r_cur = torch.norm(diff)
            if r_cur < 1e-8:
                break
            v_new = diff / r_cur

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

            # ★ 显式传 beta（虽然默认值已经是调过的，这里再明确一下）
            x_adv, qs = self.manifold_search_2d(
                self.src_img, x_b, n=inner_n, u=u_new, beta=TUNED_BETA
            )
            actual_inner = self.last_actual_inner

            r_now = float(torch.norm(x_adv - self.src_img).item())
            delta_r = r_prev - r_now

            print(f'  iter {it:3d}  r_cur={r_now:.4f}  Δr={delta_r:+.5f}  '
                  f'inner={actual_inner}/{inner_n}  q+={qs}')

            r_prev = r_now
            x_e_prev = x_adv
            x_b_prev = x_b
            u_prev = u_new
            x_b = x_adv

            q_num = q_num + qs

            x_adv_inv = self.inv_tf(copy.deepcopy(x_adv.cpu()[0, :, :, :].squeeze()),
                                    self.mean, self.std)
            norm = torch.norm(x_inv - x_adv_inv)
            norms.append(norm)
            n_query.append(q_num)

        x_adv = clip_image_values(x_adv, self.lb, self.ub)
        return x_adv, n_query, norms


# ── 跑诊断 ────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--variant', default='exploit', choices=list(VARIANTS.keys()))
    parser.add_argument('--num_img', type=int, default=3)
    parser.add_argument('--iter', dest='iteration', type=int, default=50)
    parser.add_argument('--model', default='wideresnet40_2')
    args = parser.parse_args()

    torch.manual_seed(992); np.random.seed(992)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f'\n{"="*78}')
    print(f'  Manifold2D Diagnostic — TUNED (beta/inner_n/tol)')
    print(f'  variant = {args.variant}  λ = {VARIANTS[args.variant]}  '
          f'(direction params UNCHANGED)')
    print(f'  num_img = {args.num_img}   iteration = {args.iteration}')
    print(f'  ★ TUNED: beta=π/{int(math.pi/TUNED_BETA)}  '
          f'inner_n={TUNED_INNER_N}  tol={TUNED_TOL}')
    print(f'  (original:  beta=π/30           inner_n=10        tol=1e-4)')
    print(f'  model   = {args.model}    device = {device}')
    print(f'{"="*78}')

    mean = [0.4914, 0.4822, 0.4465]
    std  = [0.2023, 0.1994, 0.2010]

    cifar10 = datasets.CIFAR10(root='./data', train=False, download=True, transform=None)
    tf_normalize = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    def to_tensor(arr_uint8):
        return tf_normalize(Image.fromarray(arr_uint8))[None].to(device)

    net = load_model(args.model, ckpt_dir='checkpoints', device=device)

    done = 0
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
        attacker = DiagnosticAttack(
            net, x_0, mean, std, lb, ub,
            iteration=args.iteration,
            lam=VARIANTS[args.variant],
            verbose_control='No',
        )
        with torch.no_grad():
            x_adv, n_query, norms = attacker.Attack()
        with torch.no_grad():
            adv_pred = torch.argmax(net(x_adv)).item()

        print(f'  [Done] final_r={float(norms[-1]):.4f}  total_q={n_query[-1]}  '
              f'pred {gt}→{adv_pred}  {"SUCCESS" if adv_pred != gt else "FAIL"}')


if __name__ == '__main__':
    main()
