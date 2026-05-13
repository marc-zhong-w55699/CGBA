"""
Manifold2D 诊断脚本
-------------------
不修改 pro_attack.py。通过 subclass 重写 manifold_search_2d 和 Attack，
在每个外循环 print:
    iter <it>  r_cur=<...>  Δr=<...>  inner=<actual>/<inner_n>  q+=<...>

跑 1~3 张 CIFAR-10 图就够看 pattern。把 print 日志贴回来即可判断卡哪个参数。

用法:
    python manifold2d_diagnose.py                       # 默认 exploit, 3 张图, iter=50
    python manifold2d_diagnose.py --variant explore     # 换变体
    python manifold2d_diagnose.py --num_img 1 --iter 30 # 更短

如果你 Kaggle 那边已经是参数化版本（不再是 5 个文件），把下面 `from pro_attack
import Proposed_attack` 改成你那个统一文件的 import 即可——其它逻辑不用动。
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
from pro_attack import Proposed_attack  # ← 改这一行就能切换 Manifold2D 实现
from utils import clip_image_values, valid_bounds


# ── 5 个 Manifold2D 变体的 λ 三元组 ──────────────────────────────────────────
VARIANTS = {
    'exploit': (0.4, 0.3, 0.3),
    'explore': (0.2, 0.2, 0.6),
    'refine':  (0.3, 0.5, 0.2),
    'random':  (0.0, 0.0, 1.0),
    # state 用状态机，这里不诊断；如要诊断改 import 为 pro_atk 即可
}


# ── Subclass：完全保留父类逻辑，只多记录 last_actual_inner + 添加 per-iter print ──
class DiagnosticAttack(Proposed_attack):
    """
    Subclass of Proposed_attack with diagnostic instrumentation.

    Changes:
      - Accepts `lam=(λ1, λ2, λ3)` kwarg to override the hardcoded weights.
      - Overrides manifold_search_2d to store the actual inner step count
        (`self.last_actual_inner`) without changing the return signature.
      - Overrides Attack() to print (it, r_cur, Δr, actual_inner, q+) each outer iter.
    """

    def __init__(self, *args, lam=(0.4, 0.3, 0.3), **kwargs):
        super().__init__(*args, **kwargs)
        self.lam1, self.lam2, self.lam3 = lam
        self.last_actual_inner = 0

    # ────────────────────────────────────────────────────────────────────────
    # 复制父类 manifold_search_2d 的完整 body，唯一改动：
    #   - 把 break 出来时的 i 存到 self.last_actual_inner
    #   - 返回签名保持 (x_e, num_calls) 不变
    # ────────────────────────────────────────────────────────────────────────
    def manifold_search_2d(self, x_o, x_b,
                           alpha=0.99,
                           beta=math.pi / 30,
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

        # 找有效旋转方向 s
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
            # 旋转方向都找不到 → 内层 0 步就出来了
            self.last_actual_inner = 0
            return x_b, num_calls

        # 主循环：同时旋转 + 收缩
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
    # 复制父类 Attack() 的完整 body，唯一改动：
    #   - 用 self.lam1/2/3 替代硬编码的 0.4/0.3/0.3
    #   - 每个外循环结束 print (it, r_cur, Δr, actual_inner, q+=qs)
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
        inner_n = 10
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

            x_adv, qs = self.manifold_search_2d(
                self.src_img, x_b, n=inner_n, u=u_new
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


# ── 跑诊断 ────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--variant', default='exploit', choices=list(VARIANTS.keys()))
    parser.add_argument('--num_img', type=int, default=3)
    parser.add_argument('--iter', dest='iteration', type=int, default=50)
    parser.add_argument('--model', default='wideresnet40_2')
    args = parser.parse_args()

    torch.manual_seed(992); np.random.seed(992)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f'\n{"="*72}')
    print(f'  Manifold2D Diagnostic')
    print(f'  variant = {args.variant}  λ = {VARIANTS[args.variant]}')
    print(f'  num_img = {args.num_img}   iteration = {args.iteration}')
    print(f'  model   = {args.model}    device = {device}')
    print(f'{"="*72}')

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
            verbose_control='No',     # 关掉父类那些 'iteration ->' 之类的 print
        )
        with torch.no_grad():
            x_adv, n_query, norms = attacker.Attack()
        with torch.no_grad():
            adv_pred = torch.argmax(net(x_adv)).item()

        print(f'  [Done] final_r={float(norms[-1]):.4f}  total_q={n_query[-1]}  '
              f'pred {gt}→{adv_pred}  {"SUCCESS" if adv_pred != gt else "FAIL"}')


if __name__ == '__main__':
    main()
