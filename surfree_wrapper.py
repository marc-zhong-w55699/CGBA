"""
SurFree 适配器：把官方 SurFree 包装成 CGBA `Proposed_attack` 风格的 .Attack() API，
方便 Non_targeted_attack_* 风格的跑批脚本无缝替换。

设计要点
--------
1. 接口对齐 CGBA 的 Proposed_attack：
       attack = SurFreeWrapper(net, x_0, mean, std, lb, ub,
                               dim_reduc_factor=..., attack_method='SurFree',
                               iteration=..., max_queries=...)
       x_adv, n_query, norms = attack.Attack()
   返回的 `n_query` / `norms` 都是 list，长度 = iteration + 1（与 CGBA 一致），
   于是外层 np.array(...) → np.median(..., axis=0) 不用改。

2. 输入域对齐：CGBA 的 x_0 / lb / ub 都是 *已归一化* 张量（减 mean 除 std），
   而 SurFree 原版假设 model 输入在 [0,1] 像素域。所以这里：
     - 把 x_0 反归一化回 [0,1] 当作 SurFree 的 X
     - 把 net 包一层 NormalizedModel，输入 [0,1] 内部做 normalize 再前向
   norms 一律在 *像素域 [0,1]* 度量，与 CGBA proposed_attack.py 中
   `torch.norm(x_inv - x_adv_inv)` 同口径。

3. 每次 model 调用都通过 `_QueryCountingModel`，自己计数 query；
   每个 SurFree step 结束记录一次 (cum_queries, best_l2_pixel)，
   最后下采样/补齐成长度 iteration+1 的曲线。

4. 返回的 x_adv 转回 *归一化空间*，这样外层脚本里
   `with torch.no_grad(): adv_label = torch.argmax(net(x_adv)).item()`
   能直接复用 CGBA 的代码不变。
"""

import torch
import numpy as np

from surfree_src.surfree import SurFree


class _NormalizedModel(torch.nn.Module):
    """Wraps a model so it can be called with inputs in [0,1] pixel space."""
    def __init__(self, net, mean, std, device):
        super().__init__()
        self.net = net
        self.mean = torch.tensor(mean, device=device).view(1, 3, 1, 1)
        self.std  = torch.tensor(std,  device=device).view(1, 3, 1, 1)

    def forward(self, x):
        x_norm = (x - self.mean) / self.std
        return self.net(x_norm)


class _QueryCountingModel(torch.nn.Module):
    """Counts every forward call (one model call == one batch query)."""
    def __init__(self, inner):
        super().__init__()
        self.inner = inner
        self.n_queries = 0

    def forward(self, x):
        # SurFree calls model on a batch of one image per attack instance.
        # We count one query per image in the batch, matching CGBA's accounting.
        self.n_queries += int(x.shape[0])
        return self.inner(x)


class SurFreeWrapper:
    """
    Drop-in replacement for `Proposed_attack` whose `.Attack()` returns
    (x_adv_normalized, n_query_list, norms_list_pixel).
    """

    def __init__(self, net, x_0, mean, std, lb, ub,
                 dim_reduc_factor=4,
                 attack_method='SurFree',
                 iteration=50,
                 max_queries=2000,
                 theta_max=30,
                 verbose=True):
        self.net  = net
        self.x_0  = x_0                # normalized tensor, (1,C,H,W)
        self.mean = mean
        self.std  = std
        self.lb   = lb                 # normalized bounds (unused inside SurFree; SurFree uses clip(0,1))
        self.ub   = ub
        self.iteration   = int(iteration)
        self.max_queries = int(max_queries)
        self.theta_max   = theta_max
        self.verbose     = verbose
        self.device      = x_0.device
        # dim_reduc_factor / attack_method are accepted for API compatibility
        # but SurFree itself doesn't use them.
        self.dim_reduc_factor = dim_reduc_factor
        self.attack_method    = attack_method

    # ────────────────────────────────────────────────────────────
    def _to_pixel(self, x_norm):
        mean_t = torch.tensor(self.mean, device=self.device).view(1, 3, 1, 1)
        std_t  = torch.tensor(self.std,  device=self.device).view(1, 3, 1, 1)
        return (x_norm * std_t + mean_t).clamp(0.0, 1.0)

    def _to_norm(self, x_pix):
        mean_t = torch.tensor(self.mean, device=self.device).view(1, 3, 1, 1)
        std_t  = torch.tensor(self.std,  device=self.device).view(1, 3, 1, 1)
        return (x_pix - mean_t) / std_t

    # ────────────────────────────────────────────────────────────
    def Attack(self):
        # 1) prepare pixel-space input + wrapped model
        x_pix = self._to_pixel(self.x_0)                    # [0,1]
        wrapped = _NormalizedModel(self.net, self.mean, self.std, self.device)
        counting = _QueryCountingModel(wrapped).to(self.device).eval()

        with torch.no_grad():
            label = counting.inner(x_pix).argmax(1)         # don't count this bookkeeping call
            counting.n_queries -= 1                         # rollback the counter for this internal probe

        # 2) instantiate SurFree with a subclass that logs per-step state
        log_entries = []   # list of (cum_queries, best_l2_pixel)

        outer = self

        class _LoggingSurFree(SurFree):
            def _is_adversarial(self_, perturbed):
                result = super()._is_adversarial(perturbed)
                # Record best L2 in pixel space at this query checkpoint
                if getattr(self_, 'best_advs', None) is not None and self_.X is not None:
                    d = (self_.best_advs - self_.X).flatten(1).norm(dim=1)[0].item()
                    log_entries.append((counting.n_queries, d))
                return result

        attacker = _LoggingSurFree(
            steps=self.iteration,
            max_queries=self.max_queries,
            theta_max=self.theta_max,
            quantification=False,    # don't snap to 1/255 grid (we work in floats)
            clip=True,
        )

        # 3) run SurFree
        with torch.no_grad():
            x_adv_pix = attacker(counting, x_pix, label)

        # 4) build curves of length iteration+1 (cum_query, norm) — like CGBA
        L = self.iteration + 1
        if len(log_entries) == 0:
            # fall back to a flat curve (attack made zero queries — unlikely)
            init_norm = (attacker.best_advs - x_pix).flatten(1).norm(dim=1)[0].item() \
                if attacker.best_advs is not None else 0.0
            n_query_curve = [counting.n_queries] * L
            norm_curve    = [init_norm] * L
        else:
            # uniformly sample L points from log_entries (hold last value if too short)
            idxs = np.linspace(0, len(log_entries) - 1, L).round().astype(int)
            n_query_curve = [int(log_entries[i][0]) for i in idxs]
            norm_curve    = [float(log_entries[i][1]) for i in idxs]

        # 5) return x_adv in normalized space (so outer net(x_adv) works unchanged)
        x_adv_norm = self._to_norm(x_adv_pix)

        if self.verbose:
            print(f'\n── SurFree summary ────────────────────────────')
            print(f'Steps (iteration) requested : {self.iteration}')
            print(f'Max queries budget          : {self.max_queries}')
            print(f'Actual queries used         : {counting.n_queries}')
            print(f'Final L2 (pixel space)      : {norm_curve[-1]:.4f}')
            print(f'────────────────────────────────────────────────')

        return x_adv_norm, n_query_curve, norm_curve
