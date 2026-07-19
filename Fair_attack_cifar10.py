# Fair_attack_cifar10.py
'''
Fair-comparison driver for CIFAR-10 decision-based attacks.

Differences vs the vanilla driver (attack_cifar10.py):
  1. append-all           : records the L2 curve for EVERY clean-classified
                            image, not just the post-verify successes.
  2. running-min          : an extra key `all_norms_running_min` is saved.
                            Per-image reported L2 = min over the trajectory.
  3. init-fallback        : for post-verify FAILED images, the running-min
                            curve is replaced by norm_initial (= norms[0],
                            the L2 of the random-init adversarial, which is
                            guaranteed valid after clip).  This prevents a
                            failed attack from being under-reported by its
                            algorithm-internal "best" state that doesn't
                            survive re-normalization.

.npz keys — original 5 kept exactly:
    norm, query, all_norms, all_queries, asr
plus 3 new extras:
    all_norms_running_min, all_postverify_ok, all_best_l2
'''
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import numpy as np
import os
from utils import valid_bounds
from PIL import Image
import time
from M2D_random_v11_cifar10 import Proposed_attack
from models_cifar10 import load_model
##############################################################################
torch.manual_seed(992)
torch.cuda.manual_seed(992)
np.random.seed(992)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using device: {device}')
##############################################################################
VARIANT_SUFFIX = 'random_v11_fair'   # ← 改这里区分变体
CIFAR10_CLASSES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]
num_img          = 50
iteration        = 2500
attack_methods   = ['Manifold2D']
dim_reduc_factor = 4

mean = [0.4914, 0.4822, 0.4465]
std  = [0.2023, 0.1994, 0.2010]

# ── 数据 ──────────────────────────────────────────────────────────────────────
cifar10_test = datasets.CIFAR10(
    root='./data', train=False, download=False, transform=None
)
tf_normalize = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std),
])

def numpy_uint8_to_tensor(arr):
    """uint8 numpy (H,W,C) → normalized tensor (1,C,H,W) on device"""
    return tf_normalize(Image.fromarray(arr))[None].to(device)

# ── 攻击模型列表 ───────────────────────────────────────────────────────────────
#MODEL_NAMES = ['preactresnet18', 'wideresnet40_2','vit']
MODEL_NAMES = ['preactresnet18', 'wideresnet40_2']
for model_name in MODEL_NAMES:
    net = load_model(model_name, ckpt_dir='checkpoints', device=device)
    for attack_method in attack_methods:
        print(f'\n{"="*60}')
        print(f'Model: {model_name}  |  Attack: {attack_method}')
        print(f'{"="*60}')
        all_norms         = []   # raw L2 curve per image
        all_queries       = []
        all_postverify_ok = []   # per-image: post-verify pass / fail
        image_iter        = 0
        success_count     = 0    # post-verify success count (for `asr`)

        for image_iter1 in range(len(cifar10_test)):
            if image_iter >= num_img:
                break
            im_pil, ground_label_int = cifar10_test[image_iter1]
            # ── 计算像素空间 bounds ───────────────────────────────
            lb_np, ub_np = valid_bounds(im_pil, delta=255)
            # ── 转为归一化 tensor ─────────────────────────────────
            x_0 = tf_normalize(im_pil)[None].to(device)
            lb  = numpy_uint8_to_tensor(lb_np)
            ub  = numpy_uint8_to_tensor(ub_np)
            # ── 模型预测 ──────────────────────────────────────────
            with torch.no_grad():
                orig_label = torch.argmax(net(x_0)).item()
            print(f'\nImage {image_iter1:05d}: '
                  f'GT={CIFAR10_CLASSES[ground_label_int]}  '
                  f'Pred={CIFAR10_CLASSES[orig_label]}')
            # ── 跳过已误分类样本 ──────────────────────────────────
            if ground_label_int != orig_label:
                print('Misclassified, skip.')
                continue
            image_iter += 1
            print(f'[{image_iter}/{num_img}]')
            print('#' * 60)
            print(f'Start: {attack_method} | '
                  f'iterations={iteration} | '
                  f'dim_reduc_factor={dim_reduc_factor}')
            print('#' * 60)
            t3 = time.time()
            attack = Proposed_attack(
                net, x_0, mean, std, lb, ub,
                dim_reduc_factor=dim_reduc_factor,
                attack_method=attack_method,
                iteration=iteration,
            )
            x_adv, n_query, norms = attack.Attack()
            t4 = time.time()
            print(f'Done in {t4 - t3:.2f}s')
            # ── 验证攻击是否成功（informational; 不 gate append） ──
            with torch.no_grad():
                adv_label = torch.argmax(net(x_adv)).item()
            if adv_label != ground_label_int:
                print(f'Attack SUCCESS: '
                      f'{CIFAR10_CLASSES[ground_label_int]} → '
                      f'{CIFAR10_CLASSES[adv_label]}')
                success_count += 1
                pv_ok = True
            else:
                print(f'Attack FAILED: '
                      f'still predicted as {CIFAR10_CLASSES[adv_label]}')
                pv_ok = False
            # ── append-all (核心改动) ─────────────────────────
            all_norms        .append(norms)
            all_queries      .append(n_query)
            all_postverify_ok.append(pv_ok)

        # ── ASR 统计 ──────────────────────────────────────────────
        asr = success_count / image_iter * 100 if image_iter > 0 else 0
        print(f'\n── Attack Summary ──────────────────────────────')
        print(f'Model             : {model_name}')
        print(f'Attack method     : {attack_method}')
        print(f'Total images      : {image_iter}')
        print(f'Success           : {success_count}')
        print(f'ASR (post-verify) : {asr:.1f}%')

        # ── 保存结果 ──────────────────────────────────────────────
        if len(all_norms) > 0:
            norm_array  = np.array(all_norms)          # (N, T) raw
            query_array = np.array(all_queries)        # (N, T)
            pv_arr      = np.array(all_postverify_ok)  # (N,)

            # ── running-min L2 curve ─────────────────────────────
            running_min_arr = np.minimum.accumulate(norm_array, axis=1)

            # ── init-fallback: 对 post-verify FAIL 图，整条曲线拉平到 norm_initial ──
            # 理由: 内部 running-min 是"虚假的好"（不 survive re-normalize），
            #       但初始 random adv 保证 valid，是这张图能拿到的真实最好战绩。
            init_norms = norm_array[:, 0]              # (N,) 每图的 norm_initial
            running_min_arr[~pv_arr] = init_norms[~pv_arr, None]

            # per-image best L2 (last value of running-min)
            all_best_l2 = running_min_arr[:, -1]

            print(f'★ Median best L2   : {np.median(all_best_l2):.3f}   ← main-table metric')
            print(f'  Mean   best L2   : {np.mean(all_best_l2):.3f}')
            print(f'  Fail-fallback    : {int((~pv_arr).sum())} img(s) replaced by norm_initial')
            print(f'────────────────────────────────────────────────')

            save_dir = 'Non_targeted_results_cifar10'
            os.makedirs(save_dir, exist_ok=True)
            save_path = (f'{save_dir}/{attack_method}_nonTar_{model_name}'
             f'_dimReducFac_{dim_reduc_factor}'
             f'_imgNum_{num_img}_iteration_{iteration}'
             f'_{VARIANT_SUFFIX}')
            np.savez(
                save_path,
                # ── 原 5 key（与老 driver 完全兼容） ──
                norm                  = np.median(running_min_arr, 0),   # ← curve 换成 running-min (fallback applied)
                query                 = np.median(query_array,     0),
                all_norms             = norm_array,
                all_queries           = query_array,
                asr                   = asr,
                # ── 新增 extras ─────────────────────
                all_norms_running_min = running_min_arr,
                all_postverify_ok     = pv_arr,
                all_best_l2           = all_best_l2,
            )
            print(f'Results saved to {save_path}.npz')
        else:
            print(f'────────────────────────────────────────────────')
            print('No images processed, nothing saved.')
