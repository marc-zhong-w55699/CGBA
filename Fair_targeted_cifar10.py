# Fair_targeted_cifar10.py
'''
Fair-comparison driver for CIFAR-10 TARGETED decision-based attacks.

Mirrors Fair_targeted_imagenet.py; adopts TtBA convention for target
selection:
  (1) target GT class  !=  source GT class
  (2) target image is correctly classified (pred == GT)
  → tar_label = target's GT class (NOT its predicted class)

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
pair_num         = 100         # ← (source, target) 对数
iteration        = 2500
attack_methods   = ['Manifold2D']

mean = [0.4914, 0.4822, 0.4465]
std  = [0.2023, 0.1994, 0.2010]

# ── 数据 ──────────────────────────────────────────────────────────────────────
cifar10_test = datasets.CIFAR10(
    root='./data', train=False, download=True, transform=None
)
tf_normalize = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std),
])

def numpy_uint8_to_tensor(arr):
    """uint8 numpy (H,W,C) → normalized tensor (1,C,H,W) on device"""
    return tf_normalize(Image.fromarray(arr))[None].to(device)

n_test = len(cifar10_test)

# ── 攻击模型列表 ───────────────────────────────────────────────────────────────
#MODEL_NAMES = ['preactresnet18', 'wideresnet40_2','vit']
MODEL_NAMES = ['preactresnet18', 'wideresnet40_2']
for model_name in MODEL_NAMES:
    net = load_model(model_name, ckpt_dir='checkpoints', device=device)
    for attack_method in attack_methods:
        print(f'\n{"="*60}')
        print(f'Model: {model_name}  |  Attack: {attack_method} (TARGETED)')
        print(f'{"="*60}')
        all_norms         = []   # raw L2 curve per pair
        all_queries       = []
        all_postverify_ok = []   # per-pair: post-verify pass / fail (pred == tar_lbl)
        image_iter        = 0
        success_count     = 0    # targeted post-verify success count

        for image_iter1 in range(n_test):
            if image_iter >= pair_num:
                break

            im_pil, ground_label_int = cifar10_test[image_iter1]

            # ── 计算像素空间 bounds ───────────────────────────────
            lb_np, ub_np = valid_bounds(im_pil, delta=255)

            # ── 转为归一化 tensor ─────────────────────────────────
            x_0 = tf_normalize(im_pil)[None].to(device)
            lb  = numpy_uint8_to_tensor(lb_np)
            ub  = numpy_uint8_to_tensor(ub_np)

            # ── 模型预测 (source) ────────────────────────────────
            with torch.no_grad():
                orig_label = torch.argmax(net(x_0)).item()

            # ── 跳过已误分类样本 ──────────────────────────────────
            if ground_label_int != orig_label:
                # 静默 skip；CIFAR clean acc 高，打印会刷屏
                continue

            # ── 挑 target image (TtBA convention, 严格 1-to-1) ─────
            # (1) target GT class != source GT class
            # (2) target image 正确分类 (pred == GT)
            # 用 per-source 独立 RandomState → 跨 attack 的 target 采样序列固定，
            # 不受 attack 内部随机数消耗影响。同一个 image_iter1 拿到的
            # target 候选列表在 M2D / CGBA / CGBA_H 之间完全一致。
            rng = np.random.RandomState(992 + image_iter1)
            MAX_TARGET_TRIES = 20
            target_found = False
            for _ in range(MAX_TARGET_TRIES):
                image_iter2 = int(rng.choice(n_test))
                if image_iter2 == image_iter1:
                    continue
                im_pil_t, tar_gt = cifar10_test[image_iter2]
                if tar_gt == ground_label_int:
                    continue     # 同类，先快速跳过再避免 forward
                x_0_t = tf_normalize(im_pil_t)[None].to(device)
                with torch.no_grad():
                    tar_pred = torch.argmax(net(x_0_t)).item()
                if tar_pred == tar_gt:
                    tar_label = tar_gt
                    target_found = True
                    break
            if not target_found:
                print(f'#{image_iter1}: 20 tries failed to find valid target, skip.')
                continue

            image_iter += 1
            print(f'\nImage {image_iter1:05d} → target class '
                  f'{CIFAR10_CLASSES[tar_label]:>10s}  '
                  f'(source={CIFAR10_CLASSES[ground_label_int]})')
            print(f'[{image_iter}/{pair_num}]')
            print('#' * 60)
            print(f'Start: {attack_method} TARGETED | '
                  f'iterations={iteration}')
            print('#' * 60)
            t3 = time.time()
            attack = Proposed_attack(
                net, x_0, mean, std, lb, ub,
                tar_img=x_0_t,
                attack_method=attack_method,
                iteration=iteration,
            )
            x_adv, n_query, norms = attack.Attack()
            t4 = time.time()
            print(f'Done in {t4 - t3:.2f}s')
            # ── 验证攻击是否成功 (targeted: pred == tar_label) ──
            with torch.no_grad():
                adv_label = torch.argmax(net(x_adv)).item()
            if adv_label == tar_label:
                print(f'Attack SUCCESS (targeted): '
                      f'{CIFAR10_CLASSES[ground_label_int]} → '
                      f'{CIFAR10_CLASSES[adv_label]}')
                success_count += 1
                pv_ok = True
            else:
                print(f'Attack FAILED: predicted as {CIFAR10_CLASSES[adv_label]} '
                      f'(target was {CIFAR10_CLASSES[tar_label]})')
                pv_ok = False
            # ── append-all (核心) ─────────────────────────────
            all_norms        .append(norms)
            all_queries      .append(n_query)
            all_postverify_ok.append(pv_ok)

        # ── ASR 统计 ──────────────────────────────────────────────
        asr = success_count / image_iter * 100 if image_iter > 0 else 0
        print(f'\n── Attack Summary (Targeted) ──────────────────')
        print(f'Model             : {model_name}')
        print(f'Attack method     : {attack_method}')
        print(f'Total pairs       : {image_iter}')
        print(f'Success           : {success_count}')
        print(f'ASR (targeted)    : {asr:.1f}%')

        # ── 保存结果 ──────────────────────────────────────────────
        if len(all_norms) > 0:
            norm_array  = np.array(all_norms)          # (N, T) raw
            query_array = np.array(all_queries)        # (N, T)
            pv_arr      = np.array(all_postverify_ok)  # (N,)

            # ── running-min L2 curve ─────────────────────────────
            running_min_arr = np.minimum.accumulate(norm_array, axis=1)

            # ── init-fallback: post-verify FAIL pair 拉平到 norm_initial ──
            # 理由: targeted 里初始 adv 就是 target 图本身 (L2=||target-source||)，
            #       它是保证 valid 的（target 已经预测为 tar_label）。攻击若失败，
            #       fallback 到 "用 target 图作为对抗样本" 这个 baseline。
            init_norms = norm_array[:, 0]              # (N,) 每对的 ||target - source||
            running_min_arr[~pv_arr] = init_norms[~pv_arr, None]

            # per-pair best L2 (last value of running-min)
            all_best_l2 = running_min_arr[:, -1]

            print(f'★ Median best L2   : {np.median(all_best_l2):.3f}   ← main-table metric')
            print(f'  Mean   best L2   : {np.mean(all_best_l2):.3f}')
            print(f'  Fail-fallback    : {int((~pv_arr).sum())} pair(s) replaced by norm_initial')
            print(f'────────────────────────────────────────────────')

            save_dir = 'Targeted_results_cifar10'
            os.makedirs(save_dir, exist_ok=True)
            save_path = (f'{save_dir}/{attack_method}_Tar_{model_name}'
             f'_imgNum_{pair_num}_iteration_{iteration}'
             f'_{VARIANT_SUFFIX}')
            np.savez(
                save_path,
                # ── 原 5 key（与老 driver 完全兼容） ──
                norm                  = np.median(running_min_arr, 0),
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
            print('No pairs processed, nothing saved.')
