# Fair_cgbah_cifar10.py
'''
Fair-comparison driver for CIFAR-10 CGBA_H (non-targeted).
Same as Fair_cgba_cifar10.py, only:
  - attack_methods = ['CGBA_H']
  - VARIANT_SUFFIX = 'cgbah_fair'
'''
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import numpy as np
import os
from utils import valid_bounds
from PIL import Image
import time
from proposed_attack import Proposed_attack
from models_cifar10 import load_model
##############################################################################
torch.manual_seed(992)
torch.cuda.manual_seed(992)
np.random.seed(992)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using device: {device}')
##############################################################################
VARIANT_SUFFIX = 'cgbah_fair'   # ← 改这里区分变体
CIFAR10_CLASSES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]
num_img          = 1000
iteration        = 59
attack_methods   = ['CGBA_H']
dim_reduc_factor = 1

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
MODEL_NAMES = ['preactresnet18', 'wideresnet40_2', 'vit']
for model_name in MODEL_NAMES:
    net = load_model(model_name, ckpt_dir='checkpoints', device=device)
    for attack_method in attack_methods:
        print(f'\n{"="*60}')
        print(f'Model: {model_name}  |  Attack: {attack_method}')
        print(f'{"="*60}')
        all_norms         = []
        all_queries       = []
        all_postverify_ok = []
        image_iter        = 0
        success_count     = 0

        for image_iter1 in range(len(cifar10_test)):
            if image_iter >= num_img:
                break
            im_pil, ground_label_int = cifar10_test[image_iter1]
            lb_np, ub_np = valid_bounds(im_pil, delta=255)
            x_0 = tf_normalize(im_pil)[None].to(device)
            lb  = numpy_uint8_to_tensor(lb_np)
            ub  = numpy_uint8_to_tensor(ub_np)
            with torch.no_grad():
                orig_label = torch.argmax(net(x_0)).item()
            print(f'\nImage {image_iter1:05d}: '
                  f'GT={CIFAR10_CLASSES[ground_label_int]}  '
                  f'Pred={CIFAR10_CLASSES[orig_label]}')
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
            all_norms        .append(norms)
            all_queries      .append(n_query)
            all_postverify_ok.append(pv_ok)

        asr = success_count / image_iter * 100 if image_iter > 0 else 0
        print(f'\n── Attack Summary ──────────────────────────────')
        print(f'Model             : {model_name}')
        print(f'Attack method     : {attack_method}')
        print(f'Total images      : {image_iter}')
        print(f'Success           : {success_count}')
        print(f'ASR (post-verify) : {asr:.1f}%')

        if len(all_norms) > 0:
            norm_array  = np.array(all_norms)
            query_array = np.array(all_queries)
            pv_arr      = np.array(all_postverify_ok)

            running_min_arr = np.minimum.accumulate(norm_array, axis=1)
            init_norms = norm_array[:, 0]
            running_min_arr[~pv_arr] = init_norms[~pv_arr, None]

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
                norm                  = np.median(running_min_arr, 0),
                query                 = np.median(query_array,     0),
                all_norms             = norm_array,
                all_queries           = query_array,
                asr                   = asr,
                all_norms_running_min = running_min_arr,
                all_postverify_ok     = pv_arr,
                all_best_l2           = all_best_l2,
            )
            print(f'Results saved to {save_path}.npz')
        else:
            print(f'────────────────────────────────────────────────')
            print('No images processed, nothing saved.')
