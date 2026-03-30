# attack_cifar10.py
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

CIFAR10_CLASSES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]

num_img          = 1000
iteration        = 93
attack_methods   = ['CGBA_H', 'CGBA']
dim_reduc_factor = 4

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


# ── 攻击模型列表 ───────────────────────────────────────────────────────────────
MODEL_NAMES = ['preactresnet18', 'wideresnet40_2']

for model_name in MODEL_NAMES:

    net = load_model(model_name, ckpt_dir='checkpoints', device=device)

    for attack_method in attack_methods:
        print(f'\n{"="*60}')
        print(f'Model: {model_name}  |  Attack: {attack_method}')
        print(f'{"="*60}')

        all_norms   = []
        all_queries = []
        image_iter  = 0

        for image_iter1 in range(len(cifar10_test)):
            if image_iter >= num_img:
                break

            im_pil, ground_label_int = cifar10_test[image_iter1]
            # im_pil 是 32×32 PIL，直接用，无需 resize

            # ── 计算像素空间 bounds ───────────────────────────────
            lb_np, ub_np = valid_bounds(im_pil, delta=255)  # uint8 (H,W,C)

            # ── 转为归一化 tensor ─────────────────────────────────
            x_0 = tf_normalize(im_pil)[None].to(device)     # (1,3,32,32)
            lb  = numpy_uint8_to_tensor(lb_np)               # (1,3,32,32)
            ub  = numpy_uint8_to_tensor(ub_np)               # (1,3,32,32)

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

            all_norms.append(norms)
            all_queries.append(n_query)

        # ── 保存结果 ──────────────────────────────────────────────
        norm_array  = np.array(all_norms)
        query_array = np.array(all_queries)

        save_dir = 'Non_targeted_results_cifar10'
        os.makedirs(save_dir, exist_ok=True)

        save_path = (f'{save_dir}/{attack_method}_nonTar_{model_name}'
                     f'_dimReducFac_{dim_reduc_factor}'
                     f'_imgNum_{num_img}_iteration_{iteration}')
        np.savez(
            save_path,
            norm        = np.median(norm_array,  0),
            query       = np.median(query_array, 0),
            all_norms   = norm_array,
            all_queries = query_array,
        )
        print(f'\nResults saved to {save_path}.npz')
