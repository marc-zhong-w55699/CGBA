'''
Fair-comparison driver for ImageNet CGBA (non-targeted).
Loads from Kaggle imagenet-mini (1000 classes, ~3923 val imgs).
Same as Fair_cgbah_imagenet.py, only:
  - attack_methods = ['CGBA']
  - VARIANT_SUFFIX = 'cgba_fair'
'''
import torchvision.transforms as transforms
import torchvision.models as torch_models
from torchvision.datasets import ImageFolder
import numpy as np
import torch
import os
from utils import valid_bounds
from PIL import Image
import time
from proposed_attack import Proposed_attack
VARIANT_SUFFIX = 'cgba_fair'

##############################################################################
torch.manual_seed(992)
torch.cuda.manual_seed(992)
np.random.seed(992)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
##############################################################################

num_img          = 1000
iteration        = 60         # ← CGBA 论文 iter；~10k query budget
model_arc        = 'resnet50'    # 'resnet50' / 'vgg19' / 'inception_v3' / 'ViT'
attack_methods   = ['CGBA']
dim_reduc_factor = 4

mean = [0.485, 0.456, 0.406]
std  = [0.229, 0.224, 0.225]

# ── Kaggle imagenet-mini val root ────────────────────────────────────────────
imnet_val_root = '/root/autodl-tmp/data/imagenet-mini/imagenet-mini/val'

for attack_method in attack_methods:
    if model_arc == 'resnet50':
        net = torch_models.resnet50(pretrained=True)
    elif model_arc == 'resnet101':
        net = torch_models.resnet101(pretrained=True)
    elif model_arc == 'vgg16':
        net = torch_models.vgg16(pretrained=True)
    elif model_arc == 'vgg19':
        net = torch_models.vgg19(pretrained=True)
    elif model_arc == 'inception_v3':
        net = torch_models.inception_v3(pretrained=True, aux_logits=True)
    elif model_arc == 'efficientnet_b0':
        net = torch_models.efficientnet_b0(pretrained=True)
    elif model_arc == 'ViT':
        net = torch_models.vit_b_32(pretrained=True)
    net = net.to(device)
    net.eval()

    im_sz = 299 if model_arc == 'inception_v3' else 224

    # ── 加载 val 集 + 固定 seed 打乱（保证跨 attack 一致 & 覆盖多类别） ──
    imnet_val = ImageFolder(imnet_val_root)   # 返回 (PIL, class_idx)
    rng = np.random.RandomState(992)
    idx_perm = rng.permutation(len(imnet_val))
    print(f'Loaded {len(imnet_val)} val images from {imnet_val_root}')
    print(f'Will iterate seeded permutation, take first {num_img} correctly-classified')

    all_norms         = []
    all_queries       = []
    all_postverify_ok = []
    image_iter        = 0
    success_count     = 0

    for perm_i, image_iter1 in enumerate(idx_perm):
        if image_iter >= num_img:
            break

        im_orig, ground_label_int = imnet_val[int(image_iter1)]
        im_orig = im_orig.convert('RGB')
        im_orig = transforms.Resize((im_sz, im_sz))(im_orig)

        delta = 255
        lb, ub = valid_bounds(im_orig, delta)

        im = transforms.Compose([
            transforms.CenterCrop(im_sz),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)])(im_orig)

        lb = transforms.Compose([transforms.ToTensor(),
                                 transforms.Normalize(mean=mean, std=std)])(lb)
        ub = transforms.Compose([transforms.ToTensor(),
                                 transforms.Normalize(mean=mean, std=std)])(ub)

        lb  = lb[None, :, :, :].to(device)
        ub  = ub[None, :, :, :].to(device)
        x_0 = im[None, :, :, :].to(device)

        with torch.no_grad():
            orig_label = torch.argmax(net(x_0)).item()

        print(f'\nImage idx={int(image_iter1):05d} (perm #{perm_i}): '
              f'GT={ground_label_int}  Pred={orig_label}')

        if ground_label_int != orig_label:
            print('Already misclassified, skip.')
            continue

        image_iter += 1
        print(f'[{image_iter}/{num_img}]')
        print('#' * 60)
        print(f'Start: {attack_method} | iterations={iteration} | '
              f'dim_reduc_factor={dim_reduc_factor}')
        print('#' * 60)

        t3 = time.time()
        attack = Proposed_attack(net, x_0, mean, std, lb, ub,
                                 dim_reduc_factor=dim_reduc_factor,
                                 attack_method=attack_method,
                                 iteration=iteration)
        x_adv, n_query, norms = attack.Attack()
        t4 = time.time()
        print(f'Done in {t4 - t3:.2f}s')

        with torch.no_grad():
            adv_label = torch.argmax(net(x_adv)).item()
        if adv_label != ground_label_int:
            print(f'Attack SUCCESS: GT={ground_label_int} → adv={adv_label}')
            success_count += 1
            pv_ok = True
        else:
            print(f'Attack FAILED: still predicted as {ground_label_int}')
            pv_ok = False

        all_norms        .append(norms)
        all_queries      .append(n_query)
        all_postverify_ok.append(pv_ok)

    asr = success_count / image_iter * 100 if image_iter > 0 else 0
    print(f'\n── Attack Summary ──────────────────────────────')
    print(f'Model         : {model_arc}')
    print(f'Attack method : {attack_method}')
    print(f'Total images  : {image_iter}')
    print(f'Success       : {success_count}')
    print(f'ASR (post-verify) : {asr:.1f}%')

    if len(all_norms) > 0:
        norm_array  = np.array(all_norms)
        query_array = np.array(all_queries)
        pv_arr      = np.array(all_postverify_ok)

        running_min_arr = np.minimum.accumulate(norm_array, axis=1)
        init_norms = norm_array[:, 0]
        running_min_arr[~pv_arr] = init_norms[~pv_arr, None]

        all_best_l2 = running_min_arr[:, -1]

        print(f'★ Median best L2 : {np.median(all_best_l2):.3f}   ← main-table metric')
        print(f'  Mean   best L2 : {np.mean(all_best_l2):.3f}')
        print(f'  Fail-fallback  : {int((~pv_arr).sum())} img(s) replaced by norm_initial')
        print(f'────────────────────────────────────────────────')

        save_dir = 'Non_targeted_results_imagenet'
        os.makedirs(save_dir, exist_ok=True)
        save_path = (f'{save_dir}/{attack_method}_nonTar_{model_arc}'
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
