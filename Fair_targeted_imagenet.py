'''
Fair-comparison driver for ImageNet TARGETED decision-based attacks.

Differences vs the vanilla Targeted_attack.py:
  1. append-all           : records the L2 curve for EVERY (source, target)
                            pair where source is clean-classified AND
                            source-class != target-class. Not gated by
                            post-verify success.
  2. running-min          : an extra key `all_norms_running_min` is saved.
                            Per-image reported L2 = min over trajectory.
  3. init-fallback        : for post-verify FAILED pairs, the running-min
                            curve is replaced by norm_initial (= norms[0],
                            the L2 between target and source, which IS a
                            guaranteed valid adversarial since target is
                            already predicted as tar_label).
  4. post-verify + ASR    : adds `adv_label == tar_label` verification and
                            reports targeted ASR (fraction of pairs where
                            final x_adv is predicted as tar_label).

Targeted-specific:
  - success = pred == tar_label (target's PREDICTED class at init time)
  - init adv = target image itself (norm_initial = ||target - source||)

.npz keys — original 5 kept exactly:
    norm, query, all_norms, all_queries, asr
plus 3 new extras:
    all_norms_running_min, all_postverify_ok, all_best_l2
'''
import torchvision.transforms as transforms
import torchvision.models as torch_models
import numpy as np
import torch
import os
from utils import get_label
from utils import valid_bounds
from PIL import Image
from torch.autograd import Variable
import time
from M2D_random_dct_v11_imagenet import Proposed_attack
VARIANT_SUFFIX = 'random_dct_v11_fair'   # ← 改这里区分变体

##############################################################################
torch.manual_seed(992)
torch.cuda.manual_seed(992)
np.random.seed(992)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
##############################################################################

pair_num         = 100         # ← 目标 (source, target) 对数量
iteration        = 2500
model_arc        = 'resnet50'  # 'resnet50' / 'vgg16' / 'vgg19' / 'ViT'
                               #  / 'inception_v3' / 'efficientnet_b0'
attack_methods   = ['Manifold2D']
dim_reduc_factor = 4

mean = [0.485, 0.456, 0.406]
std  = [0.229, 0.224, 0.225]


def load_image(image_iter1, im_sz):
    """载入一张 ImageNet val 图片 (返回 im_orig PIL)，若不存在返回 None"""
    if len(str(image_iter1)) == 1:
        temp = "000" + str(image_iter1)
    elif len(str(image_iter1)) == 2:
        temp = "00" + str(image_iter1)
    elif len(str(image_iter1)) == 3:
        temp = "0" + str(image_iter1)
    else:
        temp = str(image_iter1)
    img_name = "ILSVRC2012_val_0000" + temp + ".JPEG"
    img_path = "Image_path/ImageNet/val"
    try:
        im_orig = Image.open(os.path.join(img_path, img_name)).convert('RGB')
    except FileNotFoundError:
        return None, None
    im_orig = transforms.Compose([transforms.Resize((im_sz, im_sz))])(im_orig)
    return im_orig, temp


for attack_method in attack_methods:
    # ── Model ──────────────────────────────────────────────
    if model_arc == 'resnet50':
        net = torch_models.resnet50(pretrained=True)
    if model_arc == 'resnet101':
        net = torch_models.resnet101(pretrained=True)
    if model_arc == 'vgg16':
        net = torch_models.vgg16(pretrained=True)
    if model_arc == 'vgg19':
        net = torch_models.vgg19(pretrained=True)
    if model_arc == 'inception_v3':
        net = torch_models.inception_v3(pretrained=True, aux_logits=True)
    if model_arc == 'efficientnet_b0':
        net = torch_models.efficientnet_b0(pretrained=True)
    if model_arc == 'ViT':
        import timm
        net = timm.create_model('vit_base_patch16_224', pretrained=True)
    net = net.to(device)
    net.eval()

    im_sz = 299 if model_arc == 'inception_v3' else 224

    # ── Storage ────────────────────────────────────────────
    all_norms         = []   # raw L2 curve per pair (as returned)
    all_queries       = []
    all_postverify_ok = []   # per-pair: post-verify pass / fail (pred == tar_lbl)
    image_iter        = 0
    success_count     = 0    # targeted post-verify success count

    labels       = open('synset_words.txt', 'r').read().split('\n')
    ground_truth = open('val.txt',           'r').read().split('\n')

    def normalize_img(pil_img):
        return transforms.Compose([
            transforms.CenterCrop(im_sz),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)])(pil_img)

    for image_iter1 in range(1, 3000):
        if image_iter >= pair_num:
            break

        # ── 加载 source ─────────────────────────────────────────
        im_orig, temp = load_image(image_iter1, im_sz)
        if im_orig is None:
            continue

        # Bounds (based on source)
        delta = 255
        lb, ub = valid_bounds(im_orig, delta)

        im = normalize_img(im_orig)
        lb_t = transforms.Compose([transforms.ToTensor(),
                                   transforms.Normalize(mean=mean, std=std)])(lb)
        ub_t = transforms.Compose([transforms.ToTensor(),
                                   transforms.Normalize(mean=mean, std=std)])(ub)
        lb_t = lb_t[None, :, :, :].to(device)
        ub_t = ub_t[None, :, :, :].to(device)
        x_0  = im [None, :, :, :].to(device)

        # ── source labels & clean check ────────────────────────
        with torch.no_grad():
            orig_label = torch.argmax(net(x_0)).item()
        str_label_orig   = get_label(labels[np.int32(orig_label)].split(',')[0])
        ground_label_int = int(ground_truth[image_iter1 - 1].split()[1])

        if ground_label_int != orig_label:
            print(f'\n#{image_iter1}: Source mis-classified, skip.')
            continue

        # ── 挑 target image (TtBA convention) ──────────────────
        # 要求:
        #   (1) target GT class != source GT class
        #   (2) target 图被模型正确分类 (pred == GT)
        # 满足即用 target GT 作为 tar_label（不用 predicted）
        MAX_TARGET_TRIES = 20
        target_found = False
        for _ in range(MAX_TARGET_TRIES):
            image_iter2 = int(np.random.choice(range(1, 5000)))
            if image_iter2 == image_iter1:
                continue
            im_orig_t, temp_t = load_image(image_iter2, im_sz)
            if im_orig_t is None:
                continue
            im_t = normalize_img(im_orig_t)
            x_0_t = im_t[None, :, :, :].to(device)
            with torch.no_grad():
                tar_pred = torch.argmax(net(x_0_t)).item()
            try:
                tar_gt = int(ground_truth[image_iter2 - 1].split()[1])
            except (IndexError, ValueError):
                continue
            # TtBA 两条硬约束
            if tar_gt != ground_label_int and tar_pred == tar_gt:
                tar_label = tar_gt
                target_found = True
                break
        if not target_found:
            print(f'\n#{image_iter1}: 20 tries failed to find valid target, skip.')
            continue

        # 用于 attack 的 lb/ub（来自 source）
        lb = lb_t
        ub = ub_t

        str_label_tar = get_label(labels[np.int32(tar_label)].split(',')[0])
        print(f'\nSource {temp}: GT={ground_label_int}({str_label_orig})  '
              f'Target {temp_t}: GT={tar_label}({str_label_tar}) [pred==GT verified]')

        image_iter = image_iter + 1
        print(f'[{image_iter}/{pair_num}]')

        print('#' * 60)
        print(f'Start: {attack_method} TARGETED will be run for '
              f'{iteration} iterations with dim_reduc_factor: {dim_reduc_factor}')
        print('#' * 60)

        t3 = time.time()
        attack = Proposed_attack(net, x_0, mean, std, lb, ub,
                                 dim_reduc_factor=dim_reduc_factor,
                                 tar_img=x_0_t,
                                 attack_method=attack_method,
                                 iteration=iteration)
        x_adv, n_query, norms = attack.Attack()
        t4 = time.time()
        print(f'##################### End Itetations:  took '
              f'{t4 - t3:.3f} sec #########################')

        # ── 验证攻击是否成功 (targeted: pred == tar_label) ─────
        with torch.no_grad():
            adv_label = torch.argmax(net(x_adv)).item()
        if adv_label == tar_label:
            str_label_adv = get_label(labels[np.int32(adv_label)].split(',')[0])
            print(f'Attack SUCCESS (targeted): {str_label_orig} → {str_label_adv}')
            success_count += 1
            pv_ok = True
        else:
            str_label_adv = get_label(labels[np.int32(adv_label)].split(',')[0])
            print(f'Attack FAILED: {str_label_orig} → {str_label_adv} '
                  f'(target was {str_label_tar})')
            pv_ok = False

        # ── append-all (核心) ────────────────────────────────────
        all_norms        .append(norms)
        all_queries      .append(n_query)
        all_postverify_ok.append(pv_ok)

    # ── ASR 统计 ──────────────────────────────────────────────
    asr = success_count / image_iter * 100 if image_iter > 0 else 0
    print(f'\n── Attack Summary (Targeted) ──────────────────')
    print(f'Model             : {model_arc}')
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

        # ── init-fallback: 对 post-verify FAIL pair，整条曲线拉平到 norm_initial ──
        # 理由: targeted 里初始 adv 就是 target 图本身，L2 = ||target - source||，
        #       它是保证 valid 的（target 图预测就是 tar_label）。攻击若失败，
        #       fallback 到用 target 图作为对抗样本。
        init_norms = norm_array[:, 0]              # (N,) 每对的 ||target - source||
        running_min_arr[~pv_arr] = init_norms[~pv_arr, None]

        # per-pair best L2 (last value of running-min)
        all_best_l2 = running_min_arr[:, -1]

        print(f'★ Median best L2 : {np.median(all_best_l2):.3f}   ← main-table metric')
        print(f'  Mean   best L2 : {np.mean(all_best_l2):.3f}')
        print(f'  Fail-fallback  : {int((~pv_arr).sum())} pair(s) replaced by norm_initial')
        print(f'────────────────────────────────────────────────')

        save_dir = 'Targeted_results_imagenet'
        os.makedirs(save_dir, exist_ok=True)
        save_path = (f'{save_dir}/{attack_method}_Tar_{model_arc}'
                     f'_dimReducFac_{dim_reduc_factor}'
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
