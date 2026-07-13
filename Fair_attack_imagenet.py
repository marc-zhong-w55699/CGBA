'''
Fair-comparison driver for ImageNet decision-based attacks.

Differences vs the vanilla driver (Non_targeted_attack_imaginenet_res50.py):
  1. append-all           : records the L2 curve for EVERY clean-classified
                            image, not just the post-verify successes.
  2. running-min (capped) : an extra key `all_norms_running_min` is saved.
                            Per-image reported L2 = min over the trajectory,
                            capped at ||x0||   (matches TtBA `Reall2`).

.npz keys — original 5 kept exactly:
    norm, query, all_norms, all_queries, asr
plus 4 new extras:
    all_norms_running_min, all_x0_norms, all_postverify_ok, all_best_l2
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

num_img          = 100
iteration        = 2500
model_arc        = 'resnet50'   # 'resnet50' / 'vgg16' / 'vgg19' / 'ViT'
                                #  / 'inception_v3' / 'efficientnet_b0'
attack_methods   = ['Manifold2D']
dim_reduc_factor = 4

mean = [0.485, 0.456, 0.406]
std  = [0.229, 0.224, 0.225]

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
    all_norms         = []   # raw L2 curve per image (as returned)
    all_queries       = []
    all_x0_norms      = []   # per-image ||x0|| (pixel space)
    all_postverify_ok = []   # per-image: post-verify pass / fail
    image_iter        = 0
    success_count     = 0    # post-verify success count (for `asr`)

    for image_iter1 in range(1, 300):   # scan up to 300 to find num_img clean
        if image_iter >= num_img:
            break
        if len(str(image_iter1)) == 1:
            temp = "000" + str(image_iter1)
        if len(str(image_iter1)) == 2:
            temp = "00" + str(image_iter1)
        if len(str(image_iter1)) == 3:
            temp = "0" + str(image_iter1)
        if len(str(image_iter1)) == 4:
            temp = str(image_iter1)
        img_name = "ILSVRC2012_val_0000" + temp + ".JPEG"
        img_path = "Image_path/ImageNet/val"

        t11 = time.time()

        try:
            im_orig = Image.open(os.path.join(img_path, img_name)).convert('RGB')
        except FileNotFoundError:
            print(f'#{image_iter1}: file missing, skip')
            continue

        im_orig = transforms.Compose([transforms.Resize((im_sz, im_sz))])(im_orig)

        delta = 255
        lb, ub = valid_bounds(im_orig, delta)

        # normalized image (fed to network + attack)
        im = transforms.Compose([
            transforms.CenterCrop(im_sz),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)])(im_orig)

        # pixel-space image (used only for the ||x0|| cap)
        im_pixel = transforms.Compose([
            transforms.CenterCrop(im_sz),
            transforms.ToTensor()])(im_orig)

        lb = transforms.Compose([transforms.ToTensor(),
                                 transforms.Normalize(mean=mean, std=std)])(lb)
        ub = transforms.Compose([transforms.ToTensor(),
                                 transforms.Normalize(mean=mean, std=std)])(ub)

        lb = lb[None, :, :, :].to(device)
        ub = ub[None, :, :, :].to(device)
        x_0 = im[None, :, :, :].to(device)
        x0_norm = float(torch.norm(im_pixel).item())

        orig_label = torch.argmax(
            net.forward(Variable(x_0, requires_grad=True)).data).item()
        labels = open(os.path.join('synset_words.txt'), 'r').read().split('\n')
        str_label_orig = get_label(labels[np.int32(orig_label)].split(',')[0])

        ground_truth = open(os.path.join('val.txt'), 'r').read().split('\n')
        ground_name_label = ground_truth[image_iter1 - 1]
        ground_label = ground_name_label.split()[1]
        ground_label_int = int(ground_label)
        str_label_ground = get_label(labels[np.int32(ground_label)].split(',')[0])

        print(f'\nSource image {temp}:  Class ID: {ground_label}   '
              f'Class Name: {str_label_ground}   ||x0||={x0_norm:.2f}')

        ##############################################################

        if ground_label_int != int(orig_label):
            print('Already missclassified ... Lets try another one!')
        else:
            image_iter = image_iter + 1
            print('Image number good to go: ', image_iter)

            print('#' * 60)
            print(f'Start: {attack_method} non-targeted will be run for '
                  f'{iteration} iterations with dim_reduc_factor: {dim_reduc_factor}')
            print('#' * 60)

            t3 = time.time()
            attack = Proposed_attack(net, x_0, mean, std, lb, ub,
                                     dim_reduc_factor=dim_reduc_factor,
                                     attack_method=attack_method,
                                     iteration=iteration)
            x_adv, n_query, norms = attack.Attack()
            t4 = time.time()
            print(f'##################### End Itetations:  took '
                  f'{t4 - t3:.3f} sec #########################')

            # ── 验证攻击是否成功（informational; 不 gate append） ──
            with torch.no_grad():
                adv_label = torch.argmax(net(x_adv)).item()
            if adv_label != ground_label_int:
                str_label_adv = get_label(labels[np.int32(adv_label)].split(',')[0])
                print(f'Attack SUCCESS: {str_label_ground} → {str_label_adv}')
                success_count += 1
                pv_ok = True
            else:
                print(f'Attack FAILED: still predicted as {str_label_ground}')
                pv_ok = False

            # ── append-all (核心改动) ─────────────────────────
            all_norms        .append(norms)
            all_queries      .append(n_query)
            all_x0_norms     .append(x0_norm)
            all_postverify_ok.append(pv_ok)

    # ── ASR 统计 ──────────────────────────────────────────────
    asr = success_count / image_iter * 100 if image_iter > 0 else 0
    print(f'\n── Attack Summary ──────────────────────────────')
    print(f'Model         : {model_arc}')
    print(f'Attack method : {attack_method}')
    print(f'Total images  : {image_iter}')
    print(f'Success       : {success_count}')
    print(f'ASR (post-verify) : {asr:.1f}%')

    # ── 保存结果 ──────────────────────────────────────────────
    if len(all_norms) > 0:
        norm_array   = np.array(all_norms)          # (N, T) raw
        query_array  = np.array(all_queries)        # (N, T)
        x0_norms_arr = np.array(all_x0_norms)       # (N,)
        pv_arr       = np.array(all_postverify_ok)  # (N,)

        # ── running-min L2 curve, capped at ||x0|| ───────────
        running_min_arr = np.minimum.accumulate(norm_array, axis=1)
        running_min_arr = np.minimum(running_min_arr, x0_norms_arr[:, None])

        # per-image best L2 (last value of running-min-capped)
        all_best_l2 = running_min_arr[:, -1]

        print(f'★ Median best L2 : {np.median(all_best_l2):.3f}   ← main-table metric')
        print(f'  Mean   best L2 : {np.mean(all_best_l2):.3f}')
        print(f'  Median ||x0||  : {np.median(x0_norms_arr):.2f}')
        print(f'────────────────────────────────────────────────')

        save_dir = 'Non_targeted_results_imagenet'
        os.makedirs(save_dir, exist_ok=True)
        save_path = (f'{save_dir}/{attack_method}_nonTar_{model_arc}'
                     f'_dimReducFac_{dim_reduc_factor}'
                     f'_imgNum_{num_img}_iteration_{iteration}'
                     f'_{VARIANT_SUFFIX}')
        np.savez(
            save_path,
            # ── 原 5 key（与老 driver 完全兼容） ──
            norm                  = np.median(running_min_arr, 0),   # ← curve 换成 running-min
            query                 = np.median(query_array,     0),
            all_norms             = norm_array,
            all_queries           = query_array,
            asr                   = asr,
            # ── 新增 extras ─────────────────────
            all_norms_running_min = running_min_arr,
            all_x0_norms          = x0_norms_arr,
            all_postverify_ok     = pv_arr,
            all_best_l2           = all_best_l2,
        )
        print(f'Results saved to {save_path}.npz')
    else:
        print(f'────────────────────────────────────────────────')
        print('No images processed, nothing saved.')
