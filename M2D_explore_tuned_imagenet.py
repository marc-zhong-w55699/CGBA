'''
Manifold2D (explore_tuned variant) attack on ImageNet ResNet-50.
- variant: pro_attack_explore_tuned (λ=0.2,0.2,0.6, inner_n=15, tol=1e-5, β=π/30)
- iter=1600 targets ~10000 queries (q/iter ≈ 6 on ImageNet)
- iso-query comparison with SurFree (max_queries=10000)
Output npz fields aligned with CGBA / SurFree scripts.
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
from pro_attack_explore_tuned import Proposed_attack


##############################################################################
torch.manual_seed(992)
torch.cuda.manual_seed(992)
np.random.seed(992)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
##############################################################################


num_img          = 50
iteration        = 1600          # ★ targets ~10000 queries (q/iter ≈ 6 on ImageNet)
model_arc        = 'resnet50'
attack_methods   = ['Manifold2D']
VARIANT_SUFFIX   = 'explore_tuned'
dim_reduc_factor = 4

mean = [0.485, 0.456, 0.406]
std  = [0.229, 0.224, 0.225]

for attack_method in attack_methods:
    # Models
    if model_arc == 'resnet50':
        net = torch_models.resnet50(pretrained=True)
    if model_arc == 'resnet101':
        net = torch_models.resnet101(pretrained=True)
    if model_arc == 'vgg16':
        net = torch_models.vgg16(pretrained=True)
    if model_arc == 'ViT':
        import timm
        net = timm.create_model('vit_base_patch16_224', pretrained=True)
    net = net.to(device)
    net.eval()

    all_norms     = []
    all_queries   = []
    image_iter    = 0
    success_count = 0

    for image_iter1 in range(1, 51):
        if image_iter >= num_img:
            break
        if len(str(image_iter1)) == 1:
            temp = "000" + str(image_iter1)
        if len(str(image_iter1)) == 2:
            temp = "00"  + str(image_iter1)
        if len(str(image_iter1)) == 3:
            temp = "0"   + str(image_iter1)
        if len(str(image_iter1)) == 4:
            temp =        str(image_iter1)
        img_name = "ILSVRC2012_val_0000" + temp + ".JPEG"
        img_path = "Image_path/ImageNet/val"

        t11 = time.time()

        im_orig = Image.open(os.path.join(img_path, img_name))
        im_sz = 224
        im_orig = transforms.Compose([transforms.Resize((im_sz, im_sz))])(im_orig)

        delta = 255
        lb, ub = valid_bounds(im_orig, delta)

        im = transforms.Compose([
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)])(im_orig)

        lb = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])(lb)
        ub = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])(ub)

        lb = lb[None, :, :, :].to(device)
        ub = ub[None, :, :, :].to(device)

        x_0 = im[None, :, :, :].to(device)

        orig_label = torch.argmax(net.forward(Variable(x_0, requires_grad=True)).data).item()
        labels = open(os.path.join('synset_words.txt'), 'r').read().split('\n')
        str_label_orig = get_label(labels[np.int32(orig_label)].split(',')[0])

        ground_truth = open(os.path.join('val.txt'), 'r').read().split('\n')
        ground_name_label = ground_truth[image_iter1 - 1]
        ground_label = ground_name_label.split()[1]
        ground_label_int = int(ground_label)

        str_label_ground = get_label(labels[np.int32(ground_label)].split(',')[0])
        print(f'\nSource image {temp}:  Class ID: {ground_label}   Class Name: {str_label_ground}')

        ##############################################################################
        if ground_label_int != int(orig_label):
            print('Already missclassified ... Lets try another one!')
        else:
            image_iter = image_iter + 1
            print('Image number good to go: ', image_iter)

            print('#################################################################################')
            print(f'Start: {attack_method}-{VARIANT_SUFFIX} non-targeted, iter={iteration}, dim_reduc={dim_reduc_factor}')
            print('#################################################################################')

            t3 = time.time()
            attack = Proposed_attack(net, x_0, mean, std, lb, ub,
                                     dim_reduc_factor=dim_reduc_factor,
                                     attack_method=attack_method,
                                     iteration=iteration)
            x_adv, n_query, norms = attack.Attack()
            t4 = time.time()
            print(f'##################### End Iterations:  took {t4-t3:.3f} sec #########################')

            # ── 验证攻击是否成功 ──────────────────────────────────
            with torch.no_grad():
                adv_label = torch.argmax(net(x_adv)).item()

            if adv_label != ground_label_int:
                str_label_adv = get_label(labels[np.int32(adv_label)].split(',')[0])
                print(f'Attack SUCCESS: {str_label_ground} → {str_label_adv}')
                success_count += 1
                all_norms.append(norms)
                all_queries.append(n_query)
            else:
                print(f'Attack FAILED: still predicted as {str_label_ground}')

    # ── ASR 统计 ──────────────────────────────────────────────
    asr = success_count / image_iter * 100 if image_iter > 0 else 0
    print(f'\n── Attack Summary ──────────────────────────────')
    print(f'Model         : {model_arc}')
    print(f'Attack method : {attack_method}-{VARIANT_SUFFIX}')
    print(f'Total images  : {image_iter}')
    print(f'Success       : {success_count}')
    print(f'ASR           : {asr:.1f}%')
    print(f'────────────────────────────────────────────────')

    # ── 保存结果 ──────────────────────────────────────────────
    if len(all_norms) > 0:
        norm_array  = np.array(all_norms)
        query_array = np.array(all_queries)

        if not os.path.exists('Non_targeted_results'):
            os.makedirs('Non_targeted_results')

        save_path = (f'Non_targeted_results/{attack_method}_{VARIANT_SUFFIX}_nonTar_{model_arc}'
                     f'_dimReducFac_{dim_reduc_factor}'
                     f'_imgNum_{num_img}_iteration_{iteration}')
        np.savez(
            save_path,
            norm        = np.median(norm_array,  0),
            query       = np.median(query_array, 0),
            all_norms   = norm_array,
            all_queries = query_array,
            asr         = asr,
        )
        print(f'Results saved to {save_path}.npz')
    else:
        print('No successful attacks, nothing saved.')
