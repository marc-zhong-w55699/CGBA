# models_cifar10.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os

# ── PreActResNet18 (from kuangliu/pytorch-cifar) ──────────────────────────────
if not os.path.exists('./pytorch-cifar'):
    os.system('git clone https://github.com/kuangliu/pytorch-cifar.git')
sys.path.append('./pytorch-cifar')
from models.preact_resnet import PreActResNet18

# ── ViT (from lucidrains/vit-pytorch) ─────────────────────────────────────────
# pip install vit-pytorch
from vit_pytorch import ViT


def build_vit_cifar10():

    return ViT(
        image_size  = 32,
        patch_size  = 4,
        num_classes = 10,
        dim         = 384,
        depth       = 7,
        heads       = 12,
        mlp_dim     = 384 * 4,
        dropout     = 0.1,
        emb_dropout = 0.1,
    )


# ── WideResNet 40-2 ───────────────────────────────────────────────────────────
class WideBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1, dropout=0.3):
        super().__init__()
        self.bn1      = nn.BatchNorm2d(in_planes)
        self.conv1    = nn.Conv2d(in_planes, planes, 3, padding=1, bias=False)
        self.drop     = nn.Dropout(p=dropout)
        self.bn2      = nn.BatchNorm2d(planes)
        self.conv2    = nn.Conv2d(planes, planes, 3, stride=stride, padding=1, bias=False)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, 1, stride=stride, bias=False)
            )
    def forward(self, x):
        out = self.drop(self.conv1(F.relu(self.bn1(x))))
        out = self.conv2(F.relu(self.bn2(out)))
        out += self.shortcut(x)
        return out

class WideResNet(nn.Module):
    def __init__(self, depth=40, widen_factor=2, dropout=0.3, num_classes=10):
        super().__init__()
        assert (depth - 4) % 6 == 0
        n       = (depth - 4) // 6
        nStages = [16, 16*widen_factor, 32*widen_factor, 64*widen_factor]
        self.in_planes = nStages[0]
        self.conv1  = nn.Conv2d(3, nStages[0], 3, padding=1, bias=False)
        self.layer1 = self._make_layer(nStages[1], n, stride=1, dropout=dropout)
        self.layer2 = self._make_layer(nStages[2], n, stride=2, dropout=dropout)
        self.layer3 = self._make_layer(nStages[3], n, stride=2, dropout=dropout)
        self.bn     = nn.BatchNorm2d(nStages[3])
        self.linear = nn.Linear(nStages[3], num_classes)

    def _make_layer(self, planes, num_blocks, stride, dropout):
        strides = [stride] + [1] * (num_blocks - 1)
        layers  = []
        for s in strides:
            layers.append(WideBlock(self.in_planes, planes, s, dropout))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.relu(self.bn(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        return self.linear(out)


# ── 加载函数 ──────────────────────────────────────────────────────────────────
def load_model(model_name, ckpt_dir='checkpoints', device='cpu'):
    if model_name == 'preactresnet18':
        net = PreActResNet18()
    elif model_name == 'wideresnet40_2':
        net = WideResNet(depth=40, widen_factor=2, dropout=0.3, num_classes=10)
    elif model_name == 'vit':
        net = build_vit_cifar10()
    else:
        raise ValueError(f'Unknown model: {model_name}')

    ckpt_path = os.path.join(ckpt_dir, f'{model_name}_cifar10_best.pth')
    net.load_state_dict(torch.load(ckpt_path, map_location=device))
    net = net.to(device)
    net.eval()
    print(f'Loaded {model_name} from {ckpt_path}')
    return net