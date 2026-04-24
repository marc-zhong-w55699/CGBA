# train_cifar10_vit.py
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import math
import os

from vit_pytorch import ViT  # pip install vit-pytorch

##############################################################################
torch.manual_seed(992)
torch.cuda.manual_seed(992)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using device: {device}')
##############################################################################


# ── 数据 ──────────────────────────────────────────────────────────────────────
mean = [0.4914, 0.4822, 0.4465]
std  = [0.2023, 0.1994, 0.2010]

train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std),
])
test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std),
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,  download=True, transform=train_transform)
testset  = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True,  num_workers=4, pin_memory=True)
testloader  = torch.utils.data.DataLoader(testset,  batch_size=128, shuffle=False, num_workers=4, pin_memory=True)

os.makedirs('checkpoints', exist_ok=True)


# ── 工具函数 ──────────────────────────────────────────────────────────────────
def build_model(model_name):
    if model_name == 'vit':
        # 使用 lucidrains/vit-pytorch 的 ViT
        # 针对 CIFAR-10 (32x32) 的小型配置
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
    else:
        raise ValueError(f'Unknown model: {model_name}')


def evaluate(net):
    net.eval()
    correct = total = 0
    with torch.no_grad():
        for inputs, targets in testloader:
            inputs, targets = inputs.to(device), targets.to(device)
            correct += net(inputs).argmax(1).eq(targets).sum().item()
            total   += targets.size(0)
    return 100. * correct / total


def train_model(model_name, epochs=200, lr=1e-3, weight_decay=5e-2, warmup_epochs=10):
    print(f'\n{"="*60}')
    print(f'Training {model_name}')
    print(f'{"="*60}')

    net       = build_model(model_name).to(device)
    optimizer = optim.AdamW(net.parameters(), lr=lr, weight_decay=weight_decay)

    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / max(1, epochs - warmup_epochs)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    best_acc  = 0.0
    ckpt_path = f'checkpoints/{model_name}_cifar10_best.pth'

    for epoch in range(1, epochs + 1):
        # ── Train ────────────────────────────────────────────────
        net.train()
        train_loss = correct = total = 0

        for inputs, targets in trainloader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss    = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            correct    += outputs.argmax(1).eq(targets).sum().item()
            total      += targets.size(0)

        scheduler.step()
        train_acc  = 100. * correct / total
        train_loss = train_loss / len(trainloader)

        # ── Evaluate every 10 epochs and last epoch ──────────────
        if epoch % 10 == 0 or epoch == epochs:
            test_acc = evaluate(net)
            marker   = ' ← best' if test_acc > best_acc else ''
            print(f'Epoch [{epoch:3d}/{epochs}]  '
                  f'Loss: {train_loss:.4f}  '
                  f'Train Acc: {train_acc:.2f}%  '
                  f'Test Acc: {test_acc:.2f}%{marker}')
            if test_acc > best_acc:
                best_acc = test_acc
                torch.save(net.state_dict(), ckpt_path)
        else:
            print(f'Epoch [{epoch:3d}/{epochs}]  '
                  f'Loss: {train_loss:.4f}  '
                  f'Train Acc: {train_acc:.2f}%')

    print(f'\n{model_name} best test acc: {best_acc:.2f}%')
    print(f'Checkpoint saved to: {ckpt_path}')


if __name__ == '__main__':
    train_model('vit')
