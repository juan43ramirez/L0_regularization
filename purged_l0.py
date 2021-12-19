import torch
import numpy as np

from utils import AverageMeter, accuracy
from dataloaders import cifar10, cifar100
from models import L0WideResNet
from l0_layers import L0Conv2d
from purged_models import PurgedResNet

def get_top1(loader, model):

    top1 = AverageMeter()

    for _, (input_, target) in enumerate(loader):
        if torch.cuda.is_available():
            target = target.cuda()
            input_ = input_.cuda()

        output = model(input_)
        prec1 = accuracy(output.data, target, topk=(1,))[0]
        top1.update(100 - prec1.item(), input_.size(0))

    return top1.avg

# Config
dataset = "c100" # "c100"
lamba = 0.002

# Instantiate the model
model = L0WideResNet(
    28,
    num_classes=10 if dataset=="c10" else 100,
    widen_factor=10,
    droprate_init=0.3,
    N=50000,
    beta_ema=0.,
    weight_decay=5e-4,
    local_rep=False,
    lamba=lamba,
    temperature=2./3.
    )

# Find the model checkpoint
checkpoint_path = "runs/L0WideResNet_28_10"
if dataset == "c100": checkpoint_path += "_" + dataset
checkpoint_path += "_" + str(lamba) + "/checkpoint.pth.tar"

# Load checkpoint
checkpoint = torch.load(checkpoint_path)
prec1 = checkpoint['best_prec1']
time_acc = checkpoint['time_acc']

print("Reported best", prec1)
print("Current (time, train_error, val_error", time_acc[-1])

model.load_state_dict(checkpoint['state_dict'])
model = model.cuda()

# # Check train error
# dataload = cifar10 if dataset == 'c10' else cifar100
# train_loader, val_loader, _ = dataload(augment=True, batch_size=128)

# model.train()
# train_top1 = get_top1(train_loader, model)
# print("Train error:", train_top1)

# # Check val error
# model.eval()
# val_top1 = get_top1(val_loader, model)
# print("Val dataset error:", val_top1)

# ------------- Asses Prune degree

# Active gates and total gates.
nag, nsg = [], []

# Go over layers, select only L0 layers
for layer in model.layers:
    if isinstance(layer, L0Conv2d):
        # Get gates
        gates = layer.sample_z(1, sample=False)
        gates = gates.view(-1)
        print("Gates values", np.round(gates.detach().cpu().numpy(), 2))
        # Binarize (implicit threshold of 0.)
        gates = (gates > 0.).float() 
        nag.append(sum(gates).int().item())
        # Count
        nsg.append(len(gates))#, layer.weights.data.shape[0])

# For the last layer, check gates    

print("Active gates:", nag)
print("Sparsifiable gates:", nsg)
