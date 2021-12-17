import torch
from dataloaders import cifar10, cifar100
from models import L0WideResNet
from purged_models import PurgedResNet

# Config
dataset = "c10" # "c100"
lamba = 0.001 # 0.002

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
print("Current acc", time_acc)

model.load_state_dict(checkpoint['state_dict'])
model.cuda()

# Check train error
dataload = cifar10 if args.dataset == 'c10' else cifar100
train_loader, val_loader, _ = dataload(augment=True, batch_size=128)

model.train()
train_top1 = get_top1(train_loader, model)
print("Train error:", train_top1)

# Check val error
model.train()
val_top1 = get_top1(val_loader, model)
print("Val dataset error:", val_top1)

# Prune


# New val_error after prune


# Get number of parameters





def get_top1(loader, model):
    
    for i, (input_, target) in enumerate(loader):
        if torch.cuda.is_available():
            target = target.cuda()
            input_ = input_.cuda()
        # input_var = torch.autograd.Variable(input_)
        # target_var = torch.autograd.Variable(target)

        output = model(input_var)
        prec1 = accuracy(output.data, target, topk=(1,))[0]
        top1.update(100 - prec1.item(), input_.size(0))
    
    return top1.avg