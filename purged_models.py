import torch
import torch.nn as nn
import torch.nn.functional as F

from dataclasses import dataclass
from copy import deepcopy

from models import L0LeNet5, L0MLP, L0WideResNet
from utils import get_final_features

from typing import Type, Union, List, Dict, Optional, Tuple
from l0_layers import L0Conv2d

@dataclass
class PrunedLayerData:
    """Stores the main elements of a base pytorch layer. This is used to
    construct a pruned layer from an L0 layer."""
    weight: torch.Tensor
    bias: Optional[torch.Tensor]
    kwargs: Optional[Dict]

def purge_conv_modules(conv_layers: List, val_kwargs: Dict):
    """Purges a list of L0Conv2d modules by removing """

    conv_kwargs_list = [_.conv_kwargs for _ in conv_layers]

    layer_gates = []
    for layer in conv_layers:
        # binarize = True to get a mask for slicing
        gates = layer.sample_z(do_sample=False, collapse_batch=True,
                                val_threshold=val_kwargs['val_threshold'],
                                val_binarize=True)
        layer_gates.append(torch.flatten(gates).bool())

    purged_modules = []

    # First layer
    my_gates = layer_gates[0]
    dense_params = conv_layers[0].sample_params(do_sample=False, val_kwargs=val_kwargs)

    _weight = dense_params['weight'][my_gates, ...]
    _bias = dense_params['bias'][my_gates] if dense_params['bias'] is not None else None
    purged_modules.append(PrunedLayerData(_weight, _bias, conv_kwargs_list[0]))

    for i, foo in enumerate(zip(conv_layers[1:], conv_kwargs_list[1:]), start=1):

        layer, conv_kwargs = foo
        prev_gates = layer_gates[i-1]

        my_gates = layer_gates[i]
        dense_params = layer.sample_params(do_sample=False, val_kwargs=val_kwargs)

        _weight = dense_params['weight'][my_gates, ...][:, prev_gates, ...]
        _bias = dense_params['bias'][my_gates] if dense_params["bias"] is not None else None
        purged_modules.append(PrunedLayerData(_weight, _bias, conv_kwargs))

    return purged_modules, layer_gates[-1]

def compute_layer_nsp(layer):
    num_par = layer.weight.numel()
    if layer.bias is not None:
        num_par += layer.bias.numel()
    return num_par

# --------------- Create purged layers -------


def create_purged_conv(weight, bias, conv_kwargs) -> nn.Conv2d:
    """ Creates an nn.Conv2d module from given weight and bias
    """
    use_bias = not(bias is None)
    out_channels, in_channels, kh, kw = weight.shape

    conv_layer = nn.Conv2d(in_channels, out_channels, (kh, kw),
                           bias=use_bias, **conv_kwargs)

    conv_layer.weight.data = weight
    if use_bias:
        conv_layer.bias.data = bias

    return conv_layer

class PurgedResNet(nn.Module):
    def __init__(self, model: Type[L0WideResNet], val_kwargs):
        super().__init__()

        self.input_size = (3, 32, 32)

        # 1st conv before any network block
        _weight = model.conv1.weight.data
        _kwargs = dict(stride=1, padding=1)
        self.conv1 = create_purged_conv(_weight, None, _kwargs)

        self.l0_layers = []
        for i, network_block in enumerate([model.block1, model.block2, model.block3]):
            name = "block" + str(i + 1)
            purged_block = PurgedNetworkBlock(network_block, val_kwargs)
            setattr(self, name, purged_block)
            self.l0_layers += getattr(purged_block, "l0_layers")

        self.batch_norm = deepcopy(model.batch_norm)

        _weight = model.fcout.weight.data
        _bias = model.fcout.bias.data if model.fcout.use_bias else None
        self.fcout = create_purged_linear(_weight, _bias)

    def forward(self, x):
        out = self.conv1(x)

        out = self.block1.forward(out)
        out = self.block2.forward(out)
        out = self.block3.forward(out)

        out = F.relu(self.batch_norm(out))
        # To match fcout, must have a 1x1 matrix per channel.
        pool_k_size = (out.shape[2], out.shape[3])
        out = F.avg_pool2d(out, pool_k_size)
        out = out.view(out.size(0), -1)

        return self.fcout(out)

    def nsp_per_layer(self):
        # Just considering pruned layers here. Ignoring BatchNorm and MAP layers.
        purged_nsps = []
        for layer in self.l0_layers:
            purged_nsps.append(compute_layer_nsp(layer))
        return purged_nsps

class PurgedNetworkBlock(nn.Module):
    def __init__(self, network_block, val_kwargs):
        super().__init__()

        self.l0_layers = []
        blocks = []
        for basic_block in network_block.blocks:
            # Block with purged layers
            purged_block = PurgedBasicBlock(basic_block, val_kwargs)
            blocks.append(purged_block)

            # Store the layers which were L0-based
            self.l0_layers.append(purged_block.conv1)

        self.blocks = nn.Sequential(*blocks)

    def forward(self, x):
        return self.blocks.forward(x)

class PurgedBasicBlock(nn.Module):
    def __init__(self, block, val_kwargs):
        super().__init__( )

        self.batch_norm1 = deepcopy(block.batch_norm1)

        purged_modules, gates = purge_conv_modules([block.l0conv1], val_kwargs)
        purged_conv_module = purged_modules[0]
        self.conv1 = create_purged_conv(purged_conv_module.weight,
                                      purged_conv_module.bias,
                                      purged_conv_module.kwargs)

        # Batch Norm 2
        self.batch_norm2 = nn.BatchNorm2d(int(sum(gates).item()))
        self.batch_norm2.weight.data = block.batch_norm2.weight.data[gates]
        self.batch_norm2.bias.data = block.batch_norm2.bias.data[gates]
        self.batch_norm2.running_mean = block.batch_norm2.running_mean[gates]
        self.batch_norm2.running_var = block.batch_norm2.running_var[gates]
        self.batch_norm2.num_batches_tracked = block.batch_norm2.num_batches_tracked

        # Second conv layer
        _weight = block.conv2.weight.data[:, gates, ...]
        _bias = block.conv2.bias.data if block.conv2.use_bias else None
        _kwargs = block.conv2.conv_kwargs
        self.conv2 = create_purged_conv(_weight, _bias, _kwargs)

        # The conv shortcut is not affected by sparsification
        self.preserves_planes = block.preserves_planes
        if not self.preserves_planes:
            _weight = block.conv_shortcut.weight.data
            _bias = block.conv_shortcut.bias.data if block.conv_shortcut.use_bias else None
            _kwargs = block.conv_shortcut.conv_kwargs
            self.conv_shortcut = create_purged_conv(_weight, _bias, _kwargs)

    def forward(self, x):
        # TODO: same forward as basic block. Could inherit from BasicBlock
        # Pre-activation employed. BatchNorm -> ReLU -> Conv -> Add skip connection
        out = F.relu(self.batch_norm1(x))

        if self.preserves_planes:
            # No need to reshape x
            reshaped_x = x
        else:
            # BN + ReLU + Reshape (1x1 conv)
            reshaped_x = self.conv_shortcut(out)

        out = self.conv1(out)
        out = self.conv2(F.relu(self.batch_norm2(out)))
        out = torch.add(out, reshaped_x)

        return out