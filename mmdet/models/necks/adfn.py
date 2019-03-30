# this file is implemetation of arpn deform feature
import torch
import torch.nn as nn
from mmdet.ops import DeformConv
from mmcv.cnn import constant_init, xavier_init
from ..registry import NECKS

class ARPNDeformFeature(nn.Module):

    def __init__(self,
                 feature_in_channels,
                 feature_out_channels,
                 num_fpn_levels=4,
                 deform_kernel_size=3,
                 offset_in_channels=2,
                 offset_kernel_size=1,
                 with_activation=None,
                ):
        super(ARPNDeformFeature, self).__init__()
        self.feature_in_channels = feature_in_channels
        self.feature_out_channels = feature_out_channels
        self.num_fpn_levels = num_fpn_levels
        self.deform_kernel_size = deform_kernel_size
        self.offset_kernel_size = offset_kernel_size
        self.offset_in_channels = offset_in_channels
        self.with_activation = with_activation

        self.offset_convs = nn.ModuleList()
        self.deform_convs = nn.ModuleList()

        for i in range(self.num_fpn_levels):
            offset_conv = nn.Conv2d(
                self.offset_in_channels,
                self.deform_kernel_size**2*2,
                kernel_size = offset_kernel_size,
                padding = offset_kernel_size//2,
                stride = 1,
            )

            deform_conv = DeformConv(
                self.feature_in_channels,
                self.feature_out_channels,
                kernel_size=self.deform_kernel_size,
                stride = 1,
                padding = 1,
            )
            self.offset_convs.append(offset_conv)
            self.deform_convs.append(deform_conv)

        if self.with_activation:
            self.activate = nn.ReLU(inplace=inplace)
        
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                constant_init(m, 0)
    
    def forward(self, feature_input, offset_input):
        assert len(feature_input)==len(offset_input)==self.num_fpn_levels

        # bulid offset convs
        offsets = [
            offset_convx(offset_input[i])
            for i, offset_convx in enumerate(self.offset_convs)
        ]

        # build deform convs
        deform_outs = [
            deform_convx(feature_input[i], offsets[i])
            for i, deform_convx in enumerate(self.deform_convs)
        ]

        if self.with_activation:
            deform_outs = [
                self.activate(deform_outs[i])
                for i in range(len(deform_outs))
            ]
        
        return tuple(deform_outs)
        