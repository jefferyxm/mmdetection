import torch
import torch.nn as nn
from mmdet.models.necks import PAM_Module, CAM_Module


class AddAttention(nn.Module):
    def __init__(self, in_dim, num_lvls):
        super(PAM_Module, self).__init__()

        self.div4 = nn.MaxPool2d(in_dim, in_dim, kernel_size=4, stride=4, padding=0)
        self.div2 = nn.MaxPool2d(in_dim, in_dim, kernel_size=2, stride=2, padding=0)
        # self.div1 = nn.Conv2d(in_dim, in_dim, kernel_size=1, stride=1, padding=0)

        self.up2 = nn.UpsamplingNearest2d(scale_factor=2)
        self.up4 = nn.UpsamplingNearest2d(scale_factor=4)

        self.in_dim = in_dim
        self.num_lvls = num_lvls

        self.pams = nn.ModuleList()
        for i in range(num_lvls):
            pam = PAM_Module(in_dim)
            self.pams.append(pam)
            
        self.add_pam = PAM_Module(in_dim)
        # self.add_cam = CAM_Module(in_dim)

    def forward(self, x):
        assert self.num_lvls == len(x)
        pam_inputs = [self.div4(x[0]), self.div2(x[1]), 
                      x[2], x[3] if self.num_lvls >= 4 else None ]
        pam_inputs.remove(None)

        pam_outs = [
            self.pams[i](pam_inputs[i]) for i in range(len(pam_inputs))
        ]

        outs = [self.up4(pam_outs[0]), self.up2(pam_outs[1]), 
                pam_outs[2], pam_outs[3] if self.num_lvls >=4 else None]
        outs.remove(None)
        return tuple(outs)

