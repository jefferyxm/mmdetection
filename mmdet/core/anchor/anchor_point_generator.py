import torch


class AnchorPointGenerator(object):

    def __init__(self, base_size, scales, scale_major=True, ctr=None):
        self.base_size = base_size
        self.scales = torch.Tensor(scales)
        self.scale_major = scale_major
        self.ctr = ctr
        self.base_anchor_point = self.gen_base_anchor_point()

    @property
    def num_base_anchors(self):
        return self.base_anchors.size(0)

    def gen_base_anchor_point(self):
        w = self.base_size
        h = self.base_size
        if self.ctr is None:
            x_ctr = 0.5 * (w - 1)
            y_ctr = 0.5 * (h - 1)
        else:
            x_ctr, y_ctr = self.ctr
        return torch.tensor([x_ctr, y_ctr])

    def _meshgrid(self, x, y, row_major=True):
        xx = x.repeat(len(y))
        yy = y.view(-1, 1).repeat(1, len(x)).view(-1)
        if row_major:
            return xx, yy
        else:
            return yy, xx

    def grid_anchor_points(self, featmap_size, stride=16, device='cuda'):
        '''
            generate anchor center points for each position, 
            K*2 style,  K is number of positions on feature map(featmap_h * featmap_w), row first
            for example:
                [   
                    [0,0],
                    [1,0],
                    [2,0],
                    ...,
                    [0,1],
                    [1,1],
                    [2,1],
                    ...
                ]
        '''
        base_anchor_point = self.base_anchor_point.to(device)
        feat_h, feat_w = featmap_size
        shift_x = torch.arange(0, feat_w, device=device) * stride
        shift_y = torch.arange(0, feat_h, device=device) * stride
        shift_xx, shift_yy = self._meshgrid(shift_x, shift_y)
        shifts = torch.stack([shift_xx, shift_yy], dim=-1)
        shifts = shifts.type_as(base_anchor_point)
        anchor_points = shifts + base_anchor_point
        return anchor_points

    def valid_flags(self, featmap_size, valid_size, device='cuda'):
        '''
            if anchor points is inside of the featmap, then this points is valid
            same sequence as anchor points
        '''
        feat_h, feat_w = featmap_size
        valid_h, valid_w = valid_size
        assert valid_h <= feat_h and valid_w <= feat_w
        valid_x = torch.zeros(feat_w, dtype=torch.uint8, device=device)
        valid_y = torch.zeros(feat_h, dtype=torch.uint8, device=device)
        valid_x[:valid_w] = 1
        valid_y[:valid_h] = 1
        valid_xx, valid_yy = self._meshgrid(valid_x, valid_y)
        valid = valid_xx & valid_yy
        # valid = valid[:, None].expand(
        #     valid.size(0), self.num_base_anchors).contiguous().view(-1)
        return valid
