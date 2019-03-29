import torch

from ..bbox import assign_and_sample, build_assigner, PseudoSampler, bbox2delta
from ..utils import multi_apply
import numpy as np
import numpy.random as npr


def adaptive_anchor_target(anchor_points_list,
                  valid_flag_list,
                  gt_bboxes_list,
                  img_metas,
                  target_means,
                  target_stds,
                  cfg,
                  gt_bboxes_ignore_list=None,
                  gt_labels_list=None,
                  label_channels=1,
                  sampling=True,
                  unmap_outputs=True,
                  stride_list = [4, 8, 16, 32, 64]):
    """Compute regression and classification targets for anchors.

    Args:
        anchor_points_list (list[list]): Multi level anchor points of each image.
        valid_flag_list (list[list]): Multi level valid flags of each image.
        gt_bboxes_list (list[Tensor]): Ground truth bboxes of each image.
        img_metas (list[dict]): Meta info of each image.
        target_means (Iterable): Mean value of regression targets.
        target_stds (Iterable): Std value of regression targets.
        cfg (dict): RPN train configs.

    Returns:
        ARPN training target
        include:    cls & cls weights
                    shape_wh & shape_wh_weights
                    reg_target & reg_target_weights
                    positive_num & nagative_num
    """
    num_imgs = len(img_metas)
    assert len(anchor_points_list) == len(valid_flag_list) == num_imgs
    # anchor number of multi levels
    num_level_ap = [ap.size(0) for ap in anchor_points_list[0]]
    norm_list = np.array(stride_list) * 8.0
    num_total_pos = 0
    num_total_neg = 0

    labels_list, shape_wh_list, bbox_targets_list= [], [], []
    label_weights_list, shape_wh_weights_list, bbox_weights_list = [], [], [] 
    for im_i in range(num_imgs):
        
        labels, shape_whs, box_targets = [], [], []
        bg_list = []
        gt_rois = np.array(torch.Tensor.cpu(gt_bboxes_list[im_i]))

        for i in range(len(num_level_ap)):
            this_level_ap = np.array(torch.Tensor.cpu(anchor_points_list[im_i][i]))
            this_level_label = np.zeros((this_level_ap.shape[0], ), dtype=np.float32)
            this_level_label_weight = np.zeros((this_level_ap.shape[0], ), dtype=np.float32)
            this_level_wh = np.zeros((this_level_ap.shape[0], 2), dtype=np.float32)
            this_level_box_delta = np.zeros((this_level_ap.shape[0], 4),dtype=np.float32)
            if gt_rois.shape[0] > 0:
                norm = norm_list[i]
                gt_areas = (gt_rois[:, 2] - gt_rois[:, 0] + 1)*(gt_rois[:, 3]-gt_rois[:, 1] + 1)/(norm*norm)
                if i == 0:
                    valid_gtidx = np.where(gt_areas <= 2.0)[0]
                elif i == len(num_level_ap) - 1:
                    valid_gtidx = np.where(gt_areas >= 0.5)[0]
                else:
                    valid_gtidx = np.where((gt_areas <= 2.0) & (gt_areas >= 0.5))[0]
                valid_gts = gt_rois[valid_gtidx, :]

                valid_apidx = np.empty(0, dtype=np.int32)
                for gt in valid_gts:
                    idx = np.where( (this_level_ap[:,0] > gt[0]) & 
                                    (this_level_ap[:,0] < gt[2]) & 
                                    (this_level_ap[:,1] > gt[1]) & 
                                    (this_level_ap[:,1] < gt[3]) )[0]
                    valid_apidx = np.append(valid_apidx, idx)
                valid_apidx = np.unique(valid_apidx)
                valid_aps = this_level_ap[valid_apidx]

                m =valid_aps.shape[0]
                n =valid_gts.shape[0]

                # points transformation
                # 1 transform all points to left-up side of the gt boxes and 
                # 2 set points outside the boxes to the left-up corner
                transfm_aps = np.zeros((2,m,n), dtype=np.float32)
                tmp_aps = np.empty(valid_aps.shape, dtype=np.float32)
                for idx, gt in enumerate(valid_gts):
                    tmp_aps[:] = valid_aps
                    # 1
                    gtcx = 0.5*(gt[0] + gt[2] + 1)
                    gtcy = 0.5*(gt[1] + gt[3] + 1)
                    tmp_aps[np.where( ( gtcx - tmp_aps[:,0]) < 0 )[0], 0] \
                                = 2*gtcx - tmp_aps[ np.where( ( gtcx - tmp_aps[:,0]) < 0 )[0], 0]
                    tmp_aps[np.where( ( gtcy - tmp_aps[:,1]) < 0 )[0], 1] \
                                = 2*gtcy - tmp_aps[ np.where( ( gtcy - tmp_aps[:,1]) < 0 )[0], 1]
                    
                    # 2 add a small value to prevent D & C to be zero 
                    tmp_aps[np.where( (tmp_aps[:,0] <= gt[0]) | (tmp_aps[:,1] <= gt[1]) )[0] ] = gt[0:2] + 0.001
                    transfm_aps[:, :, idx] = tmp_aps.transpose(1,0)
                
                A = np.zeros((m, n), dtype = np.float32)
                A[:] = (valid_gts[:,2] - valid_gts[:, 0] + 1)*(valid_gts[:,3] - valid_gts[:, 1] + 1)
                C = ( transfm_aps[0] - (np.tile(valid_gts[:,0], [m, 1])) ) * 0.5
                D = ( transfm_aps[1] - (np.tile(valid_gts[:,1], [m, 1])) ) * 0.5
                B = 4*C*D

                CANDW = np.zeros((4, m, n), dtype = np.float32)
                CANDH = np.zeros((4, m, n), dtype = np.float32)
                # on  the edge of the constrains
                CANDW[0:2, :, :] = [4*C,  2*(1 + np.tile(valid_gts[:, 2], [m, 1]) - transfm_aps[0] ) ]
                CANDH[0:2, :, :] = [4*D,  2*(1 + np.tile(valid_gts[:, 3], [m, 1]) - transfm_aps[1] ) ]

                # inside constrains 
                sqdelta = np.sqrt(np.power((A-4*B),2) + 64*A*C*D)
                a1 = ((A-4*B) + sqdelta)
                a2 = ((A-4*B) - sqdelta)

                w1 = a1/(8*D)
                w1[np.where( (w1-CANDW[0,:,:] < 0) | (w1 - CANDW[1,:,:] > 0) )[0]] = 0
                w2 = a2/(8*D)
                w2[np.where( (w2-CANDW[0,:,:] < 0) | (w2 - CANDW[1,:,:] > 0) )[0]] = 0

                h1 = a1/(8*C)
                h1[np.where( (h1 - CANDH[0,:,:] < 0) | (h1 - CANDH[1,:,:] > 0) )[0]] = 0
                h2 = a2/(8*C)
                h2[np.where( (h2 - CANDH[0,:,:] < 0) | (h2 - CANDH[1,:,:] > 0) )[0]] = 0

                CANDW[2:4,:,:] = [w1, w2]
                CANDH[2:4,:,:] = [h1, h2]

                # conbination of the w & h
                CANDWS = np.tile(CANDW, [4,1,1])
                CANDHS = np.repeat(CANDH, 4, axis = 0)
                IOUS = (B+ C*CANDHS + D*CANDWS + 0.25*CANDWS*CANDHS)/(A-(B + C*CANDHS + D*CANDWS) + 0.75*CANDWS*CANDHS)
                IOUS[ np.where( (CANDWS==0) | (CANDHS==0) ) ] = 0
                IOU = np.max(IOUS, axis=0)
                WHidx = np.argmax(IOUS, axis=0)

                enable_idx = np.where(IOU>=0.7)

                # generate label map
                this_level_label[ valid_apidx[enable_idx[0]] ] = 1
                this_level_label_weight[ valid_apidx[enable_idx[0]] ] = 1

                this_level_wh[ valid_apidx[enable_idx[0]], 0 ] = \
                            CANDWS[ WHidx[enable_idx[0], enable_idx[1]], enable_idx[0], enable_idx[1] ]/norm 
                this_level_wh[ valid_apidx[enable_idx[0]], 1 ] = \
                            CANDHS[ WHidx[enable_idx[0], enable_idx[1]], enable_idx[0], enable_idx[1] ]/norm
                
                # compute box delta
                gt_widths = (valid_gts[enable_idx[1], 2] - valid_gts[enable_idx[1], 0] + 1) 
                gt_heghts = (valid_gts[enable_idx[1], 3] - valid_gts[enable_idx[1], 1] + 1) 
                gt_ctrx = valid_gts[enable_idx[1], 0] + 0.5 * gt_widths
                gt_ctry = valid_gts[enable_idx[1], 1] + 0.5 * gt_heghts

                this_level_box_delta[valid_apidx[enable_idx[0]], 0] = \
                        (gt_ctrx - this_level_ap[valid_apidx[enable_idx[0]], 0])/(this_level_wh[valid_apidx[enable_idx[0]], 0] * norm)
                this_level_box_delta[valid_apidx[enable_idx[0]], 1] = \
                        (gt_ctry - this_level_ap[valid_apidx[enable_idx[0]], 1])/(this_level_wh[valid_apidx[enable_idx[0]], 1] * norm)
                this_level_box_delta[valid_apidx[enable_idx[0]], 2] = \
                        np.log( gt_widths/(this_level_wh[valid_apidx[enable_idx[0]], 0] * norm) )
                this_level_box_delta[valid_apidx[enable_idx[0]], 3] = \
                        np.log( gt_heghts/(this_level_wh[valid_apidx[enable_idx[0]], 1] * norm) )
                
                cplogidx = np.where(this_level_wh > 0 ) 
                this_level_wh[ cplogidx ] = np.log(this_level_wh[ cplogidx ])

                DBG=0
                if DBG:
                    # show label in image
                    import matplotlib.pyplot as plt
                    import cv2


                    img_root = 'data/icdar2015/train/'
                    
                    im = cv2.imread(img_root + img_metas[0]['imname'])
                    im = cv2.resize(im, (0,0), fx=img_metas[0]['scale_factor'], fy=img_metas[0]['scale_factor'])
                    
                    im_plt = im[:,:,(2,1,0)]
                    plt.cla()
                    plt.imshow(im_plt)

                    tg_index = np.where(this_level_label==1)[0]
                    print(len(tg_index))

                    for tg in tg_index:
                        w = np.exp(this_level_wh[tg][0])*norm
                        h = np.exp(this_level_wh[tg][1])*norm
                        p1 = [(this_level_ap[tg][0] - 0.5*w), 
                                (this_level_ap[tg][1])- 0.5*h]
                        plt.gca().add_patch(plt.Rectangle((p1[0], p1[1]), w, h ,fill=False, edgecolor='r', linewidth=1))

                    for gt in valid_gts:
                        plt.gca().add_patch(plt.Rectangle((gt[0], gt[1]), gt[2]-gt[0], gt[3]-gt[1] ,fill=False, edgecolor='g', linewidth=1))

                    plt.show()
            
            # set the nearest position to be positive 
            #  not implement

            # save labels into list
            labels.append(this_level_label)
            shape_whs.append(this_level_wh)
            box_targets.append(this_level_box_delta)

            bg_map = np.zeros((this_level_ap.shape[0],), dtype=np.int32)
            bg_map[valid_apidx] = 1
            bg_list.append(bg_map)
        # lvl end

        # sampling positive and negative
        num_total = cfg.sampler.num
        pos_fraction = cfg.sampler.pos_fraction

        lvl_counts = [x.shape[0] for x in labels]
        slice_position = [sum(lvl_counts[:i+1]) for i in range(len(lvl_counts))]
        slice_position.insert(0,0)

        cat_labels = np.concatenate(tuple(labels), axis=0)       
        cat_shape_whs = np.concatenate(tuple(shape_whs), axis=0)
        cat_box_targets = np.concatenate(tuple(box_targets), axis=0)
        cat_bg = np.concatenate(tuple(bg_list), axis=0)

        valid_flag = tuple([valid_flag_list[im_i][l] for l in range(len(num_level_ap))])
        cat_valid_flag = torch.cat(valid_flag)
        cat_valid_flag = np.array(torch.Tensor.cpu(cat_valid_flag))

        assert cat_labels.shape[0] == cat_valid_flag.shape[0]

        pos_idx = np.where((cat_labels==1) & (cat_valid_flag == 1))[0]
        # sub sampling positive if too much
        if len(pos_idx) > int(num_total * pos_fraction):
            disable_inds = npr.choice(pos_idx, size=(len(pos_idx) - int(num_total * pos_fraction)), replace=False)
            cat_labels[disable_inds] = 0
            pos_idx = np.where((cat_labels==1) & (cat_valid_flag == 1))[0]
            # recover data
            cat_labels[disable_inds] = 1
        
        # sub_sampling negative if too much
        neg_idx = np.where((cat_bg==0) & (cat_valid_flag == 1))[0]
        if len(neg_idx) > int(num_total - len(pos_idx)):
            neg_idx = neg_idx[npr.randint(len(neg_idx), size=int(num_total - len(pos_idx)) )]

        num_total_pos += len(pos_idx)
        num_total_neg += len(neg_idx)
        
        # to cuda
        cat_labels = torch.from_numpy(cat_labels).to('cuda')
        cat_shape_whs = torch.from_numpy(cat_shape_whs).to('cuda')
        cat_box_targets = torch.from_numpy(cat_box_targets).to('cuda')

        # get weights
        cat_labels_w = torch.zeros((cat_labels.shape[0],), dtype=torch.float32, device = 'cuda')
        cat_shape_whs_w = torch.zeros((cat_labels.shape[0], 2), dtype=torch.float32, device = 'cuda')
        cat_box_targets_w = torch.zeros((cat_labels.shape[0], 4), dtype=torch.float32, device = 'cuda')
        
        weight_position = np.concatenate((pos_idx, neg_idx))
        cat_labels_w[weight_position] = 1.0
        cat_shape_whs_w[pos_idx, :] = 1.0
        cat_box_targets_w[pos_idx, :] = 1.0

        # recover data
        labels, shape_whs, box_targets = [], [], []
        labels_w, shape_whs_w, box_targets_w = [], [], []
        for l in range(len(slice_position)-1):
            labels.append(cat_labels[slice_position[l]:slice_position[l+1]])
            labels_w.append(cat_labels_w[slice_position[l]:slice_position[l+1]])

            shape_whs.append(cat_shape_whs[slice_position[l]:slice_position[l+1]])
            shape_whs_w.append(cat_shape_whs_w[slice_position[l]:slice_position[l+1]])

            box_targets.append(cat_box_targets[slice_position[l]:slice_position[l+1]])
            box_targets_w.append(cat_box_targets_w[slice_position[l]:slice_position[l+1]])
        
        # fit shape to loss input
        if im_i == 0:
            labels_list = labels.copy()
            shape_wh_list = shape_whs.copy()
            bbox_targets_list = box_targets.copy()

            label_weights_list = labels_w.copy()
            shape_wh_weights_list = shape_whs_w.copy()
            bbox_weights_list = box_targets_w.copy()
        else:
            for lvl in range(len(labels)):
                labels_list[lvl] = torch.cat((labels_list[lvl], labels[lvl]), 0)
                shape_wh_list[lvl] = torch.cat((shape_wh_list[lvl], shape_whs[lvl]), 0)
                bbox_targets_list[lvl] = torch.cat((bbox_targets_list[lvl],box_targets[lvl]), 0)

                label_weights_list[lvl]= torch.cat((label_weights_list[lvl], labels_w[lvl]), 0)
                shape_wh_weights_list[lvl] = torch.cat((shape_wh_weights_list[lvl], shape_whs_w[lvl]), 0)
                bbox_weights_list[lvl] = torch.cat((bbox_weights_list[lvl], box_targets_w[lvl]), 0)

        DBG=0
        if DBG :
            # show label in image
            import matplotlib.pyplot as plt
            import cv2
            img_root = 'data/icdar2015/train/'
            im = cv2.imread(img_root + img_metas[im_i]['imname'])
            im = cv2.resize(im, (0,0), fx=img_metas[im_i]['scale_factor'], fy=img_metas[im_i]['scale_factor'])
            
            im_plt = im[:,:,(2,1,0)]
            plt.cla()
            plt.subplot(2,2,1)
            plt.imshow(im_plt)

            score_map = labels_list[0]
            score_map_w = label_weights_list[0]
            shape_map = shape_wh_list[0]
            shape_map_w = shape_wh_weights_list[0]

            score_map = torch.Tensor.cpu(score_map)
            shape_map =torch.Tensor.cpu(shape_map)
            shape_map_w = torch.Tensor.cpu(shape_map_w)
            score_map_w = torch.Tensor.cpu(score_map_w)

            feat_h, feat_w = img_metas[im_i]['pad_shape'][0]//4, img_metas[im_i]['pad_shape'][1]//4
            score_map = torch.reshape(score_map, (1, feat_h, feat_w, 1))
            score_map = score_map.permute((0,3,1,2))

            shape_map = torch.reshape(shape_map, (1, feat_h, feat_w, 2))
            shape_map = shape_map.permute((0,3,1,2))

            score_map_w = torch.reshape(score_map_w, (1, feat_h, feat_w, 1))
            score_map_w = score_map_w.permute((0,3,1,2))

            plt.subplot(2,2,2)
            plt.imshow(score_map[0,0,:,:], cmap=plt.cm.hot)

            plt.subplot(2,2,3)
            plt.imshow(shape_map[0,0,:,:], cmap=plt.cm.hot)

            plt.subplot(2,2,4)
            plt.imshow(score_map_w[0,0,:,:], cmap=plt.cm.hot)
            plt.show()

    return (labels_list, label_weights_list, 
            shape_wh_list, shape_wh_weights_list,
            bbox_targets_list, bbox_weights_list, 
            num_total_pos, num_total_neg)





