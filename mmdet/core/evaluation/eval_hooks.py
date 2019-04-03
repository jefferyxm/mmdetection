import os
import os.path as osp
import shutil
import time

import mmcv
import numpy as np
import torch
from mmcv.runner import Hook, obj_from_dict
from mmcv.parallel import scatter, collate
from pycocotools.cocoeval import COCOeval
from torch.utils.data import Dataset

from .coco_utils import results2json, fast_eval_recall
from .mean_ap import eval_map
from mmdet import datasets
from mmdet.core.evaluation.icdar_evaluation import icdar_eval
import pycocotools.mask as maskUtils
import cv2


class DistEvalHook(Hook):

    def __init__(self, dataset, interval=1, cfg=None):
        if isinstance(dataset, Dataset):
            self.dataset = dataset
        elif isinstance(dataset, dict):
            self.dataset = obj_from_dict(dataset, datasets,
                                         {'test_mode': True})
        else:
            raise TypeError(
                'dataset must be a Dataset object or a dict, not {}'.format(
                    type(dataset)))
        self.interval = interval
        self.lock_dir = None
        self.cfg = cfg

    def _barrier(self, rank, world_size):
        """Due to some issues with `torch.distributed.barrier()`, we have to
        implement this ugly barrier function.
        """
        if rank == 0:
            for i in range(1, world_size):
                tmp = osp.join(self.lock_dir, '{}.pkl'.format(i))
                while not (osp.exists(tmp)):
                    time.sleep(1)
            for i in range(1, world_size):
                tmp = osp.join(self.lock_dir, '{}.pkl'.format(i))
                os.remove(tmp)
        else:
            tmp = osp.join(self.lock_dir, '{}.pkl'.format(rank))
            mmcv.dump([], tmp)
            while osp.exists(tmp):
                time.sleep(1)

    def before_run(self, runner):
        self.lock_dir = osp.join(runner.work_dir, '.lock_map_hook')
        if runner.rank == 0:
            if osp.exists(self.lock_dir):
                shutil.rmtree(self.lock_dir)
            mmcv.mkdir_or_exist(self.lock_dir)

    def after_run(self, runner):
        if runner.rank == 0:
            shutil.rmtree(self.lock_dir)

    def after_train_epoch(self, runner):
        if not self.every_n_epochs(runner, self.interval):
            return
        runner.model.eval()
        results = [None for _ in range(len(self.dataset))]
        prog_bar = mmcv.ProgressBar(len(self.dataset))
        for idx in range(runner.rank, len(self.dataset), runner.world_size):
            data = self.dataset[idx]
            data_gpu = scatter(
                collate([data], samples_per_gpu=1),
                [torch.cuda.current_device()])[0]

            # compute output
            with torch.no_grad():
                result = runner.model(
                    return_loss=False, rescale=True, **data_gpu)
            results[idx] = result

            batch_size = runner.world_size
            for _ in range(batch_size):
                prog_bar.update()

        if runner.rank == 0:
            print('\n')
            self._barrier(runner.rank, runner.world_size)
            for i in range(1, runner.world_size):
                tmp_file = osp.join(runner.work_dir, 'temp_{}.pkl'.format(i))
                tmp_results = mmcv.load(tmp_file)
                for idx in range(i, len(results), runner.world_size):
                    results[idx] = tmp_results[idx]
                os.remove(tmp_file)
            self.evaluate(runner, results)
        else:
            tmp_file = osp.join(runner.work_dir,
                                'temp_{}.pkl'.format(runner.rank))
            mmcv.dump(results, tmp_file)
            self._barrier(runner.rank, runner.world_size)
        self._barrier(runner.rank, runner.world_size)

    def evaluate(self):
        raise NotImplementedError


class DistEvalmAPHook(DistEvalHook):

    def evaluate(self, runner, results):
        gt_bboxes = []
        gt_labels = []
        gt_ignore = [] if self.dataset.with_crowd else None
        for i in range(len(self.dataset)):
            ann = self.dataset.get_ann_info(i)
            bboxes = ann['bboxes']
            labels = ann['labels']
            if gt_ignore is not None:
                ignore = np.concatenate([
                    np.zeros(bboxes.shape[0], dtype=np.bool),
                    np.ones(ann['bboxes_ignore'].shape[0], dtype=np.bool)
                ])
                gt_ignore.append(ignore)
                bboxes = np.vstack([bboxes, ann['bboxes_ignore']])
                labels = np.concatenate([labels, ann['labels_ignore']])
            gt_bboxes.append(bboxes)
            gt_labels.append(labels)
        # If the dataset is VOC2007, then use 11 points mAP evaluation.
        if hasattr(self.dataset, 'year') and self.dataset.year == 2007:
            ds_name = 'voc07'
        else:
            ds_name = self.dataset.CLASSES
        mean_ap, eval_results = eval_map(
            results,
            gt_bboxes,
            gt_labels,
            gt_ignore=gt_ignore,
            scale_ranges=None,
            iou_thr=0.5,
            dataset=ds_name,
            print_summary=True)
        runner.log_buffer.output['mAP'] = mean_ap
        runner.log_buffer.ready = True


class CocoDistEvalRecallHook(DistEvalHook):

    def __init__(self,
                 dataset,
                 proposal_nums=(100, 300, 1000),
                 iou_thrs=np.arange(0.5, 0.96, 0.05)):
        super(CocoDistEvalRecallHook, self).__init__(dataset)
        self.proposal_nums = np.array(proposal_nums, dtype=np.int32)
        self.iou_thrs = np.array(iou_thrs, dtype=np.float32)

    def evaluate(self, runner, results):
        # the official coco evaluation is too slow, here we use our own
        # implementation instead, which may get slightly different results
        ar = fast_eval_recall(results, self.dataset.coco, self.proposal_nums,
                              self.iou_thrs)
        for i, num in enumerate(self.proposal_nums):
            runner.log_buffer.output['AR@{}'.format(num)] = ar[i]
        runner.log_buffer.ready = True


class CocoDistEvalmAPHook(DistEvalHook):

    def evaluate(self, runner, results):
        tmp_file = osp.join(runner.work_dir, 'temp_0.json')
        results2json(self.dataset, results, tmp_file)

        res_types = ['bbox',
                     'segm'] if runner.model.module.with_mask else ['bbox']
        cocoGt = self.dataset.coco
        cocoDt = cocoGt.loadRes(tmp_file)
        imgIds = cocoGt.getImgIds()
        for res_type in res_types:
            iou_type = res_type
            cocoEval = COCOeval(cocoGt, cocoDt, iou_type)
            cocoEval.params.imgIds = imgIds
            cocoEval.evaluate()
            cocoEval.accumulate()
            cocoEval.summarize()
            field = '{}_mAP'.format(res_type)
            runner.log_buffer.output[field] = cocoEval.stats[0]
        runner.log_buffer.ready = True
        os.remove(tmp_file)


best_hmean = 0
class IcdarDistEvalF1Hook(DistEvalHook):
    def evaluate(self, runner, results):
        #1 get result file
        for idx in range(len(self.dataset)):
            img_id = self.dataset.img_ids[idx]
            bbox_result = results[idx]
            bboxes = np.vstack(bbox_result)

            pt_dir = self.cfg.work_dir + '/pt/'
            im_name = self.dataset.img_infos[idx]['filename']
            if not os.path.exists(pt_dir):
                os.makedirs(pt_dir)
            img_index = im_name.split('_')[1]
            img_index = img_index.split('.')[0]
            pt_file = open(pt_dir + 'res_img_' + img_index + '.txt', 'w')

            score_thr = self.cfg.test_cfg.rcnn.score_thr
            inds = np.where(bboxes[:, -1] > score_thr)[0]

            for i in inds:
                # reshape to ori image
                px = (bboxes[i, ::2]).astype(int)
                py = (bboxes[i, 1::2]).astype(int)

                line = str(px[0]) + ',' + str(py[0]) + ',' + str(px[1]) + ',' + str(py[1]) + ',' + \
                        str(px[2]) + ',' + str(py[2]) + ',' + str(px[3]) + ',' + str(py[3]) + '\r\n'
                pt_file.write(line)
            pt_file.close()
        
        # get zip file
        import zipfile
        pt_zip_dir = os.path.join(self.cfg.work_dir, 'pt.zip')
        output_pt_dir = os.path.join(self.cfg.work_dir, 'pt/')
        z = zipfile.ZipFile(pt_zip_dir, 'w', zipfile.ZIP_DEFLATED)

        for dirpath, dirnames, filenames in os.walk(output_pt_dir):
            for filename in filenames:
                z.write(os.path.join(dirpath, filename), filename)
        z.close()

        #3 use icdar eval
        gt_zip_dir = './work_dirs/gt.zip'
        param_dict = dict(
            # gt zip file path
            g = gt_zip_dir,
            # prediction zip file path
            s = pt_zip_dir,
        )
        result_dict = icdar_eval(param_dict)
        runner.log_buffer.output['P'] = result_dict['precision']
        runner.log_buffer.output['R'] = result_dict['recall']
        runner.log_buffer.output['F1'] = result_dict['hmean']
        

        global best_hmean
        if result_dict['hmean'] > 0.8 and result_dict['hmean'] > best_hmean:
            # delete
            if os.path.exists(self.cfg.work_dir + '/model_best'):
                import shutil
                shutil.rmtree(self.cfg.work_dir + '/model_best')
            # save model 
            runner.save_checkpoint(self.cfg.work_dir + '/model_best', save_optimizer=True)
            best_hmean = result_dict['hmean']
        runner.log_buffer.output['Best'] = best_hmean
        runner.log_buffer.ready = True

        print(result_dict)
        for i in range(6):
            print('')
        
        





