import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import argparse
import segearth_segmentor
import custom_datasets
import iou_metric
import time

from mmengine.config import Config
from mmengine.runner import Runner

from utils import append_experiment_result
from mmseg.visualization import SegLocalVisualizer
from typing import Optional
import numpy as np
from mmseg.structures import SegDataSample
import mmcv
from mmseg.registry import VISUALIZERS

@VISUALIZERS.register_module()
class MySegVisualizer(SegLocalVisualizer):
    def add_datasample(
            self,
            name: str,
            image: np.ndarray,
            data_sample: Optional[SegDataSample] = None,
            draw_gt: bool = True,
            draw_pred: bool = True,
            show: bool = False,
            wait_time: float = 0,
            out_file: Optional[str] = None,
            step: int = 0,
            with_labels: Optional[bool] = False) -> None:
        """Rewrite
        """
        classes = self.dataset_meta.get('classes', None)
        palette = self.dataset_meta.get('palette', None)

        gt_img_data = None
        pred_img_data = None

        if draw_gt and data_sample is not None:
            if 'gt_sem_seg' in data_sample:
                assert classes is not None, 'class information is ' \
                                            'not provided when ' \
                                            'visualizing semantic ' \
                                            'segmentation results.'
                gt_img_data = self._draw_sem_seg(image, data_sample.gt_sem_seg,
                                                 classes, palette, with_labels)

            if 'gt_depth_map' in data_sample:
                gt_img_data = gt_img_data if gt_img_data is not None else image
                gt_img_data = self._draw_depth_map(gt_img_data,
                                                   data_sample.gt_depth_map)

        if draw_pred and data_sample is not None:

            if 'pred_sem_seg' in data_sample:

                assert classes is not None, 'class information is ' \
                                            'not provided when ' \
                                            'visualizing semantic ' \
                                            'segmentation results.'
                pred_img_data = self._draw_sem_seg(image,
                                                   data_sample.pred_sem_seg,
                                                   classes, palette,
                                                   with_labels)

            if 'pred_depth_map' in data_sample:
                pred_img_data = pred_img_data if pred_img_data is not None \
                    else image
                pred_img_data = self._draw_depth_map(
                    pred_img_data, data_sample.pred_depth_map)
        self.add_image(name + '_gt', gt_img_data, step)
        self.add_image(name + '_pred', pred_img_data, step)


def parse_args():
    parser = argparse.ArgumentParser(
        description='SegEarth-OV evaluation with MMSeg')
    parser.add_argument('--config', default='./configs/cfg_openearthmap.py')
    parser.add_argument('--work-dir', default='./work')
    parser.add_argument(
        '--show', action='store_true', help='show prediction results')
    parser.add_argument(
        '--visualization',
        type=bool,
        default=False,
        help='visualizaion or not')
    parser.add_argument(
        '--show-dir',
        default='./show_dir/',
        help='directory to save visualizaion images')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    # When using PyTorch version >= 2.0.0, the `torch.distributed.launch`
    # will pass the `--local-rank` parameter to `tools/train.py` instead
    # of `--local_rank`.
    parser.add_argument('--local_rank', '--local-rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args

def trigger_visualization_hook(cfg, args):
    default_hooks = cfg.default_hooks
    if 'visualization' in default_hooks:
        visualization_hook = default_hooks['visualization']
        # Turn on visualization
        visualization_hook['draw'] = True
        if args.show:
            visualization_hook['show'] = True
            visualization_hook['wait_time'] = args.wait_time
        if args.show_dir:
            visualizer = cfg.visualizer
            visualizer['save_dir'] = args.show_dir
    else:
        raise RuntimeError(
            'VisualizationHook must be included in default_hooks.'
            'refer to usage '
            '"visualization=dict(type=\'VisualizationHook\')"')

    return cfg

def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)
    cfg.launcher = args.launcher
    cfg.work_dir = args.work_dir

    # !mark! visualization
    if args.visualization:
        trigger_visualization_hook(cfg, args)

    start_time = time.strftime('%Y%m%d_%H%M%S', time.localtime())

    runner = Runner.from_cfg(cfg)
    results = runner.test()

    results.update({'VIT': cfg.model.vit_type,
                    'ID': start_time,
                    'MODEL': cfg.model.model_type,
                    'Dataset': cfg.dataset_type})

    if runner.rank == 0:
        append_experiment_result('results.xlsx', [results])

    if runner.rank == 0:
        with open(os.path.join(cfg.work_dir, 'results.txt'), 'a') as f:
            f.write(os.path.basename(args.config).split('.')[0] + '\n')
            for k, v in results.items():
                f.write(k + ': ' + str(v) + '\n')

if __name__ == '__main__':
    main()
