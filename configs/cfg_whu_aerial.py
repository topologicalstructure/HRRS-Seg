_base_ = './base_config.py'

# model settings
model = dict(
    name_path='./configs/cls_whu.txt',
    prob_thd=0.65,
    scales_fusion=True,
    sam_parameters=[32, 0.2, 0.5],
    fusion_parameters=[0.125, 0.875, 1, 0.25, 0.875],
    geometric_correction=True,
    scale_normalize=False,
    aggregation_method='vote',
    sub_confi_fusion=False,
    experience_scale=True,
)

# dataset settings
dataset_type = 'WHUDataset'
data_root = ''

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(896, 896), keep_ratio=True),
    # add loading annotation after ``Resize`` because ground truth
    # does not need to do resize data transform
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs')
]

test_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='data/WHU-BD/val/image',
            seg_map_path='data/WHU-BD/val/label_cvt'),
        pipeline=test_pipeline))