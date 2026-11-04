_base_ = './base_config.py'

# model settings
model = dict(
    name_path='./configs/cls_udd5.txt',
    prob_thd=0.4,
    bg_idx=4,
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
dataset_type = 'UDD5Dataset'
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
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='data/UDD5/val/src',
            seg_map_path='data/UDD5/val/gt'),
        pipeline=test_pipeline))