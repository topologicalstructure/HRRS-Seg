_base_ = './base_config.py'

# model settings
model = dict(
    name_path='./configs/cls_vaihingen_without_clutter.txt',
    prob_thd=0,
    bg_idx=5,
    scales_fusion=True,
    sam_parameters=[32, 0.5, 0.8],
    fusion_parameters=[0.125, 0.875, 1, 0.25, 0.875],
    geometric_correction=True,
    scale_normalize=False,
    aggregation_method='vote',
    sub_confi_fusion=False,
    experience_scale=True,
)

# dataset settings
dataset_type = 'ISPRSDataset'
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
            img_path='data/vaihingen/img_dir/val',
            seg_map_path='data/vaihingen/ann_dir_processed/val'),
        pipeline=test_pipeline))