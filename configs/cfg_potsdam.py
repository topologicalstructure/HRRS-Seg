_base_ = './base_config.py'

# model settings
model = dict(
    name_path='./configs/cls_potsdam.txt',
    prob_thd=0.1,
    bg_idx=5,
    scales_fusion=True,
    sam_parameters=[16, 0.2, 0.5],
    fusion_parameters=[0.0, 0.75, 1.0, 0.25, 0.5],
    geometric_correction=True,
    scale_normalize=False,
    aggregation_method='vote',
    sub_confi_fusion=False,
    experience_scale=True,
)

# dataset settings
dataset_type = 'PotsdamDataset'
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
            img_path='data/potsdam/img_dir/val',
            seg_map_path='data/potsdam/ann_dir/val'),
        pipeline=test_pipeline))