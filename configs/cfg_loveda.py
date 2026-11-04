_base_ = './base_config.py'

# model settings
model = dict(
    name_path='./configs/cls_loveda.txt',
    prob_thd=0.3,
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
# Category labels: background – 1, building – 2, road – 3, water – 4, barren – 5,forest – 6, agriculture – 7. 
# And the no-data regions were assigned 0 which should be ignored. 
dataset_type = 'LoveDADataset'
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
        reduce_zero_label=True,
        data_prefix=dict(
            img_path='data/LoveDA/img_dir/val',
            seg_map_path='data/LoveDA/ann_dir/val'),
        pipeline=test_pipeline))