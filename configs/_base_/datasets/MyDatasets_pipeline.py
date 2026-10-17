# 数据处理 pipeline
# 简化版本 - 无裁剪和数据增强

# 数据集路径
dataset_type = 'MyDataset'  # 数据集类名
data_root = './data'  # 数据集路径

# 训练预处理 - 仅加载图像和标注
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs')
]

# 测试预处理 - 仅加载图像和标注
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs')
]

# TTA后处理 - 如果不需要TTA可以删除
tta_pipeline = [
    dict(type='LoadImageFromFile', file_client_args=dict(backend='disk')),
    dict(
        type='TestTimeAug',
        transforms=[
            [dict(type='Resize', scale_factor=1.0, keep_ratio=True)],
            [dict(type='RandomFlip', prob=0., direction='horizontal')],
            [dict(type='LoadAnnotations')],
            [dict(type='PackSegInputs')]
        ])
]

# 训练 Dataloader
train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='img_dir/train', seg_map_path='ann_dir/train'),
        pipeline=train_pipeline))

# 验证 Dataloader
val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='img_dir/val', seg_map_path='ann_dir/val'),
        pipeline=test_pipeline))

# 测试 Dataloader - 使用独立的测试集路径
test_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='img_dir/test',      # 对应 ./data/img_dir/text
            seg_map_path='ann_dir/test'   # 对应 ./data/ann_dir/text
        ),
        pipeline=test_pipeline
    )
)

# 验证 Evaluator
val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU', 'mDice', 'mFscore'])

# 测试 Evaluator
test_evaluator = val_evaluator
