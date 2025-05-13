_base_ = '../grounding_dino_swin-t_finetune_16xb2_1x_coco.py'

data_root = 'data/MarchDataset_MD_only/'
class_name = (
    "membrane_discoloration"
)
num_classes = len(class_name)
COLORS = [
    (0, 0, 255)        # Bright Red (General Deterioration)
]
metainfo = dict(classes=class_name, palette=COLORS)

model = dict(bbox_head=dict(num_classes=num_classes))

train_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file='annotations/trainval.json',
        data_prefix=dict(img='images/')))

val_dataloader = dict(
    dataset=dict(
        metainfo=metainfo,
        data_root=data_root,
        ann_file=f'annotations/test.json',
        data_prefix=dict(img='images/')))

test_dataloader = val_dataloader

val_evaluator = dict(ann_file=data_root + 'annotations/test.json')
test_evaluator = val_evaluator

max_epoch = 100

default_hooks = dict(
    checkpoint=dict(interval=1, max_keep_ckpts=1, save_best='auto'),
    logger=dict(type='LoggerHook', interval=5))
train_cfg = dict(max_epochs=max_epoch, val_interval=1)

# param_scheduler = [
#     dict(type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=30),
#     dict(
#         type='MultiStepLR',
#         begin=0,
#         end=max_epoch,
#         by_epoch=True,
#         milestones=[50],
#         gamma=0.1)
# ]

# https://mmengine.readthedocs.io/en/latest/tutorials/param_scheduler.html
param_scheduler = [
    dict(
        type='CosineAnnealingLR', 
        by_epoch=True, 
        T_max=max_epoch, 
        eta_min=1e-6, 
        # convert_to_iter_based=True
    )
]


optim_wrapper = dict(
    optimizer=dict(lr=1e-5),
    paramwise_cfg=dict(
        custom_keys={
            'absolute_pos_embed': dict(decay_mult=0.),
            'backbone': dict(lr_mult=0.1),
            'language_model': dict(lr_mult=0),
        }))

auto_scale_lr = dict(base_batch_size=16)

## Weights and Biases Setup
vis_backends = [
    dict(
        type='WandbVisBackend',
        init_kwargs={
            'project': 'vis_anom_grounding_dino',
            'entity': 'ai4ce'
        }
    )
]
visualizer = dict(
    type='DetLocalVisualizer',
    vis_backends=vis_backends,
    name='visualizer'
)


log_config = dict(
    interval=10,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(
            type='WandbLoggerHook',
            init_kwargs={
                'project': 'vis_anom_grounding_dino',
                'entity': 'ai4ce'
            }
        )
    ]
)

### Visualization
visualization = dict(
    type='DetVisualizationHook',
    draw=True,         # Enable drawing of predictions
    interval=1,        # Frequency of visualization (e.g., every epoch)
    show=False         # Set to False to log images to W&B instead of displaying them
)