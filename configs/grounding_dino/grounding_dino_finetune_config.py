_base_ = 'grounding_dino_swin-t_finetune_16xb2_1x_coco.py'

data_root = 'data/MarchDataset_organized/'
class_name = (
    "membrane_discoloration",
    "membrane_shrinkage",
    "membrane_blistering",
    "ponding_water",
    "seam_bonding",
    "uneven_seam_integrity",
    "membrane_tear",
    "poor_flashing",
    "roof_penetration",
    "spalling",
    "water_intrusion",
    "general_deterioration",
    "debris",
    "peeling_paint",
    "crack",
    "Alligatoring",
    "plants",
    "gravel_discoloration",
    "Rust",
)
num_classes = len(class_name)
COLORS = [
    (0, 0, 255),        # Bright Red (General Deterioration)
    (0, 128, 128),      # Changed to a more Teal Green (Debris)
    (255, 100, 0),      # Brighter Blue (Peeling Paint)
    (0, 165, 255),      # Orange (Crack)
    (0, 206, 209),      # turquoise (Membrane Tear)
    (128, 128, 0),      # Changed to Cyan (Poor Flashing)
    (0, 255, 255),      # Bright Yellow (Above Membrane Moisture)
    (0, 0, 139),        # Dark Red (Membrane Discoloration)
    (50, 255, 0),       # Brighter Lime Green (Below Membrane Moisture)
    (205, 0, 0),        # Deep Blue (Membrane Shrinkage)
    (128, 0, 0),        # Changed to Maroon (Membrane Blistering)
    (255, 0, 255),      # Vivid Purple (Spalling)
    (80, 127, 255),     # Changed to Silver (Seam Debonding)
    (147, 20, 255),     # Deep Pink (Poor Seam Integrity)
    (124, 252, 0),      # Lawn Green (Water Intrusion)
    (32, 215, 255),     # Gold (Alligatoring)
    (0, 0, 0),          # Black (Unknown)
    (255, 0, 0),        # Bright Red (Skip Prediction)
    (238, 130, 238),    # Medium Violet
    (85, 107, 47)       # Dark Olive Green
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

param_scheduler = [
    dict(type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=30),
    dict(
        type='MultiStepLR',
        begin=0,
        end=max_epoch,
        by_epoch=True,
        milestones=[50],
        gamma=0.1)
]

optim_wrapper = dict(
    optimizer=dict(lr=0.00005),
    paramwise_cfg=dict(
        custom_keys={
            'absolute_pos_embed': dict(decay_mult=0.),
            'backbone': dict(lr_mult=0.1),
            'language_model': dict(lr_mult=0),
        }))

auto_scale_lr = dict(base_batch_size=16)
