base_config = {
    "seed": 42,
    "eval_interval": 1,
    "ema_rate": 0.999,
    "csv_keys": [ "Prompt", "IoU", "Recall", "Precision", "F1", "epoch"],
    "opt": {
        "learning_rate": 1e-5,
        "weight_decay": 1e-4,#
        "decay_factor": 10,
        "steps": [3000, 8000],
        "warmup_steps": 250,
    },
    "corruptions": [
        "gaussian_noise",
        "shot_noise",
        "impulse_noise",
        "defocus_blur",
        "glass_blur",
        "motion_blur",
        "zoom_blur",
        "snow",
        "frost",
        "fog",
        "brightness",
        "contrast",
        "elastic_transform",
        "pixelate",
        "jpeg_compression",
    ],
    "model": {
        "type": "vit_b",
        "checkpoint": "./pretrain/",
        "ckpt": "",
        "freeze": {
            "image_encoder": True,
            "prompt_encoder": True,
            "mask_decoder": True,
        },
    },
    "datasets": {
        "NWPU": {
            "root_dir": "data/NWPU/Images",   
            "annotation_file_train": "data/NWPU/Annotations/NWPU_instances_train.json",
            "annotation_file_val": "data/NWPU/Annotations/NWPU_instances_val.json",
        },
        "WHU": {
            "root_dir": "data/WHU", 
            "annotation_file_train": "data/WHU/annotations/WHU_building_train.json",
            "annotation_file_val": "data/WHU/annotations/WHU_building_val.json",
        },
        "HRSID": {
            "root_dir": "data/HRSID/Images",
            "annotation_file_train": "data/HRSID/Annotations/inshore/inshore_train.json",
            "annotation_file_val": "data/HRSID/Annotations/inshore/inshore_test.json"
        },
    },
}
