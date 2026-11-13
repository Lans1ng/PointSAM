from box import Box
from configs.base_config import base_config

config = {
    "dataset": "HRSID", 
    "load_type": "soft",
    "num_points": 1,
    
    "batch_size": 1, #only support 1
    "val_batchsize": 1,
    "num_workers": 0,
    "num_epochs": 10,
    "max_nums": 50,
    "resume": False,

    "start_lora_layer": 1,
    "lora_rank": 4,
    "mem_bank_max_len": 512,
    "match_interval": 30,
    "match_partition": "closest",
    "iou_thr": 0.1,

    "prompt": "point",
    "out_dir": "",
    "name": "base",
    "corrupt": None,
    "visual": False,
    "model": {
        "type": "vit_b", 
        },
    "opt": {
        "learning_rate": 5e-4,
        "weight_decay": 1e-4,
        "decay_factor": 10,
        "steps": [1500, 2000],
        "warmup_steps": 250,
    },
}

cfg = Box(base_config)
cfg.merge_update(config)
