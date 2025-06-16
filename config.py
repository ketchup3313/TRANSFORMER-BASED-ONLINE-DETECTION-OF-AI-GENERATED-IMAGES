# config.py
def get_config():
    class Config:
        data_dir = "./data"
        output_dir = "./checkpoints"
        model_name = "aam_swin"
        batch_size = 16
        num_workers = 4
        lr = 1e-4
        weight_decay = 0.05
        phase1_epochs = 5
        phase2_epochs = 5
        phase3_epochs = 5
        save_interval = 2
        log_interval = 10
        seed = 42
        gpu = 0
        use_wandb = False
        wandb_project = "ai-image-detection"
        pretrained = True
    return Config()
