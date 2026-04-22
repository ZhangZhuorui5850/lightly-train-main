import lightly_train

if __name__ == "__main__":
    lightly_train.train_instance_segmentation(
        num_workers=0,
        out="out/my_experiment_seg",
        overwrite=True,
        model="weights/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth",
        data="datasets/dateset_seg/data.yaml",
        steps=20, 
        batch_size=2,
        save_checkpoint_args={
            "save_best": True,
            "save_every_num_steps": 10,
            "save_last": True,
        },
    )   
