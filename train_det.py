import lightly_train


def main():

    lightly_train.train_object_detection(
        out="out/0423/0423-jsmb_train",
        model="dinov3/vits16-ltdetr",
        model_args={
            "backbone_weights": "weights/dinov3_vits16_pretrain_lvd1689m-08c60483.pth",
            "backbone_freeze": True,
            "lr": 5e-5,

        },
        data="datasets/jsmb_dataset/dataset_det_A/data.yaml",
        overwrite=True,
        steps=26000,
        batch_size=96,
        num_workers="auto",
    )


if __name__ == "__main__":
    main()