import lightly_train


def main():

    lightly_train.train_object_detection(
        out="out/0414/NEU_train",
        model="dinov3/vits16-ltdetr",
        model_args={
            "backbone_weights": "weights/dinov3_vits16_pretrain_lvd1689m-08c60483.pth",
        },
        data="datasets/NEU_dataset/dataset_det/data.yaml",
        overwrite=True,
        steps=200,
        batch_size=4,
        num_workers="auto",
    )


if __name__ == "__main__":
    main()