import lightly_train
'''
windows:(cmd)
set LIGHTLY_TRAIN_MODEL_CACHE_DIR=D:\VScodeProjects\lightly-train-main\lightly-train-main\models
python train.py

linux：
cd /path/to/lightly-train-main
# 1) 指定本地模型缓存目录（你提前把权重放这里）
export LIGHTLY_TRAIN_MODEL_CACHE_DIR=./lightly-train-main/models
# 2) 开始训练
python train.py
'''

if __name__ == "__main__":
    # Train an image classification model with a DINOv3 backbone
    lightly_train.train_image_classification(
        out="out/my_experiment_cls",
        model="dinov3/vitt16",
        overwrite=True,
        data={
            "train": "datasets/pet_split_250/images/train",
            "val": "datasets/pet_split_250/images/val",
            "classes": {
                0: "Cat",
                1: "Dog",
            },
        },
        steps=20, 
        batch_size=2,
    )
