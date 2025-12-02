#!/bin/bash

# 第一批：並行執行（300 epochs）
# python3 ./SSL/simclr/run.py --lr 1e-4 --arch resnet18 --epochs 300 --batch-size 64 --gpu-index 1 -data ./ds/classification/seven_class/train &
# python3 ./SSL/simclr/run.py --lr 2e-4 --arch resnet18 --epochs 300 --batch-size 128 --gpu-index 2 -data ./ds/classification/seven_class/train &
# python3 ./SSL/simclr/run.py --lr 4e-4 --arch resnet18 --epochs 300 --batch-size 256 --gpu-index 3 -data ./ds/classification/seven_class/train &
# python3 ./SSL/simclr/run.py --lr 8e-4 --arch resnet18 --epochs 300 --batch-size 512 --gpu-index 4 -data ./ds/classification/seven_class/train &

# wait  # 等待所有背景進程完成

# echo "第一批訓練完成，開始第二批..."

# # 第二批：並行執行（500 epochs）
# python3 ./SSL/simclr/run.py --lr 1e-4 --arch resnet18 --epochs 500 --batch-size 64 --gpu-index 1 -data ./ds/classification/seven_class/train &
# python3 ./SSL/simclr/run.py --lr 2e-4 --arch resnet18 --epochs 500 --batch-size 128 --gpu-index 2 -data ./ds/classification/seven_class/train &
# python3 ./SSL/simclr/run.py --lr 4e-4 --arch resnet18 --epochs 500 --batch-size 256 --gpu-index 3 -data ./ds/classification/seven_class/train &
# python3 ./SSL/simclr/run.py --lr 8e-4 --arch resnet18 --epochs 500 --batch-size 512 --gpu-index 4 -data ./ds/classification/seven_class/train &

# wait  # 等待第二批完成

# # 第三批：並行執行（50 epochs）
# python3 ./SSL/simclr/run.py --lr 1e-4 --arch resnet18 --epochs 50 --batch-size 64 --gpu-index 1 -data ./ds/classification/seven_class/train &
# python3 ./SSL/simclr/run.py --lr 2e-4 --arch resnet18 --epochs 50 --batch-size 128 --gpu-index 2 -data ./ds/classification/seven_class/train &
# python3 ./SSL/simclr/run.py --lr 4e-4 --arch resnet18 --epochs 50 --batch-size 256 --gpu-index 3 -data ./ds/classification/seven_class/train &
# python3 ./SSL/simclr/run.py --lr 8e-4 --arch resnet18 --epochs 50 --batch-size 512 --gpu-index 4 -data ./ds/classification/seven_class/train &

# wait  # 等待第二批完成

# # 第四批：並行執行（25 epochs）
# python3 ./SSL/simclr/run.py --lr 1e-4 --arch resnet18 --epochs 25 --batch-size 64 --gpu-index 1 -data ./ds/classification/seven_class/train &
# python3 ./SSL/simclr/run.py --lr 2e-4 --arch resnet18 --epochs 25 --batch-size 128 --gpu-index 2 -data ./ds/classification/seven_class/train &
# python3 ./SSL/simclr/run.py --lr 4e-4 --arch resnet18 --epochs 25 --batch-size 256 --gpu-index 3 -data ./ds/classification/seven_class/train &
# python3 ./SSL/simclr/run.py --lr 8e-4 --arch resnet18 --epochs 25 --batch-size 512 --gpu-index 4 -data ./ds/classification/seven_class/train &
# wait  # 等待第二批完成


# # 第四批：並行執行（25 epochs）
# python3 ./SSL/simclr/run.py --lr 5e-5 --arch resnet18 --epochs 25 --batch-size 32 --gpu-index 1 -data ./ds/classification/seven_class/train &
# python3 ./SSL/simclr/run.py --lr 5e-5 --arch resnet18 --epochs 50 --batch-size 32 --gpu-index 2 -data ./ds/classification/seven_class/train &
# python3 ./SSL/simclr/run.py --lr 5e-5 --arch resnet18 --epochs 100 --batch-size 32 --gpu-index 3 -data ./ds/classification/seven_class/train &

# wait  # 等待第二批完成

# 第四批：並行執行（10 epochs）

python3 ./SSL/simclr/run.py --lr 5e-5 --arch resnet18 --epochs 10 --batch-size 32 --gpu-index 3 -data ./ds/classification/seven_class/train &
python3 ./SSL/simclr/run.py --lr 1e-4 --arch resnet18 --epochs 10 --batch-size 64 --gpu-index 2 -data ./ds/classification/seven_class/train &
python3 ./SSL/simclr/run.py --lr 2e-4 --arch resnet18 --epochs 10 --batch-size 128 --gpu-index 1 -data ./ds/classification/seven_class/train &
python3 ./SSL/simclr/run.py --lr 4e-4 --arch resnet18 --epochs 10 --batch-size 256 --gpu-index 0 -data ./ds/classification/seven_class/train &

wait  # 等待第二批完成

echo "所有訓練完成！"