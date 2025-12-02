## Start from ImageNet pretrained ckpt
# python3 ./SSL/simclr/run.py --arch resnet18 --epochs 100 --batch-size 16 --gpu-index 0 -data ./ds/classification/seven_class/train
# python3 ./SSL/simclr/run.py --arch resnet18 --epochs 100 --batch-size 128 --gpu-index 0 -data ./ds/classification/seven_class/train
# python3 ./SSL/simclr/run.py --arch resnet18 --epochs 100 --batch-size 256 --gpu-index 0 -data ./ds/classification/seven_class/train
# python3 ./SSL/simclr/run.py --arch resnet18 --epochs 100 --batch-size 512 --gpu-index 0 -data ./ds/classification/seven_class/train

# python3 ./SSL/simclr/run.py --arch resnet18 --epochs 300 --batch-size 16 --gpu-index 0 -data ./ds/classification/seven_class/train
# python3 ./SSL/simclr/run.py --arch resnet18 --epochs 300 --batch-size 128 --gpu-index 0 -data ./ds/classification/seven_class/train
# python3 ./SSL/simclr/run.py --arch resnet18 --epochs 300 --batch-size 256 --gpu-index 0 -data ./ds/classification/seven_class/train
# python3 ./SSL/simclr/run.py --arch resnet18 --epochs 300 --batch-size 512 --gpu-index 0 -data ./ds/classification/seven_class/train


# python3 ./SSL/simclr/run.py --arch resnet18 --epochs 500 --batch-size 16 --gpu-index 0 -data ./ds/classification/seven_class/train
# python3 ./SSL/simclr/run.py --arch resnet18 --epochs 500 --batch-size 128 --gpu-index 0 -data ./ds/classification/seven_class/train
# python3 ./SSL/simclr/run.py --arch resnet18 --epochs 500 --batch-size 256 --gpu-index 0 -data ./ds/classification/seven_class/train
# python3 ./SSL/simclr/run.py --arch resnet18 --epochs 500 --batch-size 512 --gpu-index 0 -data ./ds/classification/seven_class/train


## 12/1
#### 重新跑一次這個，因為他的數字最奇怪 --> 確實修正成功!
# python3 ./SSL/simclr/run.py --arch resnet18 --epochs 100 --batch-size 128 --gpu-index 0 -data ./ds/classification/seven_class/train


## 12/1, 修正正確的consine lr decay
# python3 ./SSL/simclr/run.py --lr 1e-4 --arch resnet18 --epochs 100 --batch-size 128 --gpu-index 1 -data ./ds/classification/seven_class/train
# python3 ./SSL/simclr/run.py --lr 2e-4 --arch resnet18 --epochs 100 --batch-size 128 --gpu-index 2 -data ./ds/classification/seven_class/train
# python3 ./SSL/simclr/run.py --lr 4e-4 --arch resnet18 --epochs 100 --batch-size 128 --gpu-index 3 -data ./ds/classification/seven_class/train
# python3 ./SSL/simclr/run.py --lr 5e-5 --arch resnet18 --epochs 100 --batch-size 128 --gpu-index 4 -data ./ds/classification/seven_class/train


## 12/1 使用lineary scaling到其他batch size
# python3 ./SSL/simclr/run.py --lr 1e-4 --arch resnet18 --epochs 100 --batch-size 64 --gpu-index 3 -data ./ds/classification/seven_class/train
# python3 ./SSL/simclr/run.py --lr 4e-4 --arch resnet18 --epochs 100 --batch-size 256 --gpu-index 1 -data ./ds/classification/seven_class/train
# python3 ./SSL/simclr/run.py --lr 8e-4 --arch resnet18 --epochs 100 --batch-size 512 --gpu-index 2 -data ./ds/classification/seven_class/train


## 12/1 增加epochs!
python3 ./SSL/simclr/run.py --lr 1e-4 --arch resnet18 --epochs 300 --batch-size 64 --gpu-index 1 -data ./ds/classification/seven_class/train
python3 ./SSL/simclr/run.py --lr 2e-4 --arch resnet18 --epochs 300 --batch-size 128 --gpu-index 2 -data ./ds/classification/seven_class/train
python3 ./SSL/simclr/run.py --lr 4e-4 --arch resnet18 --epochs 300 --batch-size 256 --gpu-index 3 -data ./ds/classification/seven_class/train
python3 ./SSL/simclr/run.py --lr 8e-4 --arch resnet18 --epochs 300 --batch-size 512 --gpu-index 4 -data ./ds/classification/seven_class/train


python3 ./SSL/simclr/run.py --lr 1e-4 --arch resnet18 --epochs 500 --batch-size 64 --gpu-index 1 -data ./ds/classification/seven_class/train
python3 ./SSL/simclr/run.py --lr 2e-4 --arch resnet18 --epochs 500 --batch-size 128 --gpu-index 2 -data ./ds/classification/seven_class/train
python3 ./SSL/simclr/run.py --lr 4e-4 --arch resnet18 --epochs 500 --batch-size 256 --gpu-index 3 -data ./ds/classification/seven_class/train
python3 ./SSL/simclr/run.py --lr 8e-4 --arch resnet18 --epochs 500 --batch-size 512 --gpu-index 4 -data ./ds/classification/seven_class/train
