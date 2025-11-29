# Classification

## 1126


## Full data training
```bash
## Random Weight Initialization
python3 ./Classification/run_first_iter.py --task_type 'medium' --device 'cuda:0' --portion 100 --batch_size 8  --no_use_pretrained
Acc = [0.718447]

python3 ./Classification/run_first_iter.py --task_type 'medium' --device 'cuda:0' --portion 100 --batch_size 8 --lr 1e-4 --no_use_pretrained 
Acc = [0.739806]

## ImageNet-pretrained
### Hard
python3 ./Classification/run_first_iter.py --task_type 'hard' --device 'cuda:2' --portion 100
Acc = [0.854369]

### Medium
python3 ./Classification/run_first_iter.py --task_type 'medium' --device 'cuda:2' --portion 100 --batch_size 8
Acc = [0.833010]



## SimCLR-OCT
### Hard
python3 ./Classification/run_first_iter.py --task_type 'hard' --device 'cuda:2' --portion 100 --pretrained_weights 'simclr' --simclr_path './SSL/simclr/resnet18_simclr_lr0.0002_bs128_ep100.pkl'
Acc = [0.908738]


### Medium
python3 ./Classification/run_first_iter.py --task_type 'medium' --device 'cuda:2' --portion 100 --batch_size 8  --pretrained_weights 'simclr' --simclr_path './SSL/simclr/resnet18_simclr_lr0.0002_bs128_ep100.pkl'
Acc = [0.902913]

python3 ./Classification/run_first_iter.py --task_type 'medium' --device 'cuda:2' --portion 100 --batch_size 8  --pretrained_weights 'simclr' --simclr_path './SSL/simclr/resnet18_simclr_lr0.0002_bs512_ep50.pkl'
Acc = [0.895146]

python3 ./Classification/run_first_iter.py --task_type 'medium' --device 'cuda:2' --portion 100 --batch_size 8  --pretrained_weights 'simclr' --simclr_path './SSL/simclr/resnet18_simclr_lr0.0002_bs512_ep100.pkl'
Acc = [0.900971]
```

## Partial data training

### 0.5%
```bash
## ImageNet 
python3 ./Classification/run_first_iter.py --task_type 'medium' --device 'cuda:2' --portion 0.5 --seed 24 --batch_size 2
Acc = [0.401942]

python3 ./Classification/run_first_iter.py --task_type 'medium' --device 'cuda:2' --portion 0.5 --seed 42 --batch_size 2
Acc = [0.353398]

## SimCLR-OCT
python3 ./Classification/run_first_iter.py --task_type 'medium' --device 'cuda:2' --portion 0.5 --seed 24 --batch_size 2  --pretrained_weights 'simclr' --simclr_path './SSL/simclr/resnet18_simclr_lr0.0002_bs128_ep100.pkl'
Acc = [0.473786]

python3 ./Classification/run_first_iter.py --task_type 'medium' --device 'cuda:2' --portion 0.5 --seed 42 --batch_size 2  --pretrained_weights 'simclr' --simclr_path './SSL/simclr/resnet18_simclr_lr0.0002_bs128_ep100.pkl'
Acc = [0.429126]

python3 ./Classification/run_first_iter.py --task_type 'hard' --device 'cuda:2' --portion 1.0 --seed 1 --pretrained_weights 'simclr' --simclr_path './SSL/simclr/resnet18_simclr_lr0.0002_bs128_ep100.pkl'


## sparK-OCT
python3 ./Classification/run_first_iter.py --task_type 'medium' --device 'cuda:2' --portion 0.5 --seed 24 --batch_size 2  --pretrained_weights 'sparK' --sparK_path './SSL/sparK/output/Epoch_100_Loss_0.563336887396872_resnet18_1kpretrained.pth'
Acc = [0.398058]

python3 ./Classification/run_first_iter.py --task_type 'medium' --device 'cuda:2' --portion 0.5 --seed 24 --batch_size 2  --pretrained_weights 'sparK' --sparK_path './SSL/sparK/output/Epoch_250_Loss_0.3579924898222089_resnet18_1kpretrained.pth'
Acc = [0.380583] 

python3 ./Classification/run_first_iter.py --task_type 'medium' --device 'cuda:2' --portion 0.5 --seed 24 --batch_size 2  --pretrained_weights 'sparK' --sparK_path './SSL/sparK/output/Epoch_400_Loss_0.32117805397138_resnet18_1kpretrained.pth'
Acc = [0.355340] 
```

### 5%
```bash
## ImageNet 
python3 ./Classification/run_first_iter.py --task_type 'medium' --device 'cuda:2' --portion 5 --seed 24 --batch_size 8
Acc = [0.574757]

## SimCLR-OCT
python3 ./Classification/run_first_iter.py --task_type 'medium' --device 'cuda:2' --portion 5 --seed 24 --batch_size 8  --pretrained_weights 'simclr' --simclr_path './SSL/simclr/resnet18_simclr_lr0.0002_bs128_ep100.pkl'
Acc = [0.642718]
```
