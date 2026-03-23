# Classification

```bash
conda activate oct-env 
cd classification
```


## Number of Images
Train: 2032 (75%)
Val: 254 (12.5%)
Test: 255 (12.5%)


## Check Random Seeds
+ Training data batch shuffle
+ Pytorch random seed
--> I'll just do not fix both these two. It's more simple!



## Full data training
```bash
## Random Weight Initialization
python3 ./run_first_iter.py --task_type 'hard' --device 'cuda:0' --portion 100 --pretrained_weights 'random' --seed 42
# Acc = []

## SimCLR Weight Initialization
python3 ./run_first_iter.py --task_type 'hard' --device 'cuda:2' --portion 100 --pretrained_weights 'simclr' --simclr_path './SSL/simclr/resnet18_simclr_lr0.0004_bs256_ep300.pkl' --seed 42
```

## Data augmentation
+ Fix data selection seed = 42
+ LR \in {7e-5 1e-4 3e-4 5e-4 7e-4}

```bash
chmod +x ./exp/data_aug/scripts/final_1.sh
./exp/data_aug/scripts/final_1.sh

cd ./exp/data_aug
python3 ./plot_all.py 
```





