# Skin OCT Active Learning

## Env Setup
```bash
# Create conda env
conda create -n oct-AL-env python=3.10.12

# Install torch
conda install pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 pytorch-cuda=12.4 -c pytorch -c nvidia

# Install other packages
pip3 install -r requirements.txt
pip3 install timm, einops, tensorboard
```

```bash
conda activate oct-env
```

## Self-supervised Pretraining

### SimCLR Pretraining

#### Note
+ We should apply linear scaling for different learning rate, as what is done by the SimCLR paper!
        + Before that, we should first optimize the learning rate... 
        + 直接用智皓的吧 --> batch size 是 128, epochs是100的時候使用2e-4
        + 但是我想先確認lr 2e-4在這個情況確定是最好的!


```bash
chmod +x ./SSL/simclr/custom/train_1201.sh
./SSL/simclr/custom/train_1201.sh
```
p.s. the default batch size is `128`


+ The model will be saved to 
```bash
./SSL/simclr/{self.args.arch}_simclr_lr{self.args.lr}_bs_{self.args.batch_size}_ep{self.args.epochs}.pkl
```
+ You can visualized training loss and acc via Tensorboard:
```bash
tensorboard --logdir=./SSL/simclr/tb_logs
```


## Classification


## About Data Augmentation
See `./classification/figs/data_aug/README.md`

## About Weights Initialization
See `./classificatin/weigths_init/README.md`

### Active Learning Selection
#### Uncertainty -- Confidence



#### Uncertainty -- Margin

#### Uncertainty -- Entrioy


#### Diversity -- Coreset

#### Hybrid -- Badge

