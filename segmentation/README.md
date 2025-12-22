# Active Learning Segementation

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

## Segmentation

### run 100%
智皓在paper report的是71.25 (Dice of Cell Nuclei)
```bash
# python3 ./segmentation/run_first_iter.py --dataroot ./ds/segmentation --phase train --portion 5 --seed 42 --device 'cuda:0'

chmod +x ./segmentation/custom/train_random.sh
./segmentation/custom/train_random.sh
```