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


## 