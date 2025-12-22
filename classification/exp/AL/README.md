

## AL Strategies

+ For now, let's only use pretrained weights = simclr
+ For now, let's only use fix lr = 1e-4 for each data portion!
+ 從5跑到60, interval是2.5

## Uncertainty-based approaches

```bash
# Margin
./classification/exp/AL/scripts/margin.sh
python3 ./classification/run_first_iter.py \
    --task_type 'hard' \
    --lr 1e-4 \
    --pretrained_weights simclr \
    --simclr_path ./classification/model/simclr/ckpt_w_vertical_aug/resnet18_simclr_lr0.0004_bs256_ep300.pkl \
    --device $device \
    --portion_start 5 \
    --portion_end 62.5 \
    --portion_interval 2.5 \
    --seed 42 \
    --aug_factor 4

# Conf
./classification/exp/AL/scripts/conf.sh

# Entropy
./classification/exp/AL/scripts/entropy.sh

```

## Diversity-based approaches
```bash
# coreset
./classification/exp/AL/scripts/coreset.sh

```

## Hybrid approaches
```bash
# Badge
./classification/exp/AL/scripts/badge.sh

```
