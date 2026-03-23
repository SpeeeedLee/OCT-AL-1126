seed=42
device='cuda:0'
for portion in 5 10 15 20 25 30 35 40 45; do
    for run in {1..5}; do
        python3 ./segmentation/run_first_iter.py \
            --dataroot ./ds/segmentation \
            --phase train \
            --portion ${portion}.0 \
            --seed ${seed} \
            --device ${device}
    done
done