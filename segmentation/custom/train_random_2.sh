seed=42
device='cuda:1'
for portion in 50 55 60 65 70 75 80 100; do
    for run in {1..5}; do
        python3 ./segmentation/run_first_iter.py \
            --dataroot ./ds/segmentation \
            --phase train \
            --portion ${portion}.0 \
            --seed ${seed} \
            --device ${device}
    done
done