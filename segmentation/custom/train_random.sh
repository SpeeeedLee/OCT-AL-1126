for portion in 5 10 15 20 25 30 35 40 45 50 100; do
    for seed in 1 2 3 4 5; do
        python3 ./segmentation/run_first_iter.py \
            --dataroot ./ds/segmentation \
            --phase train \
            --fold 0 \
            --portion ${portion}.0 \
            --seed ${seed} \
            --lr 0.001 \
            --epoch 25 \
            --device 'cuda:0'
    done
done