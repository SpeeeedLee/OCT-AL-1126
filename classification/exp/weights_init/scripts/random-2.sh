AUGS="aug4" MAX_RUN=4 SEEDS="10 24 38 42 57" LRS="5e-4 7e-4 1e-3 3e-3" PORTIONS="80 90" DEVICE="cuda:2" ./exp/meta_scripts/train_parallel.sh

AUGS="aug4" MAX_RUN=4 SEEDS="42" LRS="7e-4 1e-3 3e-3" PORTIONS="100" DEVICE="cuda:2" ./exp/meta_scripts/train_parallel.sh