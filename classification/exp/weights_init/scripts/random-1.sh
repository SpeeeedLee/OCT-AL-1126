AUGS="aug4" MAX_RUN=3 SEEDS="10 24 38 42 57" LRS="7e-6 1e-5 3e-5 5e-5 7e-5 1e-4 3e-4" PORTIONS="2.5 5 10" DEVICE="cuda:4" ./exp/meta_scripts/train_parallel.sh

AUGS="aug4" MAX_RUN=3 SEEDS="10 24 38 42 57" LRS="5e-5 7e-5 1e-4 3e-4 5e-4" PORTIONS="20 30 40" DEVICE="cuda:4" ./exp/meta_scripts/train_parallel.sh

AUGS="aug4" MAX_RUN=3 SEEDS="10 24 38 42 57" LRS="3e-4 5e-4 7e-4 1e-3 " PORTIONS="50 60 70" DEVICE="cuda:4" ./exp/meta_scripts/train_parallel.sh
