



python3 ./Classification/run_first_iter.py --task_type 'medium' --device 'cuda:1' --pretrained_weights 'simclr' --cold_start --extractor 'resnet18_simclr' --portion 0.5 --selection 'kmeans_centroid' --assign_initial './Classification/label_idx/resnet18_pretrained_initial.json' --trail_id 0
