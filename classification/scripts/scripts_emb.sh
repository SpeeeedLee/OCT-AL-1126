python3 ./Classification/run_first_iter.py --task_type 'medium' --device 'cuda:0' --no_use_pretrained  --portion 0.5 --cold_start --extractor 'resnet18_simclr' --selection 'kmeans_centroid' --batch_size 4  
python3 ./Classification/run_first_iter.py --task_type 'medium' --device 'cuda:0' --no_use_pretrained  --portion 0.5 --cold_start --extractor 'resnet18_pretrained' --selection 'kmeans_centroid' --batch_size 4





python3 ./Classification/run_first_iter.py --task_type 'medium' --device 'cuda:0' --no_use_pretrained  --portion 1.0 --cold_start --extractor 'resnet18_simclr' --selection 'kmeans_centroid' --batch_size 8

python3 ./Classification/run_first_iter.py --task_type 'medium' --device 'cuda:0' --no_use_pretrained  --portion 2.0 --cold_start --extractor 'resnet18_simclr' --selection 'kmeans_centroid' --batch_size 4

