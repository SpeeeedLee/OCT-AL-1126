# Use None
python3 ./Classification/run_first_iter.py --task_type 'medium' --device 'cuda:0' --no_use_pretrained  --portion 0.5 --seed 10 --batch_size 4  
python3 ./Classification/run_first_iter.py --task_type 'medium' --device 'cuda:0' --no_use_pretrained  --portion 0.5 --seed 24 --batch_size 4  
python3 ./Classification/run_first_iter.py --task_type 'medium' --device 'cuda:0' --no_use_pretrained  --portion 0.5 --seed 42 --batch_size 4  
python3 ./Classification/run_first_iter.py --task_type 'medium' --device 'cuda:0' --no_use_pretrained  --portion 0.5 --seed 92 --batch_size 4  
python3 ./Classification/run_first_iter.py --task_type 'medium' --device 'cuda:0' --no_use_pretrained  --portion 0.5 --seed 95 --batch_size 4  


# Use all three strategies
# python3 ./Classification/run_first_iter.py --task_type 'medium' --device 'cuda:0' --pretrained_weights 'simclr'  --portion 0.5 --cold_start --extractor 'resnet18_simclr' --selection 'kmeans_centroid' --batch_size 2  
python3 ./Classification/run_first_iter.py --task_type 'medium' --device 'cuda:0' --pretrained_weights 'simclr'  --portion 2.0 --cold_start --extractor 'resnet18_simclr' --selection 'kmeans_centroid' --batch_size 2  
python3 ./Classification/run_first_iter.py --task_type 'medium' --device 'cuda:0' --pretrained_weights 'simclr'  --portion 4.0 --cold_start --extractor 'resnet18_simclr' --selection 'kmeans_centroid' --batch_size 2  
python3 ./Classification/run_first_iter.py --task_type 'medium' --device 'cuda:0' --pretrained_weights 'simclr'  --portion 10.0 --cold_start --extractor 'resnet18_simclr' --selection 'kmeans_centroid' --batch_size 2 
python3 ./Classification/run_first_iter.py --task_type 'medium' --device 'cuda:0' --pretrained_weights 'simclr'  --portion 20.0 --cold_start --extractor 'resnet18_simclr' --selection 'kmeans_centroid' --batch_size 2 

# 1. No SimCLR Pretrained (i.e., use random initialized)
python3 ./Classification/run_first_iter.py --task_type 'medium' --device 'cuda:0' --no_use_pretrained  --portion 0.5 --cold_start --extractor 'resnet18_simclr' --selection 'kmeans_centroid' --batch_size 2  
python3 ./Classification/run_first_iter.py --task_type 'medium' --device 'cuda:0' --no_use_pretrained  --portion 2.0 --cold_start --extractor 'resnet18_simclr' --selection 'kmeans_centroid' --batch_size 2  
python3 ./Classification/run_first_iter.py --task_type 'medium' --device 'cuda:0' --no_use_pretrained  --portion 4.0 --cold_start --extractor 'resnet18_simclr' --selection 'kmeans_centroid' --batch_size 2  
python3 ./Classification/run_first_iter.py --task_type 'medium' --device 'cuda:0' --no_use_pretrained  --portion 10.0 --cold_start --extractor 'resnet18_simclr' --selection 'kmeans_centroid' --batch_size 2  
python3 ./Classification/run_first_iter.py --task_type 'medium' --device 'cuda:0' --no_use_pretrained  --portion 20.0 --cold_start --extractor 'resnet18_simclr' --selection 'kmeans_centroid' --batch_size 2

# 2. No SimCLR Data Selection (i.e., random selection)
# python3 ./Classification/run_first_iter.py --task_type 'medium' --device 'cuda:0' --pretrained_weights 'simclr'  --portion 0.5 --seed 10 --batch_size 2
# python3 ./Classification/run_first_iter.py --task_type 'medium' --device 'cuda:0' --pretrained_weights 'simclr'  --portion 0.5 --seed 24 --batch_size 2
# python3 ./Classification/run_first_iter.py --task_type 'medium' --device 'cuda:0' --pretrained_weights 'simclr'  --portion 0.5 --seed 42 --batch_size 2
# python3 ./Classification/run_first_iter.py --task_type 'medium' --device 'cuda:0' --pretrained_weights 'simclr'  --portion 0.5 --seed 92 --batch_size 2
# python3 ./Classification/run_first_iter.py --task_type 'medium' --device 'cuda:0' --pretrained_weights 'simclr'  --portion 0.5 --seed 95 --batch_size 2 

python3 ./Classification/run_first_iter.py --task_type 'medium' --device 'cuda:0' --pretrained_weights 'simclr'  --portion 2.0 --seed 10 --batch_size 2
python3 ./Classification/run_first_iter.py --task_type 'medium' --device 'cuda:0' --pretrained_weights 'simclr'  --portion 2.0 --seed 24 --batch_size 2
python3 ./Classification/run_first_iter.py --task_type 'medium' --device 'cuda:0' --pretrained_weights 'simclr'  --portion 2.0 --seed 42 --batch_size 2
python3 ./Classification/run_first_iter.py --task_type 'medium' --device 'cuda:0' --pretrained_weights 'simclr'  --portion 2.0 --seed 92 --batch_size 2
python3 ./Classification/run_first_iter.py --task_type 'medium' --device 'cuda:0' --pretrained_weights 'simclr'  --portion 2.0 --seed 95 --batch_size 2 

python3 ./Classification/run_first_iter.py --task_type 'medium' --device 'cuda:0' --pretrained_weights 'simclr'  --portion 4.0 --seed 10 --batch_size 2
python3 ./Classification/run_first_iter.py --task_type 'medium' --device 'cuda:0' --pretrained_weights 'simclr'  --portion 4.0 --seed 24 --batch_size 2
python3 ./Classification/run_first_iter.py --task_type 'medium' --device 'cuda:0' --pretrained_weights 'simclr'  --portion 4.0 --seed 42 --batch_size 2
python3 ./Classification/run_first_iter.py --task_type 'medium' --device 'cuda:0' --pretrained_weights 'simclr'  --portion 4.0 --seed 92 --batch_size 2
python3 ./Classification/run_first_iter.py --task_type 'medium' --device 'cuda:0' --pretrained_weights 'simclr'  --portion 4.0 --seed 95 --batch_size 2 

python3 ./Classification/run_first_iter.py --task_type 'medium' --device 'cuda:0' --pretrained_weights 'simclr'  --portion 10.0 --seed 10 --batch_size 2
python3 ./Classification/run_first_iter.py --task_type 'medium' --device 'cuda:0' --pretrained_weights 'simclr'  --portion 10.0 --seed 24 --batch_size 2
python3 ./Classification/run_first_iter.py --task_type 'medium' --device 'cuda:0' --pretrained_weights 'simclr'  --portion 10.0 --seed 42 --batch_size 2
python3 ./Classification/run_first_iter.py --task_type 'medium' --device 'cuda:0' --pretrained_weights 'simclr'  --portion 10.0 --seed 92 --batch_size 2
python3 ./Classification/run_first_iter.py --task_type 'medium' --device 'cuda:0' --pretrained_weights 'simclr'  --portion 10.0 --seed 95 --batch_size 2 

python3 ./Classification/run_first_iter.py --task_type 'medium' --device 'cuda:0' --pretrained_weights 'simclr'  --portion 20.0 --seed 10 --batch_size 2
python3 ./Classification/run_first_iter.py --task_type 'medium' --device 'cuda:0' --pretrained_weights 'simclr'  --portion 20.0 --seed 24 --batch_size 2
python3 ./Classification/run_first_iter.py --task_type 'medium' --device 'cuda:0' --pretrained_weights 'simclr'  --portion 20.0 --seed 42 --batch_size 2
python3 ./Classification/run_first_iter.py --task_type 'medium' --device 'cuda:0' --pretrained_weights 'simclr'  --portion 20.0 --seed 92 --batch_size 2
python3 ./Classification/run_first_iter.py --task_type 'medium' --device 'cuda:0' --pretrained_weights 'simclr'  --portion 20.0 --seed 95 --batch_size 2 


# 2. No Low Batch Size (i.e., 4 or 8)
python3 ./Classification/run_first_iter.py --task_type 'medium' --device 'cuda:2' --pretrained_weights 'simclr'  --portion 0.5 --cold_start --extractor 'resnet18_simclr' --selection 'kmeans_centroid' --batch_size 4  
python3 ./Classification/run_first_iter.py --task_type 'medium' --device 'cuda:2' --pretrained_weights 'simclr'  --portion 2.0 --cold_start --extractor 'resnet18_simclr' --selection 'kmeans_centroid' --batch_size 8  
python3 ./Classification/run_first_iter.py --task_type 'medium' --device 'cuda:2' --pretrained_weights 'simclr'  --portion 4.0 --cold_start --extractor 'resnet18_simclr' --selection 'kmeans_centroid' --batch_size 8  
python3 ./Classification/run_first_iter.py --task_type 'medium' --device 'cuda:2' --pretrained_weights 'simclr'  --portion 10.0 --cold_start --extractor 'resnet18_simclr' --selection 'kmeans_centroid' --batch_size 8  
python3 ./Classification/run_first_iter.py --task_type 'medium' --device 'cuda:2' --pretrained_weights 'simclr'  --portion 20.0 --cold_start --extractor 'resnet18_simclr' --selection 'kmeans_centroid' --batch_size 8
