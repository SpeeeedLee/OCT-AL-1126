python3 ./Classification/run.py --task_type 'medium' --pretrained_weights 'simclr' --portion_start 0.5 --portion_end 100.5 --portion_interval 0.5 --device 'cuda:2' --AL_strategy 'random' --seed 10

python3 ./Classification/run.py --task_type 'medium' --pretrained_weights 'simclr' --portion_start 0.5 --portion_end 100.5 --portion_interval 0.5 --device 'cuda:2' --AL_strategy 'random' --seed 24

python3 ./Classification/run.py --task_type 'medium' --pretrained_weights 'simclr' --portion_start 0.5 --portion_end 100.5 --portion_interval 0.5 --device 'cuda:2' --AL_strategy 'random' --seed 42

python3 ./Classification/run.py --task_type 'medium' --pretrained_weights 'simclr' --portion_start 0.5 --portion_end 100.5 --portion_interval 0.5 --device 'cuda:2' --AL_strategy 'random' --seed 92

python3 ./Classification/run.py --task_type 'medium' --pretrained_weights 'simclr' --portion_start 0.5 --portion_end 100.5 --portion_interval 0.5 --device 'cuda:2' --AL_strategy 'random' --seed 95

python3 ./Classification/run.py --task_type 'medium' --pretrained_weights 'simclr' --portion_start 0.5 --portion_end 19.5 --portion_interval 4.5 --device 'cuda:2' --AL_strategy 'random' --seed 95


python3 ./Classification/run.py --task_type 'medium' --pretrained_weights 'simclr' --portion_start 0.5 --portion_end 20.5 --portion_interval 0.5 --assign_initial './Classification/label_idx/resnet18_pretrained_initial.json' --extractor 'resnet18_simclr' --device 'cuda:2' --AL_strategy 'conf'

python3 ./Classification/run.py --task_type 'medium' --pretrained_weights 'simclr' --portion_start 0.5 --portion_end 19.5 --portion_interval 4.5 --assign_initial './Classification/label_idx/resnet18_pretrained_initial.json' --extractor 'resnet18_simclr' --device 'cuda:2' --AL_strategy 'conf'



python3 ./Classification/run.py --task_type 'medium' --pretrained_weights 'simclr' --portion_start 0.5 --portion_end 100.5 --portion_interval 5.0 --device 'cuda:2' --AL_strategy 'conf' --load_first_model --first_model_path './model_checkpoints/simclr/resnet18_simclr_0_0.5631.pth'
python3 ./Classification/run.py --task_type 'medium' --pretrained_weights 'simclr' --portion_start 0.5 --portion_end 100.5 --portion_interval 5.0 --device 'cuda:2' --AL_strategy 'conf' --load_first_model --first_model_path './model_checkpoints/simclr/resnet18_simclr_0_0.5631.pth' --test_id 1
python3 ./Classification/run.py --task_type 'medium' --pretrained_weights 'simclr' --portion_start 0.5 --portion_end 100.5 --portion_interval 5.0 --device 'cuda:2' --AL_strategy 'conf' --load_first_model --first_model_path './model_checkpoints/simclr/resnet18_simclr_0_0.5631.pth' --test_id 2

python3 ./Classification/run.py --task_type 'medium' --pretrained_weights 'simclr' --portion_start 0.5 --portion_end 70.5 --portion_interval 2.5 --device 'cuda:2' --AL_strategy 'conf' --load_first_model --first_model_path './model_checkpoints/simclr/resnet18_simclr_0_0.5631.pth'
python3 ./Classification/run.py --task_type 'medium' --pretrained_weights 'simclr' --portion_start 0.5 --portion_end 70.5 --portion_interval 2.5 --device 'cuda:2' --AL_strategy 'conf' --load_first_model --first_model_path './model_checkpoints/simclr/resnet18_simclr_0_0.5631.pth' --test_id 2
python3 ./Classification/run.py --task_type 'medium' --pretrained_weights 'simclr' --portion_start 0.5 --portion_end 70.5 --portion_interval 2.5 --device 'cuda:2' --AL_strategy 'conf' --load_first_model --first_model_path './model_checkpoints/simclr/resnet18_simclr_0_0.5631.pth' --test_id 3

python3 ./Classification/run.py --task_type 'medium' --pretrained_weights 'simclr' --portion_start 0.5 --portion_end 100.5 --portion_interval 10.0 --device 'cuda:7' --AL_strategy 'conf' --load_first_model --first_model_path './model_checkpoints/simclr/resnet18_simclr_0_0.5631.pth'
python3 ./Classification/run.py --task_type 'medium' --pretrained_weights 'simclr' --portion_start 0.5 --portion_end 100.5 --portion_interval 10.0 --device 'cuda:7' --AL_strategy 'conf' --load_first_model --first_model_path './model_checkpoints/simclr/resnet18_simclr_0_0.5631.pth' --test_id 2
python3 ./Classification/run.py --task_type 'medium' --pretrained_weights 'simclr' --portion_start 0.5 --portion_end 100.5 --portion_interval 10.0 --device 'cuda:7' --AL_strategy 'conf' --load_first_model --first_model_path './model_checkpoints/simclr/resnet18_simclr_0_0.5631.pth' --test_id 3




python3 ./Classification/run.py --task_type 'medium' --pretrained_weights 'simclr' --portion_start 0.5 --portion_end 100.5 --portion_interval 5.0 --device 'cuda:2' --AL_strategy 'conf' --skip_top_k 3 --load_first_model --first_model_path './model_checkpoints/simclr/resnet18_simclr_0_0.5631.pth'
python3 ./Classification/run.py --task_type 'medium' --pretrained_weights 'simclr' --portion_start 0.5 --portion_end 100.5 --portion_interval 5.0 --device 'cuda:2' --AL_strategy 'conf' --skip_top_k 10 --load_first_model --first_model_path './model_checkpoints/simclr/resnet18_simclr_0_0.5631.pth'



python3 ./Classification/run.py --task_type 'medium' --pretrained_weights 'simclr' --portion_start 0.5 --portion_end 100.5 --portion_interval 5.0 --device 'cuda:2' --AL_strategy 'conf' --seed 47
python3 ./Classification/run.py --task_type 'medium' --pretrained_weights 'simclr' --portion_start 0.5 --portion_end 100.5 --portion_interval 5.0 --device 'cuda:2' --AL_strategy 'conf' --seed 35
python3 ./Classification/run.py --task_type 'medium' --pretrained_weights 'simclr' --portion_start 0.5 --portion_end 100.5 --portion_interval 5.0 --device 'cuda:2' --AL_strategy 'conf' --seed 23

python3 ./Classification/run.py --task_type 'medium' --pretrained_weights 'simclr' --portion_start 0.5 --portion_end 100.5 --portion_interval 5.0 --device 'cuda:2' --AL_strategy 'random' --seed 31





## coreset
python3 ./Classification/run.py --task_type 'medium' --pretrained_weights 'simclr' --portion_start 0.5 --portion_end 100.5 --portion_interval 5.0 --device 'cuda:7' --AL_strategy 'coreset' --load_first_model --first_model_path './model_checkpoints/simclr/resnet18_simclr_0_0.5631.pth'


python3 ./Classification/run.py --task_type 'medium' --pretrained_weights 'simclr' --portion_start 0.5 --portion_end 100.5 --portion_interval 5.0 --device 'cuda:7' --AL_strategy 'coreset' --seed 23
python3 ./Classification/run.py --task_type 'medium' --pretrained_weights 'simclr' --portion_start 0.5 --portion_end 100.5 --portion_interval 5.0 --device 'cuda:7' --AL_strategy 'coreset' --seed 35
python3 ./Classification/run.py --task_type 'medium' --pretrained_weights 'simclr' --portion_start 0.5 --portion_end 100.5 --portion_interval 5.0 --device 'cuda:7' --AL_strategy 'coreset' --seed 47


## badge
python3 ./Classification/run.py --task_type 'medium' --pretrained_weights 'simclr' --portion_start 0.5 --portion_end 100.5 --portion_interval 5.0 --device 'cuda:7' --AL_strategy 'badge' --load_first_model --first_model_path './model_checkpoints/simclr/resnet18_simclr_0_0.5631.pth'


python3 ./Classification/run.py --task_type 'medium' --pretrained_weights 'simclr' --portion_start 0.5 --portion_end 100.5 --portion_interval 5.0 --device 'cuda:7' --AL_strategy 'badge' --seed 23
python3 ./Classification/run.py --task_type 'medium' --pretrained_weights 'simclr' --portion_start 0.5 --portion_end 100.5 --portion_interval 5.0 --device 'cuda:7' --AL_strategy 'badge' --seed 35
python3 ./Classification/run.py --task_type 'medium' --pretrained_weights 'simclr' --portion_start 0.5 --portion_end 100.5 --portion_interval 5.0 --device 'cuda:7' --AL_strategy 'badge' --seed 47
