python3 ./Classification/Active_FT/get_features.py --task_type 'medium' --extractor 'MedImageInsight' --device 'cuda:0'



['dinov2_small', 'dinov2_base', 'dinov2_large', 'dinov2_giant']
['clip_base_32', 'clip_base_16', 'clip_large_14', 'clip_large_14_336']

python3 ./Classification/Active_FT/run_first_iter_test.py --task_type 'medium' --portion 0.5 --pretrained_weights 'simclr' --extractor 'resnet18_simclr' --batch_size 2
python3 ./Classification/Active_FT/run_first_iter_test.py --task_type 'medium' --portion 0.5 --pretrained_weights 'simclr' --extractor 'resnet18_pretrained' --batch_size 2

python3 ./Classification/Active_FT/run_first_iter_test.py --task_type 'medium' --portion 0.5 --pretrained_weights 'simclr' --extractor 'dino' --batch_size 2



# python3 ./Classification/Active_FT/run_first_iter_test.py --task_type 'medium' --portion 0.5 --pretrained_weights 'simclr' --extractor 'resnet18_simclr' --batch_size 4
# python3 ./Classification/Active_FT/run_first_iter_test.py --task_type 'medium' --portion 0.5 --pretrained_weights 'simclr' --extractor 'resnet18_simclr' --batch_size 4
# python3 ./Classification/Active_FT/run_first_iter_test.py --task_type 'medium' --portion 0.5 --pretrained_weights 'simclr' --extractor 'resnet18_simclr' --batch_size 4


# python3 ./Classification/Active_FT/run_first_iter_test.py --task_type 'medium' --portion 1.0 --pretrained_weights 'simclr' --extractor 'resnet18_simclr' --batch_size 4
# python3 ./Classification/Active_FT/run_first_iter_test.py --task_type 'medium' --portion 1.0 --pretrained_weights 'simclr' --extractor 'resnet18_simclr' --batch_size 4
# python3 ./Classification/Active_FT/run_first_iter_test.py --task_type 'medium' --portion 1.0 --pretrained_weights 'simclr' --extractor 'resnet18_simclr' --batch_size 4


# python3 ./Classification/Active_FT/run_first_iter_test.py --task_type 'medium' --portion 1.5 --pretrained_weights 'simclr' --extractor 'resnet18_simclr' --batch_size 8
# python3 ./Classification/Active_FT/run_first_iter_test.py --task_type 'medium' --portion 1.5 --pretrained_weights 'simclr' --extractor 'resnet18_simclr' --batch_size 8
# python3 ./Classification/Active_FT/run_first_iter_test.py --task_type 'medium' --portion 1.5 --pretrained_weights 'simclr' --extractor 'resnet18_simclr' --batch_size 8


# python3 ./Classification/Active_FT/run_first_iter_test.py --task_type 'medium' --portion 2.0 --pretrained_weights 'simclr' --extractor 'resnet18_simclr' --batch_size 8
# python3 ./Classification/Active_FT/run_first_iter_test.py --task_type 'medium' --portion 2.0 --pretrained_weights 'simclr' --extractor 'resnet18_simclr' --batch_size 8
# python3 ./Classification/Active_FT/run_first_iter_test.py --task_type 'medium' --portion 2.0 --pretrained_weights 'simclr' --extractor 'resnet18_simclr' --batch_size 8



# python3 ./Classification/Active_FT/run_first_iter_test.py --task_type 'medium' --portion 0.5 --pretrained_weights 'simclr' --extractor 'resnet18_pretrained' --batch_size 4
# python3 ./Classification/Active_FT/run_first_iter_test.py --task_type 'medium' --portion 0.5 --pretrained_weights 'simclr' --extractor 'resnet18_pretrained' --batch_size 4
# python3 ./Classification/Active_FT/run_first_iter_test.py --task_type 'medium' --portion 0.5 --pretrained_weights 'simclr' --extractor 'resnet18_pretrained' --batch_size 4


# python3 ./Classification/Active_FT/run_first_iter_test.py --task_type 'medium' --portion 1.0 --pretrained_weights 'simclr' --extractor 'resnet18_pretrained' --batch_size 4
# python3 ./Classification/Active_FT/run_first_iter_test.py --task_type 'medium' --portion 1.0 --pretrained_weights 'simclr' --extractor 'resnet18_pretrained' --batch_size 4
# python3 ./Classification/Active_FT/run_first_iter_test.py --task_type 'medium' --portion 1.0 --pretrained_weights 'simclr' --extractor 'resnet18_pretrained' --batch_size 4


# python3 ./Classification/Active_FT/run_first_iter_test.py --task_type 'medium' --portion 1.5 --pretrained_weights 'simclr' --extractor 'resnet18_pretrained' --batch_size 8
# python3 ./Classification/Active_FT/run_first_iter_test.py --task_type 'medium' --portion 1.5 --pretrained_weights 'simclr' --extractor 'resnet18_pretrained' --batch_size 8
# python3 ./Classification/Active_FT/run_first_iter_test.py --task_type 'medium' --portion 1.5 --pretrained_weights 'simclr' --extractor 'resnet18_pretrained' --batch_size 8


# python3 ./Classification/Active_FT/run_first_iter_test.py --task_type 'medium' --portion 2.0 --pretrained_weights 'simclr' --extractor 'resnet18_pretrained' --batch_size 8
# python3 ./Classification/Active_FT/run_first_iter_test.py --task_type 'medium' --portion 2.0 --pretrained_weights 'simclr' --extractor 'resnet18_pretrained' --batch_size 8
# python3 ./Classification/Active_FT/run_first_iter_test.py --task_type 'medium' --portion 2.0 --pretrained_weights 'simclr' --extractor 'resnet18_pretrained' --batch_size 8



python3 ./Classification/Active_FT/run_first_iter_test.py --task_type 'medium' --portion 0.5 --pretrained_weights 'simclr' --extractor 'MedImageInsight' --batch_size 4
python3 ./Classification/Active_FT/run_first_iter_test.py --task_type 'medium' --portion 0.5 --pretrained_weights 'simclr' --extractor 'MedImageInsight' --batch_size 4
python3 ./Classification/Active_FT/run_first_iter_test.py --task_type 'medium' --portion 0.5 --pretrained_weights 'simclr' --extractor 'MedImageInsight' --batch_size 4


python3 ./Classification/Active_FT/run_first_iter_test.py --task_type 'medium' --portion 1.0 --pretrained_weights 'simclr' --extractor 'MedImageInsight' --batch_size 4
python3 ./Classification/Active_FT/run_first_iter_test.py --task_type 'medium' --portion 1.0 --pretrained_weights 'simclr' --extractor 'MedImageInsight' --batch_size 4
python3 ./Classification/Active_FT/run_first_iter_test.py --task_type 'medium' --portion 1.0 --pretrained_weights 'simclr' --extractor 'MedImageInsight' --batch_size 4


python3 ./Classification/Active_FT/run_first_iter_test.py --task_type 'medium' --portion 1.5 --pretrained_weights 'simclr' --extractor 'MedImageInsight' --batch_size 8
python3 ./Classification/Active_FT/run_first_iter_test.py --task_type 'medium' --portion 1.5 --pretrained_weights 'simclr' --extractor 'MedImageInsight' --batch_size 8
python3 ./Classification/Active_FT/run_first_iter_test.py --task_type 'medium' --portion 1.5 --pretrained_weights 'simclr' --extractor 'MedImageInsight' --batch_size 8


python3 ./Classification/Active_FT/run_first_iter_test.py --task_type 'medium' --portion 2.0 --pretrained_weights 'simclr' --extractor 'MedImageInsight' --batch_size 8
python3 ./Classification/Active_FT/run_first_iter_test.py --task_type 'medium' --portion 2.0 --pretrained_weights 'simclr' --extractor 'MedImageInsight' --batch_size 8
python3 ./Classification/Active_FT/run_first_iter_test.py --task_type 'medium' --portion 2.0 --pretrained_weights 'simclr' --extractor 'MedImageInsight' --batch_size 8