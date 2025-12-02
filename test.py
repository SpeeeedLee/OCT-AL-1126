import os
import sys
print(f"Current working directory: {os.getcwd()}")
sys.path.insert(0, os.getcwd())

import random
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler

from classification.utils.data import get_data, get_num_train
from classification.model.resnet import get_resnet18_classifier
from classification.model.simclr.resnet_simclr import ResNetSimCLR 
from classification.utils.train_eval import train_model


def parse_arguments():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_type', type=str, choices=['easy', 'medium', 'hard'], required=True)
    parser.add_argument('--portion', type=float, required=True) 
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--device', type=str, default='cuda:0')   
    parser.add_argument('--exp_path', type=str, default='./exp_results')   
    parser.add_argument('--epoch', type=int, default=20)   
    parser.add_argument('--lr', type=float, default=5e-5) 
    parser.add_argument('--weight_decay', type=float, default=None)   
    parser.add_argument('--no_use_pretrained', dest='use_pretrained', action='store_false', default=True)
    parser.add_argument('--pretrained_weights', type=str, choices=['simclr', 'auto_encoder'], default=None)   
    parser.add_argument('--simclr_path', type=str, default=None)
    parser.add_argument('--ask_saving', action='store_true', default=False)
    
    return parser.parse_args()


def initialize_model(num_classes, pretrained):
    model = get_resnet18_classifier(num_classes=num_classes, pretrained=pretrained)
    return model


def initialize_simclr_model(num_classes, simclr_path):
    model = ResNetSimCLR('resnet18', 32)
    state_dict = torch.load(simclr_path, map_location='cpu', weights_only=True)
    model.load_state_dict(state_dict, strict=False)
    
    in_features = model.backbone.fc[0].in_features
    model.backbone.fc = nn.Linear(in_features, num_classes, bias=True)
    return model

# def initialize_ae_model(num_classes, ae_path):
#     from SSL.auto_encoder.classes.resnet_autoencoder import AE
    
#     ae = AE('default')
#     ckpt = torch.load('./SSL/auto_encoder/autoencoder_1.ckpt', map_location='cpu')
#     state_dict = ckpt.get('model_state_dict', ckpt)
#     ae.load_state_dict(state_dict, strict=False)
    
#     encoder = ae.encoder
#     model = nn.Sequential(
#         encoder,
#         nn.AdaptiveAvgPool2d((1, 1)),
#         nn.Flatten(),
#         nn.Linear(512, num_classes, bias=True)
#     )
#     return model


def main():
    args = parse_arguments()
    
    # Determine task configuration
    task_config = {
        'easy': (2, './ds/classification/two_class'),
        'medium': (4, './ds/classification/four_class'),
        'hard': (7, './ds/classification/seven_class')
    }
    num_classes, data_dir = task_config[args.task_type]
    
    # Auto-adjust batch size
    # if args.portion <= 20:
    #     args.batch_size = 2
    # elif args.portion <= 60:
    #     args.batch_size = 4
    # else:
    #     args.batch_size = 8
    args.batch_size = 16
    
    # Generate file name
    file_name = f"random{args.seed}_lr{args.lr}_bs{args.batch_size}"
    if args.weight_decay:
        file_name += f"_wd{args.weight_decay}"
    file_name += ".json"
    
    print(f'Exp name: {file_name}')
    
    # Get training data indices
    tot_num_train = get_num_train(data_dir)
    print(f'Total Number of Train: {tot_num_train}')
    
    print(f'===== Load {args.portion}% Data =====')
    target_num = round(tot_num_train * args.portion / 100)
    
    # Sample indices
    random.seed(args.seed)
    unlabeled_idx = list(range(tot_num_train))
    label_idx = random.sample(unlabeled_idx, target_num)
    
    print(f"Number of labeled samples: {len(label_idx)}")
    # print(f"Label indices: {label_idx}")
    
    # Load data
    data_loaders, dataset_sizes = get_data(data_dir, label_idx, args.batch_size, data_aug=False)
    print(dataset_sizes)
    


if __name__ == "__main__":
    main()