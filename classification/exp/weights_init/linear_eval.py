import os
import sys
print(f"Current working directory: {os.getcwd()}")
sys.path.insert(0, os.getcwd())

import random
import json
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

from classification.utils.data import get_data, get_num_train
from classification.model.resnet import get_resnet18_classifier
from classification.model.simclr.resnet_simclr import ResNetSimCLR


def parse_arguments():
    import argparse
    parser = argparse.ArgumentParser(description='Linear evaluation of pretrained representations')
    parser.add_argument('--task_type', type=str, choices=['easy', 'medium', 'hard'], required=True)
    parser.add_argument('--portion', type=float, required=True,
                        help='Percentage of training data to use (e.g., 10.0 for 10%)')
    parser.add_argument('--seed', type=int, required=True,
                        help='Random seed for sampling training data')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--exp_path', type=str, default='./exp_results',
                        help='Path to save experiment results')
    parser.add_argument('--pretrained_weights', type=str, 
                        choices=['random', 'imagenet', 'simclr', 'auto_encoder'], 
                        required=True,
                        help='Type of pretrained weights to evaluate')
    parser.add_argument('--simclr_path', type=str, default=None,
                        help='Path to SimCLR pretrained weights (required if pretrained_weights=simclr)')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for feature extraction')
    
    # Logistic Regression hyperparameters
    parser.add_argument('--max_iter', type=int, default=1000,
                        help='Maximum iterations for logistic regression')
    parser.add_argument('--C', type=float, default=1.0,
                        help='Inverse of regularization strength for logistic regression')
    parser.add_argument('--solver', type=str, default='lbfgs',
                        choices=['lbfgs', 'liblinear', 'saga', 'sag'],
                        help='Solver for logistic regression')
    
    return parser.parse_args()


def initialize_model(num_classes, pretrained_weights, simclr_path=None):
    """Initialize model based on pretrained_weights type"""
    
    if pretrained_weights == 'simclr':
        if simclr_path is None:
            raise ValueError("simclr_path is required when pretrained_weights='simclr'")
        
        print(f'Loading SimCLR model from {simclr_path}')
        model = ResNetSimCLR('resnet18', 32)
        state_dict = torch.load(simclr_path, map_location='cpu', weights_only=True)
        model.load_state_dict(state_dict, strict=False)
        
        # Replace the final FC layer (though we won't use it)
        in_features = model.backbone.fc[0].in_features
        model.backbone.fc = nn.Linear(in_features, num_classes, bias=True)
        
        return model, 'simclr'
    
    elif pretrained_weights == 'random':
        print('Initializing ResNet18 without pretrained weights')
        model = get_resnet18_classifier(num_classes=num_classes, pretrained=False)
        return model, 'resnet'
    
    elif pretrained_weights == 'imagenet':
        print('Initializing ResNet18 with ImageNet pretrained weights')
        model = get_resnet18_classifier(num_classes=num_classes, pretrained=True)
        return model, 'resnet'
    
    elif pretrained_weights == 'auto_encoder':
        raise NotImplementedError("auto_encoder not implemented yet")
    
    else:
        raise ValueError(f"Unknown pretrained_weights: {pretrained_weights}")


def extract_features_hook(module, input, output):
    """Hook function to extract features before FC layer"""
    global features_blob
    features_blob = output.data.cpu()


def extract_all_embeddings(model, data_loaders, device, model_type='resnet'):
    """
    Extract embeddings from all data splits (train, val, test)
    
    Args:
        model: The neural network model
        data_loaders: Dictionary of DataLoaders
        device: Device to run inference on
        model_type: 'resnet' or 'simclr' to determine which layer to hook
    
    Returns:
        embeddings_dict: Dictionary with train/val/test embeddings
        labels_dict: Dictionary with train/val/test labels
    """
    model.to(device)
    model.eval()
    
    # Register hook to extract features
    global features_blob
    features_blob = None
    
    if model_type == 'simclr':
        hook = model.backbone.avgpool.register_forward_hook(extract_features_hook)
    else:
        hook = model.avgpool.register_forward_hook(extract_features_hook)
    
    embeddings_dict = {}
    labels_dict = {}
    
    for phase in ['train', 'val', 'test']:
        print(f"\nExtracting {phase} embeddings...")
        all_embeddings = []
        all_labels = []
        
        with torch.no_grad():
            for inputs, labels in tqdm(data_loaders[phase], desc=f"Extracting {phase}"):
                inputs = inputs.to(device)
                
                # Forward pass
                _ = model(inputs)
                
                # Get the hooked features and flatten
                batch_embeddings = features_blob.reshape(features_blob.shape[0], -1)
                
                all_embeddings.append(batch_embeddings.numpy())
                all_labels.append(labels.numpy())
        
        # Concatenate all batches
        embeddings_dict[phase] = np.vstack(all_embeddings)
        labels_dict[phase] = np.concatenate(all_labels)
        
        print(f"{phase.capitalize()} embeddings shape: {embeddings_dict[phase].shape}")
    
    # Remove hook
    hook.remove()
    
    return embeddings_dict, labels_dict


def train_logistic_regression(X_train, y_train, X_val, y_val, args):
    """
    Train logistic regression on the embeddings
    
    Args:
        X_train: Training embeddings
        y_train: Training labels
        X_val: Validation embeddings
        y_val: Validation labels
        args: Command line arguments
    
    Returns:
        clf: Trained logistic regression classifier
        val_acc: Validation accuracy
    """
    print("\n" + "="*60)
    print("Training Logistic Regression")
    print("="*60)
    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print(f"Feature dimension: {X_train.shape[1]}")
    print(f"Number of classes: {len(np.unique(y_train))}")
    print(f"C (inverse regularization): {args.C}")
    print(f"Max iterations: {args.max_iter}")
    print(f"Solver: {args.solver}")
    print("-"*60)
    
    # Initialize and train logistic regression
    clf = LogisticRegression(
        C=args.C,
        max_iter=args.max_iter,
        solver=args.solver,
        multi_class='multinomial',
        random_state=args.seed,
        verbose=1,
        n_jobs=-1  # Use all CPU cores
    )
    
    print("\nTraining...")
    clf.fit(X_train, y_train)
    
    # Evaluate on validation set
    val_pred = clf.predict(X_val)
    val_acc = accuracy_score(y_val, val_pred)
    
    print(f"\nValidation Accuracy: {val_acc:.4f}")
    
    return clf, val_acc


def evaluate_on_test(clf, X_test, y_test, class_names=None):
    """
    Evaluate the trained classifier on test set
    
    Args:
        clf: Trained classifier
        X_test: Test embeddings
        y_test: Test labels
        class_names: List of class names for detailed report
    
    Returns:
        test_acc: Test accuracy
    """
    print("\n" + "="*60)
    print("Evaluating on Test Set")
    print("="*60)
    
    test_pred = clf.predict(X_test)
    test_acc = accuracy_score(y_test, test_pred)
    
    print(f"Test Accuracy: {test_acc:.4f}")
    
    # Detailed classification report
    if class_names is not None:
        print("\nDetailed Classification Report:")
        print("-"*60)
        report = classification_report(
            y_test, test_pred, 
            target_names=class_names,
            digits=4
        )
        print(report)
    
    return test_acc


def save_compact_json(data, file_path):
    """Save JSON with compact list formatting"""
    def format_dict(d, indent=0):
        lines = []
        items = list(d.items())
        for i, (key, value) in enumerate(items):
            is_last = (i == len(items) - 1)
            comma = '' if is_last else ','
            
            if isinstance(value, dict):
                lines.append('  ' * indent + f'"{key}": {{')
                lines.append(format_dict(value, indent + 1))
                lines.append('  ' * indent + '}' + comma)
            elif isinstance(value, list):
                list_str = '[' + ', '.join(str(x) for x in value) + ']'
                lines.append('  ' * indent + f'"{key}": {list_str}' + comma)
            else:
                lines.append('  ' * indent + f'"{key}": {json.dumps(value)}' + comma)
        
        return '\n'.join(lines)
    
    json_str = '{\n' + format_dict(data, 1) + '\n}'
    
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(json_str)


def check_existing_results(file_path, portion_key, C_key, max_runs=5):
    """
    Check if the experiment has already been run enough times.
    """
    if not os.path.isfile(file_path):
        print(f"No existing results file found at {file_path}")
        return
    
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"Warning: JSON decode error in {file_path}: {e}")
        return
    
    # Navigate through the nested structure
    if portion_key in data:
        if C_key in data[portion_key]:
            existing_results = data[portion_key][C_key]
            num_existing = len(existing_results)
            
            if num_existing >= max_runs:
                raise RuntimeError(
                    f"\n{'='*60}\n"
                    f"Experiment already completed!\n"
                    f"Configuration: portion={portion_key}, C={C_key}\n"
                    f"Existing results: {existing_results}\n"
                    f"Number of runs: {num_existing}/{max_runs}\n"
                    f"File: {file_path}\n"
                    f"{'='*60}\n"
                )
            else:
                print(f"Found {num_existing}/{max_runs} existing results for this configuration")
        else:
            print(f"No results found for C={C_key}")
    else:
        print(f"No results found for portion={portion_key}")


def main():
    args = parse_arguments()
    
    # Determine task configuration
    task_config = {
        'easy': (2, './ds/classification/two_class'),
        'medium': (4, './ds/classification/four_class'),
        'hard': (7, './ds/classification/seven_class')
    }
    num_classes, data_dir = task_config[args.task_type]
    
    print('='*60)
    print(f"Linear Evaluation Experiment")
    print('='*60)
    print(f"Task: {args.task_type} ({num_classes} classes)")
    print(f"Pretrained weights: {args.pretrained_weights}")
    print(f"Data portion: {args.portion}%")
    print(f"Random seed: {args.seed}")
    print(f"C parameter: {args.C}")
    print('='*60)
    
    # Generate file name
    file_name = f"linear_eval_seed{args.seed}_C{args.C}_solver{args.solver}.json"
    
    # Check if experiment already completed
    portion_key = str(float(args.portion))
    C_key = str(args.C)
    
    save_path = os.path.join(
        args.exp_path, 
        f"classification_{args.task_type}", 
        f"linear_eval_{args.pretrained_weights}"
    )
    file_path = os.path.join(save_path, file_name)
    
    print(f"\nChecking existing results...")
    check_existing_results(file_path, portion_key, C_key, max_runs=5)
    print(f"Check passed. Proceeding with evaluation...\n")
    
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
    
    # Load data (NO augmentation for linear evaluation)
    data_loaders, dataset_sizes = get_data(
        data_dir, 
        label_idx, 
        args.batch_size, 
        data_aug=False
    )
    
    print(f"\nDataset sizes: {dataset_sizes}")
    
    # Get class names
    import torchvision.datasets as datasets
    temp_dataset = datasets.ImageFolder(os.path.join(data_dir, 'val'))
    class_names = temp_dataset.classes
    print(f"Classes: {class_names}")
    
    # Initialize model
    model, model_type = initialize_model(
        num_classes,
        args.pretrained_weights,
        args.simclr_path
    )
    
    # Extract embeddings from all splits
    print("\n" + "="*60)
    print("Extracting embeddings...")
    print("="*60)
    
    embeddings_dict, labels_dict = extract_all_embeddings(
        model, 
        data_loaders, 
        args.device,
        model_type
    )
    
    # Train logistic regression
    clf, val_acc = train_logistic_regression(
        embeddings_dict['train'], 
        labels_dict['train'],
        embeddings_dict['val'], 
        labels_dict['val'],
        args
    )
    
    # Evaluate on test set
    test_acc = evaluate_on_test(
        clf,
        embeddings_dict['test'],
        labels_dict['test'],
        class_names
    )
    
    # Round to 4 decimal places
    test_acc = round(test_acc, 4)
    val_acc = round(val_acc, 4)
    
    print("\n" + "="*60)
    print(f"Final Results:")
    print(f"  Validation Accuracy: {val_acc:.4f}")
    print(f"  Test Accuracy: {test_acc:.4f}")
    print("="*60)
    
    # Save results
    os.makedirs(save_path, exist_ok=True)
    
    # Load or create results file
    if os.path.isfile(file_path):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            print(f"Warning: JSON decode error in {file_path}: {e}")
            print("Creating new data structure...")
            data = {}
    else:
        data = {}
    
    # Initialize portion key if not exists
    if portion_key not in data:
        data[portion_key] = {}
    
    # Initialize C key if not exists
    if C_key not in data[portion_key]:
        data[portion_key][C_key] = []
    
    # Add result (we save test_acc)
    data[portion_key][C_key].append(test_acc)
    
    # Sort portions and C values
    sorted_data = {}
    for p_key in sorted(data.keys(), key=float):
        sorted_C = {}
        for c_key in sorted(data[p_key].keys(), key=float):
            sorted_C[c_key] = data[p_key][c_key]
        sorted_data[p_key] = sorted_C
    
    # Save with compact list format
    save_compact_json(sorted_data, file_path)
    
    print(f'\nResult saved to {file_path}!')
    print('='*60)


if __name__ == "__main__":
    main()