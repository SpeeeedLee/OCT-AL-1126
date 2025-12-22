import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from tqdm import tqdm

# Add current directory to path
sys.path.insert(0, os.getcwd())

from classification.utils.data import get_data
from classification.model.resnet import get_resnet18_classifier
from classification.model.simclr.resnet_simclr import ResNetSimCLR


def parse_arguments():
    parser = argparse.ArgumentParser(description='Plot t-SNE visualization of test set representations')
    parser.add_argument('--task_type', type=str, choices=['easy', 'medium', 'hard'], required=True,
                        help='Task difficulty determining number of classes')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='Device to use for inference')
    parser.add_argument('--pretrained_weights', type=str, 
                        choices=['random', 'imagenet', 'simclr', 'auto_encoder'], 
                        required=True,
                        help='Type of pretrained weights to use')
    parser.add_argument('--model_path', type=str, default=None,
                        help='Path to trained model checkpoint (optional, if you want to load a fine-tuned model)')
    parser.add_argument('--simclr_path', type=str, default=None,
                        help='Path to SimCLR pretrained weights (required if pretrained_weights=simclr)')
    parser.add_argument('--output_path', type=str, default='./tsne_plots',
                        help='Directory to save the t-SNE plot')
    parser.add_argument('--perplexity', type=int, default=30,
                        help='t-SNE perplexity parameter')
    parser.add_argument('--random_state', type=int, default=42,
                        help='Random state for t-SNE reproducibility')
    parser.add_argument('--figsize', type=int, nargs=2, default=[10, 8],
                        help='Figure size (width height)')
    
    return parser.parse_args()


def initialize_model(num_classes, pretrained, pretrained_weights, simclr_path=None):
    """Initialize model based on pretrained_weights type"""
    
    if pretrained_weights == 'simclr':
        if simclr_path is None:
            raise ValueError("simclr_path is required when pretrained_weights='simclr'")
        
        print(f'Loading SimCLR model from {simclr_path}')
        model = ResNetSimCLR('resnet18', 32)
        state_dict = torch.load(simclr_path, map_location='cpu', weights_only=True)
        model.load_state_dict(state_dict, strict=False)
        
        # Replace the final FC layer
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
    features_blob = output.data.cpu().numpy()


def extract_embeddings(model, data_loader, device, model_type='resnet'):
    """
    Extract embeddings from the model before the final FC layer
    
    Args:
        model: The neural network model
        data_loader: DataLoader for the test set
        device: Device to run inference on
        model_type: 'resnet' or 'simclr' to determine which layer to hook
    
    Returns:
        embeddings: numpy array of shape (N, embedding_dim)
        labels: numpy array of shape (N,)
    """
    model.to(device)
    model.eval()
    
    # Register hook to extract features
    global features_blob
    features_blob = None
    
    if model_type == 'simclr':
        # For SimCLR, hook the avgpool layer before FC
        hook = model.backbone.avgpool.register_forward_hook(extract_features_hook)
    else:
        # For regular ResNet, hook the avgpool layer
        hook = model.avgpool.register_forward_hook(extract_features_hook)
    
    all_embeddings = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in tqdm(data_loader, desc="Extracting embeddings"):
            inputs = inputs.to(device)
            
            # Forward pass
            _ = model(inputs)
            
            # Get the hooked features and flatten
            batch_embeddings = features_blob.reshape(features_blob.shape[0], -1)
            
            all_embeddings.append(batch_embeddings)
            all_labels.append(labels.numpy())
    
    # Remove hook
    hook.remove()
    
    # Concatenate all batches
    embeddings = np.vstack(all_embeddings)
    labels = np.concatenate(all_labels)
    
    print(f"Extracted embeddings shape: {embeddings.shape}")
    print(f"Labels shape: {labels.shape}")
    
    return embeddings, labels


def plot_tsne(embeddings, labels, class_names, args):
    """
    Apply t-SNE and plot the results
    
    Args:
        embeddings: numpy array of shape (N, embedding_dim)
        labels: numpy array of shape (N,)
        class_names: list of class names
        args: command line arguments
    """
    print(f"\nApplying t-SNE with perplexity={args.perplexity}...")
    
    tsne = TSNE(
        n_components=2,
        perplexity=args.perplexity,
        random_state=args.random_state,
        n_iter=1000,
        verbose=1
    )
    
    embeddings_2d = tsne.fit_transform(embeddings)
    
    print("Creating visualization...")
    
    # Create figure
    fig, ax = plt.subplots(figsize=tuple(args.figsize))
    
    # Generate colors for each class
    colors = plt.cm.tab10(np.linspace(0, 1, len(class_names)))
    
    # Plot each class with different color
    for class_idx, class_name in enumerate(class_names):
        mask = labels == class_idx
        ax.scatter(
            embeddings_2d[mask, 0],
            embeddings_2d[mask, 1],
            c=[colors[class_idx]],
            label=class_name,
            alpha=0.6,
            s=50,
            edgecolors='w',
            linewidth=0.5
        )
    
    ax.set_xlabel('t-SNE Component 1', fontsize=12)
    ax.set_ylabel('t-SNE Component 2', fontsize=12)
    ax.set_title(f't-SNE Visualization - {args.task_type.capitalize()} Task ({args.pretrained_weights})', 
                 fontsize=14, fontweight='bold')
    ax.legend(loc='best', frameon=True, fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure
    os.makedirs(args.output_path, exist_ok=True)
    filename = f"tsne_{args.task_type}_{args.pretrained_weights}.png"
    filepath = os.path.join(args.output_path, filename)
    
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to: {filepath}")
    
    # # Also save as PDF for better quality
    # pdf_filepath = filepath.replace('.png', '.pdf')
    # plt.savefig(pdf_filepath, bbox_inches='tight')
    # print(f"Plot saved to: {pdf_filepath}")
    
    plt.show()


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
    print(f"Task: {args.task_type} ({num_classes} classes)")
    print(f"Pretrained weights: {args.pretrained_weights}")
    print(f"Data directory: {data_dir}")
    print('='*60)
    
    # Load data (we only need test set, but get_data returns all)
    # Use all training data indices (not important since we only use test set)
    data_loaders, dataset_sizes = get_data(
        data_dir, 
        labeled_train_idx=None,  # Not used for test set
        batch_size=32,  # Larger batch size for faster inference
        data_aug=False  # No augmentation needed for testing
    )
    
    # Get class names
    import torchvision.datasets as datasets
    temp_dataset = datasets.ImageFolder(os.path.join(data_dir, 'val'))
    class_names = temp_dataset.classes
    print(f"\nClasses: {class_names}")
    print(f"Test set size: {dataset_sizes['test']}")
    
    # Initialize model
    model, model_type = initialize_model(
        num_classes, 
        pretrained=(args.pretrained_weights == 'imagenet'),
        pretrained_weights=args.pretrained_weights,
        simclr_path=args.simclr_path
    )
    
    # Optionally load fine-tuned checkpoint
    if args.model_path is not None:
        print(f"\nLoading fine-tuned model from {args.model_path}")
        state_dict = torch.load(args.model_path, map_location='cpu', weights_only=True)
        model.load_state_dict(state_dict)
        print("Model loaded successfully")
    
    # Extract embeddings from test set
    print("\nExtracting embeddings from test set...")
    embeddings, labels = extract_embeddings(
        model, 
        data_loaders['test'], 
        args.device,
        model_type=model_type
    )
    
    # Plot t-SNE
    plot_tsne(embeddings, labels, class_names, args)
    
    print("\nDone!")


if __name__ == "__main__":
    main()