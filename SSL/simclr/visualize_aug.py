import os
import argparse
from PIL import Image, ImageFilter
import torch
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np


class GaussianBlur(object):
    """Gaussian blur augmentation from SimCLR"""
    def __init__(self, kernel_size, sigma=[.1, 2.]):
        self.kernel_size = kernel_size
        self.sigma = sigma

    def __call__(self, x):
        from PIL import ImageFilter
        sigma = np.random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


def get_augmentation_transform(size=224):
    """
    Get data augmentation transform (without normalization for visualization)
    
    Args:
        size: Image size (default: 224)
    
    Returns:
        transforms.Compose object
    """
    data_transforms = transforms.Compose([
        transforms.RandomResizedCrop(size=size, scale=(0.3, 0.7)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomAutocontrast(),
        transforms.RandomEqualize(),
        GaussianBlur(kernel_size=int(0.1 * size)),
        transforms.ToTensor(),
        # 移除 Normalize，因為要視覺化
    ])
    
    return data_transforms


def apply_augmentations(image_path, output_dir, num_augmentations=10, size=224):
    """
    Apply augmentations multiple times and save results
    
    Args:
        image_path: Path to input image
        output_dir: Directory to save augmented images
        num_augmentations: Number of augmented versions to generate
        size: Image size
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load original image
    original_img = Image.open(image_path).convert('RGB')
    
    # Save original image
    original_save_path = os.path.join(output_dir, '00_original.png')
    original_img.save(original_save_path)
    print(f"Original image saved to: {original_save_path}")
    
    # Get augmentation transform
    transform = get_augmentation_transform(size=size)
    
    # Apply augmentations multiple times
    print(f"\nGenerating {num_augmentations} augmented versions...")
    for i in range(1, num_augmentations + 1):
        # Apply transform
        augmented_tensor = transform(original_img)
        
        # Convert tensor to PIL Image
        # Tensor is in [C, H, W] format with values in [0, 1]
        augmented_np = augmented_tensor.permute(1, 2, 0).numpy()
        augmented_np = (augmented_np * 255).astype(np.uint8)
        augmented_img = Image.fromarray(augmented_np)
        
        # Save augmented image
        save_path = os.path.join(output_dir, f'{i:02d}_augmented.png')
        augmented_img.save(save_path)
        print(f"  [{i}/{num_augmentations}] Saved: {save_path}")
    
    print(f"\nAll augmented images saved to: {output_dir}")


def create_visualization_grid(output_dir, num_images=9, figsize=(15, 15)):
    """
    Create a grid visualization of original and augmented images
    
    Args:
        output_dir: Directory containing the saved images
        num_images: Number of images to display (including original)
        figsize: Figure size
    """
    # Get list of image files
    image_files = sorted([f for f in os.listdir(output_dir) if f.endswith('.png')])
    
    # Limit to num_images
    image_files = image_files[:num_images]
    
    # Calculate grid size
    n_cols = 3
    n_rows = (len(image_files) + n_cols - 1) // n_cols
    
    # Create figure
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes
    
    # Plot images
    for idx, img_file in enumerate(image_files):
        img_path = os.path.join(output_dir, img_file)
        img = Image.open(img_path)
        
        axes[idx].imshow(img)
        
        # Set title
        if img_file.startswith('00_original'):
            title = 'Original'
        else:
            aug_num = img_file.split('_')[0]
            title = f'Augmentation {int(aug_num)}'
        
        axes[idx].set_title(title, fontsize=12, fontweight='bold')
        axes[idx].axis('off')
    
    # Hide extra subplots
    for idx in range(len(image_files), len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    
    # Save grid visualization
    grid_save_path = os.path.join(output_dir, 'augmentation_grid.png')
    plt.savefig(grid_save_path, dpi=150, bbox_inches='tight')
    print(f"\nGrid visualization saved to: {grid_save_path}")
    
    plt.show()


def parse_arguments():
    parser = argparse.ArgumentParser(description='Visualize data augmentation by applying transforms multiple times')
    parser.add_argument('--image_path', type=str, required=True,
                        help='Path to input image')
    parser.add_argument('--output_dir', type=str, default='./augmentation_results',
                        help='Directory to save augmented images')
    parser.add_argument('--num_augmentations', type=int, default=10,
                        help='Number of augmented versions to generate')
    parser.add_argument('--size', type=int, default=256, # 智皓是用256囉...
                        help='Image size for augmentation')
    parser.add_argument('--create_grid', action='store_true',
                        help='Create a grid visualization of the results')
    parser.add_argument('--grid_size', type=int, default=9,
                        help='Number of images to show in grid (including original)')
    
    return parser.parse_args()


def main():
    args = parse_arguments()
    
    # Check if image exists
    if not os.path.isfile(args.image_path):
        raise FileNotFoundError(f"Image not found: {args.image_path}")
    
    print('='*60)
    print('Data Augmentation Visualization')
    print('='*60)
    print(f"Input image: {args.image_path}")
    print(f"Output directory: {args.output_dir}")
    print(f"Number of augmentations: {args.num_augmentations}")
    print(f"Image size: {args.size}x{args.size}")
    print('='*60)
    
    # Apply augmentations
    apply_augmentations(
        args.image_path,
        args.output_dir,
        args.num_augmentations,
        args.size
    )
    
    # Create grid visualization if requested
    if args.create_grid:
        print('\n' + '='*60)
        print('Creating grid visualization...')
        print('='*60)
        create_visualization_grid(
            args.output_dir,
            args.grid_size
        )
    
    print('\nDone!')


if __name__ == "__main__":
    main()