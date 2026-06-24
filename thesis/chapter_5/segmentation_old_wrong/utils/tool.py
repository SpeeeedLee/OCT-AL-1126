import numpy as np
import matplotlib.pyplot as plt


def compute_dice_binary(pred, target):
    """
    Compute Dice coefficient for binary segmentation
    
    Args:
        pred: Predicted binary mask - shape: [batch, 1, H, W] or [1, H, W] or [H, W]
              Values: 0.0 or 1.0
        target: Ground truth binary mask - shape: same as pred
                Values: 0.0 or 1.0
    
    Returns:
        dice: Dice coefficient (float)
    """
    # Flatten arrays
    pred = pred.flatten()
    target = target.flatten()
    
    # Calculate Dice
    intersection = (pred * target).sum()
    dice = (2. * intersection) / (pred.sum() + target.sum() + 1e-9)
    
    return dice


def IOUDICE_binary(pred, target):
    """
    Calculate IoU and Dice coefficient for binary segmentation
    
    Args:
        pred: Predicted binary mask - shape: [H, W], values: 0 or 1
        target: Ground truth binary mask - shape: [H, W], values: 0 or 1
    
    Returns:
        iou: Intersection over Union
        dice: Dice coefficient
    """
    TP = np.count_nonzero((target == 1) & (pred == 1))
    FP = np.count_nonzero((target == 0) & (pred == 1))
    FN = np.count_nonzero((target == 1) & (pred == 0))
    
    # IoU
    iou = TP / (TP + FP + FN + 1e-9)
    
    # Dice
    dice = 2 * TP / (2 * TP + FP + FN + 1e-9)
    
    return iou, dice


def plot_binary(img_sub, target, pred_mask, threshold=0.5):
    """
    Visualize binary segmentation results
    
    Args:
        img_sub: Original image - shape: [H, W] or [1, H, W]
        target: Ground truth binary mask - shape: [H, W] or [1, H, W], values: 0/1
        pred_mask: Predicted mask - shape: [H, W] or [1, H, W], values: 0-1 (probabilities) or 0/1 (binary)
        threshold: Threshold for binarizing predictions, default: 0.5
    """
    # Squeeze dimensions if needed
    if len(img_sub.shape) == 3:
        img_sub = img_sub[0]
    if len(target.shape) == 3:
        target = target[0]
    if len(pred_mask.shape) == 3:
        pred_mask = pred_mask[0]
    
    # Binarize prediction if needed
    pred_binary = (pred_mask > threshold).astype(np.uint8)
    target_binary = target.astype(np.uint8)
    
    # Create colored visualization
    height, width = img_sub.shape
    cimg = np.zeros((height, width, 4))
    
    # Color coding:
    # Yellow: True Positives (correct nuclei detection)
    # Magenta: False Positives (predicted nuclei but actually background)
    # Cyan: False Negatives (missed nuclei)
    # Black: True Negatives (correct background)
    
    cimg[(target_binary == 1) & (pred_binary == 1)] = [255, 255, 0, 255]    # TP: Yellow
    cimg[(target_binary == 0) & (pred_binary == 1)] = [255, 0, 255, 255]    # FP: Magenta
    cimg[(target_binary == 1) & (pred_binary == 0)] = [0, 255, 255, 100]    # FN: Cyan
    cimg[(target_binary == 0) & (pred_binary == 0)] = [0, 0, 0, 255]        # TN: Black

    # Display original image
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.imshow(img_sub, cmap="gray")
    plt.title('Original Image')
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.imshow(target_binary, cmap="gray")
    plt.title('Ground Truth')
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.imshow(cimg.astype('uint8'))
    plt.title('Prediction (Yellow=TP, Magenta=FP, Cyan=FN)')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()


# ========== Legacy functions (kept for compatibility with multi-class code) ==========

def IOUDICE(out, gt, c):
    """
    [DEPRECATED for binary segmentation]
    Calculating the Dice coefficient for multi-class segmentation
    
    Use compute_dice_binary() or IOUDICE_binary() instead for binary segmentation.
    """
    TP = np.count_nonzero((gt == c) & (out == c))
    FP = np.count_nonzero((gt != c) & (out == c))
    FN = np.count_nonzero((gt == c) & (out != c))
    Dice = 2 * TP / (2 * TP + FP + FN + 1e-9)
    return Dice


def plot(img_sub, gim_sub, out_img, c):
    """
    [DEPRECATED for binary segmentation]
    Visualize multi-class segmentation results
    
    Use plot_binary() instead for binary segmentation.
    """
    cimg = np.zeros((384, 500, 4))
    
    # Assigning colors to different regions
    cimg[(gim_sub == c) & (out_img == c)] = [255, 255, 0, 255]    # TP
    cimg[(gim_sub != c) & (out_img == c)] = [255, 0, 255, 255]    # FP
    cimg[(gim_sub == c) & (out_img != c)] = [0, 255, 255, 100]    # FN
    cimg[(gim_sub != c) & (out_img != c)] = [0, 0, 0, 255]        # TN

    # Display original image
    plt.imshow(img_sub, cmap="gray")
    plt.axis('off')
    plt.show()

    # Display segmented image
    plt.imshow(cimg.astype('uint8'))
    plt.axis('off')
    plt.show()