import os
import sys
import itertools
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms


def visualize_contrast_jitter(image_path):
    img = Image.open(image_path).convert('L')

    values = [0.0, 0.3, 0.6, 0.9, 1.3, 1.6, 1.9]
    combos = list(itertools.product(values, values))  # (brightness, contrast), 49 combos

    save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'jitter_comparison')
    os.makedirs(save_dir, exist_ok=True)

    for b, c in combos:
        if b == 0.0 and c == 0.0:
            result = img
            fname = 'original.png'
        else:
            jitter = transforms.ColorJitter(brightness=b, contrast=c)
            result = jitter(img)
            fname = f'b{b}_c{c}.png'

        save_path = os.path.join(save_dir, fname)
        result.save(save_path)

    print(f"Saved {len(combos)} images to: {save_dir}")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python visualize_jitter.py <path_to_image.png>")
        sys.exit(1)
    visualize_contrast_jitter(sys.argv[1])

# python3 visualize_jitter.py ../../../ds/classification/seven_class/train/Normal/20201225_152846B.png