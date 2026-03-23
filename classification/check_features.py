# inspect_resnet_features.py
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import sys

def inspect_feature_map(image_path, device='cpu'):
    # Load pretrained ResNet18
    model = models.resnet18(pretrained=True)
    model.eval()
    model.to(device)

    # Hook to capture feature map before avgpool
    feature_map_info = {}

    def hook_fn(module, input, output):
        feature_map_info['shape'] = output.shape
        feature_map_info['tensor'] = output

    # Register hook on layer4 (the last conv block, right before avgpool)
    hook = model.layer4.register_forward_hook(hook_fn)

    # Preprocess image (standard ImageNet transform)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    # Load OCT image (convert to RGB in case it's grayscale)
    img = Image.open(image_path).convert('RGB')
    print(f"Original image size: {img.size}")  # (W, H)

    x = transform(img).unsqueeze(0).to(device)  # (1, 3, 224, 224)
    print(f"Input tensor shape: {x.shape}")

    # Forward pass
    with torch.no_grad():
        output = model(x)

    hook.remove()

    shape = feature_map_info['shape']
    print(f"\n=== Feature map BEFORE avgpool ===")
    print(f"Shape: {shape}")
    print(f"  Batch size : {shape[0]}")
    print(f"  Channels   : {shape[1]}")
    print(f"  Height     : {shape[2]}")
    print(f"  Width      : {shape[3]}")
    print(f"\nAfter avgpool → flattened to: (1, {shape[1]})")

if __name__ == "__main__":
    image_path = sys.argv[1] if len(sys.argv) > 1 else "./ds/classification/seven_class/train/Normal/20201223_094345B.png"
    inspect_feature_map(image_path)