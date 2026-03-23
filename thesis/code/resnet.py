import torchvision.models as models

model = models.resnet18(pretrained=False)

# 印出模型架構
print(model)

# 計算可訓練參數量
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"\n總參數量: {total_params:,}")
print(f"可訓練參數量: {trainable_params:,}")