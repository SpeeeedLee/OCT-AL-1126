import os
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import Subset, DataLoader
from torchvision import datasets, transforms

def check_dropout_status(model):
    """
    檢查模型中 dropout 層的狀態
    
    參數:
        model: PyTorch 模型
    
    返回:
        dropout_layers: list of tuples, [(layer_name, dropout_module), ...]
        active_dropout_count: int, 處於 training 模式的 dropout 層數量
    """
    dropout_layers = []
    active_dropout_count = 0
    
    for name, module in model.named_modules():
        if isinstance(module, nn.Dropout):
            dropout_layers.append((name, module))
            if module.training:  # dropout 是否啟用
                active_dropout_count += 1
    
    return dropout_layers, active_dropout_count


# def add_dropout(model: nn.Module, p: float = 0.05, architecture='simclr_resnet18') -> nn.Module:
#     if architecture != 'simclr_resnet18':
#         raise NotImplementedError()
#     class DropoutWrappedModel(nn.Module):
#         def __init__(self, base_model, p):
#             super().__init__()
#             self.backbone = base_model.backbone  # 不 clone，直接引用
#             self.dropout = nn.Dropout(p=p)
        
#         def forward(self, x):
#             x = self.backbone.conv1(x)
#             x = self.backbone.bn1(x)
#             x = self.backbone.relu(x)
#             x = self.backbone.maxpool(x)

#             x = self.backbone.layer1(x)
#             x = self.backbone.layer2(x)
#             x = self.backbone.layer3(x)
#             x = self.backbone.layer4(x)

#             x = self.backbone.avgpool(x)
#             x = torch.flatten(x, 1)

#             x = self.dropout(x)  # 在 fc 前加 dropout

#             x = self.backbone.fc(x)
#             return x

#     return DropoutWrappedModel(model, p)


def add_dropout(model: nn.Module, p: float = 0.5, architecture='simclr_resnet18') -> nn.Module:
    if architecture != 'simclr_resnet18':
        raise NotImplementedError()

    # 取得原本 fc 的 Linear 層
    old_fc = model.backbone.fc
    if isinstance(old_fc, nn.Sequential):
        # 如果原本已經是 Sequential，提取最後一層 Linear
        for layer in reversed(old_fc):
            if isinstance(layer, nn.Linear):
                old_fc = layer
                break

    # 建立新的 fc：Dropout + 原 Linear
    new_fc = nn.Sequential(
        nn.Dropout(p=p),
        nn.Linear(old_fc.in_features, old_fc.out_features, bias=old_fc.bias is not None)
    )

    # 複製原來 Linear 權重與 bias
    new_fc[1].weight.data.copy_(old_fc.weight.data)
    if old_fc.bias is not None:
        new_fc[1].bias.data.copy_(old_fc.bias.data)

    # 替換 model 中的 fc
    model.backbone.fc = new_fc
    return model


# def add_dropout(model, architecture='simclr_resnet18', 
#                 conv_dropout_prob=0.00, fc_dropout_prob=0.00):
#     """
#     為模型添加 dropout 層，返回新模型而不修改原模型
    
#     Args:
#         model: 原始模型（不會被修改）
#         architecture: 模型架構類型
#         conv_dropout_prob: 卷積層 dropout 機率
#         fc_dropout_prob: 全連接層 dropout 機率
    
#     Returns:
#         new_model: 添加了 dropout 的新模型（原模型不變）
#     """
    
#     # 🔑 關鍵：在函數開始就創建深拷貝
#     import copy
#     print("Creating deep copy of the original model...")
#     new_model = copy.deepcopy(model)
#     print("✅ Deep copy created, original model will not be modified")

#     if architecture == 'simclr_resnet18':
#         print("Adding comprehensive dropout to SimCLR ResNet18...")
        
#         # 首先在 fc layer 添加 dropout
#         if hasattr(new_model.backbone, 'fc'):
#             original_fc = new_model.backbone.fc
            
#             if isinstance(original_fc, nn.Linear):
#                 in_features = original_fc.in_features
#                 out_features = original_fc.out_features
#                 original_weight = original_fc.weight.data.clone()
#                 original_bias = original_fc.bias.data.clone() if original_fc.bias is not None else None
                
#                 new_fc = nn.Sequential(
#                     nn.Dropout(fc_dropout_prob),
#                     nn.Linear(in_features, out_features, bias=original_fc.bias is not None)
#                 )
                
#                 new_fc[1].weight.data.copy_(original_weight)
#                 if original_bias is not None:
#                     new_fc[1].bias.data.copy_(original_bias)
                
#                 new_model.backbone.fc = new_fc
#                 print(f"✅ Added dropout (p={fc_dropout_prob}) to fc layer")
        
#         # 嘗試在 ResNet layers 後添加 dropout
#         def add_dropout_after_layer(layer_name):
#             """在指定層後添加 dropout"""
#             try:
#                 layer = getattr(new_model.backbone, layer_name)
                
#                 # 創建一個包裝器，在原始層後添加 dropout
#                 class LayerWithDropout(nn.Module):
#                     def __init__(self, original_layer, dropout_prob):
#                         super().__init__()
#                         self.original_layer = original_layer
#                         self.dropout = nn.Dropout2d(dropout_prob)  # 使用 Dropout2d 用於 2D feature maps
                    
#                     def forward(self, x):
#                         x = self.original_layer(x)
#                         if self.dropout.p > 0:  # 只有當 dropout_prob > 0 時才應用
#                             x = self.dropout(x)
#                         return x
                
#                 # 替換原始層
#                 if conv_dropout_prob > 0:
#                     wrapped_layer = LayerWithDropout(layer, conv_dropout_prob)
#                     setattr(new_model.backbone, layer_name, wrapped_layer)
#                     print(f"✅ Added dropout (p={conv_dropout_prob}) after {layer_name}")
#                 else:
#                     print(f"⏭️  Skipping dropout for {layer_name} (p=0)")
                
#             except AttributeError:
#                 print(f"❌ Layer {layer_name} not found")
        
#         # 在主要的 ResNet layers 後添加 dropout
#         if conv_dropout_prob > 0:
#             for layer_name in ['layer1', 'layer2', 'layer3', 'layer4']:
#                 add_dropout_after_layer(layer_name)
#         else:
#             print("⏭️  Skipping conv dropout (p=0)")
        
#         # 驗證 dropout 層
#         dropout_count = 0
#         for name, module in new_model.named_modules():
#             if isinstance(module, (nn.Dropout, nn.Dropout2d)):
#                 dropout_count += 1
#                 print(f"Found dropout layer: {name} with p={module.p}")
        
#         print(f"Total dropout layers: {dropout_count}")
        
#         return new_model
    
#     else:
#         raise NotImplementedError(f"Architecture '{architecture}' not supported")

def verify_weights_preserved(original_model, modified_model):
    """
    驗證修改後的模型是否保持了原始權重
    """
    print("Verifying weights preservation...")
    
    # 檢查主要層的權重
    layers_to_check = [
        'backbone.conv1.weight',
        'backbone.layer1.0.conv1.weight', 
        'backbone.layer4.1.conv2.weight'
    ]
    
    for layer_name in layers_to_check:
        try:
            original_param = original_model
            modified_param = modified_model
            
            # 遍歷層級
            for attr in layer_name.split('.'):
                original_param = getattr(original_param, attr)
                modified_param = getattr(modified_param, attr)
            
            # 比較權重
            if torch.equal(original_param.data, modified_param.data):
                print(f"✓ {layer_name}: weights preserved")
            else:
                print(f"✗ {layer_name}: weights changed!")
                
        except AttributeError:
            print(f"? {layer_name}: layer not found")
    
    # 檢查 fc layer 權重（需要特別處理）
    if hasattr(original_model.backbone, 'fc') and hasattr(modified_model.backbone, 'fc'):
        original_fc = original_model.backbone.fc
        modified_fc = modified_model.backbone.fc
        
        if isinstance(original_fc, nn.Linear) and isinstance(modified_fc, nn.Sequential):
            # 比較原始 Linear 和新 Sequential 中的 Linear
            if torch.equal(original_fc.weight.data, modified_fc[1].weight.data):
                print("✓ FC layer: weights preserved")
            else:
                print("✗ FC layer: weights changed!")

def mc_bald(model, data_dir, unlabel_data_idx, num_data_to_label, device, T):
    """
    Monte Carlo BALD (Bayesian Active Learning by Disagreement) 主動學習策略
    
    通過 Monte Carlo Dropout 估計模型參數的不確定性，選擇具有最高 BALD 分數的樣本。
    BALD 分數衡量的是模型間的分歧程度（epistemic uncertainty）。
    
    參數:
        model: 已訓練的分類模型（必須包含 dropout 層）
        data_dir: 數據目錄路徑
        unlabel_data_idx: 未標記數據的索引列表
        num_data_to_label: 需要選擇的樣本數量
        device: 計算設備 (cuda/cpu)
        T: Monte Carlo dropout 的推理次數
    
    返回:
        to_label_data_idx: 選中樣本在全數據集中的索引列表
    
    BALD 計算公式:
        BALD = H[E[p(y|x,ω)]] - E[H[p(y|x,ω)]]
        其中:
        - H[E[p(y|x,ω)]] 是平均預測的熵
        - E[H[p(y|x,ω)]] 是預測熵的平均
        - ω 表示模型參數的不同採樣
    """
    
    # 數據預處理（與 entropy 函數保持一致）
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    
    # 載入數據
    full_dataset = datasets.ImageFolder(os.path.join(data_dir, 'train'), transform=transform)
    unlabel_dataset = Subset(full_dataset, unlabel_data_idx)
    unlabel_loader = DataLoader(unlabel_dataset, batch_size=32, shuffle=False, num_workers=4)
    
    model.train()  # 設置為 train 模式
    model.to(device)
    
    print(f"Running MC BALD with T={T} cycles on {len(unlabel_data_idx)} samples...")
    
    # 檢查模型是否有 dropout 層
    dropout_layers = [m for m in model.modules() if isinstance(m, nn.Dropout)]
    if len(dropout_layers) == 0:
        print("Warning: No dropout layers found in the model!")
        print("MC BALD requires dropout layers for uncertainty estimation.")
    else:
        print(f"Found {len(dropout_layers)} dropout layers")
    
    # 收集所有 Monte Carlo 預測結果
    all_mc_predictions = []  # List of [N_unlabeled, num_classes] tensors
    
    for t in tqdm(range(T), desc="MC Dropout cycles"):
        cycle_probs = []
        with torch.no_grad():
            for inputs, _ in unlabel_loader:
                inputs = inputs.to(device)
                outputs = model(inputs)
                probs = torch.softmax(outputs, dim=1)
                cycle_probs.append(probs.cpu())
        
        # 連接所有批次的結果
        cycle_probs = torch.cat(cycle_probs, dim=0)  # [N_unlabeled, num_classes]
        all_mc_predictions.append(cycle_probs)
        
    # 將預測結果轉換為 numpy array: [N_unlabeled, num_classes, T]
    mc_predictions = torch.stack(all_mc_predictions, dim=-1)  # [N_unlabeled, num_classes, T]
    mc_predictions = mc_predictions.numpy()
    
    print(f"MC predictions shape: {mc_predictions.shape}")
    
    # 計算 BALD 分數
    bald_scores = calculate_bald_scores(mc_predictions)
    
    print(f"BALD scores - Mean: {np.mean(bald_scores):.4f}, Std: {np.std(bald_scores):.4f}")
    print(f"BALD scores - Min: {np.min(bald_scores):.4f}, Max: {np.max(bald_scores):.4f}")
    
    # 選擇 BALD 分數最高的樣本
    sorted_indices = np.argsort(bald_scores)[::-1][:num_data_to_label]
    
    # 映射回原始數據集索引
    to_label_data_idx = [unlabel_data_idx[i] for i in sorted_indices]
    
    # 輸出選中樣本的 BALD 分數
    selected_scores = bald_scores[sorted_indices]
    print(f"Selected {len(to_label_data_idx)} samples with BALD scores:")
    print(f"  Top 5 scores: {selected_scores[:5]}")
    print(f"  Score range: [{selected_scores[-1]:.4f}, {selected_scores[0]:.4f}]")
    
    return to_label_data_idx

def calculate_bald_scores(mc_predictions):
    """
    根據 Monte Carlo 預測計算 BALD 分數
    
    參數:
        mc_predictions: numpy array, shape [N_samples, num_classes, T]
                       其中 T 是 Monte Carlo 採樣次數
    
    返回:
        bald_scores: numpy array, shape [N_samples]
    """
    epsilon = 1e-10  # 避免 log(0)
    
    # 計算平均預測: E[p(y|x,ω)]
    mean_predictions = np.mean(mc_predictions, axis=-1)  # [N_samples, num_classes]
    
    # 第一項: H[E[p(y|x,ω)]] - 平均預測的熵
    entropy_of_mean = -np.sum(
        mean_predictions * np.log(mean_predictions + epsilon), 
        axis=1
    )  # [N_samples]
    
    # 第二項: E[H[p(y|x,ω)]] - 預測熵的平均
    # 先計算每次採樣的熵
    entropies = -np.sum(
        mc_predictions * np.log(mc_predictions + epsilon), 
        axis=1
    )  # [N_samples, T]
    
    # 再計算平均
    mean_of_entropies = np.mean(entropies, axis=1)  # [N_samples]
    
    # BALD 分數 = 互資訊 = 第一項 - 第二項
    bald_scores = entropy_of_mean - mean_of_entropies
    
    return bald_scores

def verify_mc_bald_setup(model):
    """
    驗證模型是否適合進行 MC BALD
    """
    print("Verifying MC BALD setup...")
    
    # 檢查 dropout 層
    dropout_layers, active_dropout_count = check_dropout_status(model)
    
    if len(dropout_layers) == 0:
        print("❌ No dropout layers found!")
        print("   MC BALD requires dropout layers for uncertainty estimation.")
        return False
    
    if active_dropout_count == 0:
        print("❌ Dropout layers found but none are active!")
        print("   MC BALD requires active dropout layers.")
        return False
    
    print(f"✅ Found {len(dropout_layers)} dropout layers, {active_dropout_count} active:")
    for name, layer in dropout_layers:
        status = "ACTIVE" if layer.training else "INACTIVE"
        print(f"   - {name}: p={layer.p} ({status})")
    
    print("✅ MC BALD setup verified successfully!")
    return True