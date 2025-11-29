import time
import torch.nn.functional as F
import torch
from tqdm import tqdm
import torch.nn as nn
from torchvision import models

def get_resnet18_classifier(num_classes: int, pretrained: bool = True) -> nn.Module:
    if pretrained == True:
        model = models.resnet18(weights='IMAGENET1K_V1')
    else:
        model = models.resnet18()
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model

def eval_model(model, device, data_loader, dataset_size, criterion):
    model.to(device)
    model.eval()

    running_loss = 0.0
    running_corrects = 0

    with torch.no_grad():
        for inputs, labels in tqdm(data_loader, desc="Evaluating"):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

    epoch_loss = running_loss / dataset_size
    epoch_acc  = running_corrects.double() / dataset_size

    return epoch_loss, epoch_acc.item()


def train_model(model, device, data_loaders, dataset_sizes, criterion, optimizer, scheduler, num_epochs=20):
    model.to(device)
    since = time.time()

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                print('train...')
                model.train()
            else:
                print('validate...')
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in tqdm(data_loaders[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # 紀錄最後一輪 val acc（僅當 phase == 'val'）
            if phase == 'val':
                last_val_acc = epoch_acc.item()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Final val Acc: {:4f}'.format(last_val_acc))

    return model, last_val_acc


def train_model_w_confusion(model, device, data_loaders, dataset_sizes, criterion, optimizer, scheduler, num_epochs=20):
    """
    訓練模型並返回最終驗證準確率和confusion matrix
    
    返回:
        model: 訓練後的模型
        last_val_acc: 最終驗證準確率
        confusion_info: 字典包含confusion matrix相關信息
            - 'matrix': confusion matrix (numpy array)
            - 'class_names': 類別名稱列表（按指定順序排列）
            - 'description': 說明文字
    """
    model.to(device)
    since = time.time()
    
    # 指定的類別順序
    expected_class_order = ['Normal', 'Eczema', 'Psoriasis', 'Nevus', 'Seborrhoeic keratosis', 'Solar lentigo', 'Vitiligo']
    
    # 獲取原始類別名稱
    if hasattr(data_loaders['val'].dataset, 'classes'):
        original_class_names = data_loaders['val'].dataset.classes
    elif hasattr(data_loaders['val'].dataset, 'dataset'):
        # 如果是Subset，需要從原始dataset獲取
        original_class_names = data_loaders['val'].dataset.dataset.classes
    else:
        raise ValueError("Cannot extract class names from dataset")
    
    # 檢查類別名稱是否完全匹配
    original_class_set = set(original_class_names)
    expected_class_set = set(expected_class_order)
    
    if original_class_set != expected_class_set:
        missing_from_original = expected_class_set - original_class_set
        extra_in_original = original_class_set - expected_class_set
        
        error_msg = f"Class names mismatch!\n"
        error_msg += f"Expected: {expected_class_order}\n"
        error_msg += f"Found: {list(original_class_names)}\n"
        
        if missing_from_original:
            error_msg += f"Missing from dataset: {list(missing_from_original)}\n"
        if extra_in_original:
            error_msg += f"Extra in dataset: {list(extra_in_original)}\n"
        
        raise ValueError(error_msg)
    
    # 使用指定的類別順序
    class_names = expected_class_order
    num_classes = len(class_names)
    
    # 創建原始類別索引到新順序索引的映射
    # original_class_names 中的索引 -> expected_class_order 中的索引
    original_to_custom_mapping = {}
    for orig_idx, class_name in enumerate(original_class_names):
        custom_idx = expected_class_order.index(class_name)
        original_to_custom_mapping[orig_idx] = custom_idx
    
    print(f"Class order mapping: {original_to_custom_mapping}")
    print(f"Final class order: {class_names}")

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                print('train...')
                model.train()
            else:
                print('validate...')
                model.eval()

            running_loss = 0.0
            running_corrects = 0
            
            # 在最後一個epoch的驗證階段收集預測結果
            if epoch == num_epochs - 1 and phase == 'val':
                all_preds = []
                all_labels = []

            for inputs, labels in tqdm(data_loaders[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                
                # 在最後一個epoch的驗證階段收集數據
                if epoch == num_epochs - 1 and phase == 'val':
                    # 將原始標籤和預測轉換為自定義順序
                    custom_preds = [original_to_custom_mapping[p] for p in preds.cpu().numpy()]
                    custom_labels = [original_to_custom_mapping[l] for l in labels.cpu().numpy()]
                    
                    all_preds.extend(custom_preds)
                    all_labels.extend(custom_labels)

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # 紀錄最後一輪 val acc（僅當 phase == 'val'）
            if phase == 'val':
                last_val_acc = epoch_acc.item()
                
                # 在最後一個epoch計算confusion matrix
                if epoch == num_epochs - 1:
                    # 創建confusion matrix (使用自定義順序)
                    confusion_matrix = np.zeros((num_classes, num_classes), dtype=int)
                    
                    for true_label, pred_label in zip(all_labels, all_preds):
                        confusion_matrix[pred_label, true_label] += 1
                    
                    # 準備返回的confusion matrix信息
                    confusion_info = {
                        'matrix': confusion_matrix,
                        'class_names': class_names,
                        'description': f'Confusion Matrix: rows=predicted, columns=ground_truth. Class order: {class_names}. Each column sums to the number of samples in that true class.'
                    }
                    
                    # 打印confusion matrix信息
                    print(f"\nFinal Validation Confusion Matrix:")
                    print(f"Class order: {class_names}")
                    print(f"Matrix shape: {confusion_matrix.shape}")
                    print(f"Matrix (rows=predicted, cols=ground_truth):")
                    print(confusion_matrix)
                    
                    # 驗證每列的和
                    print(f"\nColumn sums (number of samples per true class):")
                    for i, class_name in enumerate(class_names):
                        print(f"{class_name}: {confusion_matrix[:, i].sum()}")

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Final val Acc: {:4f}'.format(last_val_acc))

    return model, last_val_acc, confusion_info, tsne_map

    
def train_model_w_sam(model, device, data_loaders, dataset_sizes, criterion, optimizer, scheduler, num_epochs=20):
    model.to(device)
    since = time.time()

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                print('train...')
                model.train()
            else:
                print('validate...')
                model.eval()

            running_loss = 0.0
            running_corrects = 0
            
            for inputs, labels in tqdm(data_loaders[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        def closure():
                            outputs = model(inputs)
                            _, preds = torch.max(outputs, 1)
                            loss = criterion(outputs, labels)
                            loss.backward()
                            return loss                
                        optimizer.step(closure)
    
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # 紀錄最後一輪 val acc（僅當 phase == 'val'）
            if phase == 'val':
                last_val_acc = epoch_acc.item()
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Final val Acc: {:4f}'.format(last_val_acc))

    return model, last_val_acc


def train_model_w_trustAL(model, device, data_loaders, dataset_sizes, criterion, optimizer, scheduler, num_epochs, teacher_model, alpha):
    # prepare teacher
    teacher_model.to(device)
    teacher_model.eval()
    model.to(device)

    since = time.time()
    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs-1} with teacher loss, alpha = {alpha}')
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                print('train...')
                model.train()
            else:
                print('validate...')
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in tqdm(data_loaders[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                # forward pass
                with torch.set_grad_enabled(phase == 'train'):
                    # student outputs
                    student_logits = model(inputs)
                    _, preds = torch.max(student_logits, dim=1)

                    # standard CE loss
                    loss_ce = criterion(student_logits, labels)

                    if phase == 'train':
                        # teacher outputs (no grad)
                        with torch.no_grad():
                            teacher_logits = teacher_model(inputs)

                        # KL divergence: D_KL( softmax(teacher) || softmax(student) )
                        loss_kl = F.kl_div(
                            F.log_softmax(student_logits, dim=1),
                            F.softmax(teacher_logits, dim=1),
                            reduction='batchmean'
                        )

                        # total loss: L_CE + alpha * L_KL
                        loss = loss_ce + alpha * loss_kl
                        # loss = (loss_ce + alpha * loss_kl) / (1.0 + alpha) # normalize, since we use the same learning rate across AL strategy

                        # backward + step
                        loss.backward()
                        optimizer.step()
                    else:
                        # validation: just CE
                        loss = loss_ce

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            if phase == 'val':
                last_val_acc = epoch_acc.item()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed//60:.0f}m {time_elapsed%60:.0f}s')
    print(f'Final val Acc: {last_val_acc:.4f}')

    return model, last_val_acc


# def train_model(model, device, data_loaders, dataset_sizes, criterion, optimizer, scheduler, num_epochs=20):
#     model.to(device)
#     since = time.time()
#     best_model_wts = copy.deepcopy(model.state_dict())
#     best_acc = 0.0
#     # softmax = nn.Softmax(dim=1)
#     for epoch in range(num_epochs):
#         print('Epoch {}/{}'.format(epoch, num_epochs - 1))
#         print('-' * 10)

#         # Each epoch has a training and validation phase
#         for phase in ['train', 'val']:
#             if phase == 'train':
#                 print('train...')
#                 model.train()  # Set model to training mode
#             else:
#                 print('validate...')
#                 model.eval()   # Set model to evaluate mode

#             running_loss = 0.0
#             running_corrects = 0

#             # Iterate over data.
#             for inputs, labels in tqdm(data_loaders[phase]):
#                 inputs = inputs.to(device)
#                 labels = labels.to(device)

#                 # zero the parameter gradients
#                 optimizer.zero_grad()

#                 # forward
#                 # track history if only in train
#                 with torch.set_grad_enabled(phase == 'train'):
#                     outputs = model(inputs)
#                     #outputs = softmax(outputs).to(dtype = torch.float)
#                     _, preds = torch.max(outputs, 1)
#                     loss = criterion(outputs, labels)
                    
#                     # backward + optimize only if in training phase
#                     if phase == 'train':
#                         loss.backward()
#                         optimizer.step()

#                 # statistics
#                 running_loss += loss.item() * inputs.size(0)
#                 running_corrects += torch.sum(preds == labels.data)
#             if phase == 'train':
#                 scheduler.step()
#             epoch_loss = running_loss / dataset_sizes[phase]
#             epoch_acc = running_corrects.double() / dataset_sizes[phase]
#             if(phase=='train'):    
#                 train_epoch_loss = epoch_loss
#                 train_epoch_acc = epoch_acc
#             else:
#                 valid_epoch_loss = epoch_loss
#                 valid_epoch_acc = epoch_acc

#             print('{} Loss: {:.4f} Acc: {:.4f}'.format(
#                 phase, epoch_loss, epoch_acc))
            
#             # deep copy the model
#             if phase == 'val' and epoch_acc > best_acc:
#                 best_acc = epoch_acc
#                 best_model_wts = copy.deepcopy(model.state_dict())
#         # print('Train Loss: {:.4f} Acc: {:.4f} | Val Loss: {:.4f} Acc: {:.4f}'.format(
#         # train_epoch_loss, train_epoch_acc, valid_epoch_loss, valid_epoch_acc))

#     time_elapsed = time.time() - since
#     print('Training complete in {:.0f}m {:.0f}s'.format(
#         time_elapsed // 60, time_elapsed % 60))
#     print('Best val Acc: {:4f}'.format(best_acc))

#     # load best model weights
#     model.load_state_dict(best_model_wts)

#     return model, best_acc.item()