import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import cv2
import pydicom
import numpy as np

# Import kiến trúc ResNet 
from mri_model import ResNetMRI

class BinaryBrainDataset(Dataset):
    def __init__(self, data_dir):
        self.file_paths = []
        self.labels = []

        normal_dir = os.path.join(data_dir, 'normal')
        if os.path.exists(normal_dir):
            for f in os.listdir(normal_dir):
                if f.lower().endswith(('.dcm', '.jpg', '.jpeg', '.png', '.ima')):
                    self.file_paths.append(os.path.join(normal_dir, f))
                    self.labels.append(0)

        abnormal_dir = os.path.join(data_dir, 'abnormal')
        if os.path.exists(abnormal_dir):
            for f in os.listdir(abnormal_dir):
                if f.lower().endswith(('.dcm', '.jpg', '.jpeg', '.png', '.ima')):
                    self.file_paths.append(os.path.join(abnormal_dir, f))
                    self.labels.append(1)

        print(f"📊 Đã nạp {self.labels.count(0)} Normal và {self.labels.count(1)} Abnormal.")

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        path = self.file_paths[idx]
        label = self.labels[idx]

        if path.lower().endswith(('.dcm', '.ima')):
            ds = pydicom.dcmread(path)
            image = ds.pixel_array.astype(np.float32)
        else:
            image = cv2.imread(path, cv2.IMREAD_GRAYSCALE).astype(np.float32)

        if image.ndim == 3:
            image = image[image.shape[0] // 2, :, :]

        # Chuẩn hóa nhanh bằng Numpy
        img_min = image.min()
        img_max = image.max()
        image = (image - img_min) / (img_max - img_min + 1e-8)
        
        # TỐI ƯU 1: Dùng cv2.resize siêu tốc thay vì skimage
        image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_LINEAR)

        image = (image - 0.5) / 0.5 
        tensor_img = torch.from_numpy(image).unsqueeze(0)

        return tensor_img, label

def train_smart_ai():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 KÍCH HOẠT CHẾ ĐỘ TURBO TRÊN: {device}")
    
    data_dir = '../data/mri_smart_train' 
    if not os.path.exists(data_dir):
        data_dir = 'data/mri_smart_train'
        
    dataset = BinaryBrainDataset(data_dir=data_dir)

    # TỐI ƯU 2 & 4: Tăng batch_size, bật num_workers và pin_memory
    # Nếu máy báo lỗi RAM, hãy hạ num_workers xuống 4 hoặc batch_size xuống 32
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=8, pin_memory=True)
    
    model = ResNetMRI(num_classes=2).to(device)
    
    normal_count = dataset.labels.count(0)
    abnormal_count = dataset.labels.count(1)
    if abnormal_count > 0:
        weight_normal = 1.0 / normal_count
        weight_abnormal = 1.0 / abnormal_count
        weights = torch.tensor([weight_normal, weight_abnormal]).float().to(device)
        criterion = nn.CrossEntropyLoss(weight=weights)
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    
    # GIẢM EPOCHS: Với 14.000 ảnh, 15 Epochs là quá đủ để hội tụ, không cần chạy 40 Epochs
    epochs = 15 
    best_loss = float('inf')
    
    # TỐI ƯU 3: Khởi tạo bộ khuếch đại cho Mixed Precision (AMP)
    scaler = torch.amp.GradScaler('cuda')
    
    os.makedirs('models', exist_ok=True)
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            
            optimizer.zero_grad()
            
            # Ép GPU chạy ở chuẩn 16-bit
            with torch.amp.autocast(device_type='cuda'):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            
            # Scale loss và cập nhật trọng số siêu tốc
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
        avg_loss = running_loss / len(dataloader)
        accuracy = 100 * correct / total
        print(f"Epoch [{epoch+1}/{epochs}] | Loss: {avg_loss:.4f} | Accuracy: {accuracy:.2f}%")
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), 'models/mri_model_best.pth')
            print("   ⚡ Đã lưu phiên bản AI thông minh nhất!")

if __name__ == "__main__":
    train_smart_ai()