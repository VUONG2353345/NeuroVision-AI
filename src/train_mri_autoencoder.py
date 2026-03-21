import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Import kiến trúc mạng và bộ nạp dữ liệu vừa viết
from mri_autoencoder import BrainAutoencoder
from mri_autoencoder_dataset import HealthyBrainDataset

def train_autoencoder():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Đang khởi động huấn luyện trên: {device}")
    
    data_dir = '../data/mri_normal' # Trỏ tới thư mục chứa ảnh vừa tải
    
    # Sửa lại đường dẫn nếu chạy từ thư mục gốc
    if not os.path.exists(data_dir):
        data_dir = 'data/mri_normal'
        
    if not os.path.exists(data_dir) or len(os.listdir(data_dir)) == 0:
        print(f"❌ LỖI: Thư mục {data_dir} đang trống hoặc không tồn tại.")
        print("Vui lòng copy ảnh từ Kaggle vào thư mục này trước khi chạy.")
        return

    # Khởi tạo DataLoader
    dataset = HealthyBrainDataset(data_dir=data_dir) 
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
    
    model = BrainAutoencoder().to(device)
    criterion = nn.MSELoss() # Hàm tính độ chênh lệch
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    epochs = 100 # Chạy 100 vòng để mô hình học thật kỹ cấu trúc não
    best_loss = float('inf')
    
    # Đảm bảo thư mục lưu model tồn tại
    os.makedirs('models', exist_ok=True)
    save_path = 'models/mri_autoencoder_best.pth'
    if not os.path.exists('src'): # Xử lý đường dẫn tương đối
        save_path = 'models/mri_autoencoder_best.pth'
        os.makedirs('models', exist_ok=True)
    
    print("Bắt đầu quá trình tự học (Unsupervised Learning)...")
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
        avg_loss = running_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{epochs}] | Loss tái tạo: {avg_loss:.6f}")
        
        # Lưu lại bản có độ sai số thấp nhất
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), save_path)
            print(f"   💾 Đã lưu mô hình tốt nhất! (Loss: {best_loss:.6f})")

if __name__ == "__main__":
    train_autoencoder()