import os
import torch
from torch.utils.data import Dataset
import pydicom
import numpy as np
import cv2
from skimage.transform import resize

class HealthyBrainDataset(Dataset):
    def __init__(self, data_dir):
        # Lấy tất cả đường dẫn file ảnh (Hỗ trợ cả DICOM y tế và JPG/PNG Kaggle)
        self.file_paths = []
        for f in os.listdir(data_dir):
            if f.lower().endswith(('.dcm', '.jpg', '.jpeg', '.png', '.ima')):
                self.file_paths.append(os.path.join(data_dir, f))
                
        print(f"Đã tìm thấy {len(self.file_paths)} ảnh não khỏe mạnh để huấn luyện.")

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        path = self.file_paths[idx]
        
        # Đọc ảnh tùy theo định dạng
        if path.lower().endswith(('.dcm', '.ima')):
            ds = pydicom.dcmread(path)
            image = ds.pixel_array.astype(np.float32)
        else:
            # Đọc ảnh JPG/PNG dưới dạng ảnh xám (Grayscale)
            image = cv2.imread(path, cv2.IMREAD_GRAYSCALE).astype(np.float32)

        # Xử lý nếu là ảnh 3D
        if image.ndim == 3:
            image = image[image.shape[0] // 2, :, :]

        # Chuẩn hóa giá trị pixel về dải [0, 1] để mô hình dễ học
        image = (image - image.min()) / (image.max() - image.min() + 1e-8)
        
        # Đưa tất cả về một kích thước chuẩn 224x224
        image = resize(image, (224, 224), preserve_range=True).astype(np.float32)
        
        # Chuyển thành PyTorch Tensor (thêm chiều kênh: [1, 224, 224])
        tensor_img = torch.from_numpy(image).unsqueeze(0)
        
        # Autoencoder tự học trên chính nó: Input = Target
        return tensor_img, tensor_img