import torch
import torch.nn as nn
import torchvision.models as models

class ResNetMRI(nn.Module):
    def __init__(self, num_classes=2, model_name='resnet50', pretrained=True):
        super(ResNetMRI, self).__init__()
        if model_name == 'resnet50':
            self.backbone = models.resnet50(weights='IMAGENET1K_V1' if pretrained else None)
        elif model_name == 'resnet101':
            self.backbone = models.resnet101(weights='IMAGENET1K_V1' if pretrained else None)
        else:
            raise ValueError("Unsupported model")

        # Chuyển đổi lớp conv đầu tiên cho 1 kênh
        old_conv = self.backbone.conv1
        self.backbone.conv1 = nn.Conv2d(1, old_conv.out_channels,
                                         kernel_size=old_conv.kernel_size,
                                         stride=old_conv.stride,
                                         padding=old_conv.padding,
                                         bias=False)
        with torch.no_grad():
            self.backbone.conv1.weight.data = old_conv.weight.data.mean(dim=1, keepdim=True)

        # Lấy số đặc trưng trước lớp FC
        num_ftrs = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()  # bỏ lớp FC gốc

        # Thêm các lớp mới: Attention + Dropout + FC
        self.attention = nn.Sequential(
            nn.Linear(num_ftrs, 512),
            nn.Tanh(),
            nn.Linear(512, 1)
        )
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        features = self.backbone(x)  # (batch, num_ftrs)
        # Attention weights
        attn_weights = self.attention(features)  # (batch, 1)
        attn_weights = torch.softmax(attn_weights, dim=0) 
        
        out = self.dropout(features)
        out = self.fc(out)
        return out