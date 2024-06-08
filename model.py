import torch
import torch.nn as nn
import torch.nn.functional as F

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        
        # Encoder 
        self.enc1 = self.conv_block(3, 8)
        self.enc2 = self.conv_block(8, 16)
        self.enc3 = self.conv_block(16, 32)
        self.enc4 = self.conv_block(32, 64)
        self.enc5 = self.conv_block(64, 128)
        
        # Decoder
        self.up6 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec6 = self.conv_block(128, 64)
        self.up7 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec7 = self.conv_block(64, 32)
        self.up8 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
        self.dec8 = self.conv_block(32, 16)
        self.up9 = nn.ConvTranspose2d(16, 8, kernel_size=2, stride=2)
        self.dec9 = self.conv_block(16, 8)

        self.outc = nn.Conv2d(8, 1, kernel_size=1)
    
    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_channels),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_channels)
        )
    
    def forward(self, x):
        # Encoder
        c1 = self.enc1(x)
        p1 = F.max_pool2d(c1, kernel_size=2)
        c2 = self.enc2(p1)
        p2 = F.max_pool2d(c2, kernel_size=2)
        c3 = self.enc3(p2)
        p3 = F.max_pool2d(c3, kernel_size=2)
        c4 = self.enc4(p3)
        p4 = F.max_pool2d(c4, kernel_size=2)
        c5 = self.enc5(p4)
        
        # Decoder
        u6 = self.up6(c5)
        u6 = torch.cat([u6, c4], dim=1)
        c6 = self.dec6(u6)
        u7 = self.up7(c6)
        u7 = torch.cat([u7, c3], dim=1)
        c7 = self.dec7(u7)
        u8 = self.up8(c7)
        u8 = torch.cat([u8, c2], dim=1)
        c8 = self.dec8(u8)
        u9 = self.up9(c8)
        u9 = torch.cat([u9, c1], dim=1)
        c9 = self.dec9(u9)
        
        # outputs = torch.sigmoid(self.outc(c9))
        outputs = self.outc(c9) 
        return outputs

if __name__ == "__main__":
    model = UNet()
    print(model)
