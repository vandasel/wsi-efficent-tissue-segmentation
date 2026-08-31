import torch
import torch.nn as nn

class ResidualConvBlock(nn.Module):
    """Res, GroupNorm i opcjonalny Spatial Dropout."""
    def __init__(self, in_channels, out_channels, dropout_prob=0.0):
        super().__init__()
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.GroupNorm(8, out_channels)
        ) if in_channels != out_channels else nn.Identity()

        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, out_channels),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, out_channels)
        ]
        
        if dropout_prob > 0.0:
            layers.append(nn.Dropout2d(p=dropout_prob))

        self.conv_block = nn.Sequential(*layers)
        self.final_relu = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x):
        res = self.shortcut(x)
        x = self.conv_block(x)
        return self.final_relu(x + res) 

class AttentionGate2D(nn.Module):
    """Czyste, dwuwymiarowe Attention Gate"""
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=False),
            nn.GroupNorm(4, F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=False),
            nn.GroupNorm(4, F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=False),
            nn.GroupNorm(1, 1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi  

class ModernAttentionUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=4, features=[32, 64, 128, 256]):
        super().__init__()
        
        self.enc1 = ResidualConvBlock(in_channels, features[0])
        self.enc2 = ResidualConvBlock(features[0], features[1])
        self.enc3 = ResidualConvBlock(features[1], features[2])
        self.enc4 = ResidualConvBlock(features[2], features[3], dropout_prob=0.3)
        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        bottleneck_features = features[3] * 2
        self.bottleneck = ResidualConvBlock(features[3], bottleneck_features, dropout_prob=0.4)

        self.up4 = nn.ConvTranspose2d(bottleneck_features, features[3], kernel_size=2, stride=2)
        self.att4 = AttentionGate2D(F_g=features[3], F_l=features[3], F_int=features[2])
        self.dec4 = ResidualConvBlock(features[3] * 2, features[3], dropout_prob=0.3)

        self.up3 = nn.ConvTranspose2d(features[3], features[2], kernel_size=2, stride=2)
        self.att3 = AttentionGate2D(F_g=features[2], F_l=features[2], F_int=features[1])
        self.dec3 = ResidualConvBlock(features[2] * 2, features[2])

        self.up2 = nn.ConvTranspose2d(features[2], features[1], kernel_size=2, stride=2)
        self.att2 = AttentionGate2D(F_g=features[1], F_l=features[1], F_int=features[0])
        self.dec2 = ResidualConvBlock(features[1] * 2, features[1])

        self.up1 = nn.ConvTranspose2d(features[1], features[0], kernel_size=2, stride=2)
        self.dec1 = ResidualConvBlock(features[0] * 2, features[0])

        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        b = self.bottleneck(self.pool(e4))

        d4 = self.up4(b)
        x4 = self.att4(g=d4, x=e4)
        d4 = torch.cat((x4, d4), dim=1)
        d4 = self.dec4(d4)

        d3 = self.up3(d4)
        x3 = self.att3(g=d3, x=e3)
        d3 = torch.cat((x3, d3), dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        x2 = self.att2(g=d2, x=e2)
        d2 = torch.cat((x2, d2), dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat((e1, d1), dim=1)
        d1 = self.dec1(d1)

        return self.final_conv(d1)