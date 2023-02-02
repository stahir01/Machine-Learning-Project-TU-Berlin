import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF

class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels) -> None:
        super(DownBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding = 'same', padding_mode = 'reflect')
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding = 'same', padding_mode = 'reflect')
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))

        h = x.clone()

        x = F.max_pool2d(x, kernel_size=2, stride=2)

        return x, h

class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels) -> None:
        super(UpBlock, self).__init__()
        self.up_conv = nn.ConvTranspose2d(in_channels, out_channels, 2, 2)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding = 'same', padding_mode = 'reflect')
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding = 'same', padding_mode = 'reflect')
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x, h):
        x = self.up_conv(x)
        c = self.crop(h, h.shape[2], x.shape[2])
        x = torch.cat((c, x), dim=1)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))

        return x

    def crop(self, x, in_size, out_size):
        start = int((in_size - out_size)/2)
        stop  = int((in_size + out_size)/2)

        return x[:,:,start:stop,start:stop]


class NewUNet(nn.Module):
    def __init__(self, in_channel) -> None:
        super(NewUNet, self).__init__()
        self.down_block_1 = DownBlock(in_channels=in_channel, out_channels=64)
        self.down_block_2 = DownBlock(in_channels=64, out_channels=128)
        self.down_block_3 = DownBlock(in_channels=128, out_channels=256)
        self.down_block_4 = DownBlock(in_channels=256, out_channels=512)

        self.middle_conv_1 = nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, padding = 'same', padding_mode = 'reflect')
        self.middle_conv_2 = nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, padding = 'same', padding_mode = 'reflect')

        self.up_block_1 = UpBlock(in_channels=1024, out_channels=512)
        self.up_block_2 = UpBlock(in_channels=512, out_channels=256)
        self.up_block_3 = UpBlock(in_channels=256, out_channels=128)
        self.up_block_4 = UpBlock(in_channels=128, out_channels=64)

        self.out_conv = nn.Conv2d(in_channels=64, out_channels=2, kernel_size=1)

    def forward(self, x):
        x, h_1 = self.down_block_1(x)
        x, h_2 = self.down_block_2(x)
        x, h_3 = self.down_block_3(x)
        x, h_4 = self.down_block_4(x)

        x = self.middle_conv_1(x)
        x = self.middle_conv_2(x)

        x = self.up_block_1(x, h_4)
        x = self.up_block_2(x, h_3)
        x = self.up_block_3(x, h_2)
        x = self.up_block_4(x, h_1)

        x = self.out_conv(x)

        return x


def build_model():
    
    model = NewUNet()

    return model

if __name__ == '__main__':
    # model = build_model()
    model = NewUNet(1)

    X = torch.rand(1, 1, 512, 512)
    X = model(X)

    print(X.shape)