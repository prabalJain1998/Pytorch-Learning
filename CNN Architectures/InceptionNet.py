import torch
import torch.nn as nn


class GoogleNet(nn.Module):
    def __init__(self, num_classes=1000):
        super(GoogleNet, self).__init__()
        self.conv1 = Conv_Block(in_channels=3, out_channels=64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3))
        self.maxpool1 = nn.MaxPool2d(kernel_size=(3, 3), stride=2, padding=1)
        self.conv2 = Conv_Block(in_channels=64, out_channels=192, kernel_size=3, stride=1, padding=1)
        self.maxpool2 = nn.MaxPool2d(kernel_size=(3, 3), stride=2, padding=1)

        self.inception3a = InceptionBlock(192, 64, 96, 128, 16, 32, 32)
        self.inception3b = InceptionBlock(256, 128, 128, 192, 32, 96, 64)
        self.maxpool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.inception4a = InceptionBlock(480, 192, 96, 208, 16, 48, 64)
        self.inception4b = InceptionBlock(512, 160, 112, 224, 24, 64, 64)
        self.inception4c = InceptionBlock(512, 128, 128, 256, 24, 64, 64)
        self.inception4d = InceptionBlock(512, 112, 144, 288, 32, 64, 64)
        self.inception4e = InceptionBlock(528, 256, 160, 320, 32, 128, 128)
        self.maxpool4 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.inception5a = InceptionBlock(832, 256, 160, 320, 32, 128, 128)
        self.inception5b = InceptionBlock(832, 384, 192, 384, 48, 128, 128)

        self.avgpool = nn.AvgPool2d(kernel_size=7, stride=1)
        self.dropout = nn.Dropout(p=0.4)
        self.fc1 = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.maxpool2(x)

        x = self.maxpool3(self.inception3b(self.inception3a(x)))

        x = self.maxpool4(self.inception4e(self.inception4d(self.inception4c(self.inception4b(self.inception4a(x))))))

        x = self.avgpool(self.inception5b(self.inception5a(x)))

        x = x.reshape(x.shape[0], -1)
        x = self.dropout(x)
        x = self.fc1(x)

        return x


class InceptionBlock(nn.Module):
    def __init__(self, in_channels, out_1x1, red_3x3, out_3x3, red_5x5, out_5x5, out1x1_pool):
        super(InceptionBlock, self).__init__()

        self.branch_1 = Conv_Block(in_channels, out_1x1, kernel_size=(1,1))
        self.branch_2 = nn.Sequential(
            Conv_Block(in_channels, red_3x3, kernel_size=1),
            Conv_Block(red_3x3, out_3x3, kernel_size=(3, 3), padding=1)
        )

        self.branch_3 = nn.Sequential(
            Conv_Block(in_channels, red_5x5, kernel_size=1),
            Conv_Block(red_5x5, out_5x5, kernel_size=(5, 5), padding=2)
        )

        self.branch_4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            Conv_Block(in_channels, out1x1_pool, kernel_size=(1, 1))
        )

    def forward(self, x):
        # X shape = N xChannels x 224 x 224
        return torch.cat(
            [self.branch_1(x), self.branch_2(x), self.branch_3(x), self.branch_4(x)], 1
        )


class Conv_Block(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(Conv_Block, self).__init__()
        self.relu = nn.ReLU()
        self.conv = nn.Conv2d(in_channels, out_channels, **kwargs)
        self.batchNorm = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        return self.relu(self.batchNorm(self.conv(x)))


if __name__ == "__main__":
    x = torch.randn(3,3,224,224)
    model = GoogleNet(num_classes=1000)
    print(model(x).shape)
