import torch
import torch.nn as nn


class resnetblock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1):
        super(resnetblock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride)
        if in_channels != out_channels:
            self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
        self.activation = nn.LeakyReLU(0.2)

    def forward(self, x):
        residual = x
        x = self.activation(self.conv1(x))
        x = self.activation(self.conv2(x))
        if x.shape[1] != residual.shape[1]:
            residual = self.conv3(residual)
        x = x + residual
        return x


class generator(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.sz1 = 8
        # self.total_size = 65536
        '''self.pre_process = nn.Sequential(
            nn.Linear(self.total_size, 256 * self.sz1 * self.sz1),
            nn.ReLU()
        )'''
        self.encoder = pre_process_encoder()
        # self.pre_process = nn.Conv2d(1, 256, kernel_size=3, padding=1)
        self.up_sample = nn.Upsample(scale_factor=2)
        self.startup1 = resnetblock(256, 256)
        self.startup2 = resnetblock(256, 256)
        self.conv1 = nn.Sequential(
            resnetblock(256, 256, kernel_size=5, padding=2),
            nn.BatchNorm2d(256),
            # nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            resnetblock(256, 128, kernel_size=5, padding=2),
            nn.BatchNorm2d(128),
            # nn.ReLU()
        )
        self.conv3 = nn.Sequential(
            resnetblock(128, 128, kernel_size=5, padding=2),
            nn.BatchNorm2d(128),
            # nn.ReLU()
        )
        self.conv4 = nn.Sequential(
            resnetblock(128, 64, kernel_size=5, padding=2),
            nn.BatchNorm2d(64),
            # nn.ReLU()
        )
        self.conv5 = nn.Sequential(
            resnetblock(64, 64, kernel_size=5, padding=2),
            nn.BatchNorm2d(64),
            # nn.ReLU()
        )
        # self.conv6 = nn.ConvTranspose2d(64, 3, kernel_size=1)
        self.conv6 = nn.Sequential(
            resnetblock(64, 3, kernel_size=5, padding=2),
            # nn.BatchNorm2d(64),
            # nn.ReLU()
        )
        # self.final_activation = nn.LeakyReLU(0.2)

    def forward(self, sketch):
        '''
        主要的结构是：首先进行projection，然后用conv2d和up进行upsampling，中间的conv2d被resnet替换
        注意这里我没法运行所以形状是没有调的，需要自己调一下
        upsample
        resnet(256,256)
        resnet(256,256)
        upsample
        resnet(256,128)
        upsample
        resnet(128,128)
        upsample
        resnet(128,64)
        upsample
        resnet(64,64)
        leakyrelu
        tanh
        这是目前的格式，可能需要调整
        '''
        # sketch = torch.nn.functional.interpolate(sketch, (64, 64))
        # sketch = self.pre_process(sketch)
        '''preprocess_label = torch.zeros([label.shape[0],10]).to(label.device)
        label = torch.scatter(preprocess_label,1,label.unsqueeze(1),value = 1).to(label.device)
        total_label = torch.cat((z,label),dim = 1)
        #total label shape
        total_label = self.projection(total_label)
        print(total_label.shape)
        total_label = total_label.reshape(-1,256,self.sz1,self.sz1)'''
        sketch = self.encoder(sketch)
        # sketch = self.up_sample(sketch)
        sketch = self.startup1(sketch)
        sketch = self.startup2(sketch)
        # sketch = self.up_sample(sketch)
        x2 = self.conv1(sketch)
        x2 = self.up_sample(x2)
        x3 = self.conv2(x2)
        x3 = self.up_sample(x3)
        x4 = self.conv3(x3)
        x4 = self.up_sample(x4)
        x5 = self.conv4(x4)
        x5 = self.up_sample(x5)
        x6 = self.conv5(x5)
        x6 = self.up_sample(x6)
        x7 = self.conv6(x6)
        x8 = torch.tanh(x7)
        return x8


class pre_process_encoder(nn.Module):
    def __init__(self):
        super(pre_process_encoder, self).__init__()
        self.conv1 = resnetblock(1, 32)
        self.conv2 = resnetblock(32, 64)
        self.conv3 = resnetblock(64, 128)
        self.conv4 = resnetblock(128, 256)
        self.conv5 = resnetblock(256, 256)
        self.pool = nn.MaxPool2d(kernel_size=2)

    def forward(self, x):
        x = self.pool(self.conv1(x))
        x = self.pool(self.conv2(x))
        x = self.pool(self.conv3(x))
        x = self.pool(self.conv4(x))
        x = self.pool(self.conv5(x))
        return x


class discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2)
        )
        self.final_linear = nn.Linear(64 * 16 * 16, 1)
        self.maxpool = nn.MaxPool2d(kernel_size=2)

    def forward(self, img, sketch):
        input = torch.cat((img, sketch), dim=-3)
        x = self.conv1(input)
        x = self.maxpool(x)
        x = self.conv2(x)
        x = self.maxpool(x)
        x = self.conv3(x)
        x = self.maxpool(x)
        x = self.conv4(x)
        x = self.maxpool(x)
        x = self.final_linear(torch.flatten(x, start_dim=1))
        return torch.sigmoid(x)





