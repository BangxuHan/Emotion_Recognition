import torch
from torch import nn


class Block(nn.Module):
    def __init__(self, p_c, i, t, c, n, s):
        super(Block, self).__init__()
        # 每个重复的最后一次负责下采样和通道处理，所以i=n-1的时候进行操作
        self.i = i
        self.n = n

        _s = s if i == n - 1 else 1  # 判断是否是最后一次重复，最后一次步长为2
        _c = c if i == n - 1 else p_c  # 判断是否是最后一次重复，最后一次负责将通道变换为下层的输入

        _p_c = p_c * t  # 输入通道的扩增倍数
        self.layer = nn.Sequential(
            nn.Conv2d(p_c, _p_c, 1, 1, bias=False),
            nn.BatchNorm2d(_p_c),
            nn.ReLU6(),
            nn.Conv2d(_p_c, _p_c, 3, _s, padding=1, groups=_p_c, bias=False),
            nn.BatchNorm2d(_p_c),
            nn.ReLU6(),
            nn.Conv2d(_p_c, _c, 1, 1, bias=False),
            nn.BatchNorm2d(_c)
        )

    def forward(self, x):
        if self.i == self.n - 1:
            return self.layer(x)
        else:
            return self.layer(x) + x


class MobileNet_v2(nn.Module):
    def __init__(self):
        super(MobileNet_v2, self).__init__()
        self.config = [
                        [-1, 32, 1, 2],
                        [1, 16, 1, 1],
                        [6, 24, 2, 2],
                        [6, 32, 3, 2],
                        [6, 64, 4, 2],
                        [6, 96, 3, 1],
                        [6, 160, 3, 2],
                        [6, 320, 1, 1]
        ]
        self.input_layer = nn.Sequential(
            nn.Conv2d(1, 32, 3, 2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU6()
        )

        self.blocks = []
        p_c = self.config[0][1]
        for t, c, n, s in self.config[1:]:
            for i in range(n):
                self.blocks.append(Block(p_c, i, t, c, n, s))
            p_c = c
        self.hidden_layer = nn.Sequential(*self.blocks)  # 星号代表序列（可变参数）

        self.output_layer = nn.Sequential(
            nn.Conv2d(320, 1280, 1, 1, bias=False),
            nn.BatchNorm2d(1280),
            nn.ReLU6(),
            nn.AvgPool2d(1, 1),
            nn.Conv2d(1280, 7, 2, 1),
            # nn.BatchNorm2d(7),  # 一般最后一层不加BatchNormal
            nn.Flatten()
        )

    def forward(self, x):
        h = self.input_layer(x)
        h = self.hidden_layer(h)
        h = self.output_layer(h)
        return h


if __name__ == '__main__':
    net = MobileNet_v2()
    x = torch.randn(1024, 1, 48, 48)
    y = net(x)
    print(x.shape)
    print(y.shape)
