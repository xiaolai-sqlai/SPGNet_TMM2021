import torch
import torch.nn as nn


class ConvX(nn.Module):
    def __init__(self, in_planes, out_planes, groups, kernel_size=3, stride=1, act="relu"):
        super(ConvX, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, groups=groups, padding=kernel_size//2, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)

        self.act = None
        if act == "relu":
            self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.bn(self.conv(x))
        if self.act != None:
            out = self.act(out)
        return out


class SqueezeExcitation(nn.Module):
    def __init__(self, inplanes, se_ratio=0.25):
        super(SqueezeExcitation, self).__init__()
        hidden_dim = int(inplanes * se_ratio)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv2d(in_channels=inplanes, out_channels=hidden_dim, kernel_size=1, bias=False)
        self.conv2 = nn.Conv2d(in_channels=hidden_dim, out_channels=inplanes, kernel_size=1, bias=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.avg_pool(x)
        out = self.conv1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = torch.sigmoid(out)
        return x * out


class SPGModule(nn.Module):
    def __init__(self, in_planes, out_planes, kernel=3, stride=1, reduce_num=2, group_num=8, use_se=False):
        super(SPGModule, self).__init__()
        self.use_se = use_se
        if self.use_se:
            self.se = SqueezeExcitation(out_planes * 2)

        self.conv_in = ConvX(in_planes, int(out_planes*0.5), groups=1, kernel_size=1, stride=1, act="relu")
        self.conv1 = ConvX(int(out_planes*0.5), int(out_planes*0.5), groups=group_num, kernel_size=kernel, stride=stride, act="relu")
        self.conv2 = ConvX(int(out_planes*0.5), int(out_planes*0.5), groups=group_num, kernel_size=kernel, stride=1, act="relu")
        self.conv_out = ConvX(int(out_planes*1.0), out_planes, groups=1, kernel_size=1, stride=1, act=None)

        self.act = nn.ReLU(inplace=True)
        # self.act = nn.PReLU()

        self.stride = stride
        self.skip = None
        if stride == 1 and in_planes != out_planes:
            self.skip = nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_planes)
            )

        if stride == 2 and in_planes != out_planes:
            self.skip = nn.Sequential(
                nn.Conv2d(in_planes, in_planes, kernel_size=3, stride=2, padding=1, groups=in_planes, bias=False),
                nn.BatchNorm2d(in_planes),
                nn.Conv2d(in_planes, out_planes, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_planes)
            )


    def forward(self, x):
        skip = x
        out = self.conv_in(x)
        out1 = self.conv1(out)
        out2 = self.conv2(out1)
        out_cat = torch.cat((out1, out2), dim=1)

        if self.use_se:
            out_cat = self.se(out_cat)
        out = self.conv_out(out_cat)

        if self.skip is not None:
            skip = self.skip(skip)
        out += skip
        return self.act(out)


class SPGNet(nn.Module):
    # (out_planes, num_blocks, stride, group_num, kernel)
    cfgs = {
        "s2p6": [12, 24, 
            (96 , 4, 2, 6, 3),
            (192, 8, 2, 6, 3),
            (384, 3, 2, 6, 3)],
        "s2p7": [14, 28, 
            (112, 4, 2, 7, 3),
            (224, 8, 2, 7, 3),
            (448, 3, 2, 7, 3)],
        "s2p8": [16, 32, 
            (128, 4, 2, 8, 3),
            (256, 8, 2, 8, 3),
            (512, 3, 2, 8, 3)],
        "s2p9": [18, 36, 
            (144, 4, 2, 9, 3),
            (288, 8, 2, 9, 3),
            (576, 3, 2, 9, 3)]
    }

    def __init__(self, num_classes=1000, dropout=0.2, version="s2p6"):
        super(SPGNet, self).__init__()
        cfg = self.cfgs[version]

        self.first_conv = nn.Sequential(
            ConvX(3 , cfg[0], 1, 3, 2, "relu"),
            ConvX(cfg[0], cfg[1], 1, 3, 2, "relu")
        )

        self.layers = self._make_layers(in_planes=cfg[1], cfg=cfg[2:])

        self.conv_last = ConvX(cfg[4][0], 1024, 1, 1, 1, "relu")

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(1024, 1024, bias=False)
        self.bn = nn.BatchNorm1d(1024)
        self.relu = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(p=dropout)
        self.linear = nn.Linear(1024, num_classes, bias=False)
        self.init_params()

    def init_params(self):
        for name, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                if 'first' in name:
                    nn.init.normal_(m.weight, 0, 0.01)
                else:
                    nn.init.normal_(m.weight, 0, 1.0 / m.weight.shape[1])
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                if name.endswith("conv_out.bn.weight"):
                    nn.init.constant_(m.weight, 0)
                else:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0001)
                nn.init.constant_(m.running_mean, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0001)
                nn.init.constant_(m.running_mean, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _make_layers(self, in_planes, cfg):
        layers = []
        for out_planes, num_blocks, stride, group_num, kernel in cfg:
            strides = [stride] + [1] * (num_blocks - 1)
            for stride in strides:
                layers.append(SPGModule(in_planes, out_planes, kernel, stride, reduce_num=2, group_num=group_num))
                in_planes = out_planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.first_conv(x)
        out = self.layers(out)
        out = self.conv_last(out)
        out = self.gap(out).flatten(1)
        out = self.relu(self.bn(self.fc(out)))
        out = self.drop(out)
        out = self.linear(out)
        return out
