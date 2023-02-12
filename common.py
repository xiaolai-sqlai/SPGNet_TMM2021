import torch
import torch.nn as nn


class IBN(nn.Module):
    def __init__(self, planes):
        super(IBN, self).__init__()
        half1 = int(planes/2)
        self.half = half1
        half2 = planes - half1
        self.IN = nn.InstanceNorm2d(half1, affine=True)
        self.BN = nn.BatchNorm2d(half2)
    
    def forward(self, x):
        split = torch.split(x, self.half, 1)
        out1 = self.IN(split[0].contiguous())
        out2 = self.BN(split[1].contiguous())
        out = torch.cat((out1, out2), 1)
        return out


class HardSigmoid(nn.Module):
    def forward(self, x):
        return torch.clamp((x + 1) / 2, min=0, max=1)

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class HardSwish(nn.Module):
    def forward(self, x):
        return x * torch.clamp((x + 1) / 2, min=0, max=1)

class Mish(nn.Module):
    def forward(self, x):
        return x * torch.tanh(nn.functional.softplus(x))

class Act(nn.Module):
    def __init__(self, out_planes=None, act_type="relu"):
        super(Act, self).__init__()

        self.act = None
        if act_type == "relu":
            self.act = nn.ReLU(inplace=True)
        elif act_type == "prelu":
            self.act = nn.PReLU(out_planes)
        elif act_type == "swish":
            self.act = Swish()
        elif act_type == "hardswish":
            self.act = HardSwish()
        elif act_type == "mish":
            self.act = Mish()

    def forward(self, x):
        if self.act is not None:
            x = self.act(x)
        return x


class ConvX(nn.Module):
    def __init__(self, in_planes, out_planes, groups=1, kernel_size=3, stride=1, act_type="relu", ibn=False):
        super(ConvX, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, groups=groups, padding=kernel_size//2, bias=False)
        if ibn:
            self.norm = IBN(out_planes)
        else:
            self.norm = nn.BatchNorm2d(out_planes)

        self.act = Act(out_planes, act_type)

    def forward(self, x):
        out = self.norm(self.conv(x))
        out = self.act(out)
        return out


class SqueezeExcitation(nn.Module):
    def __init__(self, inplanes, ratio=16, sigmoid="soft"):
        super(SqueezeExcitation, self).__init__()

        if ratio == 0:
            self.context = nn.AdaptiveAvgPool2d(1)
            self.fusion = nn.Conv2d(in_channels=inplanes, out_channels=inplanes, kernel_size=1, bias=True)
        elif ratio > 0:
            hidden_dim = inplanes // ratio
            self.context = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(in_channels=inplanes, out_channels=hidden_dim, kernel_size=1, bias=True),
                nn.ReLU(inplace=True)
            )
            self.fusion = nn.Conv2d(in_channels=hidden_dim, out_channels=inplanes, kernel_size=1, bias=True)

        self.sigmoid = None
        if sigmoid == "soft":
            self.sigmoid = nn.Sigmoid()
        elif sigmoid == "hard":
            self.sigmoid = HardSigmoid()
        
    def forward(self, x):
        out = self.context(x)
        out = self.fusion(out)
        if self.sigmoid is not None:
            out = self.sigmoid(out)
        return x * out


def init_params(model):
    for name, m in model.named_modules():
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