import torch
from thop import profile, clever_format
from network.spgnet import SPGNet

input = torch.randn(1, 3, 224, 224)

version = "s2p6"
model = SPGNet(version=version)
model.load_state_dict(torch.load("network/spgnet_{}.tar".format(version), "cpu")["state_dict"])
print(model)
model.eval()      # Don't forget to call this before inference.



macs, params = profile(model, inputs=(input, ))
macs, params = clever_format([macs, params], "%.3f")

print('Flops:  ', macs)
print('Params: ', params)
