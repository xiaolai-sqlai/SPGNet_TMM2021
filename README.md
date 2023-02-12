# SPGNet_TMM2021
```
from network.spgnet import SPGNet
version = "s2p6"
model = SPGNet(version=version)
model.load_state_dict(torch.load("network/spgnet_{}.tar".format(version), "cpu")["state_dict"])
```

### Citation
If you find this project useful for your research, please use the following BibTeX entry.

```
@article{DBLP:journals/tmm/WangLCZQ22,
  author    = {Xuan Wang and
               Shenqi Lai and
               Zhenhua Chai and
               Xingjun Zhang and
               Xueming Qian},
  title     = {SPGNet: Serial and Parallel Group Network},
  journal   = {{IEEE} Trans. Multim.},
  volume    = {24},
  pages     = {2804--2814},
  year      = {2022},
  url       = {https://doi.org/10.1109/TMM.2021.3088639},
  doi       = {10.1109/TMM.2021.3088639}
}
```
