# GNF-torch
 Pytorch-Implementation of Graph Normalizing Flows


GNF-part adapted from: [GNF](https://github.com/johncava/GNF-pytorch/blob/master/README.md)  (Pulled Request)

Training session adapted from: [GVAE](https://github.com/DaehanKim/vgae_pytorch) 


Related papers:
1. Variational Graph Auto-Encoders
```
@misc{kipf2016variational,
      title={Variational Graph Auto-Encoders},
      author={Thomas N. Kipf and Max Welling},
      year={2016},
      eprint={1611.07308},
      archivePrefix={arXiv},
      primaryClass={stat.ML}
}
```

2. Graph Normalizing Flows
```
@article{liu2019graph,
  title={Graph normalizing flows},
  author={Liu, Jenny and Kumar, Aviral and Ba, Jimmy and Kiros, Jamie and Swersky, Kevin},
  journal={Advances in Neural Information Processing Systems},
  volume={32},
  year={2019}
}
```

How to run:
1. run GNF
```python
python gnf.py --epochs 10 --lr 0.015 --dataset PubMed
```
2. run GVAE
```python
python train.py --load_config 'config/vgae.yaml'
```

Currently there's some bugs in the code that the AUC/ROC doesnt change. Also, QM9 needs to be included as mentioned in the paper as well. 