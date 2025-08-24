QuPeD:Quantized Personalization via Distillation with Applications to Federated Learning
=======================================================================

**Authors: Kaan Ozkara, Navjot Singh, Deepesh Data, Suhas Diggavi**


This code implements quantization aware training at [QuPeD:Quantized Personalization via Distillation with Applications to Federated Learning](https://proceedings.neurips.cc/paper/2021/hash/1dba3025b159cd9354da65e2d0436a31-Abstract.html). This corresponds to Algorithm 1 of the paper. 


Dependencies
------------
Python 			          3.8
cudatoolkit               10.1.243
numpy                     1.18.5
pytorch                   1.5.1
scipy                     1.5.0
torchvision               0.6.1


Directory structure
-------------------
The codes folder consists of the following files:

- utils.py: Includes functions that are utilized in main scripts. It includes a function to simulate proximal functions, center initialization function and a quantization function.

- main_centralized.py: This is the main function to run the centralized case in the paper (Algorithm 1 in paper)

- resnet.py: Implementation of ResNet's, this file is from https://github.com/akamaster/pytorch_resnet_cifar10/

- two .pt files for pretrained ResNet networks.

Running code
------------

To run our code the main script is:


- main_centralized.py:
	Parameters between lines 101-109 can be adjusted to simulate different settings. Line 91 also allows to choose type of model (between ResNet-20 or -32), and on line 92 number of centers can be chosen.


* Please see the comments in the code and the paper for detailed information about hyperparameters. 


Output
------
The metrics we observe are training loss and top1 test accuracy (logged in by epoch), and the best performing model.
The output of the code is a series of text files dumped in the Results folders.

Citing
------
If you use this work, please consider citing
```
@inproceedings{ozkara2021,
 author = {Ozkara, Kaan and Singh, Navjot and Data, Deepesh and Diggavi, Suhas},
 booktitle = {Advances in Neural Information Processing Systems},
 title = {QuPeD: Quantized Personalization via Distillation with Applications to Federated Learning},
 url = {https://proceedings.neurips.cc/paper_files/paper/2021/file/1dba3025b159cd9354da65e2d0436a31-Paper.pdf},
 year = {2021}
}
 ```

