# LDA-inspired Domain Adaptation
**An Embarrassingly Simple Approach to Visual Domain Adaptation**

IEEE Transactions on Image Processing, 2018

By [Hao Lu](http://sites.google.com/site/poppinace)<sup>1</sup>, [Chunhua Shen](https://cs.adelaide.edu.au/~chhshen/)<sup>2</sup>, Zhiguo Cao<sup>1</sup>, Yang Xiao<sup>1</sup>, [Anton van den Hengel](https://cs.adelaide.edu.au/~hengel/)<sup>2</sup>
  
<sup>1</sup>Huazhong University of Science and Technology, China

<sup>2</sup>The University of Adelaide, Australia
  
### Introduction

This repository includes the implimentation of LDA-inspired Domain Adaptation (LDADA) proposed in our TIP paper. LDADA can achieve high-quality domain adaptation without explicit adaptation. It is conceptually simple, effective, robust, fast, parameter-free, and applicable to both unsupervised and semi-supervised DA setting.

**Prerequisites**
1. Matlab is required. This repository has been tested on 64-bit Mac OS X Matlab2016a and on 64-bit Window 10 Matlab2017a.
2. LibLinear toolbox at: https://www.csie.ntu.edu.tw/~cjlin/liblinear/. Please remember to install it following the instruction on the website, especially for Windows and Ubuntun users.

**Usage**
1. choose your options in the paramInit.m function (e.g., setting opt.dataset to specify a dataset or opt.nclasstrain to specify the expermental protocol.
2. run batchOfficeCaltech10.m/batchOffice.m/batchSatelliteScene5.m/batchMTFS3.m to reproduce the results of corresponding dataset.

### Citation

If you use our codes in your research, please cite:

	@inproceedings{hao2018ldada,
		author = {Hao Lu and Chunhua Shen and Zhiguo Cao and Yang Xiao and Anton van den Hengel},
		title = {An Embarrassingly Simple Approach to Visual Domain Adaptation},
		journal = {IEEE Transactions on Image Processing},
		year = {2018}
	}
  
