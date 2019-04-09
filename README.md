# isbi2019miccan

##MRI Reconstruction via Cascaded Channel-wise Attention Network
Qiaoying Huang, Dong Yang, Pengxiang Wu, Hui Qu, Jingru Yi, Dimitris Metaxas
* [Paper](https://arxiv.org/abs/1810.08229) is accepted by The IEEE International Symposium on Biomedical Imaging (ISBI) 2019.

###Prerequisites
python 3.6
Pytorch 0.4.1

###Quick start
Training: run main function with toy dataset
```
python main.py
```
Test: run main function with trained model
```
python main.py --train false
```

###Toy dataset
We create a toy dataset with 2D cardiac SAX data from *[ACDC Challenge](https://www.creatis.insa-lyon.fr/Challenge/acdc/). In the data folder, we have kdata(undersample k-space data), mask(undersampling mask, undersampling rate=0.875), fully(fully sampled k-space data) and image(fully sampled magnitude image). If you want to run your own dataset, just place the your data into the data folder.