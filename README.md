# DNN learning

DNN Code including classification,landmark,text recognition...  
Training and testing script of each DNN project is included.

## Classification:
Classification models are trained and evaluated on the task of character classification including 36 characters(10 digits and 26 uppercase letter).  
input: 28 x 28 x 3  
environment: openvino 2019  、CORE i7、GTX 1060

Names of Classfier are just the general type of CNN net,readers can go into codes to find specific type.
Readers can fine-tuning the network architecture for your own tasks.
(The accuarcy and speed of every classifier is only for your reference,  
it may floats because of training steps 、learning rate and any other hyper parameter.)

|   Classifier   | Accuarcy (%)|  Speed (ms)|
|:------------:|:-------------------:|:-------------------:|
| my classification    |        99.6        |  1.62|
|    MobileNetV1   |        99.3        |  0.63|
|    MobileNetV2   |        99.7        |  1.12|
|   ResNetV1    |     99.8    |4.22|
|   ResNetV2    |     99.2    |4.03|
|  ShuffleNetV1  |     92.7    |    1.52|
|  ShuffleNetV2  |     97.7    |    1.75|
|  ResNeXt(C=1)  |     99.0    |    1.02|
|  ResNeXt(C=32)  |     97.8    |    12.80|
|  DenseNet  |     98.4    |    5.40|
|  DenseNet(My)  |     99.3    |    3.40|
|  WideResNet    |     98.7    |    4.50|
|  SqueezeNet    |     98.8    |    0.67|
|  SENet    |     97.8    |    2.8|
|  MobileNetV3    |     99.2    |    1.3|
|  EfficientNet    |     98.8    |    1.57|

## landmark:

#### License plate corner detection
![](https://github.com/qzq2514/ImageForGithubMakdown/blob/master/DNNCode/landmark1.PNG)
![](https://github.com/qzq2514/ImageForGithubMakdown/blob/master/DNNCode/landmark2.PNG) 
![](https://github.com/qzq2514/ImageForGithubMakdown/blob/master/DNNCode/landmark3.PNG)
![](https://github.com/qzq2514/ImageForGithubMakdown/blob/master/DNNCode/landmark4.PNG) 


## text recognition:

#### CTC Loss - base
![](https://github.com/qzq2514/ImageForGithubMakdown/blob/master/Patents/SecondPatent/CTC_rec_res1.png)
![](https://github.com/qzq2514/ImageForGithubMakdown/blob/master/Patents/SecondPatent/CTC_rec_res2.png)


## face recognition:

#### Triplet Loss - base
number between images is the distance of the pair images  
threshold can be set 1.0 to determine if the two images are from the same person.  
![](https://github.com/qzq2514/ImageForGithubMakdown/blob/master/DNNCode/face.PNG)