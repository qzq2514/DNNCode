# DNN learning

DNN Code including classification,landmark,text recognition...  
Training and testing script of each DNN project is included.

## Classification:
Classification models are trained and evaluated on the task of character classification including 36 characters(10 digits and 26 uppercase letter).  
input: 28 x 28 x 3  
environment: openvino 2019  、CORE i7、GTX 1060

Names of Classfier are just the general type of CNN net,readers can go into codes to find specific type.
Readers can fine-tuning the network architecture for your own tasks.

|   Classifier   | Accuarcy (%)|  Speed (ms)|
|:------------:|:-------------------:|:-------------------:|
| my classification    |        99.6        |  1.62|
|    MobileNetV1   |        99.3        |  0.63|
|    MobileNetV2   |        99.7        |  1.12|
|   ResNetV1    |     99.8    |4.22|
|  ShuffleNetV1  |     92.7    |    1.52|
|  ShuffleNetV2  |     97.7    |    1.75|



## landmark:

#### License plate corner detection
![](https://github.com/qzq2514/ImageForGithubMakdown/blob/master/DNNCode/landmark1.PNG)
![](https://github.com/qzq2514/ImageForGithubMakdown/blob/master/DNNCode/landmark2.PNG) 
![](https://github.com/qzq2514/ImageForGithubMakdown/blob/master/DNNCode/landmark3.PNG)
![](https://github.com/qzq2514/ImageForGithubMakdown/blob/master/DNNCode/landmark4.PNG) 



