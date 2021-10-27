# FANet
Feature Aggregation Network (FANet) is a scene segmentation model which is designed for resource constrained embedded devices. By leveraging a new multi-scale feature fusion technique, it maps rich semantic features at different labels and improves model performance. This model has less parameters and FLOPS compare to many existing real-time scene segmentation model and it also produces better prediction accuracy. Model performance is evaluated on cityscapes and CamVid dataset. FANet can handle high resolution input images and can use less memory footprint. To compare our model performance with other existing semantice segmentation models, we also trained FAST-SCNN, bayes-segnet, Deeplab and separable UNet models. Separable UNet is basically an UNet model in which all convolution layers are replaced by separable convolution layers to reduce number of parameters and FLOPS. In DeepLab model, we have used exception as a backbone network. Separable UNet and Deeplab are off-line segmentation model whereas FAST-SCNN and Bayes-segnet are real-time segmentation model. Our experiment exhibits that FANet outperforms than these models. We have achieved 65.9% class mean IoU and 83.6% category mean IoU on Cityscapes vlidation dataset. Here, we are uploading scripts for existing models and our results.

### Complete pipeline of FANet
![pipeline](https://github.com/tanmaysingha/FANet/blob/master/Predicted_images/FANet_pipeline.png?raw=true)

## Datasets
For this research work, we have used cityscapes benchmark datasets and CamVid dataset.
* Cityscapes - To access this benchmark, user needs an account. https://www.cityscapes-dataset.com/downloads/     
* CamVid - To access this benchmark, visit this link: http://mi.eng.cam.ac.uk/research/projects/VideoRec/CamVid/

## Metrics
To understand the metrics used for model performance evaluation, please  refer here: https://www.cityscapes-dataset.com/benchmarks/#pixel-level-results

## Transfer Learning
For performance comparison, we trained few off-line and real-time existing models under same configuration and compared their performance with FANet. Some existing models require the use of ImageNet pretrained models to initialize their weights. Details will be given soon.

## Requirements for Project
* TensorFlow 2.1
  * This requires CUDA >= 10.1
  * TensorRT will require an NVIDIA GPU with Tensor Cores.
  * Horovod framework (for effective utilization of resources and speed up GPUs)
* Keras 2.3.1
* Python >= 3.7

## Results
We trained our model with different input resolutions for cityscapes dataset. Cityscapes provides 1024 x 2048 px resolution images. We mainly focus full resolution of cityscapes images. For CamVid dataset, we use 512 x 1024px resolution altough original image size is 720 x 960px. We trained other models with different resolutions of cityscapes images. 
### Separable UNet
![Separable UNet](https://github.com/tanmaysingha/FANet/blob/master/Predicted_images/separable_UNet.png?raw=true)

### DeepLab
![DeepLabV3+](https://github.com/tanmaysingha/FANet/blob/master/Predicted_images/DeepLab.png?raw=true)

### Bayesian SegNet
![Bayesian SegNet](https://github.com/tanmaysingha/FANet/blob/master/Predicted_images/bayes_segnet.png?raw=true)

### FAST-SCNN
![FAST-SCNN](https://github.com/tanmaysingha/FANet/blob/master/Predicted_images/fast_scnn.png?raw=true)
<b><u>IoU Over Classes on Validation Set</b></u>

classes       |  IoU  |   nIoU
--------------|-------|---------
road          | 0.943 |    nan
sidewalk      | 0.761 |    nan
building      | 0.876 |    nan
wall          | 0.444 |    nan
fence         | 0.433 |    nan
pole          | 0.434 |    nan
traffic light | 0.511 |    nan
traffic sign  | 0.595 |    nan
vegetation    | 0.889 |    nan
terrain       | 0.546 |    nan
sky           | 0.908 |    nan
person        | 0.667 |  0.396
rider         | 0.437 |  0.228
car           | 0.899 |  0.787
truck         | 0.552 |  0.196
bus           | 0.650 |  0.365
train         | 0.451 |  0.197
motorcycle    | 0.395 |  0.186
bicycle       | 0.631 |  0.351
<b>Score Average | <b>0.633 | <b>0.338
 
 <b><u>IoU Over Categories </b></u>

categories    |  IoU   |  nIoU
--------------|--------|--------
flat          | 0.955  |   nan
construction  | 0.882  |   nan
object        | 0.529  |   nan
nature        | 0.891  |   nan
sky           | 0.908  |   nan
human         | 0.708  | 0.426
vehicle       | 0.878  | 0.756
<b>Score Average | <b>0.822  | <b>0.591
 
 <b><u>To see the performance of FAST-SCNN on test dataset, you can view the .csv file from here: </b></u>
 (https://github.com/tanmaysingha/FANet/blob/master/Predicted_images/FAST_SCNN_Test_Results_Evaluated_By_Cityscapes_Server.csv)

### FANet
![FANet](https://github.com/tanmaysingha/FANet/blob/master/Predicted_images/FANet.png?raw=true)
<b><u>IoU Over Classes on Validation Set of Cityscapes</b></u>

classes       |  IoU  |   nIoU
--------------|-------|---------
road          | 0.962 |    nan
sidewalk      | 0.751 |    nan
building      | 0.893 |    nan
wall          | 0.527 |    nan
fence         | 0.473 |    nan
pole          | 0.470 |    nan
traffic light | 0.535 |    nan
traffic sign  | 0.646 |    nan
vegetation    | 0.898 |    nan
terrain       | 0.552 |    nan
sky           | 0.925 |    nan
person        | 0.702 |  0.459
rider         | 0.456 |  0.272
car           | 0.909 |  0.785
truck         | 0.470 |  0.201
bus           | 0.704 |  0.358
train         | 0.615 |  0.311
motorcycle    | 0.388 |  0.186
bicycle       | 0.652 |  0.403
<b>Score Average | <b>0.659 | <b>0.372

<b><u>IoU Over Categories on validation set of Cityscapes</b></u>

categories    |  IoU   |  nIoU
--------------|--------|--------
flat          | 0.967  |   nan
construction  | 0.894  |   nan
object        | 0.556  |   nan
nature        | 0.901  |   nan
sky           | 0.925  |   nan
human         | 0.715  | 0.489
vehicle       | 0.892  | 0.767
<b>Score Average | <b>0.836  | <b>0.628
 
 <b><u>To see the performance of FANet on test dataset, you can view the .csv file from here:</b></u>
  (https://github.com/tanmaysingha/FANet/blob/master/Predicted_images/FANet_Test_Results_evaluated_by_Cityscapes_server.csv)

### Model prediction on CamVid dataset
![FANet_Vs_FAST_SCNN](https://github.com/tanmaysingha/FANet/blob/master/Predicted_images/CamVid_prediction.png?raw=true)
 
 ### Citation
 ```yaml
cff-version: 1.2.0
If FANet is useful for your research work, please consider for citing the paper:
@inproceedings{singha2020fanet,
  title={FANet: Feature Aggregation Network for Semantic Segmentation},
  author={Singha, Tanmay and Pham, Duc-Son and Krishna, Aneesh},
  booktitle={2020 Digital Image Computing: Techniques and Applications (DICTA)},
  pages={1--8},
  year={2020},
  organization={IEEE}
}
```
 Refer the following link for FANet paper: https://ieeexplore.ieee.org/abstract/document/9363370

