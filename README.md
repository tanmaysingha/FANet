# FANet
Feature Aggregation Network (FANet) is a scene segmentation model which is designed for resource constrained embedded devices. By leveraging a new multi-scale feature fusion technique, it maps rich semantic features at different labels and improves model performance. This model has less parameters and FLOPS compare to many existing real-time scene segmentation model and it also produces better prediction accuracy. Model performance is evaluated on cityscapes and CamVid dataset. FANet can handle high resolution input images and can use less memory footprint. To compare our model performance with other existing semantice segmentation models, we also trained FAST-SCNN, bayes-segnet, Deeplab and separable UNet models. Separable UNet is basically an UNet model in which all convolution layers are replaced by separable convolution layers to reduce number of parameters and FLOPS. In DeepLab model, we have used exception as a backbone network. Separable UNet and Deeplab are off-line segmentation model whereas FAST-SCNN and Bayes-segnet are real-time segmentation model. Our experiment exhibits that FANet outperforms than these models. We have achieved 65.9% class mean IoU and 83.6% category mean IoU on Cityscapes vlidation dataset. Here, we are uploading scripts for existing models and our results. More details will be available upon acceptance of the paper. 

## Datasets
For this research work, we have used cityscapes benchmark datasets and CamVid dataset.
* Cityscapes - To access this benchmark, user needs an account. https://www.cityscapes-dataset.com/downloads/     
* CamVid - To access this benchmark, visit this link: http://mi.eng.cam.ac.uk/research/projects/VideoRec/CamVid/

## Metrics
To understand the metrics used for model performance evaluation, please  refer here: https://www.cityscapes-dataset.com/benchmarks/#pixel-level-results

## Transfer Learning
For performance comparison, we trained few off-line and real-time existing models under same configuration and compared their performance with ESPNet. Some existing models require the use of ImageNet pretrained models to initialize their weights. Details will be given soon.

## Requirements for Project
* TensorFlow 2.1
  * This requires CUDA >= 10.1
  * TensorRT will require an NVIDIA GPU with Tensor Cores.
  * Horovod framework (for effective utilization of resources and speed up GPUs)
* Keras 2.3.1
* Python >= 3.7

## Results
We trained our model with different input resolutions for cityscapes dataset. Cityscapes provides 1024 x 2048 px resolution images. We mainly focus full resolution of cityscapes images. For CamVid dataset, we use 512 x 1024px resolution altough original image size is 720 x 960px. We trained other existng models with full resolution of cityscapes images. 
### Separable UNet
![Separable UNet](https://github.com/tanmaysingha/FANet/blob/master/Predicted_images/separable_UNet.png?raw=true)

### DeepLab
![DeepLabV3+](https://github.com/tanmaysingha/FANet/blob/master/Predicted_images/DeepLab.png?raw=true)

### Bayesian SegNet
![Bayesian SegNet](https://github.com/tanmaysingha/FANet/blob/master/Predicted_images/bayes_segnet.png?raw=true)

### FAST-SCNN
![FAST-SCNN](https://github.com/tanmaysingha/FANet/blob/master/Predicted_images/fast_scnn.png?raw=true)

### FANet
![FANet](?raw=true)

### Model prediction on CamVid dataset
![FANet_Vs_FAST_SCNN](https://github.com/tanmaysingha/FANet/blob/master/Predicted_images/CamVid_prediction.png?raw=true)
