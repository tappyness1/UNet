# UNet

## What this is about
Just a simple implementation based on the UNet which is the convolutional neural network used in Medical Field and Denoising and hence was used by Latent Diffusion Models. Hence, pretty important and my whole impetus for trying to implement it. 

## What has been done 
1. Set up the Architecture
1. Set up the dataset and dataloader
1. Set up the training, which could be better implemented admittedly.
1. Set up validation, but only takes accuracy and loss. 
1. Results visualisation

## What else needs to be done
1. Remove hardcoded weights

## How to run 

Make sure you change the directory of your data. I used the VOCSegmentation dataset which has 20 classes. 

```
python -m src.main
```