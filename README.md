# Data-Centric-AI
This is an attempt to get good accuracy in AI solution by modifying the existing data. So here we are doing various data augmentation techniques in the data.

This is a R&D as a part of Data-Centric-AI competition in deeplearning.ai.

## Steps-Taken-For-Data-Preprocess
> Download the dataset from the competition website.

> Removed the mislabelled datas.

> Rearranged the datas.
## Competition-Instructions
> Submission data must have less than 10,000 images combined in training and validation.

> Images will resize to **32/32/3** before training, validation and testing.

## Augmentations
Here we tried augmentation techniques are given below.
> CutOut Augmentation.

> Augmix Augmentation.

> Custom Crop Augmentation.

## Process-Steps
 >  Started the training with raw data from the website and analysed the accuracy. Its more less than the baseline accuracy.
 
 >  Then we used the mentioned preprocess and cleaned the data. It gives the promising accuracy as compared to baseline accuracy in the website.
 
 >  Applied the CutOut Augmentation on cleaned data and checked the results and it was not good when compared to baseline accuracy.
 
 >  Applied the Augmix Augmentation on cleaned data and checked the results and it was not good when compared to baseline accuracy.
 
 >  After these expirements we adopt a custom crop augmentation and it gives the good results only when apply this mechanism on test set before the testing.
 
 >  Experiment results are given in the **Training_Stats** directory.
