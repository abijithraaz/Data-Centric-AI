import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from augmix import AugMix
import cv2
import os

# precalculated means and stds of the dataset (in RGB order)
# means = [0.44892993872313053, 0.4148519066242368, 0.301880284715257]
# stds = [0.24393544875614917, 0.2108791383467354, 0.220427056859487]

# calculate mean and stds of the dataset (in RGB order)
means = [0.44769294, 0.44769294, 0.44769294]
stds = [0.19916161, 0.19916161, 0.19916161]
ag = AugMix(means, stds)

directory = "D:/Data-centric-AI/Data_preperation _clear"
output_dir = "D:/Data-centric-AI/cleaned_augmix_data"

if __name__ == "__main__":
    for folders in (os.listdir(directory)):
        folder = directory + "/" + folders
        op_folder = output_dir+"/"+folders
        if not (os.path.exists(op_folder)):
            os.makedirs(op_folder)
        for classes in (os.listdir(folder)):
            class_name = folder + "/" + classes
            op_class_name = op_folder + "/" + classes
            if not (os.path.exists(op_class_name)):
                os.makedirs(op_class_name)
            for images in (os.listdir(class_name)):
                image_name = class_name + "/" + images
                img = cv2.imread(image_name)
                image = tf.convert_to_tensor(img, dtype=tf.float32)
                h, w, c = image.shape
                if (h <= w):
                    image = tf.image.resize(image, (h, h)) # resize to square
                else:
                    image = tf.image.resize(image, (w, w)) # resize to square
                image /=  255  # scale to [0, 1]

                # augment
                augmented = ag.transform(image)

                #writing
                aug = augmented.numpy()
                aug = np.clip(aug, 0, 1)
                op_image = op_class_name + "/"+images
                plt.imsave(op_image, aug)
