from numpy.core.fromnumeric import shape
import tensorflow as tf
import tensorflow_addons as tfa
import os
import cv2

directory = "D:/Data-centric-AI/Data_preperation"
output_dir = "D:/Data-centric-AI/aug_cutout_data"

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
                image = class_name + "/" + images
                img = cv2.imread(image)
                tensor = tf.convert_to_tensor(img, dtype=tf.float32)
                tensor = tf.expand_dims(tensor, axis=0)

                aug_img = tfa.image.random_cutout(tensor, (50,50), constant_values = 255)
                n, h, w, c = aug_img.shape
                aug_shape = (h, w, c)
                aug_img = tf.reshape(aug_img, aug_shape, name=None)
                aug = aug_img.numpy()

                op_image = op_class_name + "/"+images
                cv2.imwrite(op_image, aug)

