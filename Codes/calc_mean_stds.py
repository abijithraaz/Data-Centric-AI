import cv2
import os
import tqdm
import numpy as np

def cal_mean_std(images_dir):
    """
    calculate mean and stds of dataset.
    """
    m_list, s_list = [], []
    for folders in (os.listdir(directory)):
        folder = directory + "/" + folders
        for classes in (os.listdir(folder)):
            class_name = folder + "/" + classes
            for images in (os.listdir(class_name)):
                img = cv2.imread(class_name + '/' + images)
                img = img / 255.0
                m, s = cv2.meanStdDev(img)

                m_list.append(m.reshape((3,)))
                s_list.append(s.reshape((3,)))
    m_array = np.array(m_list)
    s_array = np.array(s_list)
    m = m_array.mean(axis=0, keepdims=True)
    s = s_array.mean(axis=0, keepdims=True)
    print('mean: ',m[0][::-1])
    print('std:  ',s[0][::-1])
    return m, s

if __name__ == "__main__":
    directory = "D:/Data-centric-AI/Data_preperation"
    m, s = cal_mean_std(directory)
