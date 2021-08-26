import os
import cv2
import numpy as np

threshold_level = 1
padd = 10
directory = "D:/Data-centric-AI/Data_preperation _clear"
output_dir = "D:/Data-centric-AI/cleaned_crop_custom_data"

def normalise_coord(y1, x1, y2, x2):
    if (y1 < 0):
        y1 = 0

    if(y2 > height):
        y2 = height

    if(x1 < 0):
        x1 = 0

    if (x2 > width):
        x2 = width
    
    return y1, x1, y2, x2

def coordinates(image, height, width):
    coords = np.column_stack(np.where(image < threshold_level))
    min = np.amin(coords, axis=0)
    max = np.amax(coords, axis=0)
    y1, x1, y2, x2 = min[0], min[1], max[0], max[1]
    y1 = y1 - padd
    y2 = y2 + padd
    x1 = x1 - padd
    x2 = x2 + padd
    y1, x1, y2, x2 = normalise_coord(y1, x1, y2, x2)
    
    if((y2 - y1) < 32):
        y_padd = abs(32-(y2-y1))
        if (y_padd%2 != 0):
            y_padd = y_padd+1
        if (y1 == 0 and y2 != height):
            y2 = y2 + y_padd
        elif (y1 != 0 and y2 == height):
            y1 = y1 - y_padd
        else:
            y1 = y1 - (y_padd/2)
            y2 = y2 + (y_padd/2)
    
    if((x2 - x1) < 32):
        x_padd = abs(32-(x2-x1))
        if (x_padd%2 != 0):
            x_padd = x_padd+1
        if (x1 == 0 and x2 != width):
            x2 = x2 + x_padd
        elif (x1 != 0 and x2 == width):
            x1 = x1 - x_padd
        else:
            x1 = x1 - (x_padd/2)
            x2 = x2 + (x_padd/2)

    y1, x1, y2, x2 = normalise_coord(y1, x1, y2, x2)

    return int(y1), int(x1), int(y2), int(x2)

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
                image = cv2.imread(image_name)
                height, width, channels = image.shape

                img = cv2.imread(image_name, 2)
                ret, bw_img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

                y1, x1, y2, x2 = coordinates(bw_img, height, width)

                crop_image = image[y1:y2, x1:x2]
                op_image = op_class_name + "/"+images
                cv2.imwrite(op_image, crop_image)
