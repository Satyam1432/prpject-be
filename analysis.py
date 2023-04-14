# -*- coding: utf-8 -*-
import os

import tensorflow as tf

classifierLoad = tf.keras.models.load_model('model.h5')
main_dir = 'dataset/test/'
import numpy as np
from keras.preprocessing import image

sub_dir = ["Open_Eyes","Closed_Eyes"]



def confusion_matrix():
    total_images = 0
    correct_prediction = 0
    actual_values=[]
    predicted_values=[]
    for dir_ in sub_dir:
        for image_ in os.listdir(main_dir + dir_ + "/"):
            print(dir_)
            total_images += 1
            test_image = tf.keras.utils.load_img(main_dir + dir_ + "/" + image_, target_size=(200, 200))
            test_image = np.expand_dims(test_image, axis=0)
            result = classifierLoad.predict(test_image)
            if "Open_Eyes" in dir_:
                actual_values.append(0)
            elif "Closed_Eyes" in dir_:
                actual_values.append(1)

            if result[0][0] == 1: 
                print(result[0][0])
                predicted_values.append(0)
            elif result[0][1] == 1:
                predicted_values.append(1)

    return actual_values,predicted_values

