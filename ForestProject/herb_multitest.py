from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import tensorflow as tf
from keras.preprocessing import image
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
from keras.models import model_from_json

def whatHerb(img):
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)
    #load model
    with open("./mashroom_weight/model_mashroomv3.json", "r") as json_file:
        model_json=json_file.read()
        json_file.close()
        model=tf.keras.models.model_from_json(model_json)

    #single test
    #test_image = image.load_img('/home/yujinlee/gongmo/mornot/test/버섯아님/자연_45.jpg', target_size = (64, 64))
    test_image = img

    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis = 0)

    #weight load
    #model.load_weights("./mashroom_weight/0626/mashroom_weight-13-0.9725444912910461.h5")
    model.load_weights("./herb_weight/herb_weight-03-0.87890625.h5")

    #predict model
    result = model.predict(test_image)[0]
    res=np.argmax(result)

    if res == 2:
        prediction = '개나리'
    elif res == 0:
        prediction = '가죽나무'
    else :
        prediction = '갈대'

    return prediction

