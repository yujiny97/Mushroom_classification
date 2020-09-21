#keras에서는 폴더 이름에 따라서 classification 이름이 정해진다
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from keras.preprocessing import image
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
from keras.models import model_from_json

def ispoison(img):
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)

    #load model
    with open("./mashroom_weight/model_mashroom.json", "r") as json_file:
        model_json=json_file.read()
        json_file.close()
        model=model_from_json(model_json)

    #single test

    #test_image = image.load_img(img, target_size = (64, 64))
    test_image = img
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis = 0)

    #weight load
    model.load_weights("./mashroom_weight/0626/mashroom_weight-18-0.7787500023841858.h5")

    #predict model
    result = model.predict(test_image)
    prediction=''
    if result[0][0] == 1:
        #먹을 수 있는 것
        prediction = 1
    else:
        #먹을 수 없는 것
        prediction = 0

    print(prediction)
    return prediction
