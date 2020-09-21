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


config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)
categories=["독버섯_muscaria", "독버섯_개나리광대버섯","독버섯_독우산광대버섯","독버섯_비듬땀버섯", "독버섯_큰갓버섯","식용_귀신그물버섯", "식용_느타리버섯", "식용_볏집버섯", "식용_송이버섯","식용_표고버섯","식용_흑갈색벚꽃버섯"]
#load model

#with open("/home/yujinlee/gongmo/10mushroom/model/v4/model_mashroomv3.json", "r") as json_file:
with open("/home/yujinlee/gongmo/10mushroom/model/v4/model_mashroomv3.json", "r") as json_file:
    model_json=json_file.read()
    json_file.close()
    model=model_from_json(model_json)

print("Loaded model from disk")

#single test
#test_image = image.load_img('/home/yujinlee/gongmo/mornot/test/버섯아님/자연_45.jpg', target_size = (64, 64))
test_image = image.load_img('/home/yujinlee/gongmo/10mushroom/test/식용_표고버섯/표고버섯2_45.jpg', target_size = (64, 64))
#test_image = image.load_img('/home/yujinlee/gongmo/test/식용버섯/송이버섯_105.jpg', target_size = (64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)

#weight load
#model.load_weights("./mashroom_weight/0626/mashroom_weight-13-0.9725444912910461.h5")
model.load_weights("/home/yujinlee/gongmo/10mushroom/weight/v4/wfile-64-0.6361607313156128.h5")
print("Loaded weight from disk")


#predict model
result = model.predict(test_image)[0]
res=np.argmax(result)
prediction=categories[res]
print(result)
print(res)
print(prediction)
