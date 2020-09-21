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
categories=["notmushroom", "독버섯_muscaria", "독버섯_갈색고리갓버섯", "독버섯_갈황색미치광이버섯", "독버섯_개나리광대버섯", "독버섯_검은망그물버섯", "독버섯_게딱지버섯", "독버섯_긴골광대버섯아재비", "독버섯_깔때기무당버섯", "독버섯_꼭지버섯", "독버섯_냄새무당버섯", "독버섯_노란각시버섯", "독버섯_노란다발버섯", "독버섯_노란싸리버섯", "독버섯_독우산광대버섯", "독버섯_마귀곰보버섯", "독버섯_마귀광대버섯", "독버섯_뱀껍질광대버섯", "독버섯_붉은사슴뿔버섯", "독버섯_붉은싸리버섯", "독버섯_비듬땀버섯", "독버섯_짧은대꽃잎버섯", "독버섯_큰갓버섯", "독버섯_화경버섯", "독버섯_흰독큰갓버섯(독흰갈대버섯)", "독버섯_흰알광대버섯", "독버섯_흰오뚜기광대버섯", "식용_개암버섯(주의.노란다발버섯과 유사)", "식용_곰보버섯(주의.마귀곰보 버섯과 유사)", "식용_굴털이젖버섯", "식용_귀신그물버섯", "식용_그물버섯아재비", "식용_금빛비늘버섯", "식용_기와버섯", "식용_꾀꼬리버섯", "식용_노루궁뎅이", "식용_느타리버섯", "식용_능이버섯", "식용_다발방패버섯", "식용_다색벚꽃버섯", "식용_달걀버섯", "식용_먹물버섯(주의.검은색 물든것 식용불가)", "식용_볏집버섯", "식용_뿔나팔버섯", "식용_새잣버섯", "식용_송이버섯", "식용_싸리버섯", "식용_연기색만가닥버섯", "식용_영지버섯", "식용_잎새버섯", "식용_자주방망이버섯아재비", "식용_점마개버섯", "식용_팽나무버섯", "식용_팽이버섯", "식용_표고버섯", "식용_하늘색깔대기버섯", "식용_흑갈색벚꽃버섯", "식용_흰주름버섯"]
#load model
with open("/home/yujinlee/gongmo/0714newversion/model/v4_58/model_mashroomv3.json", "r") as json_file:
    model_json=json_file.read()
    json_file.close()
    model=model_from_json(model_json)

print("Loaded model from disk")

#single test
#test_image = image.load_img('/home/yujinlee/gongmo/mornot/test/버섯아님/자연_45.jpg', target_size = (64, 64))
test_image = image.load_img('/home/yujinlee/gongmo/0714newversion/test/식용_느타리버섯/느타리버섯2_2.jpg', target_size = (64, 64))
#test_image = image.load_img('/home/yujinlee/gongmo/test/식용버섯/송이버섯_105.jpg', target_size = (64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)

#weight load
#model.load_weights("./mashroom_weight/0626/mashroom_weight-13-0.9725444912910461.h5")
model.load_weights("/home/yujinlee/gongmo/0714newversion/weight/v4_58/wfile-298-0.3829941749572754.h5")
print("Loaded weight from disk")


#predict model
result = model.predict(test_image)[0]
res=np.argmax(result)
prediction=categories[res]
print(result)
print(res)
print(prediction)
