#keras에서는 폴더 이름에 따라서 classification 이름이 정해진다
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense, Dropout, Activation
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from keras.preprocessing import image
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
from keras.callbacks import ModelCheckpoint
import cv2

filepath='/home/yujinlee/gongmo/0714newversion/weight/v4_58/wfile-{epoch:02d}-{val_accuracy}.h5'
checkpoint=ModelCheckpoint(filepath,monitor='val_accuracy',verbose=1,save_weights_only=True)

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

categories=["notmushroom", "독버섯_muscaria", "독버섯_갈색고리갓버섯", "독버섯_갈황색미치광이버섯", "독버섯_개나리광대버섯", "독버섯_검은망그물버섯", "독버섯_게딱지버섯", "독버섯_긴골광대버섯아재비", "독버섯_깔때기무당버섯", "독버섯_꼭지버섯", "독버섯_냄새무당버섯", "독버섯_노란각시버섯", "독버섯_노란다발버섯", "독버섯_노란싸리버섯", "독버섯_독우산광대버섯", "독버섯_마귀곰보버섯", "독버섯_마귀광대버섯", "독버섯_뱀껍질광대버섯", "독버섯_붉은사슴뿔버섯", "독버섯_붉은싸리버섯", "독버섯_비듬땀버섯", "독버섯_짧은대꽃잎버섯", "독버섯_큰갓버섯", "독버섯_화경버섯", "독버섯_흰독큰갓버섯(독흰갈대버섯)", "독버섯_흰알광대버섯", "독버섯_흰오뚜기광대버섯", "식용_개암버섯(주의.노란다발버섯과 유사)", "식용_곰보버섯(주의.마귀곰보 버섯과 유사)", "식용_굴털이젖버섯", "식용_귀신그물버섯", "식용_그물버섯아재비", "식용_금빛비늘버섯", "식용_기와버섯", "식용_꾀꼬리버섯", "식용_노루궁뎅이", "식용_느타리버섯", "식용_능이버섯", "식용_다발방패버섯", "식용_다색벚꽃버섯", "식용_달걀버섯", "식용_먹물버섯(주의.검은색 물든것 식용불가)", "식용_볏집버섯", "식용_뿔나팔버섯", "식용_새잣버섯", "식용_송이버섯", "식용_싸리버섯", "식용_연기색만가닥버섯", "식용_영지버섯", "식용_잎새버섯", "식용_자주방망이버섯아재비", "식용_점마개버섯", "식용_팽나무버섯", "식용_팽이버섯", "식용_표고버섯", "식용_하늘색깔대기버섯", "식용_흑갈색벚꽃버섯", "식용_흰주름버섯"]
nb_classes=len(categories)


model= Sequential()
model.add(Conv2D(kernel_size=(3,3), filters=32, activation='relu', input_shape=(64,64,3,)))
model.add(Dropout(0.25))

model.add(Conv2D(filters=64,kernel_size = (3,3),activation='relu'))
model.add(Dropout(0.25))
model.add(MaxPooling2D(2,2))

model.add(Conv2D(filters=128,kernel_size = (3,3),activation='relu'))
model.add(Dropout(0.25))
model.add(MaxPooling2D(2,2))

model.add(Conv2D(filters=30,kernel_size = (3,3),activation='relu'))
model.add(Dropout(0.25))
model.add(MaxPooling2D(2,2))

model.add(Flatten())
model.add(Dense(512,activation='relu'))
model.add(Dense(512,activation='relu'))
model.add(Dense(nb_classes,activation = 'softmax'))
    
model.compile(
              loss='categorical_crossentropy', 
              metrics=['accuracy'],
              optimizer='adam'
             )

#save model
# model_json = model.to_json()
# with open("/home/yujinlee/gongmo/0714newversion/model/v4_58/model_mashroomv3.json", "w") as json_file:
#   json_file.write(model_json)

#weight load (currently made weight file and train more)
model.load_weights("/home/yujinlee/gongmo/0714newversion/weight/v4_58/wfile-47-0.3869912922382355.h5")
#print("Loaded weight from disk")

train_datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest',
        brightness_range=[0.2,1.0]
        )

test_datagen = ImageDataGenerator(rescale = 1./255)

#training set
training_set = train_datagen.flow_from_directory('train',
target_size = (64, 64),
batch_size = 64,
class_mode = 'categorical')

#test set
test_set = test_datagen.flow_from_directory('test',
target_size = (64, 64),
batch_size = 64,
class_mode = 'categorical')


# #테스트해보기
# model.fit_generator(training_set,
# steps_per_epoch = 2390,
# epochs = 20,
# validation_data = test_set,
# validation_steps = 400,
# callbacks=[checkpoint]
# )

#테스트해보기
# model.fit_generator(training_set,
# steps_per_epoch = 75,
# epochs = 20,
# validation_data = test_set,
# validation_steps = 13,
# callbacks=[checkpoint]
# )

model.fit_generator(training_set,
steps_per_epoch = 199,
epochs = 300,
validation_data = test_set,
validation_steps = 86,
callbacks=[checkpoint]
)



