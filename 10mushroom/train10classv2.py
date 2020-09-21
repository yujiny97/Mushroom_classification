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
import matplotlib.pyplot as plt
import tensorflow as tf


np.random.seed(1)

filepath='/home/yujinlee/gongmo/10mushroom/weight/v4/wfile-{epoch:02d}-{val_accuracy}.h5'
checkpoint=ModelCheckpoint(filepath,monitor='val_accuracy',verbose=1,save_weights_only=True)

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

categories=["독버섯_muscaria", "독버섯_개나리광대버섯","독버섯_독우산광대버섯","독버섯_비듬땀버섯", "독버섯_큰갓버섯","식용_귀신그물버섯", "식용_느타리버섯", "식용_볏집버섯", "식용_송이버섯","식용_표고버섯","식용_흑갈색벚꽃버섯"]
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
#adam optimizer로 코스트 줄이기


#save model
# model_json = model.to_json()
# with open("/home/yujinlee/gongmo/10mushroom/model/v4/model_mashroomv3.json", "w") as json_file:
#     json_file.write(model_json)

#weight load (currently made weight file and train more)
model.load_weights("/home/yujinlee/gongmo/10mushroom/weight/v4/wfile-75-0.6153273582458496.h5")
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


model.fit_generator(training_set,
steps_per_epoch = 50,
epochs = 100,
validation_data = test_set,
validation_steps = 21,
callbacks=[checkpoint]
)



