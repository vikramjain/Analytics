from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Input
import numpy as np         # dealing with arrays
import os                  # dealing with directories
import keras
from keras.layers import Activation, Dense
from keras import backend as K
from keras.applications.imagenet_utils import preprocess_input
from keras.preprocessing.image import img_to_array, load_img
import glob       

img_width, img_height = 299, 299
input_shape = (img_width, img_height)
train_data_dir = 'data/train' #contains two classes cats and dogs
validation_data_dir = 'data/validation' #contains two classes cats and dogs
test_data_dir = 'data/test/'
nb_train_samples = 159
nb_validation_samples = 58
nb_epochs =5
batch_size = 27

num_classes = len(glob.glob(train_data_dir + "/*"))
print('Number of classes found:', num_classes)

base_model = InceptionV3(weights='imagenet', include_top=False)

x = base_model.output
x = GlobalAveragePooling2D()(x)
# Add a fully-connected layer
x = Dense(1024, activation='relu')(x)
# and a sigmoid layer
predictions = Dense(6, activation='softmax')(x)

# this is the model we will train
model = Model(input=base_model.input, output=predictions)

model.summary()

# first: train only the top layers (which were randomly initialized)
# i.e. freeze all convolutional InceptionV3 layers
for layer in base_model.layers:
    layer.trainable = False
    
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
validation_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=input_shape,
    batch_size=batch_size,
    class_mode='categorical')

validation_generator = validation_datagen.flow_from_directory(
    validation_data_dir,
    target_size=input_shape,
    batch_size=batch_size,
    class_mode='categorical')

model.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_samples // batch_size,
    epochs=nb_epochs,
    validation_data=validation_generator,
    validation_steps=nb_validation_samples // batch_size)

for filename in os.listdir(test_data_dir):
    img = load_img(test_data_dir + filename ,False,target_size=(img_width,img_height))
    x = img_to_array(img)
    x = np.expand_dims(x, axis=0)
    preds = np.argmax(model.predict(x),axis=1)
    #	prob = model.predict_proba(x)
    print("Ground Truth: " , str(filename) , "Prediction: " , str(preds))