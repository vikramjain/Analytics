from keras.utils import to_categorical
from keras.models import load_model
from keras.preprocessing.image import img_to_array, load_img
import numpy as np
import inspect
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
from keras import applications

# dimensions of our images.
img_width, img_height = 150, 150

top_model_weights_path = 'bottleneck_fc_model.h5'
train_data_dir = 'data/train'
validation_data_dir = 'data/validation'
test_data_dir = 'data/test'
nb_train_samples = 600
nb_validation_samples = 102
epochs = 50
batch_size = 16

def train_top_model():
    datagen = ImageDataGenerator(rescale=1. / 255)
    print (inspect.currentframe().f_lineno)
    # build the VGG16 network
    model = applications.VGG16(include_top=False, weights='imagenet')
    print (inspect.currentframe().f_lineno)
    generator = datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode=None,
        shuffle=False)
    bottleneck_features_train = model.predict_generator(
        generator, nb_train_samples // batch_size)
    np.save(open('bottleneck_features_train.npy', 'wb'),
            bottleneck_features_train)
    print (inspect.currentframe().f_lineno)
    generator = datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode=None,
        shuffle=False)
    bottleneck_features_validation = model.predict_generator(
        generator, nb_validation_samples // batch_size)
    np.save(open('bottleneck_features_validation.npy', 'wb'),
            bottleneck_features_validation)
    print (inspect.currentframe().f_lineno)
    generator = datagen.flow_from_directory(
        test_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode=None,
        shuffle=False)
    bottleneck_features_test = model.predict_generator(
        generator, nb_validation_samples // batch_size)
    np.save(open('bottleneck_features_test', 'wb'),
            bottleneck_features_test)
    print (inspect.currentframe().f_lineno)			
    train_data = np.load(open('bottleneck_features_train.npy', 'rb'))
    train_labels = np.array(
        [0] * (nb_train_samples // 2) + [1] * (nb_train_samples // 2))
    print (inspect.currentframe().f_lineno)
    validation_data = np.load(open('bottleneck_features_validation.npy', 'rb'))
    validation_labels = np.array(
        [0] * (nb_validation_samples // 2) + [1] * (nb_validation_samples // 2))
    print (inspect.currentframe().f_lineno)
    model = Sequential()
    model.add(Flatten(input_shape=train_data.shape[1:]))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    print (inspect.currentframe().f_lineno)
    model.compile(optimizer='rmsprop',
                  loss='binary_crossentropy', metrics=['accuracy'])
    print (inspect.currentframe().f_lineno)
    model.fit(train_data, train_labels,
              epochs=epochs,
              batch_size=batch_size,
              validation_data=(validation_data, validation_labels))
    # show the accuracy on the testing set
    print("[INFO] evaluating on testing set...")
    (loss, accuracy) = model.evaluate(validation_data, validation_labels, batch_size=batch_size, verbose=1)
    print("[INFO] loss={:.4f}, accuracy: {:.4f}%".format(loss,accuracy * 100))
    print (inspect.currentframe().f_lineno)

#save_bottlebeck_features()
train_top_model()