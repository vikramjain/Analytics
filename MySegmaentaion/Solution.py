import cv2
import os
import numpy as np
from keras import backend as K
from keras.models import Model
from keras.layers import Input
from keras.layers import Conv2D
from keras.layers import MaxPool2D
from keras.layers import UpSampling2D
from keras.layers import Dropout
from keras.layers import concatenate
from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping
from keras.callbacks import ReduceLROnPlateau
from keras.optimizers import Adam
from skimage.measure import label
from skimage.measure import regionprops

TRAIN_DATA_PATH = 'data/train/'
NEW_TRAIN_DATA_PATH = 'data/new_train/'
TEST_DATA_PATH = 'data/test/'
SUBMISSION_DATA_PATH = 'data/submission/'
MODEL_CHECKPOINT_DIR = 'Checkpoints/'
WEIGHTS = 'Model_Weights.hdf5'
AUGMENT_TRAIN_DATA = False
CREATE_EXTRA_DATA = True
IMG_ROWS = 128
IMG_COLS = 128
IMG_START_NUM = 164
SMOOTH = 1.0
CLEAN_THRESH = 20
THRESH = 100
BATCH_SIZE = 3
EPOCHS = 20
BASE_LR = 1e-04
PATIENCE = 10

srcDir = 'data/train/'
maskTrainDir = 'data/train_mask/'

######################Processing Train data to get binary masked images #####################################
for filename in os.listdir(srcDir):
	im_gray = cv2.imread(srcDir + filename,0)
	(thresh, im_bw) = cv2.threshold(im_gray, 128, 255, cv2.THRESH_BINARY)
	cv2.imwrite(srcDir + filename.split('.')[0] + '-mask.jpg', cv2.bitwise_not(im_bw))

image_names = os.listdir(TRAIN_DATA_PATH)
image_names.sort()

img_num = IMG_START_NUM
for img_name in image_names:
    if 'mask' in img_name:
        continue

    img = cv2.imread(TRAIN_DATA_PATH + img_name)
    mask_img = cv2.imread(TRAIN_DATA_PATH + img_name.split('.')[0] + '-mask.jpg')
    if img.shape == (IMG_ROWS, IMG_COLS, 3):
        cv2.imwrite(NEW_TRAIN_DATA_PATH + img_name, img)
        cv2.imwrite(NEW_TRAIN_DATA_PATH + img_name.split('.')[0] + '-mask.jpg', mask_img)
        continue
    if CREATE_EXTRA_DATA == False:
        continue

    for row in range(0, img.shape[0], IMG_ROWS):
        for col in range(0, img.shape[1], IMG_COLS):
            new_img = img[row:row + IMG_ROWS, col:col + IMG_COLS, :]
            new_mask_img = mask_img[row:row + IMG_ROWS, col:col + IMG_COLS, :]
            if new_img.shape != (IMG_ROWS, IMG_COLS, 3) or np.max(new_mask_img) != 255.0:
                continue
            cv2.imwrite(NEW_TRAIN_DATA_PATH + 'train-' + str(img_num) + '.jpg', new_img)
            cv2.imwrite(NEW_TRAIN_DATA_PATH + 'train-' + str(img_num) + '-mask.jpg', new_mask_img)
            img_num = img_num + 1
			
def get_train_data(path=NEW_TRAIN_DATA_PATH, augment=AUGMENT_TRAIN_DATA):
    image_names = os.listdir(path)
    image_names.sort()
    images_count = len(image_names) / 2
    
    if augment == True:
        images_count = images_count * 3
    X_train = np.ndarray((int(images_count), 1, IMG_ROWS, IMG_COLS), dtype=np.uint8)
    Y_train = np.ndarray((int(images_count), 1, IMG_ROWS, IMG_COLS), dtype=np.uint8)

    i = 0
    for img_name in image_names:
        if 'mask' in img_name:
            continue
        mask_img_name = img_name.split('.')[0] + '-mask.jpg'
        img = cv2.imread(path + img_name, cv2.IMREAD_GRAYSCALE)
        img[img <= CLEAN_THRESH] = 255
        mask_img = cv2.imread(path + mask_img_name, cv2.IMREAD_GRAYSCALE)
        X_train[i] = np.array([img])
        Y_train[i] = np.array([mask_img])
        i = i + 1

        if augment == True:
            X_train[i] = np.array([img[:, ::-1]])
            Y_train[i] = np.array([mask_img[:, ::-1]])
            i = i + 1
            X_train[i] = np.array([img[::-1, :]])
            Y_train[i] = np.array([mask_img[::-1, :]])
            i = i + 1

    X_train = X_train.transpose((0, 2, 3, 1))
    Y_train = Y_train.transpose((0, 2, 3, 1))
    return X_train, Y_train
	
def dice_coef(y_true, y_pred):
	y_true_flat = K.flatten(y_true)
	y_pred_flat = K.flatten(y_pred)
	intersection = K.sum(y_true_flat * y_pred_flat)
	return (2.0 * intersection + SMOOTH) / (K.sum(y_true_flat) + K.sum(y_pred_flat) + SMOOTH)

def dice_coef_loss(y_true, y_pred):
	return 1.0 - dice_coef(y_true, y_pred)

def np_dice_coef(y_true, y_pred):
	y_true_flat = y_true.flat[:]
	y_pred_flat = y_pred.flat[:]
	intersection = np.sum(y_true_flat * y_pred_flat)
	return (2.0 * intersection + SMOOTH) / (np.sum(y_true_flat) + np.sum(y_pred_flat) + SMOOTH)

def np_dice_coef_loss(y_true, y_pred):
	return 1.0 - np_dice_coef(y_true, y_pred)
	
def get_model(input_shape=(IMG_ROWS, IMG_COLS, 1), train=True):
    layers = {}
    layers['inputs'] = Input(shape=input_shape, name='inputs')

    layers['conv1_1'] = Conv2D(32, (3, 3), padding='same', activation='relu', name='conv1_1')(layers['inputs'])
    layers['conv1_2'] = Conv2D(32, (3, 3), padding='same', activation='relu', name='conv1_2')(layers['conv1_1'])
    layers['pool_1'] = MaxPool2D(pool_size=(2, 2), name='pool_1')(layers['conv1_2'])
    if train == True:
        layers['dropout_1'] = Dropout(0.25, name='dropout_1')(layers['pool_1'])
        layers['conv2_1'] = Conv2D(64, (3, 3), padding='same', activation='relu', name='conv2_1')(layers['dropout_1'])
    else:
        layers['conv2_1'] = Conv2D(64, (3, 3), padding='same', activation='relu', name='conv2_1')(layers['pool_1'])
    layers['conv2_2'] = Conv2D(64, (3, 3), padding='same', activation='relu', name='conv2_2')(layers['conv2_1'])
    layers['pool_2'] = MaxPool2D(pool_size=(2, 2), name='pool_2')(layers['conv2_2'])
    if train == True:
        layers['dropout_2'] = Dropout(0.25, name='dropout_2')(layers['pool_2'])
        layers['conv3_1'] = Conv2D(128, (3, 3), padding='same', activation='relu', name='conv3_1')(layers['dropout_2'])
    else:
        layers['conv3_1'] = Conv2D(128, (3, 3), padding='same', activation='relu', name='conv3_1')(layers['pool_2'])
    layers['conv3_2'] = Conv2D(128, (3, 3), padding='same', activation='relu', name='conv3_2')(layers['conv3_1'])
    layers['pool_3'] = MaxPool2D(pool_size=(2, 2), name='pool_3')(layers['conv3_2'])
    if train == True:
        layers['dropout_3'] = Dropout(0.25, name='dropout_3')(layers['pool_3'])
        layers['conv4_1'] = Conv2D(256, (3, 3), padding='same', activation='relu', name='conv4_1')(layers['dropout_3'])
    else:
        layers['conv4_1'] = Conv2D(256, (3, 3), padding='same', activation='relu', name='conv4_1')(layers['pool_3'])
    layers['conv4_2'] = Conv2D(256, (3, 3), padding='same', activation='relu', name='conv4_2')(layers['conv4_1'])
    layers['pool_4'] = MaxPool2D(pool_size=(2, 2), name='pool_4')(layers['conv4_2'])
    if train == True:
        layers['dropout_4'] = Dropout(0.25, name='dropout_4')(layers['pool_4'])
        layers['conv5_1'] = Conv2D(512, (3, 3), padding='same', activation='relu', name='conv5_1')(layers['dropout_4'])
    else:
        layers['conv5_1'] = Conv2D(512, (3, 3), padding='same', activation='relu', name='conv5_1')(layers['pool_4'])
    layers['conv5_2'] = Conv2D(512, (3, 3), padding='same', activation='relu', name='conv5_2')(layers['conv5_1'])

    layers['upsample_1'] = UpSampling2D(size=(2, 2), name='upsample_1')(layers['conv5_2'])
    layers['concat_1'] = concatenate([layers['upsample_1'], layers['conv4_2']], name='concat_1')
    layers['conv6_1'] = Conv2D(256, (3, 3), padding='same', activation='relu', name='conv6_1')(layers['concat_1'])
    layers['conv6_2'] = Conv2D(256, (3, 3), padding='same', activation='relu', name='conv6_2')(layers['conv6_1'])
    if train == True:
        layers['dropout_6'] = Dropout(0.25, name='dropout_6')(layers['conv6_2'])
        layers['upsample_2'] = UpSampling2D(size=(2, 2), name='upsample_2')(layers['dropout_6'])
    else:
        layers['upsample_2'] = UpSampling2D(size=(2, 2), name='upsample_2')(layers['conv6_2'])
    layers['concat_2'] = concatenate([layers['upsample_2'], layers['conv3_2']], name='concat_2')
    layers['conv7_1'] = Conv2D(128, (3, 3), padding='same', activation='relu', name='conv7_1')(layers['concat_2'])
    layers['conv7_2'] = Conv2D(128, (3, 3), padding='same', activation='relu', name='conv7_2')(layers['conv7_1'])
    if train == True:
        layers['dropout_7'] = Dropout(0.25, name='dropout_7')(layers['conv7_2'])
        layers['upsample_3'] = UpSampling2D(size=(2, 2), name='upsample_3')(layers['dropout_7'])
    else:
        layers['upsample_3'] = UpSampling2D(size=(2, 2), name='upsample_3')(layers['conv7_2'])
    layers['concat_3'] = concatenate([layers['upsample_3'], layers['conv2_2']], name='concat_3')
    layers['conv8_1'] = Conv2D(64, (3, 3), padding='same', activation='relu', name='conv8_1')(layers['concat_3'])
    layers['conv8_2'] = Conv2D(64, (3, 3), padding='same', activation='relu', name='conv8_2')(layers['conv8_1'])
    if train == True:
        layers['dropout_8'] = Dropout(0.25, name='dropout_8')(layers['conv8_2'])
        layers['upsample_4'] = UpSampling2D(size=(2, 2), name='upsample_4')(layers['dropout_8'])
    else:
        layers['upsample_4'] = UpSampling2D(size=(2, 2), name='upsample_4')(layers['conv8_2'])
    layers['concat_4'] = concatenate([layers['upsample_4'], layers['conv1_2']], name='concat_4')
    layers['conv9_1'] = Conv2D(32, (3, 3), padding='same', activation='relu', name='conv9_1')(layers['concat_4'])
    layers['conv9_2'] = Conv2D(32, (3, 3), padding='same', activation='relu', name='conv9_2')(layers['conv9_1'])
    if train == True:
        layers['dropout_9'] = Dropout(0.25, name='dropout_9')(layers['conv9_2'])
        layers['outputs'] = Conv2D(1, (1, 1), activation='sigmoid', name='outputs')(layers['dropout_9'])
    else:
        layers['outputs'] = Conv2D(1, (1, 1), activation='sigmoid', name='outputs')(layers['conv9_2'])

    model = Model(inputs=layers['inputs'], outputs=layers['outputs'])
    return model
	
X_train, Y_train = get_train_data()
X_train = X_train.astype('float32')
Y_train = Y_train.astype('float32')
X_train /= 255.0
Y_train /= 255.0

model = get_model((IMG_ROWS, IMG_COLS, 1), True)
model.summary()
model.compile(optimizer=Adam(lr=BASE_LR), loss=dice_coef_loss, metrics=[dice_coef])
callbacks = [ModelCheckpoint(MODEL_CHECKPOINT_DIR + '{epoch:02d}_{loss:.06f}.hdf5', monitor='loss', save_best_only=True),
             ReduceLROnPlateau(monitor='loss', factor=0.1, patience=PATIENCE, min_lr=1e-07, verbose=1)]
model.fit(X_train, Y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, callbacks=callbacks)
#model.save_weights(WEIGHTS)

image_names = os.listdir(TEST_DATA_PATH)
image_names.sort()
#model = get_model(train=False)
#model.load_weights(WEIGHTS)

for img_name in image_names:
    mask_img_name = img_name.split('.')[0] + '-mask.jpg'
    img = cv2.imread(TEST_DATA_PATH + img_name, cv2.IMREAD_GRAYSCALE)
    img[img <= CLEAN_THRESH] = 255
    mask_img = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)

    ret, thresh_img = cv2.threshold(img, THRESH, 255, cv2.THRESH_BINARY_INV)
    img_label = label(thresh_img)
    for region in regionprops(img_label):
        minr, minc, maxr, maxc = region.bbox
        if region.area < 500:
            continue
        r, c = (minr + maxr - IMG_ROWS) / 2, (minc + maxc - IMG_COLS) / 2
        if r < 0:
            r = 0
        if c < 0:
            c = 0
        if r + IMG_ROWS > img.shape[0]:
            r = img.shape[0] - IMG_ROWS
        if c + IMG_COLS > img.shape[1]:
            c = img.shape[1] - IMG_COLS
        test_img = img[r:r + IMG_ROWS, c:c + IMG_COLS]
        if test_img.shape != (IMG_ROWS, IMG_COLS):
            continue

        test_img = np.array([[test_img]], dtype=np.float32)
        test_img /= 255.0
        test_mask_img = model.predict(test_img.transpose(0, 2, 3, 1), verbose=1)
        test_mask_img = (test_mask_img * 255.0).astype(np.uint8)
        test_mask_img = test_mask_img.transpose(0, 3, 1, 2)[0][0]
        
        mask_img[r:r + IMG_ROWS, c:c + IMG_COLS] = test_mask_img
        cv2.imwrite(SUBMISSION_DATA_PATH + mask_img_name, mask_img)