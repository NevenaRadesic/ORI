import cv2
import matplotlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from keras_preprocessing.image import load_img, img_to_array
from sklearn.metrics import f1_score

from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras import optimizers
from keras.models import Sequential, Model
from keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D
from keras.callbacks import ModelCheckpoint
import keras.preprocessing.image
from keras.utils import np_utils
from keras.optimizers import SGD

# from IPython.core.display import display, HTML
from PIL import Image
from io import BytesIO
import base64

plt.style.use('ggplot')

# %matplotlib inline

import tensorflow as tf

print(tf.__version__)

# set variables
main_folder = 'C:/Users/LENOVO/Desktop/ORIPROJEKAT/main_folder/'
images_folder = main_folder + 'img_align_celeba/'
# images_folder = 'E:/Downloads/NevenaOriTEmp/img_align_celeba/img_align_celeba/'
EXAMPLE_PIC = images_folder + '000006.jpg'

TRAINING_SAMPLES = 700
VALIDATION_SAMPLES = 150
TEST_SAMPLES = 150
IMG_WIDTH = 178
IMG_HEIGHT = 218
BATCH_SIZE = 16
NUM_EPOCHS = 21

# import the data set that include the attribute for each picture
df_attr = pd.read_csv(main_folder + 'list_attr_celeba.csv')
df_attr.set_index('image_id', inplace=True)
df_attr.replace(to_replace=-1, value=0, inplace=True)  # replace -1 by 0

df_attr.shape

# List of available attributes
for i, j in enumerate(df_attr.columns):
    print(i, j)

# Recomended partition
df_partition = pd.read_csv(main_folder + 'list_eval_partition.csv')
df_partition.head()

# display counter by partition
# 0 -> TRAINING
# 1 -> VALIDATION
# 2 -> TEST
df_partition['partition'].value_counts().sort_index()

# join the partition with the attributes
df_partition.set_index('image_id', inplace=True)
df_par_attr = df_partition.join(df_attr['Male'], how='inner')
df_par_attr.head()


def load_reshape_img(fname):
    img = load_img(fname)
    x = img_to_array(img) / 255.
    x = x.reshape((1,) + x.shape)

    return x


def generate_df(partition, attr, num_samples):
    '''
    partition
        0 -> train
        1 -> validation
        2 -> test

    '''

    df_ = df_par_attr[(df_par_attr['partition'] == partition) & (df_par_attr[attr] == 0)].sample(int(num_samples / 2))

    df_ = pd.concat([df_,
                     df_par_attr[(df_par_attr['partition'] == partition) & (df_par_attr[attr] == 1)].sample(int(num_samples / 2), replace=True)])

    # for Train and Validation
    if partition != 2:
        x_ = np.array([load_reshape_img(images_folder + fname) for fname in df_.index])
        x_ = x_.reshape(x_.shape[0], 218, 178, 3)
        y_ = np_utils.to_categorical(df_[attr], 2)
    # for Test
    else:
        x_ = []
        y_ = []

        for index, target in df_.iterrows():
            im = cv2.imread(images_folder + index)
            im = cv2.resize(cv2.cvtColor(im, cv2.COLOR_BGR2RGB), (IMG_WIDTH, IMG_HEIGHT)).astype(np.float32) / 255.0
            im = np.expand_dims(im, axis=0)
            x_.append(im)
            y_.append(target[attr])

    return x_, y_   # slika, 0 ili jedan (nije li jeste musko)


# # Train data
# x_train, y_train = generate_df(0, 'Male', TRAINING_SAMPLES) # slika, 0 ili jedan (nije li jeste musko)
#
# # Validation Data
# x_valid, y_valid = generate_df(1, 'Male', VALIDATION_SAMPLES)  # slika, 0 ili jedan (nije li jeste musko) za validauju

# Import InceptionV3 Model
inc_model = InceptionV3(
    weights='C:/Users/LENOVO/Desktop/ORIPROJEKAT/inceptionv3/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5',
    include_top=False,
    input_shape=(IMG_HEIGHT, IMG_WIDTH, 3))

print("number of layers:", len(inc_model.layers))
inc_model.summary()


# Adding custom Layers
x = inc_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation="relu")(x)
x = Dropout(0.5)(x)
x = Dense(512, activation="relu")(x)
predictions = Dense(2, activation="softmax")(x)

# creating the final model
model_ = Model(inputs=inc_model.input, outputs=predictions)

# # compile the model
# model_.compile(optimizer=SGD(lr=0.0001, momentum=0.9)
#                , loss='categorical_crossentropy'
#                , metrics=['accuracy'])
#
# # https://keras.io/models/sequential/ fit generator
# checkpointer = ModelCheckpoint(filepath='weights.best.inc.male.hdf5',
#                                verbose=1, save_best_only=True)
#
# hist = model_.fit(x = x_train,y = y_train,callbacks=[checkpointer],batch_size = BATCH_SIZE, epochs=NUM_EPOCHS, verbose=1, validation_data=(x_valid, y_valid), steps_per_epoch=TRAINING_SAMPLES / BATCH_SIZE)
#
# # Plot loss function value through epochs
# plt.figure(figsize=(18, 4))
# plt.plot(hist.history['loss'], label='train')
# plt.plot(hist.history['val_loss'], label='valid')
# plt.legend()
# plt.title('Loss Function')
# plt.show()
#
# # Plot accuracy through epochs
# plt.figure(figsize=(18, 4))
# plt.plot(hist.history['accuracy'], label='train')
# plt.plot(hist.history['val_accuracy'], label='valid')
# plt.legend()
# plt.title('Accuracy')
# plt.show()

# load the best model
model_.load_weights('weights.best.inc.male.hdf5')

# Test Data
x_test, y_test = generate_df(2, 'Male', TEST_SAMPLES)

# generate prediction
model_predictions = [np.argmax(model_.predict(feature)) for feature in x_test]

# report test accuracy
test_accuracy = 100 * np.sum(np.array(model_predictions) == y_test) / len(model_predictions)
print('Model Evaluation')
print('Test accuracy: %.4f%%' % test_accuracy)
print('f1_score:', f1_score(y_test, model_predictions))
