from logging import getLogger, INFO
from os import listdir
from os.path import isfile, join

import matplotlib.pylab as plt
import numpy as np
import pandas as pd
from keras import backend as K
from keras.callbacks import ModelCheckpoint, EarlyStopping, History
from keras.layers import (
    Input, MaxPooling2D, UpSampling2D)
from keras.layers import merge, Conv2D, Dropout
from keras.models import Model
from keras.optimizers import Adam
from scipy import misc
import tensorflow as tf
from src.plot_callback import PlotCallback
import imutils

# params
epochs = 50
patience = 60
lambda_ = 0.0005
beta = 0.0

# data
path_prefix = "images/"
train_path = path_prefix + "images"
label_path = path_prefix + "labels"
all_data_files = [f for f in listdir(train_path) if isfile(join(train_path, f))]
labeled_data_files = [f for f in listdir(label_path) if isfile(join(label_path, f))]
prediction_data = list(filter(lambda x: x not in labeled_data_files, all_data_files))

# model path
MODEL_DIR = "model/"
NEW_MODEL_PATH = MODEL_DIR + "new_model"
NEW_MODEL_HIST = MODEL_DIR + "val_hist_{}_.csv"
BEST_MODEL_PATH = MODEL_DIR + "best_model.hdf5"

# check gpu
gpu_cores = K.tensorflow_backend._get_available_gpus()

logger = getLogger('image_segmentation')
logger.setLevel(INFO)
logger.info("Using: " + str(gpu_cores) + " Gpus")
print("Using: " + str(gpu_cores) + " Gpus")
print(prediction_data)


def correlation_coefficient_loss(y_true, y_pred):
    """
    The idea here is to exploit prior knowledge about roof's. They are usually homogeneous.
    To exploit that this custom objective function increases the loss for regions in the prediction
    that do not auto-correlate.
    :rtype: object
    """
    y_true_squeezed = tf.squeeze(y_true)
    y_true_mean1 = tf.reduce_mean(y_true_squeezed, axis=1)
    y_true_centered1 = tf.transpose(tf.subtract(tf.transpose(y_true_squeezed), tf.transpose(y_true_mean1)))
    cov1_true = tf.abs(tf.matmul(y_true_centered1, y_true_centered1, transpose_b=True)) - beta

    y_true_mean2 = tf.reduce_mean(y_true_squeezed, axis=2)
    y_pred_centered2 = tf.transpose(tf.subtract(tf.transpose(y_true_squeezed, [1, 2, 0]), tf.transpose(y_true_mean2)),
                                    [1, 2, 0])
    cov2_true = tf.abs(tf.matmul(y_pred_centered2, y_pred_centered2, transpose_a=True)) - beta
    cov_mean_true = K.mean(K.mean(cov1_true)) + K.mean(K.mean(cov2_true))

    y_pred_squeezed = tf.squeeze(y_pred)
    y_pred_mean1 = tf.reduce_mean(y_pred_squeezed, axis=1)
    y_pred_centered1 = tf.transpose(tf.subtract(tf.transpose(y_pred_squeezed), tf.transpose(y_pred_mean1)))
    cov1 = tf.abs(tf.matmul(y_pred_centered1, y_pred_centered1, transpose_b=True)) - beta

    y_pred_mean2 = tf.reduce_mean(y_pred_squeezed, axis=2)
    y_pred_centered2 = tf.transpose(tf.subtract(tf.transpose(y_pred_squeezed, [1, 2, 0]), tf.transpose(y_pred_mean2)),
                                    [1, 2, 0])
    cov2 = tf.abs(tf.matmul(y_pred_centered2, y_pred_centered2, transpose_a=True)) - beta
    cov_mean = K.mean(K.mean(cov1)) + K.mean(K.mean(cov2))
    return K.binary_crossentropy(y_true, y_pred) + lambda_ * tf.sqrt(tf.square(cov_mean_true / cov_mean - 1))


def load_training_data() -> (np.ndarray, np.ndarray):
    x_train = [misc.imread(train_path + "/" + path)[:, :, :3] for path in labeled_data_files]
    y_train = [misc.imread(label_path + "/" + path).reshape((256, 256, 1)) for path in labeled_data_files]
    y_train = np.asarray(y_train)
    y_train[y_train > 0] = 1.0
    return np.asarray(x_train), y_train


def load_prediction_data() -> (np.ndarray, np.ndarray):
    x_pred = [misc.imread(train_path + "/" + path)[:, :, :3] for path in prediction_data]
    return np.asarray(x_pred)


def create_rotation_croping_generator(n_train: int, batch_size: int) -> (np.ndarray, np.ndarray):
    """
    Generates new "verteilungstreue" Samples from the given corpus by using three transformations.
    First rotation of the image
    Second cropping
    Last mirroring
    :param n_train:
    :return:
    """
    x, y = load_training_data()
    rand_idx = list(range(np.shape(x)[0]))
    np.random.shuffle(rand_idx)
    x = x[rand_idx]
    y = y[rand_idx]
    x_val, y_val = (x[n_train:], y[n_train:])
    x_trn = x[:n_train]
    y_trn = y[:n_train]
    assemble_validation_x = list()
    assemble_validation_y = list()
    angles = np.linspace(0, 300, 4)
    for idx_val, xs_val in enumerate(x_val):
        for angle in angles:
            x_val_rotated = np.copy(xs_val)
            y_val_rotated = np.copy(y_val[idx_val])
            x_val_rotated = imutils.rotate(x_val_rotated, angle)
            y_val_rotated = imutils.rotate(y_val_rotated, angle).reshape((256, 256, 1))
            assemble_validation_x.append(x_val_rotated)
            assemble_validation_y.append(y_val_rotated)

        x_val_flipped = np.copy(xs_val)
        y_val_flipped = np.copy(y_val[idx_val])

        x_val_flipped = np.flipud(x_val_flipped)
        y_val_flipped = np.flipud(y_val_flipped)
        assemble_validation_x.append(x_val_flipped)
        assemble_validation_y.append(y_val_flipped)
        assemble_validation_x.append(xs_val)
        assemble_validation_y.append(y_val[idx_val])

    x_val = np.asarray(assemble_validation_x)
    y_val = np.asarray(assemble_validation_y)

    def generator():
        while True:
            assemble_x_rotate_cropped = list()
            assemble_y_rotate_cropped = list()
            for i in range(batch_size):
                idx = np.random.randint(0, 19)
                rotate_croped_x = np.copy(x_trn[idx])
                rotate_croped_y = np.copy(y_trn[idx])

                cointoss_rotate = np.random.uniform(0., 1.)
                if cointoss_rotate < 0.5:
                    angle = np.random.uniform(0, 359)
                    rotate_croped_x = imutils.rotate(rotate_croped_x, angle)
                    rotate_croped_y = imutils.rotate(rotate_croped_y, angle).reshape((256, 256, 1))

                cointoss_cropp = np.random.uniform(0., 1.)
                if cointoss_cropp < 0.5:
                    m = np.random.randint(0, 4)
                    xdim = np.random.randint(0, 100)
                    if m == 0:
                        rotate_croped_x[:xdim, :, :] = 0
                        rotate_croped_y[:xdim, :, :] = 0
                    if m == 1:
                        rotate_croped_x[-xdim:, :, :] = 0
                        rotate_croped_y[-xdim:, :, :] = 0
                    if m == 2:
                        rotate_croped_x[:, :xdim, :] = 0
                        rotate_croped_y[:, :xdim, :] = 0
                    if m == 3:
                        rotate_croped_x[:, -xdim:, :] = 0
                        rotate_croped_y[:, -xdim:, :] = 0

                cointoss_flipp = np.random.uniform(0., 1.)
                if cointoss_flipp < 0.5:
                    rotate_croped_x = np.flipud(rotate_croped_x)
                    rotate_croped_y = np.flipud(rotate_croped_y)

                assemble_x_rotate_cropped.append(rotate_croped_x)
                assemble_y_rotate_cropped.append(rotate_croped_y)
            yield (np.asarray(assemble_x_rotate_cropped), np.asarray(assemble_y_rotate_cropped))

    return generator, x_val, y_val


def validate_generator(idx: str):
    generator, x_val, y_val = create_rotation_croping_generator(19, 10)

    x_pred = load_prediction_data()
    plot_callback = PlotCallback(x_pred)

    model = unet()
    model_checkpoint = ModelCheckpoint(
        BEST_MODEL_PATH,
        monitor='val_jaccard_coef',
        save_best_only=True
    )
    model_earlystop = EarlyStopping(
        monitor='val_jaccard_coef',
        patience=patience,
        verbose=0,
        mode='max')
    model_history = History()

    logger.info("Fit")
    model.fit_generator(
        generator(),
        steps_per_epoch=100,
        epochs=epochs,
        verbose=1,
        validation_data=(x_val, y_val),
        callbacks=[model_checkpoint, model_earlystop, model_history, plot_callback]
    )
    del x_val
    del y_val

    model.save_weights(NEW_MODEL_PATH)

    # Save evaluation history
    pd.DataFrame(model_history.history) \
        .to_csv(NEW_MODEL_HIST.format(idx), index=False)
    logger.info(">> validate sub-command: {} ... Done")


def unet(input_size=(256, 256, 3)):
    inputs = Input(input_size)
    conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
    conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
    conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
    conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
    conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
    drop5 = Dropout(0.5)(conv5)

    up6 = Conv2D(512, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(drop5))
    merge6 = merge([drop4, up6], mode='concat', concat_axis=3)
    conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
    conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)

    up7 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv6))
    merge7 = merge([conv3, up7], mode='concat', concat_axis=3)
    conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
    conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)

    up8 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv7))
    merge8 = merge([conv2, up8], mode='concat', concat_axis=3)
    conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
    conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)

    up9 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv8))
    merge9 = merge([conv1, up9], mode='concat', concat_axis=3)
    conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
    conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    conv9 = Conv2D(2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    conv10 = Conv2D(1, 1, activation='sigmoid')(conv9)

    model = Model(input=inputs, output=conv10)

    model.compile(optimizer=Adam(lr=1e-4), loss=K.binary_crossentropy,
                  metrics=['accuracy', jaccard_coef, jaccard_coef_int])
    return model


def jaccard_coef(y_true, y_pred):
    """
    computes the jaccard coefficient metric between target and prediction.
    Defined as the intersection divided by union of the two sets.
    Much more sensitive to the task at hand as simple accuracy
    Much more sensitive with respect to the
    :param y_true:
    :param y_pred:
    :return:
    """
    smooth = 1e-12
    intersection = K.sum(y_true * y_pred, axis=[0, -1, -2])
    sum_ = K.sum(y_true + y_pred, axis=[0, -1, -2])
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return K.mean(jac)


def jaccard_coef_int(y_true, y_pred):
    """
    computes the jaccard coefficient metric between target and rounded prediction e.g. {0,1}.
    :param y_true:
    :param y_pred:
    :return:
    """
    smooth = 1e-12
    y_pred_pos = K.round(K.clip(y_pred, 0, 1))
    intersection = K.sum(y_true * y_pred_pos, axis=[0, -1, -2])
    sum_ = K.sum(y_true + y_pred_pos, axis=[0, -1, -2])
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return K.mean(jac)


def predict(path: str, storage_path: str):
    x_pred = load_prediction_data()
    x_pred_pred = x_pred
    model = unet()
    model.load_weights(path)
    y_pred = model.predict(x_pred_pred, batch_size=8, verbose=1)
    del model

    for idx, i in enumerate(y_pred):
        misc.imsave(storage_path + "_" + str(idx), i)

    fig, ax = plt.subplots(1, 1, figsize=(40, 40))
    for idx, im in enumerate(prediction_data):
        fig.add_subplot(len(prediction_data) * 2, 2, 2 * idx + 1)
        plt.imshow(x_pred[idx], interpolation='nearest')
        fig.add_subplot(len(prediction_data) * 2, 2, 2 * idx + 2)
        plt.imshow(y_pred[idx].reshape((256, 256)), interpolation='nearest')

    plt.show()


for i in range(8):
    validate_generator(
        "correlation_coefficient_loss_{}_epochs".format(str(epochs)) + "_{}_lambda".format(
            str(lambda_) + "_{} trial").format(str(i)))
#     # predict(NEW_MODEL_PATH)
#     # predict(BEST_MODEL_PATH)
