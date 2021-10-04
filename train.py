# -*- coding:utf-8 -*-
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import os
import tensorflow as tf
from sklearn.model_selection import train_test_split
from AlexNet import AlexNet
from keras.optimizers import Adam


"""define image properties"""
Image_Width = 224
Image_Height = 224
Image_Size = (Image_Width, Image_Height)
Image_Channel = 3
batch_size = 50

filenames = os.listdir('./dogs-vs-cats/train')

categories = []
for f_name in filenames:
    category = f_name.split('.')[0]
    if category == 'dog':
        categories.append(1)
    else:
        categories.append(0)

df = pd.DataFrame({'filename': filenames, 'category': categories})


"""model prams"""
epochs = 15
model = AlexNet(2)
model.compile(loss='categorical_crossentropy',
              optimizer=Adam(lr=1e-3),
              metrics=['accuracy'])
model.summary()

"""define callbacks and learning rate"""
earlystop = tf.keras.callbacks.EarlyStopping(patience=10)
learning_rate_reduction = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_acc', patience=2,
                                                               verbose=1, factor=0.5, min_lr=0.00001)
callbacks = [earlystop, learning_rate_reduction]


"""manage data"""
df['category'] = df['category'].replace({0: 'cat', 1: 'dog'})
train_df, validate_df = train_test_split(df, test_size=0.20, random_state=42)
train_df = train_df.reset_index(drop=True)
validate_df = validate_df.reset_index(drop=True)

total_train = train_df.shape[0]
total_validate = validate_df.shape[0]

"""prepare training dataset"""
"""data augmentation"""
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rotation_range=15,
                                                               rescale=1./255,
                                                               shear_range=0.1,
                                                                zoom_range=0.2,
                                                                horizontal_flip=True,
                                                                width_shift_range=0.1,
                                                                height_shift_range=0.1)

train_generator = train_datagen.flow_from_dataframe(train_df, './dogs-vs-cats/train/',
                                                    x_col='filename',
                                                    y_col='category',
                                                    target_size=Image_Size,
                                                    class_mode='categorical',
                                                    batch_size=batch_size)

validation_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

validation_generator = validation_datagen.flow_from_dataframe(validate_df, './dogs-vs-cats/train/',
                                                                x_col='filename',
                                                                y_col='category',
                                                                target_size=Image_Size,
                                                                class_mode='categorical',
                                                                batch_size=batch_size)

"""model training"""
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    history = model.fit_generator(train_generator,
                              epochs=epochs,
                              validation_data=validation_generator,
                              validation_steps=total_validate//batch_size,
                              steps_per_epoch=total_train//batch_size,
                              callbacks=callbacks)

    """save the model"""
    model.save('model_catsVSdogs_epoch.h5')



