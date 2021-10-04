# -*- coding:utf-8 -*-
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import os
import tensorflow as tf
from sklearn.model_selection import train_test_split
from AlexNet import AlexNet

"""define image properties"""
Image_Width = 224
Image_Height = 224
Image_Size = (Image_Width, Image_Height)
Image_Channel = 3
batch_size = 100

"""test data preparation"""
model = tf.keras.models.load_model('model_catsVSdogs_epoch.h5')
test_filenames = os.listdir('./dogs-vs-cats/test1')
test_df = pd.DataFrame({'filename': test_filenames})

test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rotation_range=15,
                                                               rescale=1./255,
                                                               shear_range=0.1,
                                                                zoom_range=0.2,
                                                                horizontal_flip=True,
                                                                width_shift_range=0.1,
                                                                height_shift_range=0.1)
test_generator = test_datagen.flow_from_dataframe(test_df,'./dogs-vs-cats/test1/',
                                                    x_col='filename',
                                                    y_col='category',
                                                    target_size=Image_Size,
                                                    class_mode=None,
                                                    batch_size=batch_size)

nb_samples = test_df.shape[0]
predict = model.predict_generator(test_generator, steps=np.ceil(nb_samples/batch_size))
test_df['category']=np.argmax(predict, axis=-1)
# label_map = dict((v,k) for k,v in train_generator.class_indices.items())
# label_map = (train_generator.class_indices)
label_map = {'cat': 0,'dog': 1}
label_map = dict((v, k) for k,v in label_map.items()) #flip k,v
test_df['category'] = test_df['category'].replace(label_map)
test_df['category'] = test_df['category'].replace({'dog': 1, 'cat': 0})

"""visualize the prediction results"""
sample_test = test_df.head(18)
sample_test.head()
plt.figure(figsize=(12,24))
for index, row in sample_test.iterrows():
    filename = row['filename']
    category = row['category']
    img = tf.keras.preprocessing.image.load_img('./dogs-vs-cats/test1/'+filename,
                                                target_size=Image_Size)
    plt.subplot(6, 3, index+1)
    plt.imshow(img)
    plt.xlabel(filename + '(' + '{}'.format(category) + ')')
plt.tight_layout()
plt.savefig('predict_results.png')
plt.show()



