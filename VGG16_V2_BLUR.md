<h1> Modelo VGG16 </h1>

<h3> Imports </h3>


```python
import numpy as np
import pandas as pd
import random
from tensorflow.keras.applications import VGG16
from tensorflow.keras import models, layers, optimizers, regularizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ReduceLROnPlateau , ModelCheckpoint, EarlyStopping
import matplotlib.pyplot as plt
from pathlib import Path
import tensorflow as tf
import cv2 as cv
```


```python
def random_crop(image):
    
    image = cv.bilateralFilter(image,7,20,20)
    
    rand_number = random.randint(0, 1)
    if rand_number:
        height, width, _ = image.shape
        esquina_sup_izq = (random.randint(0, height), random.randint(0, width))
        image[esquina_sup_izq[0]:esquina_sup_izq[0]+int(0.4*height), esquina_sup_izq[1]:esquina_sup_izq[1]+int(0.4*width), :] = 0

    return image
```


```python
if tf.test.gpu_device_name():
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
else:
    print("Please install GPU version of TF")
```

    Default GPU Device: /device:GPU:0
    


```python
tf.test.is_built_with_cuda()
```




    True




```python
tf.test.is_gpu_available(cuda_only=False, min_cuda_compute_capability=None)
```

    WARNING:tensorflow:From <ipython-input-5-78f884b5c1a9>:1: is_gpu_available (from tensorflow.python.framework.test_util) is deprecated and will be removed in a future version.
    Instructions for updating:
    Use `tf.config.list_physical_devices('GPU')` instead.
    




    True



<h3> Data directories </h3>


```python
data_dir = Path('D:/DATASETS/TFG/TRAIN_CLASIFICACION_A_T_O')

train_dir = data_dir / 'IMAGES'

labels_dir = data_dir / 'etiquetas.xlsx'
```


```python
df = pd.read_excel(labels_dir, index_col=0)
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Image name</th>
      <th>Label</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>street_71.jpg</td>
      <td>Other</td>
    </tr>
    <tr>
      <th>1</th>
      <td>motorbike_130.jpg</td>
      <td>Other</td>
    </tr>
    <tr>
      <th>2</th>
      <td>road_108.png</td>
      <td>Other</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Image0164.png</td>
      <td>Tower</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Image0582_A.png</td>
      <td>Insulator</td>
    </tr>
  </tbody>
</table>
</div>




```python
train_datagen = ImageDataGenerator(
    preprocessing_function=random_crop,
    rescale=1./255,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    brightness_range=(0.5,1),
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode='constant',
    cval=0,
    validation_split=0.3,
    dtype='float32')
```


```python
train_generator = train_datagen.flow_from_dataframe(
    dataframe=df,
    directory=train_dir,
    x_col='Image name',
    y_col='Label',
    subset='training',
    target_size=(150,150),
    batch_size=64,
    class_mode='categorical',
    seed=0)

val_generator = train_datagen.flow_from_dataframe(
    dataframe=df,
    directory=train_dir,
    x_col='Image name',
    y_col='Label',
    subset='validation',
    target_size=(150,150),
    batch_size=1,
    class_mode='categorical',
    seed=0)
```

    Found 2001 validated image filenames belonging to 3 classes.
    Found 857 validated image filenames belonging to 3 classes.
    


```python
# fig=plt.figure(figsize=(20, 20))
# columns = 4
# rows = 5

# for i in range(1, columns*rows +1):
#     img = next(train_generator)[0][i]
#     fig.add_subplot(rows, columns, i)
#     plt.imshow(img)
# plt.show()
```


```python
n_classes = np.unique(train_generator.classes, return_counts=True)[1]
class_weights = {0: n_classes.sum()/n_classes[0], 1: n_classes.sum()/n_classes[1], 2: n_classes.sum()/n_classes[2]}
```


```python
class_weights
```




    {0: 3.0, 1: 1.9502923976608186, 2: 6.496753246753247}



<h3> Creo el modelo </h3>


```python
for i in range(5):
    
    conv_base = VGG16(weights='imagenet',
                  include_top=False,
                  input_shape=(150,150,3))
    x = conv_base.output

    x = layers.Flatten()(x)

    x = layers.Dense(4096, kernel_initializer='glorot_normal', kernel_regularizer=regularizers.l2(0.001))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    x = layers.Dropout(0.4)(x)

    x = layers.Dense(2048, kernel_initializer='glorot_normal', kernel_regularizer=regularizers.l2(0.001))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    x = layers.Dropout(0.4)(x)

    x = layers.Dense(1024, kernel_initializer='glorot_normal', kernel_regularizer=regularizers.l2(0.001))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    x = layers.Dropout(0.4)(x)

    predictions = layers.Dense(3, kernel_initializer='glorot_normal', activation='softmax')(x)

    model = models.Model(inputs=conv_base.input, outputs=predictions)

    conv_base.trainable = True
    set_trainable = False
    for layer in conv_base.layers:

        if layer.name in ['block4_conv1', 'block4_conv2', 'block4_conv3', 'block4_pool',
                          'block5_conv1', 'block5_conv2', 'block5_conv3', 'block5_pool',]:

            set_trainable = True

        if set_trainable:

            layer.trainable = True

        else:

            layer.trainable = False

    opt = optimizers.RMSprop(learning_rate=5e-5, rho=0.9)

    lr_reduce = ReduceLROnPlateau(monitor='accuracy', factor=0.1, min_delta=0.001, patience=5, cooldown=2, verbose=1)

    early_stop = EarlyStopping(monitor='val_loss', mode='min', patience=15, verbose=1)

    filepath="VGG16_V2_BIBLUR_" + str(i+1) + ".hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')

    model.compile(loss='categorical_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])
    model.fit(train_generator,
              class_weight=class_weights,
              steps_per_epoch=30,
              epochs=150,
              validation_data=val_generator,
              validation_steps=857,
              callbacks=[lr_reduce,checkpoint,early_stop])
```

    WARNING:tensorflow:sample_weight modes were coerced from
      ...
        to  
      ['...']
    WARNING:tensorflow:sample_weight modes were coerced from
      ...
        to  
      ['...']
    Train for 30 steps, validate for 857 steps
    Epoch 1/150
    29/30 [============================>.] - ETA: 3s - loss: 11.9227 - accuracy: 0.6833
    Epoch 00001: val_loss improved from inf to 13.87456, saving model to VGG16_V2_BIBLUR_1.hdf5
    30/30 [==============================] - 150s 5s/step - loss: 11.9090 - accuracy: 0.6829 - val_loss: 13.8746 - val_accuracy: 0.5741
    Epoch 2/150
    29/30 [============================>.] - ETA: 1s - loss: 10.7082 - accuracy: 0.8154
    Epoch 00002: val_loss improved from 13.87456 to 12.50198, saving model to VGG16_V2_BIBLUR_1.hdf5
    30/30 [==============================] - 97s 3s/step - loss: 10.7012 - accuracy: 0.8174 - val_loss: 12.5020 - val_accuracy: 0.6348
    Epoch 3/150
    29/30 [============================>.] - ETA: 2s - loss: 10.1310 - accuracy: 0.8856
    Epoch 00003: val_loss did not improve from 12.50198
    30/30 [==============================] - 94s 3s/step - loss: 10.1198 - accuracy: 0.8857 - val_loss: 13.5603 - val_accuracy: 0.6721
    Epoch 4/150
    29/30 [============================>.] - ETA: 1s - loss: 9.8886 - accuracy: 0.8994
    Epoch 00004: val_loss improved from 12.50198 to 10.74360, saving model to VGG16_V2_BIBLUR_1.hdf5
    30/30 [==============================] - 98s 3s/step - loss: 9.8904 - accuracy: 0.8980 - val_loss: 10.7436 - val_accuracy: 0.8530
    Epoch 5/150
    29/30 [============================>.] - ETA: 2s - loss: 9.5785 - accuracy: 0.9259
    Epoch 00005: val_loss improved from 10.74360 to 9.33816, saving model to VGG16_V2_BIBLUR_1.hdf5
    30/30 [==============================] - 98s 3s/step - loss: 9.5678 - accuracy: 0.9274 - val_loss: 9.3382 - val_accuracy: 0.9487
    Epoch 6/150
    29/30 [============================>.] - ETA: 2s - loss: 9.2978 - accuracy: 0.9342
    Epoch 00006: val_loss did not improve from 9.33816
    30/30 [==============================] - 94s 3s/step - loss: 9.3095 - accuracy: 0.9333 - val_loss: 9.9110 - val_accuracy: 0.9218
    Epoch 7/150
    29/30 [============================>.] - ETA: 1s - loss: 9.1542 - accuracy: 0.9353
    Epoch 00007: val_loss did not improve from 9.33816
    30/30 [==============================] - 87s 3s/step - loss: 9.1538 - accuracy: 0.9365 - val_loss: 9.7859 - val_accuracy: 0.9218
    Epoch 8/150
    29/30 [============================>.] - ETA: 1s - loss: 8.9867 - accuracy: 0.9447
    Epoch 00008: val_loss did not improve from 9.33816
    30/30 [==============================] - 87s 3s/step - loss: 8.9719 - accuracy: 0.9461 - val_loss: 9.7295 - val_accuracy: 0.9067
    Epoch 9/150
    29/30 [============================>.] - ETA: 1s - loss: 8.7740 - accuracy: 0.9480
    Epoch 00009: val_loss did not improve from 9.33816
    30/30 [==============================] - 86s 3s/step - loss: 8.7931 - accuracy: 0.9471 - val_loss: 9.5512 - val_accuracy: 0.9288
    Epoch 10/150
    29/30 [============================>.] - ETA: 1s - loss: 8.5923 - accuracy: 0.9574
    Epoch 00010: val_loss improved from 9.33816 to 9.23144, saving model to VGG16_V2_BIBLUR_1.hdf5
    30/30 [==============================] - 96s 3s/step - loss: 8.5948 - accuracy: 0.9557 - val_loss: 9.2314 - val_accuracy: 0.9125
    Epoch 11/150
    29/30 [============================>.] - ETA: 1s - loss: 8.4850 - accuracy: 0.9541
    Epoch 00011: val_loss improved from 9.23144 to 8.43417, saving model to VGG16_V2_BIBLUR_1.hdf5
    30/30 [==============================] - 95s 3s/step - loss: 8.4715 - accuracy: 0.9557 - val_loss: 8.4342 - val_accuracy: 0.9615
    Epoch 12/150
    29/30 [============================>.] - ETA: 1s - loss: 8.1897 - accuracy: 0.9591
    Epoch 00012: val_loss improved from 8.43417 to 8.25951, saving model to VGG16_V2_BIBLUR_1.hdf5
    30/30 [==============================] - 95s 3s/step - loss: 8.1867 - accuracy: 0.9594 - val_loss: 8.2595 - val_accuracy: 0.9638
    Epoch 13/150
    29/30 [============================>.] - ETA: 1s - loss: 8.0957 - accuracy: 0.9652
    Epoch 00013: val_loss improved from 8.25951 to 8.02618, saving model to VGG16_V2_BIBLUR_1.hdf5
    30/30 [==============================] - 97s 3s/step - loss: 8.0831 - accuracy: 0.9664 - val_loss: 8.0262 - val_accuracy: 0.9615
    Epoch 14/150
    29/30 [============================>.] - ETA: 1s - loss: 7.9122 - accuracy: 0.9652
    Epoch 00014: val_loss did not improve from 8.02618
    30/30 [==============================] - 92s 3s/step - loss: 7.9095 - accuracy: 0.9653 - val_loss: 8.1172 - val_accuracy: 0.9417
    Epoch 15/150
    29/30 [============================>.] - ETA: 2s - loss: 7.7859 - accuracy: 0.9668
    Epoch 00015: val_loss did not improve from 8.02618
    30/30 [==============================] - 94s 3s/step - loss: 7.7894 - accuracy: 0.9658 - val_loss: 8.5110 - val_accuracy: 0.9417
    Epoch 16/150
    29/30 [============================>.] - ETA: 2s - loss: 7.6521 - accuracy: 0.9641
    Epoch 00016: val_loss improved from 8.02618 to 7.99043, saving model to VGG16_V2_BIBLUR_1.hdf5
    30/30 [==============================] - 98s 3s/step - loss: 7.6404 - accuracy: 0.9653 - val_loss: 7.9904 - val_accuracy: 0.9673
    Epoch 17/150
    29/30 [============================>.] - ETA: 2s - loss: 7.4792 - accuracy: 0.9740
    Epoch 00017: val_loss improved from 7.99043 to 7.85026, saving model to VGG16_V2_BIBLUR_1.hdf5
    30/30 [==============================] - 98s 3s/step - loss: 7.4755 - accuracy: 0.9728 - val_loss: 7.8503 - val_accuracy: 0.9568
    Epoch 18/150
    29/30 [============================>.] - ETA: 2s - loss: 7.3547 - accuracy: 0.9698
    Epoch 00018: val_loss did not improve from 7.85026
    30/30 [==============================] - 95s 3s/step - loss: 7.3495 - accuracy: 0.9698 - val_loss: 7.9725 - val_accuracy: 0.9428
    Epoch 19/150
    29/30 [============================>.] - ETA: 2s - loss: 7.2512 - accuracy: 0.9746
    Epoch 00019: val_loss improved from 7.85026 to 7.58514, saving model to VGG16_V2_BIBLUR_1.hdf5
    30/30 [==============================] - 98s 3s/step - loss: 7.2485 - accuracy: 0.9744 - val_loss: 7.5851 - val_accuracy: 0.9475
    Epoch 20/150
    29/30 [============================>.] - ETA: 2s - loss: 7.1470 - accuracy: 0.9707
    Epoch 00020: val_loss improved from 7.58514 to 7.53653, saving model to VGG16_V2_BIBLUR_1.hdf5
    30/30 [==============================] - 99s 3s/step - loss: 7.1429 - accuracy: 0.9701 - val_loss: 7.5365 - val_accuracy: 0.9498
    Epoch 21/150
    29/30 [============================>.] - ETA: 2s - loss: 7.0624 - accuracy: 0.9757
    Epoch 00021: val_loss did not improve from 7.53653
    30/30 [==============================] - 94s 3s/step - loss: 7.0513 - accuracy: 0.9765 - val_loss: 7.6246 - val_accuracy: 0.9673
    Epoch 22/150
    29/30 [============================>.] - ETA: 2s - loss: 6.8861 - accuracy: 0.9806
    Epoch 00022: val_loss did not improve from 7.53653
    30/30 [==============================] - 95s 3s/step - loss: 6.8806 - accuracy: 0.9807 - val_loss: 7.8350 - val_accuracy: 0.9137
    Epoch 23/150
    29/30 [============================>.] - ETA: 1s - loss: 6.7511 - accuracy: 0.9801
    Epoch 00023: val_loss improved from 7.53653 to 7.28861, saving model to VGG16_V2_BIBLUR_1.hdf5
    30/30 [==============================] - 97s 3s/step - loss: 6.7466 - accuracy: 0.9808 - val_loss: 7.2886 - val_accuracy: 0.9487
    Epoch 24/150
    29/30 [============================>.] - ETA: 2s - loss: 6.7123 - accuracy: 0.9757
    Epoch 00024: val_loss improved from 7.28861 to 6.95253, saving model to VGG16_V2_BIBLUR_1.hdf5
    30/30 [==============================] - 98s 3s/step - loss: 6.7045 - accuracy: 0.9760 - val_loss: 6.9525 - val_accuracy: 0.9662
    Epoch 25/150
    29/30 [============================>.] - ETA: 2s - loss: 6.5577 - accuracy: 0.9768
    Epoch 00025: val_loss improved from 6.95253 to 6.93143, saving model to VGG16_V2_BIBLUR_1.hdf5
    30/30 [==============================] - 98s 3s/step - loss: 6.5667 - accuracy: 0.9765 - val_loss: 6.9314 - val_accuracy: 0.9568
    Epoch 26/150
    29/30 [============================>.] - ETA: 1s - loss: 6.4857 - accuracy: 0.9779
    Epoch 00026: val_loss did not improve from 6.93143
    30/30 [==============================] - 94s 3s/step - loss: 6.4794 - accuracy: 0.9786 - val_loss: 7.1613 - val_accuracy: 0.9300
    Epoch 27/150
    29/30 [============================>.] - ETA: 1s - loss: 6.3592 - accuracy: 0.9818
    Epoch 00027: ReduceLROnPlateau reducing learning rate to 4.999999873689376e-06.
    
    Epoch 00027: val_loss did not improve from 6.93143
    30/30 [==============================] - 93s 3s/step - loss: 6.3600 - accuracy: 0.9813 - val_loss: 8.1611 - val_accuracy: 0.9323
    Epoch 28/150
    29/30 [============================>.] - ETA: 2s - loss: 6.3205 - accuracy: 0.9801
    Epoch 00028: val_loss improved from 6.93143 to 6.73553, saving model to VGG16_V2_BIBLUR_1.hdf5
    30/30 [==============================] - 98s 3s/step - loss: 6.3194 - accuracy: 0.9797 - val_loss: 6.7355 - val_accuracy: 0.9650
    Epoch 29/150
    29/30 [============================>.] - ETA: 2s - loss: 6.2130 - accuracy: 0.9884
    Epoch 00029: val_loss improved from 6.73553 to 6.47329, saving model to VGG16_V2_BIBLUR_1.hdf5
    30/30 [==============================] - 97s 3s/step - loss: 6.2120 - accuracy: 0.9883 - val_loss: 6.4733 - val_accuracy: 0.9673
    Epoch 30/150
    29/30 [============================>.] - ETA: 1s - loss: 6.2055 - accuracy: 0.9906
    Epoch 00030: val_loss did not improve from 6.47329
    30/30 [==============================] - 87s 3s/step - loss: 6.2028 - accuracy: 0.9909 - val_loss: 6.5498 - val_accuracy: 0.9615
    Epoch 31/150
    29/30 [============================>.] - ETA: 1s - loss: 6.1690 - accuracy: 0.9939
    Epoch 00031: val_loss improved from 6.47329 to 6.38330, saving model to VGG16_V2_BIBLUR_1.hdf5
    30/30 [==============================] - 91s 3s/step - loss: 6.1665 - accuracy: 0.9941 - val_loss: 6.3833 - val_accuracy: 0.9790
    Epoch 32/150
    29/30 [============================>.] - ETA: 1s - loss: 6.1525 - accuracy: 0.9934
    Epoch 00032: val_loss improved from 6.38330 to 6.35387, saving model to VGG16_V2_BIBLUR_1.hdf5
    30/30 [==============================] - 90s 3s/step - loss: 6.1515 - accuracy: 0.9936 - val_loss: 6.3539 - val_accuracy: 0.9720
    Epoch 33/150
    29/30 [============================>.] - ETA: 1s - loss: 6.1294 - accuracy: 0.9912
    Epoch 00033: val_loss did not improve from 6.35387
    30/30 [==============================] - 86s 3s/step - loss: 6.1274 - accuracy: 0.9915 - val_loss: 6.3570 - val_accuracy: 0.9767
    Epoch 34/150
    29/30 [============================>.] - ETA: 1s - loss: 6.1085 - accuracy: 0.9939
    Epoch 00034: val_loss did not improve from 6.35387
    30/30 [==============================] - 87s 3s/step - loss: 6.1192 - accuracy: 0.9931 - val_loss: 6.3942 - val_accuracy: 0.9767
    Epoch 35/150
    29/30 [============================>.] - ETA: 1s - loss: 6.1169 - accuracy: 0.9895
    Epoch 00035: val_loss did not improve from 6.35387
    30/30 [==============================] - 87s 3s/step - loss: 6.1138 - accuracy: 0.9899 - val_loss: 6.3895 - val_accuracy: 0.9708
    Epoch 36/150
    29/30 [============================>.] - ETA: 1s - loss: 6.0843 - accuracy: 0.9912
    Epoch 00036: ReduceLROnPlateau reducing learning rate to 4.999999873689376e-07.
    
    Epoch 00036: val_loss did not improve from 6.35387
    30/30 [==============================] - 87s 3s/step - loss: 6.0824 - accuracy: 0.9915 - val_loss: 6.4676 - val_accuracy: 0.9580
    Epoch 37/150
    29/30 [============================>.] - ETA: 1s - loss: 6.0485 - accuracy: 0.9957
    Epoch 00037: val_loss improved from 6.35387 to 6.31527, saving model to VGG16_V2_BIBLUR_1.hdf5
    30/30 [==============================] - 92s 3s/step - loss: 6.0516 - accuracy: 0.9953 - val_loss: 6.3153 - val_accuracy: 0.9720
    Epoch 38/150
    29/30 [============================>.] - ETA: 1s - loss: 6.1196 - accuracy: 0.9917
    Epoch 00038: val_loss did not improve from 6.31527
    30/30 [==============================] - 87s 3s/step - loss: 6.1159 - accuracy: 0.9920 - val_loss: 6.4674 - val_accuracy: 0.9638
    Epoch 39/150
    29/30 [============================>.] - ETA: 1s - loss: 6.0647 - accuracy: 0.9950
    Epoch 00039: val_loss improved from 6.31527 to 6.30554, saving model to VGG16_V2_BIBLUR_1.hdf5
    30/30 [==============================] - 92s 3s/step - loss: 6.0631 - accuracy: 0.9952 - val_loss: 6.3055 - val_accuracy: 0.9743
    Epoch 40/150
    29/30 [============================>.] - ETA: 1s - loss: 6.0769 - accuracy: 0.9917
    Epoch 00040: val_loss did not improve from 6.30554
    30/30 [==============================] - 87s 3s/step - loss: 6.0750 - accuracy: 0.9920 - val_loss: 6.4180 - val_accuracy: 0.9755
    Epoch 41/150
    29/30 [============================>.] - ETA: 1s - loss: 6.0739 - accuracy: 0.9934
    Epoch 00041: val_loss did not improve from 6.30554
    30/30 [==============================] - 86s 3s/step - loss: 6.0714 - accuracy: 0.9936 - val_loss: 6.4337 - val_accuracy: 0.9743
    Epoch 42/150
    29/30 [============================>.] - ETA: 1s - loss: 6.0610 - accuracy: 0.9956
    Epoch 00042: ReduceLROnPlateau reducing learning rate to 4.999999987376214e-08.
    
    Epoch 00042: val_loss did not improve from 6.30554
    30/30 [==============================] - 86s 3s/step - loss: 6.0600 - accuracy: 0.9957 - val_loss: 6.4486 - val_accuracy: 0.9720
    Epoch 43/150
    29/30 [============================>.] - ETA: 1s - loss: 6.0457 - accuracy: 0.9962
    Epoch 00043: val_loss did not improve from 6.30554
    30/30 [==============================] - 87s 3s/step - loss: 6.0568 - accuracy: 0.9958 - val_loss: 6.4644 - val_accuracy: 0.9685
    Epoch 44/150
    29/30 [============================>.] - ETA: 1s - loss: 6.0471 - accuracy: 0.9961
    Epoch 00044: val_loss did not improve from 6.30554
    30/30 [==============================] - 87s 3s/step - loss: 6.0466 - accuracy: 0.9963 - val_loss: 6.3467 - val_accuracy: 0.9685
    Epoch 45/150
    29/30 [============================>.] - ETA: 1s - loss: 6.0972 - accuracy: 0.9878
    Epoch 00045: val_loss did not improve from 6.30554
    30/30 [==============================] - 87s 3s/step - loss: 6.0939 - accuracy: 0.9883 - val_loss: 6.3168 - val_accuracy: 0.9755
    Epoch 46/150
    29/30 [============================>.] - ETA: 1s - loss: 6.0600 - accuracy: 0.9945
    Epoch 00046: val_loss improved from 6.30554 to 6.29788, saving model to VGG16_V2_BIBLUR_1.hdf5
    30/30 [==============================] - 90s 3s/step - loss: 6.0583 - accuracy: 0.9947 - val_loss: 6.2979 - val_accuracy: 0.9755
    Epoch 47/150
    29/30 [============================>.] - ETA: 1s - loss: 6.0685 - accuracy: 0.9946
    Epoch 00047: val_loss did not improve from 6.29788
    30/30 [==============================] - 88s 3s/step - loss: 6.0678 - accuracy: 0.9948 - val_loss: 6.3507 - val_accuracy: 0.9685
    Epoch 48/150
    29/30 [============================>.] - ETA: 1s - loss: 6.0447 - accuracy: 0.9939
    Epoch 00048: ReduceLROnPlateau reducing learning rate to 5.000000058430488e-09.
    
    Epoch 00048: val_loss did not improve from 6.29788
    30/30 [==============================] - 87s 3s/step - loss: 6.0430 - accuracy: 0.9941 - val_loss: 6.3794 - val_accuracy: 0.9673
    Epoch 49/150
    29/30 [============================>.] - ETA: 1s - loss: 6.0460 - accuracy: 0.9945
    Epoch 00049: val_loss did not improve from 6.29788
    30/30 [==============================] - 86s 3s/step - loss: 6.0490 - accuracy: 0.9941 - val_loss: 6.4017 - val_accuracy: 0.9720
    Epoch 50/150
    29/30 [============================>.] - ETA: 1s - loss: 6.0409 - accuracy: 0.9928
    Epoch 00050: val_loss did not improve from 6.29788
    30/30 [==============================] - 88s 3s/step - loss: 6.0397 - accuracy: 0.9931 - val_loss: 6.3875 - val_accuracy: 0.9755
    Epoch 51/150
    29/30 [============================>.] - ETA: 1s - loss: 6.0496 - accuracy: 0.9939
    Epoch 00051: val_loss did not improve from 6.29788
    30/30 [==============================] - 87s 3s/step - loss: 6.0493 - accuracy: 0.9941 - val_loss: 6.3886 - val_accuracy: 0.9673
    Epoch 52/150
    29/30 [============================>.] - ETA: 1s - loss: 6.0437 - accuracy: 0.9950
    Epoch 00052: val_loss did not improve from 6.29788
    30/30 [==============================] - 87s 3s/step - loss: 6.0421 - accuracy: 0.9952 - val_loss: 6.4515 - val_accuracy: 0.9673
    Epoch 53/150
    29/30 [============================>.] - ETA: 1s - loss: 6.0471 - accuracy: 0.9928
    Epoch 00053: val_loss did not improve from 6.29788
    30/30 [==============================] - 86s 3s/step - loss: 6.0465 - accuracy: 0.9931 - val_loss: 6.4345 - val_accuracy: 0.9720
    Epoch 54/150
    29/30 [============================>.] - ETA: 1s - loss: 6.0583 - accuracy: 0.9934
    Epoch 00054: ReduceLROnPlateau reducing learning rate to 4.999999969612646e-10.
    
    Epoch 00054: val_loss improved from 6.29788 to 6.25906, saving model to VGG16_V2_BIBLUR_1.hdf5
    30/30 [==============================] - 91s 3s/step - loss: 6.0562 - accuracy: 0.9936 - val_loss: 6.2591 - val_accuracy: 0.9778
    Epoch 55/150
    29/30 [============================>.] - ETA: 1s - loss: 6.0529 - accuracy: 0.9939
    Epoch 00055: val_loss did not improve from 6.25906
    30/30 [==============================] - 87s 3s/step - loss: 6.0517 - accuracy: 0.9941 - val_loss: 6.4206 - val_accuracy: 0.9720
    Epoch 56/150
    29/30 [============================>.] - ETA: 1s - loss: 6.0197 - accuracy: 0.9978
    Epoch 00056: val_loss did not improve from 6.25906
    30/30 [==============================] - 87s 3s/step - loss: 6.0191 - accuracy: 0.9979 - val_loss: 6.3951 - val_accuracy: 0.9708
    Epoch 57/150
    29/30 [============================>.] - ETA: 1s - loss: 6.0550 - accuracy: 0.9923
    Epoch 00057: val_loss did not improve from 6.25906
    30/30 [==============================] - 87s 3s/step - loss: 6.0559 - accuracy: 0.9920 - val_loss: 6.3906 - val_accuracy: 0.9720
    Epoch 58/150
    29/30 [============================>.] - ETA: 1s - loss: 6.0417 - accuracy: 0.9939
    Epoch 00058: val_loss did not improve from 6.25906
    30/30 [==============================] - 86s 3s/step - loss: 6.0465 - accuracy: 0.9936 - val_loss: 6.4058 - val_accuracy: 0.9685
    Epoch 59/150
    29/30 [============================>.] - ETA: 1s - loss: 6.0369 - accuracy: 0.9956
    Epoch 00059: val_loss did not improve from 6.25906
    30/30 [==============================] - 85s 3s/step - loss: 6.0376 - accuracy: 0.9952 - val_loss: 6.3958 - val_accuracy: 0.9802
    Epoch 60/150
    29/30 [============================>.] - ETA: 1s - loss: 6.0426 - accuracy: 0.9939
    Epoch 00060: val_loss did not improve from 6.25906
    30/30 [==============================] - 88s 3s/step - loss: 6.0430 - accuracy: 0.9936 - val_loss: 6.3583 - val_accuracy: 0.9778
    Epoch 61/150
    29/30 [============================>.] - ETA: 1s - loss: 6.0548 - accuracy: 0.9956
    Epoch 00061: ReduceLROnPlateau reducing learning rate to 4.999999858590343e-11.
    
    Epoch 00061: val_loss did not improve from 6.25906
    30/30 [==============================] - 87s 3s/step - loss: 6.0565 - accuracy: 0.9952 - val_loss: 6.3357 - val_accuracy: 0.9720
    Epoch 62/150
    29/30 [============================>.] - ETA: 1s - loss: 6.0387 - accuracy: 0.9950
    Epoch 00062: val_loss did not improve from 6.25906
    30/30 [==============================] - 87s 3s/step - loss: 6.0379 - accuracy: 0.9952 - val_loss: 6.2682 - val_accuracy: 0.9732
    Epoch 63/150
    29/30 [============================>.] - ETA: 1s - loss: 6.0645 - accuracy: 0.9912
    Epoch 00063: val_loss did not improve from 6.25906
    30/30 [==============================] - 88s 3s/step - loss: 6.0644 - accuracy: 0.9909 - val_loss: 6.4054 - val_accuracy: 0.9673
    Epoch 64/150
    29/30 [============================>.] - ETA: 1s - loss: 6.0602 - accuracy: 0.9945
    Epoch 00064: val_loss did not improve from 6.25906
    30/30 [==============================] - 87s 3s/step - loss: 6.0595 - accuracy: 0.9947 - val_loss: 6.3444 - val_accuracy: 0.9755
    Epoch 65/150
    29/30 [============================>.] - ETA: 1s - loss: 6.0505 - accuracy: 0.9928
    Epoch 00065: val_loss did not improve from 6.25906
    30/30 [==============================] - 86s 3s/step - loss: 6.0490 - accuracy: 0.9931 - val_loss: 6.4008 - val_accuracy: 0.9755
    Epoch 66/150
    29/30 [============================>.] - ETA: 1s - loss: 6.0518 - accuracy: 0.9945
    Epoch 00066: val_loss did not improve from 6.25906
    30/30 [==============================] - 86s 3s/step - loss: 6.0503 - accuracy: 0.9947 - val_loss: 6.3139 - val_accuracy: 0.9743
    Epoch 67/150
    29/30 [============================>.] - ETA: 1s - loss: 6.0443 - accuracy: 0.9950
    Epoch 00067: ReduceLROnPlateau reducing learning rate to 4.999999719812465e-12.
    
    Epoch 00067: val_loss did not improve from 6.25906
    30/30 [==============================] - 88s 3s/step - loss: 6.0467 - accuracy: 0.9947 - val_loss: 6.4435 - val_accuracy: 0.9697
    Epoch 68/150
    29/30 [============================>.] - ETA: 1s - loss: 6.0322 - accuracy: 0.9957
    Epoch 00068: val_loss did not improve from 6.25906
    30/30 [==============================] - 86s 3s/step - loss: 6.0312 - accuracy: 0.9957 - val_loss: 6.4616 - val_accuracy: 0.9697
    Epoch 69/150
    29/30 [============================>.] - ETA: 1s - loss: 6.0520 - accuracy: 0.9956
    Epoch 00069: val_loss did not improve from 6.25906
    30/30 [==============================] - 86s 3s/step - loss: 6.0511 - accuracy: 0.9957 - val_loss: 6.3447 - val_accuracy: 0.9673
    Epoch 00069: early stopping
    WARNING:tensorflow:sample_weight modes were coerced from
      ...
        to  
      ['...']
    WARNING:tensorflow:sample_weight modes were coerced from
      ...
        to  
      ['...']
    Train for 30 steps, validate for 857 steps
    Epoch 1/150
    12/30 [===========>..................] - ETA: 52s - loss: 12.8570 - accuracy: 0.4787WARNING:tensorflow:Can save best model only with val_loss available, skipping.
    WARNING:tensorflow:Early stopping conditioned on metric `val_loss` which is not available. Available metrics are: loss,accuracy,lr
    


    ---------------------------------------------------------------------------

    ResourceExhaustedError                    Traceback (most recent call last)

    <ipython-input-13-9082c8e429ec> in <module>
         65               validation_data=val_generator,
         66               validation_steps=857,
    ---> 67               callbacks=[lr_reduce,checkpoint,early_stop])
    

    ~\anaconda3\envs\tensorflow-gpu\lib\site-packages\tensorflow_core\python\keras\engine\training.py in fit(self, x, y, batch_size, epochs, verbose, callbacks, validation_split, validation_data, shuffle, class_weight, sample_weight, initial_epoch, steps_per_epoch, validation_steps, validation_freq, max_queue_size, workers, use_multiprocessing, **kwargs)
        817         max_queue_size=max_queue_size,
        818         workers=workers,
    --> 819         use_multiprocessing=use_multiprocessing)
        820 
        821   def evaluate(self,
    

    ~\anaconda3\envs\tensorflow-gpu\lib\site-packages\tensorflow_core\python\keras\engine\training_v2.py in fit(self, model, x, y, batch_size, epochs, verbose, callbacks, validation_split, validation_data, shuffle, class_weight, sample_weight, initial_epoch, steps_per_epoch, validation_steps, validation_freq, max_queue_size, workers, use_multiprocessing, **kwargs)
        340                 mode=ModeKeys.TRAIN,
        341                 training_context=training_context,
    --> 342                 total_epochs=epochs)
        343             cbks.make_logs(model, epoch_logs, training_result, ModeKeys.TRAIN)
        344 
    

    ~\anaconda3\envs\tensorflow-gpu\lib\site-packages\tensorflow_core\python\keras\engine\training_v2.py in run_one_epoch(model, iterator, execution_function, dataset_size, batch_size, strategy, steps_per_epoch, num_samples, mode, training_context, total_epochs)
        126         step=step, mode=mode, size=current_batch_size) as batch_logs:
        127       try:
    --> 128         batch_outs = execution_function(iterator)
        129       except (StopIteration, errors.OutOfRangeError):
        130         # TODO(kaftan): File bug about tf function and errors.OutOfRangeError?
    

    ~\anaconda3\envs\tensorflow-gpu\lib\site-packages\tensorflow_core\python\keras\engine\training_v2_utils.py in execution_function(input_fn)
         96     # `numpy` translates Tensors to values in Eager mode.
         97     return nest.map_structure(_non_none_constant_value,
    ---> 98                               distributed_function(input_fn))
         99 
        100   return execution_function
    

    ~\anaconda3\envs\tensorflow-gpu\lib\site-packages\tensorflow_core\python\eager\def_function.py in __call__(self, *args, **kwds)
        566         xla_context.Exit()
        567     else:
    --> 568       result = self._call(*args, **kwds)
        569 
        570     if tracing_count == self._get_tracing_count():
    

    ~\anaconda3\envs\tensorflow-gpu\lib\site-packages\tensorflow_core\python\eager\def_function.py in _call(self, *args, **kwds)
        597       # In this case we have created variables on the first call, so we run the
        598       # defunned version which is guaranteed to never create variables.
    --> 599       return self._stateless_fn(*args, **kwds)  # pylint: disable=not-callable
        600     elif self._stateful_fn is not None:
        601       # Release the lock early so that multiple threads can perform the call
    

    ~\anaconda3\envs\tensorflow-gpu\lib\site-packages\tensorflow_core\python\eager\function.py in __call__(self, *args, **kwargs)
       2361     with self._lock:
       2362       graph_function, args, kwargs = self._maybe_define_function(args, kwargs)
    -> 2363     return graph_function._filtered_call(args, kwargs)  # pylint: disable=protected-access
       2364 
       2365   @property
    

    ~\anaconda3\envs\tensorflow-gpu\lib\site-packages\tensorflow_core\python\eager\function.py in _filtered_call(self, args, kwargs)
       1609          if isinstance(t, (ops.Tensor,
       1610                            resource_variable_ops.BaseResourceVariable))),
    -> 1611         self.captured_inputs)
       1612 
       1613   def _call_flat(self, args, captured_inputs, cancellation_manager=None):
    

    ~\anaconda3\envs\tensorflow-gpu\lib\site-packages\tensorflow_core\python\eager\function.py in _call_flat(self, args, captured_inputs, cancellation_manager)
       1690       # No tape is watching; skip to running the function.
       1691       return self._build_call_outputs(self._inference_function.call(
    -> 1692           ctx, args, cancellation_manager=cancellation_manager))
       1693     forward_backward = self._select_forward_and_backward_functions(
       1694         args,
    

    ~\anaconda3\envs\tensorflow-gpu\lib\site-packages\tensorflow_core\python\eager\function.py in call(self, ctx, args, cancellation_manager)
        543               inputs=args,
        544               attrs=("executor_type", executor_type, "config_proto", config),
    --> 545               ctx=ctx)
        546         else:
        547           outputs = execute.execute_with_cancellation(
    

    ~\anaconda3\envs\tensorflow-gpu\lib\site-packages\tensorflow_core\python\eager\execute.py in quick_execute(op_name, num_outputs, inputs, attrs, ctx, name)
         65     else:
         66       message = e.message
    ---> 67     six.raise_from(core._status_to_exception(e.code, message), None)
         68   except TypeError as e:
         69     keras_symbolic_tensors = [
    

    ~\anaconda3\envs\tensorflow-gpu\lib\site-packages\six.py in raise_from(value, from_value)
    

    ResourceExhaustedError: 2 root error(s) found.
      (0) Resource exhausted:  OOM when allocating tensor with shape[64,64,150,150] and type float on /job:localhost/replica:0/task:0/device:GPU:0 by allocator GPU_0_bfc
    	 [[node model_1/block1_conv2/Conv2D (defined at <ipython-input-13-9082c8e429ec>:67) ]]
    Hint: If you want to see a list of allocated tensors when OOM happens, add report_tensor_allocations_upon_oom to RunOptions for current allocation info.
    
    	 [[loss/dense_7_loss/weighted_loss/broadcast_weights/assert_broadcastable/AssertGuard/pivot_f/_15/_43]]
    Hint: If you want to see a list of allocated tensors when OOM happens, add report_tensor_allocations_upon_oom to RunOptions for current allocation info.
    
      (1) Resource exhausted:  OOM when allocating tensor with shape[64,64,150,150] and type float on /job:localhost/replica:0/task:0/device:GPU:0 by allocator GPU_0_bfc
    	 [[node model_1/block1_conv2/Conv2D (defined at <ipython-input-13-9082c8e429ec>:67) ]]
    Hint: If you want to see a list of allocated tensors when OOM happens, add report_tensor_allocations_upon_oom to RunOptions for current allocation info.
    
    0 successful operations.
    0 derived errors ignored. [Op:__inference_distributed_function_196219]
    
    Function call stack:
    distributed_function -> distributed_function
    



```python
for i in range(5):
    filepath="VGG16_V2_BIBLUR_" + str(i+1) + ".hdf5"
    model = models.load_model(filepath)
    loss, acc = model.evaluate(val_generator)
    print(loss, acc)
```


```python

```


```python

```


```python

```
