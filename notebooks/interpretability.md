
# Using a multi output convolutional network to interpret inherent concepts

### 1. Import the libraries


```python
# This Source Code Form is subject to the terms of the MIT
# License. If a copy of the same was not distributed with this
# file, You can obtain one at
# https://github.com/akhilpandey95/interpretability/blob/master/LICENSE.

import os
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm
import tensorflow as tf
from PIL import Image
from IPython.display import display
import matplotlib.pyplot as plt
import tensorflow.keras as keras
from tensorflow.keras import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator
```

### 2. Read the dataset

#### 2.1 Textual attributes pertaining to the public individuals


```python
# set the path
PATH = '/media/hector/data/datasets/interpretability/'

# change the current working directory
os.chdir(PATH)

# read the attribute information
attributes = pd.read_csv('list_attr_celeba.csv')

# print the clumns
attributes.columns
```




    Index(['image_id', '5_o_Clock_Shadow', 'Arched_Eyebrows', 'Attractive',
           'Bags_Under_Eyes', 'Bald', 'Bangs', 'Big_Lips', 'Big_Nose',
           'Black_Hair', 'Blond_Hair', 'Blurry', 'Brown_Hair', 'Bushy_Eyebrows',
           'Chubby', 'Double_Chin', 'Eyeglasses', 'Goatee', 'Gray_Hair',
           'Heavy_Makeup', 'High_Cheekbones', 'Male', 'Mouth_Slightly_Open',
           'Mustache', 'Narrow_Eyes', 'No_Beard', 'Oval_Face', 'Pale_Skin',
           'Pointy_Nose', 'Receding_Hairline', 'Rosy_Cheeks', 'Sideburns',
           'Smiling', 'Straight_Hair', 'Wavy_Hair', 'Wearing_Earrings',
           'Wearing_Hat', 'Wearing_Lipstick', 'Wearing_Necklace',
           'Wearing_Necktie', 'Young'],
          dtype='object')



##### 2.1.2 Visualize the distribution of males to females


```python
plt.figure(num=None, figsize=(10, 8), dpi=300, facecolor='w', edgecolor='k')
sns.set(font_scale = 1.2, style='ticks')

sns.countplot(attributes.Male.
              apply(lambda x: 'Male' if x == 1 else 'Female'))
plt.xlabel('\nGender', fontdict={'size': 14})
plt.ylabel('Count', fontdict={'size': 14})
plt.title('Number of males to females samples in the dataset', fontdict={'size': 14})
plt.show()
```


![png](output_7_0.png)


#### 2.2 Image data for all the public individuals


```python
 # function for gathering image data
def generate_samples():
    """
    Gather the cats and dogs image dataset using tensorflow
    
    Parameters
    ----------
    None
    
    Returns
    -------
    String, String, String, String
        str, str, str, str
    """
    try:
        # set the URI
        _URL = 'https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip'
        
        # set the path to the zip file
        path_to_zip = tf.keras.utils.get_file('cats_and_dogs.zip', origin=_URL, extract=True)
        
        # set the PATH
        PATH = os.path.join(os.path.dirname(path_to_zip), 'cats_and_dogs_filtered')
        
        # assemble the training images into train directory
        train_dir = os.path.join(PATH, 'train')
        
        # assemble the validation images into validation directory
        validation_dir = os.path.join(PATH, 'validation')

        # return the data
        return train_dir, validation_dir
    except:
        return np.zeros(1)

 # function for preparding the image data for training
def prepare_data(train_dir, validation_dir, batch_size, img_height, img_width):
    """
    Prepare the image data by extracting
    the images from respective training and
    validation directories
    Parameters
    ----------
    arg1 | train_dir: str
        The directory path where training images are located
    arg2 | validation_dir: str
        The directory path where validation images are located
    arg3 | batch_size: int
        The size of the batch of images to be used while training/validation
    arg4 | img_height: int
        The directory path where validation images are located
    arg5 | img_width: int
        The size of the batch of images to be used while training/validation
    Returns
    -------
    Array
        numpy.ndarray
    """
    try:
        # create a image generator object for the training data
        train_image_generator = ImageDataGenerator(rescale=1./255,
                                                   rotation_range=55,
                                                   width_shift_range=.35,
                                                   height_shift_range=.35,
                                                   horizontal_flip=True,
                                                   zoom_range=0.3)

        # create a image generator object for the validation data
        val_image_generator = ImageDataGenerator(rescale=1./255,
                                                   rotation_range=55,
                                                   width_shift_range=.35,
                                                   height_shift_range=.35,
                                                   horizontal_flip=True,
                                                   zoom_range=0.3)

        # augment the training images using the train image generator
        train_data_gen = train_image_generator.flow_from_directory(batch_size=batch_size,
                                                           directory=train_dir,
                                                           shuffle=True,
                                                           target_size=(img_height, img_width),
                                                           class_mode='binary')

        # augment the training images using the train image generator
        val_data_gen = val_image_generator.flow_from_directory(batch_size=batch_size,
                                                            directory=validation_dir,
                                                            target_size=(img_height, img_width),
                                                            class_mode='binary')
        
        # return 
        return train_data_gen, val_data_gen
    except:
        return np.zeros(1)
```

#### 2.2.2 Prepare the Inputs


```python
dir_data      = "./img_align_celeba/"
Ntrain        = 140000 
Ntest         = 60100
nm_imgs       = np.sort(os.listdir(dir_data))
## name of the jpg files for training set
nm_imgs_train = nm_imgs[:Ntrain]
## name of the jpg files for the testing data
nm_imgs_test  = nm_imgs[Ntrain:Ntrain + Ntest]
img_shape     = (32, 32, 3)

def get_npdata(nm_imgs_train):
    X_train = []
    for i, myid in enumerate(tqdm(nm_imgs_train)):
        image = load_img(dir_data + "/" + myid,
                         target_size=img_shape[:2])
        image = img_to_array(image)/255.0
        X_train.append(image)
    X_train = np.array(X_train)
    return(X_train)

X_train = get_npdata(nm_imgs_train)
print("X_train.shape = {}".format(X_train.shape))

X_test  = get_npdata(nm_imgs_test)
print("X_test.shape = {}".format(X_test.shape))
```

##### 2.2.3 Prepare the output labels


```python
# Prepare the labels for the train images
Y_train = np.array(list(map(lambda x: \
                            attributes.loc[attributes.image_id == x]['Male'].values[0],\
                            tqdm(nm_imgs_train))))

# Reshape the output labels for the train data
Y_train = Y_train.reshape(-1, 1)

# Prepare the labels for the test images
Y_test = np.array(list(map(lambda x: \
                            attributes.loc[attributes.image_id == x]['Male'].values[0],\
                            tqdm(nm_imgs_test))))

# Reshape the output labels for the test data
Y_test = Y_test.reshape(-1, 1)
```

##### 2.2.4 Visualize sample images


```python
fig= plt.figure(figsize=(30,10))

for img_index in range(1, 10):
    ax = fig.add_subplot(1,10,img_index)
    ax.imshow(X_train[img_index])
plt.tight_layout()
plt.show()
```

### 3. Helper methods for defining the CNN model


```python
# function for creating the base model
def base_model(inputs):
    """
    Prepare the base model of the CNN with input layer
    and two convolutional layers
    Parameters
    ----------
    arg1 | inputs: numpy.ndarray
        The array storing training images are located
    Returns
    -------
    Array
        numpy.ndarray
    """
    try:
        # add the input layer
        model = Input(shape=inputs)

        # return the base model
        return model
    except:
        # return empty model
        return tf.keras.Model

# function for creating the concept model
def concept_model(base, inputs, outputs):
    """
    Prepare the concept model that is attached
    to the left side of the base model
    Parameters
    ----------
    arg1 | model: tf.keras.Model
        The base model
    arg2 | outputs: int
        The number of concepts we are inferring
    Returns
    -------
    Array
        numpy.ndarray
    """
    try:
        # add the first conv layer with relu activation
        model = Conv2D(16, 3, padding='same', activation='relu', name='Base-Convleft1')(base)

        # add a max pooling layer
        model = MaxPooling2D(name='MaxPool-left1')(model)

        # add a dropout layer
        model = Dropout(0.2)(model)

        # add the second conv layer with relu activation
        model = Conv2D(32, 3, padding='same', activation='relu', name='Base-Convleft2')(model)

        # add a max pooling layer
        model = MaxPooling2D(name='MaxPool-left2')(model)
        
        # flatten the activations in order to connect it to a fully connected layer
        concepts = Flatten(input_shape=inputs)(model)

        # add a fully connected layer with relu activation
        concepts = Dense(1024, activation='relu')(concepts)
        
        # add another fully connected layer with relu activation
        concepts = Dense(512, activation='relu')(concepts)

        # add the output layer
        concepts = Dense(outputs, activation='sigmoid', name='Concept-Activation')(concepts)

        # return the combined model
        return concepts
    except:
        # return empty model
        return tf.keras.Model

# function for creating the classification model
def classification_model(base, inputs, outputs):
    """
    Prepare the classification model that is attached
    to the right side of the base model
    Parameters
    ----------
    arg1 | model: tf.keras.Model
        The base model
    arg2 | outputs: int
        The number of classes we are predicting
    Returns
    -------
    Array
        numpy.ndarray
    """
    try:
        # add the first conv layer with relu activation
        model = Conv2D(16, 3, padding='same', activation='relu', name='Base-Conv-right1')(base)

        # add a max pooling layer
        model = MaxPooling2D(name='MaxPool-right1')(model)

        # add a dropout layer
        model = Dropout(0.2)(model)

        # add the second conv layer with relu activation
        model = Conv2D(32, 3, padding='same', activation='relu', name='Base-Convright2')(model)

        # add a max pooling layer
        model = MaxPooling2D(name='MaxPool-right2')(model)

        # add the third conv layer with relu activation
        classes = Conv2D(64, 3, padding='same', activation='relu', input_shape= inputs)(model)

        # add a max pooling layer
        classes = MaxPooling2D()(classes)

        # add a dropout layer
        classes = Dropout(0.2)(classes)

        # flatten the activations in order to connect it to a fully connected layer
        classes = Flatten()(classes)

        # add a fully connected layer with relu activation
        classes = Dense(1024, activation='relu')(classes)
        
        # add another fully connected layer with relu activation
        classes = Dense(512, activation='relu')(classes)

        # add the output layer
        classes = Dense(outputs, activation='softmax', name='Label-Activation')(classes)

        # return the combined model
        return classes
    except:
        # return empty model
        return tf.keras.Model
```

#### 3.2 Build the model


```python
# build the base model
base = base_model((150, 150, 3))

# add the left side of the network
concepts_nn = concept_model(base, (8, 8, 32), 3)

# add the right side of the network
classification_nn = classification_model(base, (8, 8, 32), 2)

# build the model
model = tf.keras.Model(inputs=base, \
                       outputs=[concepts_nn, classification_nn], \
                      name = 'ConcUndNN')

# print the model summary
model.summary()
```

    Model: "ConcUndNN"
    __________________________________________________________________________________________________
    Layer (type)                    Output Shape         Param #     Connected to                     
    ==================================================================================================
    input_6 (InputLayer)            [(None, 150, 150, 3) 0                                            
    __________________________________________________________________________________________________
    Base-Conv-right1 (Conv2D)       (None, 150, 150, 16) 448         input_6[0][0]                    
    __________________________________________________________________________________________________
    MaxPool-right1 (MaxPooling2D)   (None, 75, 75, 16)   0           Base-Conv-right1[0][0]           
    __________________________________________________________________________________________________
    dropout_14 (Dropout)            (None, 75, 75, 16)   0           MaxPool-right1[0][0]             
    __________________________________________________________________________________________________
    Base-Convleft1 (Conv2D)         (None, 150, 150, 16) 448         input_6[0][0]                    
    __________________________________________________________________________________________________
    Base-Convright2 (Conv2D)        (None, 75, 75, 32)   4640        dropout_14[0][0]                 
    __________________________________________________________________________________________________
    MaxPool-left1 (MaxPooling2D)    (None, 75, 75, 16)   0           Base-Convleft1[0][0]             
    __________________________________________________________________________________________________
    MaxPool-right2 (MaxPooling2D)   (None, 37, 37, 32)   0           Base-Convright2[0][0]            
    __________________________________________________________________________________________________
    dropout_13 (Dropout)            (None, 75, 75, 16)   0           MaxPool-left1[0][0]              
    __________________________________________________________________________________________________
    conv2d_5 (Conv2D)               (None, 37, 37, 64)   18496       MaxPool-right2[0][0]             
    __________________________________________________________________________________________________
    Base-Convleft2 (Conv2D)         (None, 75, 75, 32)   4640        dropout_13[0][0]                 
    __________________________________________________________________________________________________
    max_pooling2d_5 (MaxPooling2D)  (None, 18, 18, 64)   0           conv2d_5[0][0]                   
    __________________________________________________________________________________________________
    MaxPool-left2 (MaxPooling2D)    (None, 37, 37, 32)   0           Base-Convleft2[0][0]             
    __________________________________________________________________________________________________
    dropout_15 (Dropout)            (None, 18, 18, 64)   0           max_pooling2d_5[0][0]            
    __________________________________________________________________________________________________
    flatten_10 (Flatten)            (None, 43808)        0           MaxPool-left2[0][0]              
    __________________________________________________________________________________________________
    flatten_11 (Flatten)            (None, 20736)        0           dropout_15[0][0]                 
    __________________________________________________________________________________________________
    dense_22 (Dense)                (None, 1024)         44860416    flatten_10[0][0]                 
    __________________________________________________________________________________________________
    dense_24 (Dense)                (None, 1024)         21234688    flatten_11[0][0]                 
    __________________________________________________________________________________________________
    dense_23 (Dense)                (None, 512)          524800      dense_22[0][0]                   
    __________________________________________________________________________________________________
    dense_25 (Dense)                (None, 512)          524800      dense_24[0][0]                   
    __________________________________________________________________________________________________
    Concept-Activation (Dense)      (None, 3)            1539        dense_23[0][0]                   
    __________________________________________________________________________________________________
    Label-Activation (Dense)        (None, 2)            1026        dense_25[0][0]                   
    ==================================================================================================
    Total params: 67,175,941
    Trainable params: 67,175,941
    Non-trainable params: 0
    __________________________________________________________________________________________________


#### 3.3 Visualize the model structure


```python
from tensorflow.keras.utils import plot_model
plot_model(model, show_shapes=True, expand_nested=True)
```




![png](output_21_0.png)



#### 3.4 Compile the model


```python
# add the optimizer
rms = tf.keras.optimizers.RMSprop(lr=0.001)

# compile the model
model.compile(optimizer=rms,\
              loss = {"Label-Activation": "binary_crossentropy", \
                      "Concept-Activation": "categorical_crossentropy"}, \
              loss_weights = {"Label-Activation": 0.0, "Concept-Activation": 0.0},\
              metrics = ["accuracy"])
```
