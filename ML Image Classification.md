---
layout: post
title: Blog Post 5
---

## Introduction

In this blog post, we will implement some cool Machine Learning models for an image classification task using Tensorflow.

Our goal will be to build model(s) that are able to distinguish between pictures of dogs and cats.

## Loading Packages and Obtaining Data

We first import and load the libraries that we will be needing for this task.

```python
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
from tensorflow.keras import utils
```

Now in order to access the data we will use a sample set provided by the TensorFlow team that contains labeled images of cats and dogs.

```python
# location of data
_URL = 'https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip'

# download the data and extract it
path_to_zip = utils.get_file('cats_and_dogs.zip', origin=_URL, extract=True)

# construct paths
PATH = os.path.join(os.path.dirname(path_to_zip), 'cats_and_dogs_filtered')

train_dir = os.path.join(PATH, 'train')
validation_dir = os.path.join(PATH, 'validation')

# parameters for datasets
BATCH_SIZE = 32
IMG_SIZE = (160, 160)

# construct train and validation datasets 
train_dataset = utils.image_dataset_from_directory(train_dir,
                                                   shuffle=True,
                                                   batch_size=BATCH_SIZE,
                                                   image_size=IMG_SIZE)

validation_dataset = utils.image_dataset_from_directory(validation_dir,
                                                        shuffle=True,
                                                        batch_size=BATCH_SIZE,
                                                        image_size=IMG_SIZE)

# construct the test dataset by taking every 5th observation out of the validation dataset
val_batches = tf.data.experimental.cardinality(validation_dataset)
test_dataset = validation_dataset.take(val_batches // 5)
validation_dataset = validation_dataset.skip(val_batches // 5)
```

We have now created the datasets for training, validation, and testing.

We also run the following block of code.

```python
AUTOTUNE = tf.data.AUTOTUNE

train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
validation_dataset = validation_dataset.prefetch(buffer_size=AUTOTUNE)
test_dataset = test_dataset.prefetch(buffer_size=AUTOTUNE)
```

We can get a batch of data set using the `take` method. For example, the following line of code will retrieve one batch (32 images with labels) from the training data.

```python
train_dataset.take(1)
```

The following function creates a two-row visualization with the first row showing three random pictures of cats and the second row showing three random pictures of dogs.

```python
# Specifying the size of each image to be displayed
plt.figure(figsize = (10, 10))

# This list will contain the label 'Cat' or 'Dog'
names = []

for images, labels in train_dataset.take(1):
    
    # Rescaling images to have RGB values between 0 and 1
    images = images.numpy() / 255.0

    # Transforming the labels tensor into a numpy array
    labels = labels.numpy()

    # Creating descriptive labels to our images
    names = ['Dog' if labels[i] == 1 else 'Cat' for i in range(len(labels))]

# Determining the which indices of the images array correspond to cats and dogs
for i in range(len(labels)):
  dog_indices = [elem[0] for elem in enumerate(labels) if elem[1] == 1]
  cat_indices = [elem[0] for elem in enumerate(labels) if elem[1] == 0]

for i in range(6):
  ax = plt.subplot(3, 3, i+1)
  plt.axis('off')

  # Displaying cats in the first row
  if i < 3:
    plt.imshow(images[cat_indices[i]])
    plt.title(names[cat_indices[i]])

  # Displaying dogs in the second row
  if i >= 3:
    plt.imshow(images[dog_indices[i]])
    plt.title(names[dog_indices[i]])

# plt.savefig(fname = "blog-post-5-9.png", bbox_inches = 'tight')
# files.download("blog-post-5-9.png")
```

![blog-post-5-9](/images/blog-post-5-9.png)

The following line of code will create an iterator called `labels`.

```python
labels_iterator= train_dataset.unbatch().map(lambda image, label: label).as_numpy_iterator()
```

The following block of code computes the number of images in training data with label `0` (corresponding to `'cat'`) and label `1` (corresponding to `'dog'`).

```python
dog = []
cat = []

for i in range(len(label)):
  if label[i] == 1:
    dog.append(i)
  if label[i] == 0:
    cat.append(i)

print('Number of cats:', len(cat))
print("Number of dogs:", len(dog))
```

We get the output as follows:

- Number of cats: 15
- Number of dogs: 17

If the **baseline machine learning model** guesses the most frequent label every time, then it would be right $$\frac{17}{32} \times 100\% = 53.125%$$ of the times, or in other words, it would have about $$53\%$$ accuracy.

## First Model

Our first model, named `model_1` will be a `tf.keras.Sequential` model using the following layers:

- Two `Conv2D` layers
- Two `MaxPooling2D` layers
- One `Flatten` layer
- One `Dense` layer
- One `Dropout` layer

We build it as follows:

```python
model_1 = tf.keras.Sequential([

    # First layer: 2D Convulation layer
    layers.Conv2D(32, (3, 3),
                  activation = 'relu',
                  input_shape = (160, 160, 3),
                  dtype = 'float64'),

    # Second layer: 2D Max Pooling layer
    layers.MaxPooling2D((2, 2)),
    
    # Third layer: 2D Convulation layer
    layers.Conv2D(32, (3, 3),
                  activation = 'relu'),

    # Fourth layer: 2D Max Pooling layer
    layers.MaxPooling2D((2, 2)),

    # Fifth layer: Flatten layer to remove all dimensions but one
    layers.Flatten(),

    # Sixth layer: Dropout layer which randomly sets input units to 0 with a frequency of rate at each step during training time
    layers.Dropout(0.2),

    # Seventh layer: Dense layer
    layers.Dense(64,
                 activation = 'relu'),

])
```

Computing the summary using `model_1.summary()`, we see that the model has a total of $$2,967,520$$ parameters.

We compile the model as follows.

```python
model_1.compile(optimizer = 'adam',
                loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True),
                metrics = ['accuracy'])
```

Finally, we train the model on the `train_dataset` using the code below using 20 iterations as mentioned in `epochs = 20`.

```python
history = model_1.fit(train_dataset,
                      epochs = 20,
                      validation_data = validation_dataset)
```

Plotting the graph of Training Dataset Accuracy versus the Validation Dataset Accuracy, we obtain the following plot.

```python
plt.plot(history.history['accuracy'], label = 'training')
plt.plot(history.history['val_accuracy'], label = 'validation')
plt.gca().set(xlabel = 'epoch', ylabel = 'accuracy')
plt.legend()
# plt.savefig(fname = "blog-post-5-1.png", bbox_inches = 'tight')
# files.download("blog-post-5-1.png")
plt.show()
```

![blog-post-5-1](/images/blog-post-5-1.png)

### Observations

- **The validation accuracy of my model stabilized between 59% and 64% during training.**
- It is comparably better than the baseline accuracy of 52%.
- The training accuracy is 98%, which is much higher than the validation accuracy of about 62%. Hence, we observe overfitting in our data.

## Model with Data Augmentation

### Creating a `tf.keras.layers.RandomFlip()` layer

```python
plt.figure(figsize = (10, 10))
for images, labels in train_dataset.take(1):
  
  # Scaling the RGB values of the images tensor
  images = images / 255.0

  # Creating the flipped_image layer
  flipped_image = tf.keras.layers.RandomFlip()

  image = images[0]
  for i in range(4):
    ax = plt.subplot(1, 4, i + 1)
    plt.axis('off')
    plt.imshow(image)
    image = flipped_image(image)
```

![blog-post-5-2](/images/blog-post-5-2.png)

### Creating a `tf.keras.layers.RandomRotation()` layer

```python
plt.figure(figsize = (10, 10))
for images, labels in train_dataset.take(1):

    # Scaling the RGB values of the images tensor
    images = images / 255.0

    # Creating the rotated_image layer
    rotated_image = tf.keras.layers.RandomRotation(factor = (-0.3, 0.2))
    image = images[0]
    for i in range(4):
        ax = plt.subplot(1, 4, i + 1)
        plt.axis('off')
        plt.imshow(image)
        image = rotated_image(image)
```

![blog-post-5-3](/images/blog-post-5-3.png)

We now create a `tf.keras.models.Sequential` model, named `model_2`, in which the first two layers are `tf.keras.layers.RandomFlip()` and `tf.keras.layers.RandomRotation()` as described above, and the subsequent layers are the same as described in `model_1` above.

We build the model as follows:

```python
model_2 = tf.keras.models.Sequential([
                        
    # First layer: Data Augmentation layer
    layers.RandomFlip(),

    # Second layer: Data Augmentation layer
    layers.RandomRotation(factor = (-0.3, 0.2)),
    
    # Third layer: 2D Convulation layer
    layers.Conv2D(32, (3, 3),
                  activation = 'relu',
                  input_shape = (160, 160, 3),
                  dtype = 'float64'),

    # Fourth layer: 2D Max Pooling layer
    layers.MaxPooling2D((2, 2)),
    
    # Fifth layer: 2D Convulation layer
    layers.Conv2D(32, (3, 3),
                  activation = 'relu'),

    # Sixth layer: 2D Max Pooling layer
    layers.MaxPooling2D((2, 2)),

    # Seventh layer: Flatten layer to remove all dimensions but one
    layers.Flatten(),

    # Eighth layer: Dropout layer which randomly sets input units to 0 with a frequency of rate at each step during training time
    layers.Dropout(0.2),

    # Ninth layer: Dense layer
    layers.Dense(64,
                 activation = 'relu'),

])
```

Computing the summary using `model_2.summary()`, we see that the model has a total of $$2,967,520$$ parameters.

We compile the model as follows.

```python
model_2.compile(optimizer = 'adam',
                loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True),
                metrics = ['accuracy'])
```

Finally, we train the model on the `train_dataset` using the code below using 20 iterations as mentioned in `epochs = 20`.

```python
history = model_2.fit(train_dataset,
                      epochs = 20,
                      validation_data = validation_dataset)
```

Plotting the graph of Training Dataset Accuracy versus the Validation Dataset Accuracy, we obtain the following plot.

```python
plt.plot(history.history['accuracy'], label = 'training')
plt.plot(history.history['val_accuracy'], label = 'validation')
plt.gca().set(xlabel = 'epoch', ylabel = 'accuracy')
plt.legend()
# plt.savefig(fname = "blog-post-5-4.png", bbox_inches = 'tight')
# files.download("blog-post-5-4.png")
plt.show()
```

![blog-post-5-4](/images/blog-post-5-4.png)

### Observations

- **The validation accuracy of my model stabilized between 56% and 60% during training.**
- It is a little less than the validation accuracy we obtained in the first model.
- The training accuracy is around 57%, which is not much different than the validation accuracy of 60%. Hence, we do not observe overfitting in our data.

## Data Preprocessing

We now create a model with a preprocessing layer that will scale the RGB values between 0 to 255 to be between 0 and 1, or -1 and 1. Since we are just scaling the weights, this isn't fundamentally different than what we were doing before but the advantage to this scaling process is that we we can spend more of our training energy handling actual signal in the data and less energy having the weights adjust to the data scale.

The following code will create a preprocessing layer called `preprocessor` which we can use in your model pipeline.

```python
i = tf.keras.Input(shape=(160, 160, 3))
x = tf.keras.applications.mobilenet_v2.preprocess_input(i)
preprocessor = tf.keras.Model(inputs = [i], outputs = [x])
```

We build the model as follows:

```python
model_3 = tf.keras.models.Sequential([

    # First layer: Preprocessing layer as created above
    preprocessor,

    # Second layer: Data Augmentation layer
    layers.RandomFlip(),

    # Third layer: Data Augmentation layer
    layers.RandomRotation(factor = (-0.3, 0.2)),
    
    # Fourth layer: 2D Convulation layer
    layers.Conv2D(32, (3, 3),
                  activation = 'relu',
                  input_shape = (160, 160, 3),
                  dtype = 'float64'),

    # Fifth layer: 2D Max Pooling layer
    layers.MaxPooling2D((2, 2)),
    
    # Sixth layer: 2D Convulation layer
    layers.Conv2D(32, (3, 3),
                  activation = 'relu'),

    # Seventh layer: 2D Max Pooling layer
    layers.MaxPooling2D((2, 2)),

    # Eigth layer: Flatten layer to remove all dimensions but one
    layers.Flatten(),

    # Ninth layer: Dropout layer which randomly sets input units to 0 with a frequency of rate at each step during training time
    layers.Dropout(0.2),

    # Tenth layer: Dense layer
    layers.Dense(64,
                 activation = 'relu'),

])
```

Computing the summary using `model_3.summary()`, we see that the model has a total of $$2,967,520$$ parameters.

We compile the model as follows.

```python
model_3.compile(optimizer = 'adam',
                loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True),
                metrics = ['accuracy'])
```

Finally, we train the model on the `train_dataset` using the code below using 20 iterations as mentioned in `epochs = 20`.

```python
history = model_3.fit(train_dataset,
                      epochs = 20,
                      validation_data = validation_dataset)
```

Plotting the graph of Training Dataset Accuracy versus the Validation Dataset Accuracy, we obtain the following plot.

```python
plt.plot(history.history['accuracy'], label = 'training')
plt.plot(history.history['val_accuracy'], label = 'validation')
plt.gca().set(xlabel = 'epoch', ylabel = 'accuracy')
plt.legend()
# plt.savefig(fname = "blog-post-5-5.png", bbox_inches = 'tight')
# files.download("blog-post-5-5.png")
plt.show()
```

![blog-post-5-5](/images/blog-post-5-5.png)

### Observations

- **The validation accuracy of my model stabilized between 70% and 73.2% during training.**
- It is a lot better than the validation accuracy we obtained in the first model.
- The training accuracy is around 70% and 73%, which is pretty close to the validation accuracy of 70% and 73.2%. Hence, we do not observe overfitting in our data.

## Transfer Learning

Now, we will try and use a pre-existing model for our task of image classification. We do so by accessing a pre-existing "base model", incorporating it into a full model and then training it for our task.

We download the `MobileNetV2` model using the code below.

```python
IMG_SHAPE = IMG_SIZE + (3,)
base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
                                               include_top=False,
                                               weights='imagenet')
base_model.trainable = False

i = tf.keras.Input(shape=IMG_SHAPE)
x = base_model(i, training = False)
base_model_layer = tf.keras.Model(inputs = [i], outputs = [x])
```

This can now be configured as a layer in our own model.

We now create a model with a preprocessing layer that will scale the RGB values between 0 to 255 to be between 0 and 1, or -1 and 1. Since we are just scaling the weights, this isn't fundamentally different than what we were doing before but the advantage to this scaling process is that we we can spend more of our training energy handling actual signal in the data and less energy having the weights adjust to the data scale.

The following code will create a new model `model_4` that used `MobileNetV2` containing the following layers:

- `preprocessing` layer from `model_3`
- Data augmentation layers from `model_2`
- `base_layer_model` constructed above
- `Dense(2)` layer at the very end to perform the classification
- Few `GlobalMaxPooling2D` layers
- One `Dropout` layer

We build the model as follows:

```python
model_4 = tf.keras.models.Sequential([

    # First layer: Preprocessing layer as created above
    preprocessor,

    # Second layer: Data Augmentation layer
    layers.RandomFlip(),

    # Third layer: Data Augmentation layer
    layers.RandomRotation(factor = (-0.3, 0.2)),
    
    # Fourth layer: Layer using the MobileNetV2 model throgh Transfer Learning
    base_model_layer,

    # Fifth layer: 2D Convulation layer
    layers.Conv2D(32, (3, 3),
                  activation = 'relu',
                  dtype = 'float64'),
    
    # Sixth layer: 2D Convulation layer
    layers.Conv2D(32, (3, 3),
                  activation = 'relu'),

    # Seventh layer: 2D Global Max Pooling layer
    layers.GlobalMaxPooling2D(keepdims = True),

    # Eigth layer: Dropout layer which randomly sets input units to 0 with a frequency of rate at each step during training time
    layers.Dropout(0.3),

    # Ninth layer: Flatten layer to remove all dimensions but one
    layers.Flatten(),

    # Tenth layer: Dropout layer which randomly sets input units to 0 with a frequency of rate at each step during training time
    layers.Dropout(0.2),

    # Eleventh layer: Dense layer
    layers.Dense(2)

])
```

Here, we make use of the `GlobalMaxPooling2D` layer and additional `Dropout` layers in order to prevent overfitting in our data. Finally, the `Dense(2)` layer at the very end performs the main classification task.

Computing the summary using `model_4.summary()`, shown below, we see that the model has a total of $$2,635,970$$ parameters out of which $$377,986$$ are trainable and $$2,257,984$$ are non-trainable.

![blog-post-5-6](/images/blog-post-5-6.png)

We compile the model as follows.

```python
model_4.compile(optimizer = 'adam',
                loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True),
                metrics = ['accuracy'])
```

Finally, we train the model on the `train_dataset` using the code below using 20 iterations as mentioned in `epochs = 20`.

```python
history = model_4.fit(train_dataset,
                      epochs = 20,
                      validation_data = validation_dataset)
```

Plotting the graph of Training Dataset Accuracy versus the Validation Dataset Accuracy, we obtain the following plot.

```python
plt.plot(history.history['accuracy'], label = 'training')
plt.plot(history.history['val_accuracy'], label = 'validation')
plt.gca().set(xlabel = 'epoch', ylabel = 'accuracy')
plt.legend()
# plt.savefig(fname = "blog-post-5-7.png", bbox_inches = 'tight')
# files.download("blog-post-5-7.png")
plt.show()
```

![blog-post-5-7](/images/blog-post-5-7.png)

### Observations

- **The validation accuracy of my model stabilized between 96% and 98% during training.**
- It is a lot better than the validation accuracy between 70% and 73.2% that we were able to obtain in `model_3`.
- The training accuracy is around 93%, which is not very different than the validation accuracy of about 97%. Hence, we do not observe overfitting in our data.

## Score on Test Data

Now time for the final test - we run our most performant model, `model_4`, on the unseen `test_dataset` using 20 iterations as mentioned in `epochs = 20`. Let's see how it performs.

```python
history = model_4.fit(test_dataset,
                      epochs = 20,
                      validation_data = validation_dataset)
```

Plotting the graph of Training Dataset Accuracy versus the Validation Dataset Accuracy, we obtain the following plot.

```python
plt.plot(history.history['accuracy'], label = 'training')
plt.plot(history.history['val_accuracy'], label = 'validation')
plt.gca().set(xlabel = 'epoch', ylabel = 'accuracy')
plt.legend()
# plt.savefig(fname = "blog-post-5-8.png", bbox_inches = 'tight')
# files.download("blog-post-5-8.png")
plt.show()
```

![blog-post-5-8](/images/blog-post-5-8.png)

### Observations

- **The validation accuracy of the model on unseen data stabilized between 97% and 98.4%.**
- It is better than the validation accuracy between 96% and 98% that we were able to obtain in `model_4`.
- The testing accuracy ranges between 94% and 97%, which is pretty close to the validation accuracy of about 97.5%. Hence, we do not observe overfitting in our data.