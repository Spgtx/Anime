import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm

from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Dropout, Concatenate
from tensorflow.keras.layers import BatchNormalization, Activation, ZeroPadding2D
from tensorflow.keras.layers import UpSampling2D, Conv2D
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def build_generator():

    noise_shape = (100,)

    model = Sequential()

    model.add(Dense(256 * 6 * 6, activation="relu", input_shape=noise_shape))
    model.add(Reshape((6, 6, 256)))

    model.add(UpSampling2D())
    model.add(Conv2D(256, kernel_size=3, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Activation("relu"))

    model.add(UpSampling2D())
    model.add(Conv2D(256, kernel_size=3, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Activation("relu"))
   
    model.add(UpSampling2D())
    model.add(Conv2D(128, kernel_size=3, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Activation("relu"))
    
    model.add(UpSampling2D())
    model.add(Conv2D(64, kernel_size=3, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Activation("relu"))

    model.add(Conv2D(3, kernel_size=3, padding="same"))
    model.add(Activation("tanh"))

    model.summary()

    noise = Input(shape=noise_shape)
    img = model(noise)

    return Model(noise, img)

def build_discriminator():

    img_shape = (64, 64, 3)

    model = Sequential()

    model.add(Conv2D(32, kernel_size=3, strides=2, input_shape=img_shape, padding="same"))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, kernel_size=3, strides=2, padding="same"))
    model.add(ZeroPadding2D(padding=((0,1),(0,1))))
    model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25))

    model.add(Conv2D(128, kernel_size=3, strides=2, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25))

    model.add(Conv2D(256, kernel_size=3, strides=1, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))

    model.summary()

    img = Input(shape=img_shape)
    validity = model(img)

    return Model(img, validity)

def load_data(directory, batch_size=32, target_size=(64, 64), color_mode='rgb'):
    """
    Load and augment data using Keras ImageDataGenerator.

    Args:
    - directory (str): path to data directory
    - batch_size (int): batch size for training
    - target_size (tuple): image size after resizing
    - color_mode (str): color mode for images

    Returns:
    - generator: Keras ImageDataGenerator object for generating batches of data
    """
    datagen = ImageDataGenerator(rescale=1./255,
                                 shear_range=0.2,
                                 zoom_range=0.2,
                                 horizontal_flip=True,
                                 validation_split=0.2)

    generator = datagen.flow_from_directory(directory=directory,
                                            target_size=target_size,
                                            color_mode=color_mode,
                                            batch_size=batch_size,
                                            class_mode='binary',
                                            subset='training')

    return generator

# Load the dataset
data_dir = 'data'
batch_size = 32
img_shape = (64, 64, 3)

data_generator = load_data(data_dir, batch_size=batch_size, target_size=img_shape[:2])

# Define the models
discriminator = build_discriminator(img_shape)
generator = build_generator(img_shape)

# Define the combined model
z = Input(shape=(100,))
img = generator(z)
discriminator.trainable = False
validity = discriminator(img)
combined_model = Model(z, validity)
combined_model.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.0002, beta_1=0.5))

# Train the GAN model
epochs = 30000
sample_interval = 200

for epoch in range(epochs):

    # Select a batch of real samples
    real_imgs, _ = data_generator.next()

    # Generate a batch of fake images
    noise = np.random.normal(0, 1, (batch_size, 100))
    fake_imgs = generator.predict(noise)

    # Train the discriminator
    d_loss_real = discriminator.train_on_batch(real_imgs, np.ones((batch_size, 1)))
    d_loss_fake = discriminator.train_on_batch(fake_imgs, np.zeros((batch_size, 1)))
    d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

    # Train the generator
    noise = np.random.normal(0, 1, (batch_size, 100))
    g_loss = combined_model.train_on_batch(noise, np.ones((batch_size, 1)))

    # Print the progress and save images
    if epoch % sample_interval == 0:
        print(f"Epoch {epoch}, Discriminator Loss: {d_loss}, Generator Loss: {g_loss}")
        save_imgs(epoch, generator)

# Save the trained generator model
generator.save('anime_generator.h5')
