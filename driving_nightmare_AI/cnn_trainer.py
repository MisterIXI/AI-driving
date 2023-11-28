import tensorflow as tf
import numpy as np
import os
import h5py
import tensorflow.keras.layers as tfl

script_dir = os.path.dirname(__file__)
# Load the data
train_data = h5py.File(os.path.join(
    script_dir, 'learning_data', 'training_data.h5'), 'r')
test_data = h5py.File(os.path.join(
    script_dir, 'learning_data', 'validation_data.h5'), 'r')

train_ds = tf.data.Dataset.from_tensor_slices(
    (train_data['training_data'], train_data['training_result']))
test_ds = tf.data.Dataset.from_tensor_slices(
    (test_data['validation_data'], test_data['validation_result']))
data = train_data['training_data'][0]


model = tf.keras.Sequential([
    tfl.Rescaling(1./255, input_shape=(1080//4, 1920//4, 3)),
    tfl.Conv2D(12, 5, activation='relu', kernel_initializer='HeNormal'),
    tfl.Conv2D(12, 5, activation='relu', kernel_initializer='HeNormal'),
    tfl.Conv2D(12, 5, activation='relu', kernel_initializer='HeNormal'),
    tfl.MaxPooling2D(5),
    tfl.Flatten(),
    tfl.Dense(2, activation='tanh', kernel_initializer="HeNormal")
])
model.build((len(train_data['training_data']), 1080//4, 1920//4, 3))
model.summary()

model.compile(
    optimizer=tf.keras.optimizers.Adagrad(learning_rate=0.002),
    loss=tf.keras.losses.MeanSquaredError()
)
model.fit(train_ds.batch(64), epochs=30)

model.save(os.path.join(script_dir, "the_model"))
