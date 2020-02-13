from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping

import tensorflow as tf

import tensorflow.keras.layers as layers

import BreastCancerData as bcd

model = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=[len(bcd.train_X.keys())]),
    layers.Dense(64, activation='relu'),
    layers.Dense(1)
])

optimizer = tf.keras.optimizers.RMSprop(0.001)

model.compile(optimizer=optimizer, loss='mse', metrics=['mae', 'mse'])

early_stopping_monitor = EarlyStopping(patience=3)

history = model.fit(bcd.train_X, bcd.train_Y, validation_split=0.2, epochs=100, callbacks=[early_stopping_monitor])
