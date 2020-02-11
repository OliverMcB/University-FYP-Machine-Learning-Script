from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping

import tensorflow as tf

import tensorflow.keras.layers as layers

import BreastCancerData as bcd

# n_cols = bcd.train_X.shape[1]
#
# embedding = 'https://tfhub.dev/google/tf2-preview/gnews-swivel-20dim/1'
# hub_layer = layers.Dense(embedding, input_shape=[len(bcd.train_X.keys())], dtype=tf.string, trainable=True)


model = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=[len(bcd.train_X.keys())]),
    layers.Dense(64, activation='relu'),
    layers.Dense(1)
])

# model.add(layers.Dense(64, activation='relu', input_shape=[len(bcd.train_X.keys())]))
# model.add(layers.Dense(64, activation='relu'))
# model.add(layers.Dense(1))

optimizer = tf.keras.optimizers.RMSprop(0.001)

#compile model using mse as a measure of model performance
model.compile(optimizer=optimizer, loss='mse', metrics=['mae', 'mse'])

#set early stopping monitor so the model stops training when it won't improve anymore
early_stopping_monitor = EarlyStopping(patience=3)
#train model
print(model.fit(bcd.train_X, bcd.train_y, validation_split=0.2, epochs=30, callbacks=[early_stopping_monitor]))
