from tensorflow import keras

import tensorflow_docs as tfdocs
import tensorflow_docs.plots
import tensorflow_docs.modeling

import tensorflow as tf
import tensorflow.keras.layers as layers

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import BreastCancerData as bcd


def build_model():
    network_model = keras.Sequential([
        layers.Dense(64, activation='relu', input_shape=[len(bcd.train_dataset.keys())]),
        layers.Dense(64, activation='relu'),
        layers.Dense(1)
    ])

    optimizer = tf.keras.optimizers.RMSprop(0.001)

    network_model.compile(optimizer=optimizer, loss='mse', metrics=['mae', 'mse'])

    return network_model


model = build_model()

model.summary()

example_batch = bcd.normed_train_data[:10]
example_result = model.predict(example_batch)
print(example_result)

EPOCHS = 1000

early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)

history = model.fit(
    bcd.normed_train_data,
    bcd.train_labels,
    epochs=EPOCHS,
    validation_split=0.2,
    verbose=0,
    callbacks=[tfdocs.modeling.EpochDots()]
)

# early_history = model.fit(
#     bcd.normed_train_data,
#     bcd.train_labels,
#     epochs=EPOCHS,
#     validation_split=0.2,
#     verbose=0,
#     callbacks=[early_stop, tfdocs.modeling.EpochDots()]
# )

# hist = pd.DataFrame(history.history)
# hist['epoch'] = history.epoch
# print(hist.tail())

plotter = tfdocs.plots.HistoryPlotter(smoothing_std=2)

# These show mea and mse in graph form

# plotter.plot({'Basic': history}, metric="mean_absolute_error")
# # plt.ylim([0, 10])
# # plt.ylabel('MAE [Overall Survival (Months)]')
# #
# # plotter.plot({'Basic': history}, metric="mean_squared_error")
# # plt.ylim([0, 20])
# # plt.ylabel('MSE [Overall Survival (Months)^2]')


# plotter.plot({'Early Stopping': early_history}, metric="mean_absolute_error")
# plt.ylim([0, 10000])
# plt.ylabel('MAE [Overall Survival (Months)]')

test_predictions = model.predict(bcd.normed_test_data).flatten()

a = plt.axes(aspect='equal')
plt.scatter(bcd.test_labels, test_predictions)
plt.xlabel('True Values [Overall Survival (Months)]')
plt.ylabel('Predictions [Overall Survival (Months)]')
lims = [0, 400]
plt.xlim(lims)
plt.ylim(lims)
_ = plt.plot(lims, lims)

