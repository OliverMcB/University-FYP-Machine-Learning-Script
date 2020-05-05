from tensorflow import keras

import tensorflow_docs as tfdocs
import tensorflow_docs.plots
import tensorflow_docs.modeling

import tensorflow as tf
import tensorflow.keras.layers as layers

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import BreastCancerDeepLearning.breastcancerdata as bcd

model = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=[len(bcd.train_dataset.keys())]),
    layers.Dense(64, activation='relu'),
    layers.Dense(1)
])

optimizer = tf.keras.optimizers.RMSprop(0.001)

model.compile(optimizer=optimizer, loss='mse', metrics=['mae', 'mse'])


def initialise():

    epochs = 1000

    model.fit(
        bcd.normed_train_data,
        bcd.train_labels,
        epochs=epochs,
        validation_split=0.2,
        verbose=0,
        callbacks=[tfdocs.modeling.EpochDots()]
    )


def predict(data):

    data = bcd.convert_data_types(data)
    data = bcd.convert_data(data)
    data = bcd.remove_survival_months(data)
    data = bcd.remove_goals(data)

    normed_data = bcd.norm(data, bcd.train_stats)

    return model.predict(normed_data).flatten()


def model_accuracy():

    return model.predict(bcd.normed_test_data).flatten()


# plotter = tfdocs.plots.HistoryPlotter(smoothing_std=2)

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


initialise()

test_predictions = model_accuracy()

a = plt.axes(aspect='equal')
plt.scatter(bcd.test_labels, test_predictions)
plt.xlabel('True Values [Overall Survival (Months)]')
plt.ylabel('Predictions [Overall Survival (Months)]')
lims = [0, 400]
plt.xlim(lims)
plt.ylim(lims)
_ = plt.plot(lims, lims)

