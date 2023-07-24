import openml
from sklearn import model_selection, preprocessing
import keras_core as keras

from keras_neural_trees.latent_trees import LatentTree

x, y = openml.tasks.get_task(361091).get_X_and_y()
y = y.reshape(-1, 1)

x, x_test, y, y_test = model_selection.train_test_split(x, y, train_size=100000, test_size=10000)
_x_prepr = preprocessing.StandardScaler()
x = _x_prepr.fit_transform(x).astype("float32")
x_test = _x_prepr.transform(x_test).astype("float32")
_y_prepr = preprocessing.StandardScaler()
y = _y_prepr.fit_transform(y).astype("float32")
y_test = _y_prepr.transform(y_test).astype("float32")

model = keras.Sequential(
    [
        keras.layers.Dense(256, activation='elu'), keras.layers.Dropout(0.1),
        keras.layers.Dense(128, activation='elu'), keras.layers.Dropout(0.1),
        keras.layers.Dense(64, activation='elu'), keras.layers.Dropout(0.1),
    ]
    +
    [
        LatentTree(5, reg_lambda=1.0),
        keras.layers.Dense(1)
    ]
)

model.compile(loss='mse', optimizer=keras.optimizers.Adam(1e-3), jit_compile=False)

model.fit(x, y, validation_data=(x_test, y_test), epochs=100, batch_size=512)
