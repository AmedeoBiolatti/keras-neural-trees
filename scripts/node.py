import openml
from sklearn import model_selection, preprocessing
import keras_core as keras

from keras_neural_trees.node import NODE

x, y = openml.tasks.get_task(361091).get_X_and_y()
y = y.reshape(-1, 1)

x, x_test, y, y_test = model_selection.train_test_split(x, y, train_size=100000, test_size=10000)
_x_prepr = preprocessing.QuantileTransformer(output_distribution='normal')
x = _x_prepr.fit_transform(x)
x_test = _x_prepr.transform(x_test)
_y_prepr = preprocessing.StandardScaler()
y = _y_prepr.fit_transform(y)
y_test = _y_prepr.transform(y_test)

model = keras.Sequential(
    [
        NODE(units=4, n_layers=4, n_trees_per_layer=32, depth=4, oblivious=True),
        keras.layers.Dense(1)
    ]
)
model.compile(loss='mse', optimizer=keras.optimizers.Adam(1e-3), jit_compile=True)
model.fit(x, y, validation_data=(x_test, y_test), epochs=100, batch_size=256)
