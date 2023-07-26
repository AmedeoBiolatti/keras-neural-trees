import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import lightgbm
import keras_core as keras

from keras_neural_trees.base_tree import BaseTrees

pd.options.display.max_columns = 20

x = np.random.normal(size=(256, 3))
y = np.sum(x, axis=-1)

model = lightgbm.LGBMRegressor(n_estimators=10, max_depth=8, learning_rate=0.3)
model.fit(x, y)

tree = BaseTrees.from_lightgbm(model)

t0 = time.time()
p_xgb = model.predict(x)
t1 = time.time()
p_tree = tree(x)
t2 = time.time()

print("%.4f" % (t1 - t0))
print("%.4f" % (t2 - t1))

plt.scatter(y, p_xgb, color='tab:cyan')
plt.scatter(y, p_tree, color='tab:orange', marker='x')
plt.show()
