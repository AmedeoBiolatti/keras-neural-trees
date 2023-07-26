import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import lightgbm
import keras_core as keras

from keras_neural_trees.base_tree import BaseTrees

pd.options.display.max_columns = 20

x = np.random.normal(size=(512, 1))
y = (np.sum(x, axis=-1) > 0.0).astype(int)

model = lightgbm.LGBMClassifier(n_estimators=1, max_depth=2, learning_rate=0.3)
model.fit(x, y)

tree = BaseTrees.from_lightgbm(model)

p_xgb = model.predict_proba(x)[..., 1]

p_tree = tree(x)

plt.scatter(y, p_xgb, color='tab:cyan')
plt.scatter(y, p_tree, color='tab:orange', marker='x')
plt.show()

plt.figure()
plt.scatter(x, y, color='tab:red')
plt.scatter(x, p_xgb, color='tab:cyan')
plt.scatter(x, p_tree, color='tab:orange', marker='x')
plt.show()
