import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import xgboost
import keras_core as keras

from keras_neural_trees.base_tree import BaseTrees

pd.options.display.max_columns = 20

x = np.random.normal(size=(256, 3))
y = (np.sum(x, axis=-1) > 0.0).astype(int)

model = xgboost.XGBClassifier(n_estimators=10, max_depth=4, learning_rate=0.3)
model.fit(x, y)
model_df = model.get_booster().trees_to_dataframe()[['Tree', 'Node', 'ID', 'Feature', 'Split', 'Yes', 'No', 'Gain']]

tree = BaseTrees.from_xgboost(model)

p_xgb = model.predict_proba(x)[..., 1]

p_tree = tree(x)

plt.scatter(y, p_xgb, color='tab:cyan')
plt.scatter(y, p_tree, color='tab:orange', marker='x')
plt.show()

