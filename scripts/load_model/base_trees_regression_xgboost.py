import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import xgboost
import keras_core as keras

from keras_neural_trees.base_tree import BaseTrees

pd.options.display.max_columns = 20

x = np.random.normal(size=(256, 3))
y = np.sum(x, axis=-1)

model = xgboost.XGBRegressor(n_estimators=10, max_depth=4, learning_rate=0.3, base_score=-0.1)
model.fit(x, y)
model_df = model.get_booster().trees_to_dataframe()[['Tree', 'Node', 'ID', 'Feature', 'Split', 'Yes', 'No', 'Gain']]

tree = BaseTrees.from_xgboost(model)

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

# individual_preds = []
# for tree_ in model.get_booster():
#     individual_preds.append(
#         tree_.predict(xgboost.DMatrix(x))
#     )
# individual_preds = np.stack(individual_preds, axis=-1)
# fig, ax = plt.subplots(1, 3)
# for i in range(3):
#     ax[i].scatter(y, individual_preds[:, i], color='tab:cyan')
#     ax[i].scatter(y, p_tree[:, i], color='tab:orange', marker="x")
# plt.show()
