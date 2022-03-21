"""
This python file run multiple different models that vary in some of their
hyperparameters, which determine the model complexity. These hyperparameters
consists of 'number of hidden layers' and 'number of units'. For each model
the average validation accuracy for the last 5 epochs is measured, stored and
visualized.
"""

import os
import json
import copy
import numpy as np
import matplotlib.pyplot as plt
from forcaster import Model, Data
from keras import backend as K

with open(os.path.join("./", "parameters.json")) as fl:
    par = json.load(fl)
    
num_layers = np.arange(1, 5, 1)
num_units = np.arange(100, 401, 100)
n_l_comb, n_u_comb = np.meshgrid(num_layers, num_units)
n_l_comb = n_l_comb.flatten()
n_u_comb = n_u_comb.flatten()
acc = np.zeros(n_l_comb.shape)

for c_ind in range(n_l_comb.size):
    print("\nTraining the model with {:.0f} layer(s)".format(n_l_comb[c_ind]) + \
          " and {:.0f} units in each layer ...".format(n_u_comb[c_ind]))
    data = Data()
    feature_space = ["user_age", "user_languages", "user_interests"]
    for feat in feature_space:
        data.transform_features(feat)
    data.concat_features()
    data.transform_target()
    data.split_train_val_test()

    model = Model(data)
    model.setup(n_l_comb[c_ind], n_u_comb[c_ind])
    model.fit(40)
    acc[c_ind] = np.mean(model.model.history.history['val_accuracy'][-5:])
    K.clear_session()

acc = acc.reshape(num_layers.size, num_units.size, order='F')
fig, ax = plt.subplots()    
for l_i, lay in enumerate(num_layers):
    ax.plot(num_layers, acc[l_i], label="n_layer(s)={}".format(lay))

ax.set_xlabel("accuracy")
ax.set_xlabel("number of units")
ax.legend()
fig.savefig("model_selection.pdf")