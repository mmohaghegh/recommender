#!/usr/bin/env python3

import pickle
import numpy as np
import matplotlib.pyplot as plt
from flask import Flask, request
from forcaster import Model, Data

data = Data()
feature_space = ["user_age", "user_languages", "user_interests"]
for feat in feature_space:
    data.transform_features(feat)
data.concat_features()
data.transform_target()
data.split_train_val_test()

model = Model(data)
model.setup()
model.fit(20)
# Store the model and data objects into 'trained_model' file.
with open("trained_model", "wb") as fl:
    pickle.dump([model, data], fl)