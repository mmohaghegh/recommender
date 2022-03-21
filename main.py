#!/usr/bin/env python3

import pickle
import numpy as np
import matplotlib.pyplot as plt
from flask import Flask, request
from Model import model, data


data_obj = data("adition")
feature_space = ["user_age", "user_languages", "user_interests"]
for feat in feature_space:
    data_obj.transform_features(feat)
data_obj.concat_features()
data_obj.transform_target()
data_obj.split_train_val_test()

model_obj = model(data_obj)
model_obj.setup()
model_obj.fit()

with open("trained_model", "wb") as fl:
    pickle.dump([model_obj, data_obj], fl)
    

# with open('data.json') as fl:
#     dic = json.load(fl)
    
# out = data_obj.transform_posted_data(dic)
# y_pred = model_obj.model.predict(out)
# show_ind = y_pred.mean(axis=0).argsort()[-3:]
# show_id = list(data_obj.show_id[show_ind])

app = Flask(__name__)

@app.route('/predict-shows', methods=['POST'])
def post_route():
    if request.method == 'POST':
        print('Data Received: "{data}"'.format(data=data))
        out = request.get_json()
        out = data_obj.transform_posted_data(out)
        y_pred = model_obj.model.predict(out)
        show_ind = y_pred.mean(axis=0).argsort()[-3:]
        show_id = list(data_obj.show_id[show_ind])
        # print('Data Received: "{data}"'.format(data=data))
        return {"predicted_shows": show_id}

app.run()