from flask import Flask, request
import json
import pickle

with open("trained_model", "rb") as fl:
    model_obj, data_obj = pickle.load(fl)
    
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
        # print('Data Received: "{data}"'.format(data=data))
        out = request.get_json()
        out = data_obj.transform_posted_data(out)
        y_pred = model_obj.model.predict(out)
        show_ind = y_pred.mean(axis=0).argsort()[-3:]
        show_id = list(data_obj.show_id[show_ind])
        # print('Data Received: "{data}"'.format(data=data))
        return {"predicted_shows": show_id}

app.run()#(host="localhost", port=8080)