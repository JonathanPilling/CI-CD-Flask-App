import pandas as pd
from flask import Flask, jsonify, request
import pickle
import os

# load model
model = pickle.load(open('model.pkl','rb'))

# app
app = Flask(__name__)

# routes
@app.route('/', methods=['POST'])

def predict():
    # get data
    data = request.get_json(force=True)

    # convert data into dataframe
    data.update((x, [y]) for x, y in data.items()) # Square brackets are needed for scalar values, because dictionaries don't have implicit ordering
    data_df = pd.DataFrame.from_dict(data)

    # predictions
    result = model.predict(data_df)

    # send back to browser
    output = {'results': int(result[0])}

    # return data
    return jsonify(results=output)

if __name__ == '__main__':
    # Bind to the port if defined, otherwise default to 5001
    port = int(os.environ.get('PORT', 5000))
    app.run("0.0.0.0", port = port, debug=True)