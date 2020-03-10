import pandas as pd
from flask import Flask, jsonify, request
import pickle
import os
import psycopg2

# load model
model = pickle.load(open('model.pkl','rb'))

# app
app = Flask(__name__)

# routes
@app.route('/', methods=['POST','GET'])
def predict():
    if request.method == "POST":
        # get data
        data = request.get_json(force=True)

        # Add db data before converting to df
        DATABASE_URL = os.environ['DATABASE_URL']
        conn = psycopg2.connect(DATABASE_URL, sslmode='require')

        cursor = conn.cursor()

        query = "select Age from passengers where Name = 'Jonathan"
        cursor.execute(query)
        record = cursor.fetchall()

        data['Age'] = record

        # convert data into dataframe
        data.update((x, [y]) for x, y in data.items()) # Square brackets are needed for scalar values, because dictionaries don't have implicit ordering
        data_df = pd.DataFrame.from_dict(data)

        # predictions
        result = model.predict(data_df)

        # send back to browser
        output = {'results': int(result[0])}

        # return data
        return jsonify(results=output)
    else: # request == GET
        return ("<h1>Hi! I'm the Progrexion API</h1>")

if __name__ == '__main__':
    # Bind to the port if defined, otherwise default to 5001
    port = int(os.environ.get('PORT', 5000))
    app.run("0.0.0.0", port = port, debug=True)