import pandas as pd
import numpy as np
from flask import Flask, jsonify, request
import pickle
import os
import xgboost as xgb
from collections import OrderedDict

# load model
# model = pickle.load(open('model.pkl','rb'))
collections_model = pickle.load(open('models/p60_nopaid_random_0211.sav', 'rb')) 
conversion_model = pickle.load(open('models/conversion_random_122919.sav', 'rb'))

## Input Lists
inputs_conversion = [
    'OfferProductID',
    'conversion_180_day_avg',
    'ptid_2356',
    'AgentType_Outsourcing Training',
    'AgentTenure',
    'AgentType_PGX Inbound',
    'AgentType_PGX Intro Hot Swap',
    'AgentType_PGX Hourly',
    'ptid_2195',
    'callgroup_hotswap',
    'avg_age_of_car_loans',
    'channel_Lead Gen',
    'ptid_1858',
    'utilization_ratio',
    'med_home_value',
    'Bal_BankCardCredit',
    'score',
    'TUCTS_Rank'
    ]

inputs_collections = [
    'TUCTS_Rank',
    'avg_age_of_revolving_accounts',
    'avg_age_of_car_loans',
    'age',
    'amt_inst_paid_closed',
    'score',
    'utilization_ratio',
    'OfferProductID',
    'motivation_Personal loan',
    'channel_Paid Media',
    'channel_Hotswap External',
    'channel_Natural Search',
    'max_latePayRev',
    'callgroup_hotswap',
    'month',
    'channel_Lead Gen',
    'avg_age_of_mortgages',
    'qtr',
    'previous_paid'
    ]

# app
app = Flask(__name__)

# routes
@app.route('/', methods=['POST', 'GET'])
def predict():
    if request.method == "POST":
        # get data
        data = request.get_json(force=True)

        # Preserve ordering for pandas df
        collections_data = OrderedDict()
        conversion_data = OrderedDict()

        # Initialize preserved feature ordering
        for entry in inputs_conversion:
            conversion_data[entry] = np.nan
        for entry in inputs_collections:
            collections_data[entry] = np.nan

        # Split data into conversion/collections model,
        # Note that some keys are in both ml models
        for key, val in data.items():
            if key in conversion_data:
                conversion_data[key] = val
            if key in collections_data:
                collections_data[key] = val
   
        # convert data into dataframe
        collections_data.update((x, [y]) for x, y in collections_data.items())
        conversion_data.update((x, [y]) for x, y in conversion_data.items()) 
        collections_data_df = pd.DataFrame.from_dict(collections_data)
        conversion_data_df = pd.DataFrame.from_dict(conversion_data)

        # predictions
        collections_result = collections_model.predict(collections_data_df)[0]
        conversion_result = conversion_model.predict_proba(conversion_data_df)[0,1]
        final_prediction = collections_result * conversion_result

        # send back to browser
        output = {'collections_result': collections_result.item(), 'conversion_result': conversion_result.item(), 'final_prediction': final_prediction.item()}

        # return data
        return jsonify(output)
    else: # request == GET
        return ("<h1>Hi! I'm the Progrexion API</h1>")

if __name__ == '__main__':
    # Bind to the port if defined, otherwise default to 5001
    port = int(os.environ.get('PORT', 5000))
    app.run("0.0.0.0", port = port, debug=True)