import pandas as pd
import numpy as np
from flask import Flask, jsonify, request
import pickle
import os
import xgboost as xgb
from collections import OrderedDict
import jsonschema
from jsonschema import validate

# Database Conn
import psycopg2

#DATABASE_URL = os.environ['DATABASE_URL']

#conn = psycopg2.connect(DATABASE_URL, sslmode='require')

# load model
# model = pickle.load(open('model.pkl','rb'))
collections_model = pickle.load(open('models/p60_nopaid_random_0211.sav', 'rb')) 
conversion_model = pickle.load(open('models/conversion_random_122919.sav', 'rb'))

# Offer Product Ids HARD CODED FOR NOW
productIds = [2, 15, 36]

# Create dictionary for camelCase (from payload) -> feature name in RSRT models
camelCaseDict = {
    'conversion180DayAvg': 'conversion_180_day_avg',
    'medHomeValue': 'med_home_value',
    'agentTenure': 'AgentTenure',
    'balBankCardCredit': 'Bal_BankCardCredit',
    'age': 'age',
    'callgroupHotSwap': 'callgroup_hotswap',
    'channelNaturalSearch': 'channel_Natural Search',
    'agentTypePGXHourly': 'AgentType_PGX Hourly',
    'agentTypePGXInbound': 'AgentType_PGX Inbound',
    'motivationPersonalLoan': 'motivation_Personal loan',
    'agentTypePGXIntroHotSwap': 'AgentType_PGX Intro Hot Swap',
    'agentTypeOutsourcingTraining': 'AgentType_Outsourcing Training',
    'avgAgeOfCarLoans': 'avg_age_of_car_loans',
    'utilizationRatio': 'utilization_ratio',
    'score': 'score',
    'tuctsRank': 'TUCTS_Rank',
}

## Static Input lists for machine learning models, order matters
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

# Schema for validating payload
pgx_schema = { 'type': 'object',
    'properties': {
        'conversion180DayAvg': {'type': 'number'},
        'medHomeValue': {'type': 'number'},
        'agentTenure': {'type': 'number'},
        'balBankCardCredit': {'type': 'number'},
        'age': {'type': 'number', 'minimum': 0},
        'callgroupHotSwap': {'type': 'boolean'},
        'agentTypePGXHourly': {'type': 'boolean'},
        'agentTypePGXInbound': {'type': 'boolean'},
        'motivationPersonalLoan': {'type': 'boolean'},
        'agentTypePGXIntroHotSwap': {'type': 'boolean'},
        'agentTypeOutsourcingTraining': {'type': 'boolean'},
        'avgAgeOfCarLoans': {'type': 'boolean'},
        'utilizationRatio': {'type': 'boolean'},
        'score': {'type': 'number'},
        'tuctsRank': {'type': 'number'},
        'zip': {'type': 'string', 'minLength': 5, 'maxLength': 5},
        'agentId': {'type': 'number'},
        'tid': {'type': 'string', 'maxLength': 4}, # Can be 1858, 2356, or 2195
        'channel': {'type': 'string'}, # Need a max length for this for SQL database/ Possible values are 'Natural Search', Lead Gen, Paid Media, Hot Swap External
        }
}

# Helper function for enabling a particular channel
def channelSelect(arg):
    channelSwitch = {
        'Natural Search': 'channel_Natural Search',
        'Paid Media': 'channel_Paid Media',
        'Lead Gen': 'channel_Lead Gen',
        'Hotswap External': 'channel_Hotswap External'
    }
    return channelSwitch.get(arg, 'Invalid Key')

# app
app = Flask(__name__)

# routes
@app.route('/', methods=['POST', 'GET'])
def predict():
    if request.method == 'POST':
        # get data
        data = request.get_json(force=True)

        # Perform schema validation for payload
        try:
            validate(instance=data, schema=pgx_schema)
        except jsonschema.exceptions.ValidationError as err:
            return {"Error Output": err.message}

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
            if key in camelCaseDict:
                realKey = camelCaseDict[key] # Use real key so feature names match
                if realKey in conversion_data:
                    conversion_data[realKey] = val
                if realKey in collections_data:
                    collections_data[realKey] = val

        # Perform month/quarter calculations

        # Enable bits based on channel and tid
        
   
        # convert data into dataframe
        collections_data.update((x, [y]) for x, y in collections_data.items())
        conversion_data.update((x, [y]) for x, y in conversion_data.items()) 
        collections_data_df = pd.DataFrame.from_dict(collections_data)
        conversion_data_df = pd.DataFrame.from_dict(conversion_data)

        # Iterate through possible product ids
        bestPrediction = 0
        bestCollection = 0
        bestConversion = 0
        bestProduct = -1

        for val in productIds:
            collections_data_df.at[0, 'OfferProductID'] = val
            conversion_data_df.at[0, 'OfferProductID'] = val
            # predictions
            collections_result = collections_model.predict(collections_data_df)[0]
            conversion_result = conversion_model.predict_proba(conversion_data_df)[0,1]
            final_prediction = collections_result * conversion_result

            if final_prediction > bestPrediction:
                bestPrediction = final_prediction
                bestCollection = collections_result
                bestConversion = conversion_result
                bestProduct = val

        # send back to browser
        output = {'collections_result': bestCollection.item(), 'conversion_result': bestConversion.item(), 'final_prediction': bestPrediction.item(), 'OfferProductId': bestProduct}

        # write a row to audit db

        # return data
        return jsonify(output)
    else: # request == GET
        return ("<h1>Hi! I'm the Progrexion API (DEV Edition)</h1>")

if __name__ == '__main__':
    # Bind to the port if defined, otherwise default to 5001
    port = int(os.environ.get('PORT', 5000))
    app.run('0.0.0.0', port = port, debug=True)