import datetime
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

# Load in models
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
    'avgAgeOfRevolvingAccounts': 'avg_age_of_revolving_accounts',
    'utilizationRatio': 'utilization_ratio',
    'score': 'score',
    'tuctsRank': 'TUCTS_Rank',
    'amtInstPaidClosed': 'amt_inst_paid_closed',
    'maxLatePayRev': 'max_latePayRev',
    'avgAgeOfMortages': 'avg_age_of_mortgages',
    'previousPaid': 'previous_paid',
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
        # Retrieved from agent_id in the future
        'conversion180DayAvg': {'type': 'number'},
        'agentTypeOutsourcingTraining': {'type': 'integer', 'minimum': 0, 'maximum': 1},
        'agentTypePGXHourly': {'type': 'integer', 'minimum': 0, 'maximum': 1},
        'agentTypePGXInbound': {'type': 'integer', 'minimum': 0, 'maximum': 1},
        'agentTenure': {'type': 'number'},
        'agentTypePGXIntroHotSwap': {'type': 'integer', 'minimum': 0, 'maximum': 1},

        # IT will be sending a field called calltype, if that field contains the work "hotswap" then callgroup_hotswap = 1
        'callgroupHotSwap': {'type': 'integer', 'minimum': 0, 'maximum': 1},

        # Lookup from zip code
        'medHomeValue': {'type': 'number'},     
        'balBankCardCredit': {'type': 'number'},

        # Retrieved from dyson session/report id in the future
        'age': {'type': 'number', 'minimum': 0},      
        'motivationPersonalLoan': {'type': 'integer', 'minimum': 0, 'maximum': 1},      
        'avgAgeOfCarLoans': {'type': 'number'},
        'utilizationRatio': {'type': 'integer', 'minimum': 0, 'maximum': 1},
        'amtInstPaidClosed': {'type': 'number'},
        'maxLatePayRev': {'type': 'number'},
        'score': {'type': 'number'},
        'tuctsRank': {'type': 'number'},
        'avgAgeOfRevolvingAccounts': {'type': 'number'},
        'avgAgeOfMortgages': {'type': 'number'},

        # IT is calculating this value in Offers Engine, we'll need to do the same
        'previousPaid': {'type': 'integer', 'minimum': 0, 'maximum': 1},

        # Actual payload values
        'zip': {'type': 'string', 'pattern': "^[0-9]{5}$"},
        'agentId': {'type': 'number'},
        'tid': {'type': 'string', 'maxLength': 4}, # Not sure how we want to use this
        'channel': {'type': 'string'}, # Need a max length for this for SQL database/ Possible values are 'Natural Search', Lead Gen, Paid Media, Hot Swap External?
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

# Method for enabling a particular tid
def tidSelect(arg):
    tidSwitch = {
        '1858': 'ptid_1858',
        '2356': 'ptid_2356',
        '2195': 'ptid_2195',
    }
    return tidSwitch.get(arg, 'Invalid Key')

# Write to the audit analytics table in cloud Postgres
def writeToDb(data, collections_data, conversion_data, bestCollection, bestConversion, bestPrediction, bestProduct):

    conn = None
    try:
        DATABASE_URL = os.environ['DATABASE_URL']
        conn = psycopg2.connect(host="localhost", database="184973", user="184973", password="Goldenratio66")
        cursor = conn.cursor()
        print(conversion_data['conversion_180_day_avg'])
        cursor.execute('''        
            INSERT INTO analytics_audit (logTime, offerProductId, conversion180DayAvg, agentTenure,
            medHomeValue, balBankCardCredit, age, callgroupHotswap, currMonth, qtr,
            agentTypePGXHourly, agentTypePGXInbound, motivationPersonalLoan, agentTypePGXIntroHotSwap,
            agentTypeOutsourcingTraining, avgAgeOfCarLoans, utilizationRatio, score, tuctsRank, zip,
            agentId, tid, channel, collectionsResult, conversionResult, finalResult
            ) 
            VALUES ( %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s);''', 
                    (datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), bestProduct, conversion_data['conversion_180_day_avg'],
                    conversion_data['AgentTenure'], conversion_data['med_home_value'], conversion_data['Bal_BankCardCredit'],
                    collections_data['age'], bool(collections_data['callgroup_hotswap']), collections_data['month'], collections_data['qtr'],
                    bool(conversion_data['AgentType_PGX Hourly']), bool(conversion_data['AgentType_PGX Inbound']), bool(collections_data['motivation_Personal loan']),
                    bool(conversion_data['AgentType_PGX Intro Hot Swap']), bool(conversion_data['AgentType_Outsourcing Training']),
                    collections_data['avg_age_of_car_loans'], collections_data['utilization_ratio'], conversion_data['score'],
                    conversion_data['TUCTS_Rank'], data['zip'], data['agentId'], data['tid'], data['channel'],
                    float(bestCollection), float(bestConversion), float(bestPrediction)))
    except Exception as e:
        print('Error: ', repr(e))
    finally:
        # Close connection
        if conn is not None:
            conn.close() 

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
        today = datetime.datetime.today()
        collections_data['month'] = today.month
        collections_data['qtr'] = (today.month-1)//3 + 1

        # Enable bits based on channel
        if "channel" in data.keys():
            channel = channelSelect(data['channel'])
            if channel in inputs_collections:
                collections_data[channel] = 1
            if channel in inputs_conversion:
                conversion_data[channel] = 1

        # Enable bits based on tid
        if "tid" in data.keys():
            tid = tidSelect(data['tid'])
            conversion_data[tid] = 1
 
        # convert data into dataframe
        collections_data_df = pd.DataFrame(collections_data, index=[0])
        conversion_data_df = pd.DataFrame(conversion_data, index=[0])

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

        # Hacky way of converting np.nan to null for db insert. If we follow API contract we shouldn't have to deal with this issue
        for key, val in collections_data.items():
            if val == np.nan:
                collections_data[key] = None
        for key, val in conversion_data.items():
            if val == np.nan:
                conversion_data[key] = None
                
        # write a row to audit db
        writeToDb(data, collections_data, conversion_data, bestCollection, bestConversion, bestPrediction, bestProduct)

        # return data
        return jsonify(output)
    else: # request == GET
        return ("<h1>Hi! I'm the Progrexion API (DEV Edition)</h1>")

if __name__ == '__main__':
    # Bind to the port if defined, otherwise default to 5000
    port = int(os.environ.get('PORT', 5000))
    app.run('0.0.0.0', port = port, debug=True)