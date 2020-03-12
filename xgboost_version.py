import xgboost as xgb

bst = xgb.Booster({'nthread': 4})
bst.load_model('conversion_random_031120.dat')
