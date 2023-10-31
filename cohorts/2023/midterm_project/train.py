import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# %matplotlib inline
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import roc_auc_score, mutual_info_score, confusion_matrix
from sklearn import tree
#import xgboost as xgb
from xgboost import XGBClassifier
import pickle 

numerical = ["adults", "lead_time", "previous_cancellations", "previous_bookings_not_canceled", "booking_changes", "adr", "required_car_parking_spaces", "total_of_special_requests"]
categorical = [ "hotel", "deposit_type", "customer_type", "different_room_type", "arrival_date_month", "wait_listed"]
 
def train_fit_model(X_train, y_train, X_test, y_test, min_child_weight, max_depth, lr): 
    model = XGBClassifier(min_child_weight = min_child_weight, max_depth=max_depth, learning_rate=lr, objective='binary:logistic', enable_categorical=True)
    model.fit(X_train, y_train)
    y_pred = model.predict_proba(X_test)
    return roc_auc_score(y_test, y_pred[:,1]), model

df = pd.read_csv("hotel_booking.csv")
df.columns = df.columns.str.replace("-", "_")
df.columns = df.columns.str.replace(" ", "_")
df_fulltrain, df_test = train_test_split(df, test_size=.2, random_state=11)

# Feature Engineering
df_fulltrain.lead_time = np.log1p(df_fulltrain.lead_time)
df_test.lead_time = np.log1p(df_test.lead_time)
### Add 2 new variables to reduce category cardinality but keep the data 
df_fulltrain["wait_listed"] = df_fulltrain["days_in_waiting_list"] > 0
df_test["wait_listed"] = df_test["days_in_waiting_list"] > 0
df_fulltrain["different_room_type"] = df_fulltrain["reserved_room_type"] != df_fulltrain["assigned_room_type"]
df_test["different_room_type"] = df_test["reserved_room_type"] != df_test["assigned_room_type"]

# Fill NA
mode_values = df_fulltrain[categorical].mode().iloc[0]
df_fulltrain[categorical] = df_fulltrain[categorical].fillna(mode_values)
df_test[categorical] = df_test[categorical].fillna(mode_values)
       
# pass it into XGBoost that thse are categorical
for c in categorical:
    df_fulltrain[c] = df_fulltrain[c].astype('category')
    df_test[c] = df_test[c].astype('category')
df_fulltrain.dtypes
df_test = df_test[numerical+categorical+["is_canceled"]]
df_fulltrain = df_fulltrain[numerical+categorical+["is_canceled"]]

df_fulltrain = df_fulltrain.reset_index(drop=True)
#df_val = df_val.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)
y_fulltrain = df_fulltrain["is_canceled"].values
#y_val = df_val["is_canceled"].values
y_test = df_test["is_canceled"].values
del df_fulltrain["is_canceled"]
#del df_val["is_canceled"]
del df_test["is_canceled"]
train_dicts = df_fulltrain.to_dict(orient='records')
dv = DictVectorizer(sparse=True)
X_fulltrain = dv.fit_transform(train_dicts)
test_dicts = df_test.to_dict(orient='records')
X_test = dv.transform(test_dicts)
final_auc, model = train_fit_model(X_fulltrain, y_fulltrain, X_test, y_test, 169, 22, .3)
print(final_auc)
model.save_model('xgb_model.json')
output_file = "dv.bin"
with open(output_file, "wb") as f_out:
    pickle.dump((dv), f_out)