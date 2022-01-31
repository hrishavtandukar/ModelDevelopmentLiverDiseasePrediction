import pandas as pd
import numpy as np
import joblib
import pickle

ml_df_liver = pd.read_csv("indian_liver_patient.csv")

ml_df_liver['Dataset_Binary'] = ml_df_liver['Dataset'].apply(lambda x:0 if x==2 else 1) #using lambda function, if x=="1" then return value as Patient with liver disease else return Patient with no liver disease.

ml_df_liver = ml_df_liver.drop(columns=['Dataset'])

ml_df_liver = pd.concat([ml_df_liver]*2, ignore_index=True)

ml_df_liver = ml_df_liver.dropna() #Dropping the null values

ml_df_liver['Gender_Binary'] = ml_df_liver['Gender'].apply(lambda x:0 if x=="Female" else 1) #using lambda function, if x=="Female" then return value as 1 else return 0.

ml_df_liver = ml_df_liver.drop(columns=['Gender'])

x = ml_df_liver.drop(columns=['Dataset_Binary'])

print(x.head())
print(x.columns)

y = ml_df_liver["Dataset_Binary"]

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42) #Train: 80%, Test: 20%

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators = 100)
rfc.fit(x_train,y_train)

RandomForestClassifierScore = rfc.score(x_test, y_test)
print("Accuracy obtained by Random Forest Classifier model:",RandomForestClassifierScore*100)

with open('liver_model.pkl', 'wb') as files:
    pickle.dump(rfc, files)