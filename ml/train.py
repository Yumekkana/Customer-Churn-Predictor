import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

df = pd.read_csv("data/cleaned_df.csv")

X = df.iloc[:, :-1]
y = df.iloc[:, -1]

scaler = StandardScaler()
X = scaler.fit_transform(X)

def model_selection():
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
    return X_train, X_test, y_train, y_test

def train():
    classifier = XGBClassifier(objective='binary:logistic',base_score=0.5, booster='gbtree', colsample_bylevel=1,
                colsample_bynode=1, colsample_bytree=1, gamma=1,max_depth=10,random_state=0,reg_alpha=0, reg_lambda=12,
                tree_method='exact', validate_parameters=1, verbosity=None)
    
    X_train, X_test, y_train, y_test = model_selection()
    classifier.fit(X_train, y_train)

    with open('ml/model.pkl', 'wb') as file:
        pickle.dump(classifier, file, protocol=4)

    with open('ml/scaler.pkl', 'wb') as scaler_file:
        pickle.dump(scaler, scaler_file)

train()