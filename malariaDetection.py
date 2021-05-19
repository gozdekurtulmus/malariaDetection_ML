# -*- coding: utf-8 -*-
"""
Created on Wed May 19 16:43:53 2021

"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import joblib

# Load dataset
dataframe = pd.read_csv("csv/dataset.csv")
print(dataframe.head())
print("\n")

# Split into training and test data
x= dataframe.drop(["label"], axis=1)
y = dataframe["label"]
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size= 0.2, random_state = 42 )

# Build model
model = RandomForestClassifier(n_estimators = 100, max_depth = 5)
model.fit(x_train, y_train)

joblib.dump(model, "rf_malaria_100_5")  #store the trained model

# Make predictions

predictions = model.predict(x_test)
print(metrics.classification_report(predictions, y_test))

# Precision: tp/ (tp + fp)
# Recall   : tp / (tp + fn)
# F1-score : 2 * precision * recall / (precision + recall)
# Suppor   : Number of occurrences of each class in y_true