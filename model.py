import pandas as pd
import sklearn
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np
import joblib

df = pd.read_csv('Salary_dataset.csv')
X = df.iloc[:,1]
y = df.iloc[:,2]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

X_train = np.array(X_train).reshape(-1,1)
X_test = np.array(X_test).reshape(-1,1)

model = LinearRegression()

model.fit(X_train,y_train)
joblib.dump(model, "model.joblib")
print("Model Trained")

y_pred = model.predict(X_test)

with open("metrics.txt", 'w') as fw:
  fw.write(f"Mean Squared Error of current model is: {mean_squared_error(y_test, y_pred)}")
