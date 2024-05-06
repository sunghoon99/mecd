import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from joblib import dump

# test dummy data + label
df = pd.read_csv('test.csv', header=None, names=['ax', 'ay', 'az', 'rx', 'ry', 'rz', 'label'])

# Independent var dependent var classification
X = df[['ax', 'ay', 'az', 'rx', 'ry', 'rz']]
y = df['label']

# training set / test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# normalization
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# model
model = LogisticRegression()
model.fit(X_train, y_train)

print("finished training")

# evaluation
predictions = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, predictions))

new_data = pd.DataFrame([[0,0,0,0,0,40]], columns=['ax', 'ay', 'az', 'rx', 'ry', 'rz'])
new_data = scaler.transform(new_data)  # input data normalization
prediction = model.predict(new_data)
print("Predicted label:", prediction[0])

# model save
dump(model, 'model.joblib')

# scaler save
dump(scaler, 'scaler.joblib')
