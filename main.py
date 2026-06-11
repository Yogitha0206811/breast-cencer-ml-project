import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

df = pd.read_csv('Breast_Cancer-selected-columns (1).csv') # Load Data
print("Original Data Loaded:", df.shape)

df = df.drop_duplicates() # Data Cleaning
df = df.fillna(df.median(numeric_only=True))
df = df.fillna(df.mode().iloc[0])
print("After Cleaning:",df.shape)
# pre-processing
for c in df.select_dtypes('object'): # Convert String → Numbers
    df[c] = LabelEncoder().fit_transform(df[c])
print("After Encoding:",df.head())

corr = df.corr()['Tumor Size'].abs() # Feature Selection (keep only useful features)
features = corr[corr>0.1].index.drop('Tumor Size')
print("co-relations:\n",corr,"\nSelected Features:",list(features))

x = df[features] # Split Data for training
y = df['Tumor Size']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(x_train, y_train)

# Prediction
y_pred = model.predict(x_test)

# Evaluation
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("MAE:", mae)
print("R2 Score:", r2)

# Save Model
joblib.dump(model, "cancer_model.pkl")

print("cancer_model.pkl file created successfully")
