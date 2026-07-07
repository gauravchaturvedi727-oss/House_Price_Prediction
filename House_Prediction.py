import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv("housing_price_dataset.csv")
print("\nDataset Loaded Successfully")


print("\nFirst 5 Rows:")
print(df.head())

print("\nDataset Shape:")
print(df.shape)

print("\nDataset Columns:")
print(df.columns)

print("\nDataset Information:")
df.info()

print("\nStatistical Summary:")
print(df.describe())

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDuplicate Rows:")
print(df.duplicated().sum())

df.drop_duplicates(inplace=True)
df.dropna(inplace=True)

print("\nDataset Cleaned Successfully")
print("\nDataset Shape After Cleaning:")
print(df.shape)

plt.figure(figsize=(8, 5))

plt.hist(
    df["price_in_lakhs"],
    bins=30
)

plt.xlabel("House Price in Lakhs")
plt.ylabel("Number of Houses")
plt.title("House Price Distribution")
plt.show()

numeric_df = df.select_dtypes(
    include=["int64", "float64"]
)

correlation = numeric_df.corr()[
    "price_in_lakhs"
].sort_values(
    ascending=False
)

print("\nFeature Correlation With House Price:")
print(correlation)

correlation_matrix = numeric_df.corr()

plt.figure(figsize=(15, 10))

plt.imshow(correlation_matrix,interpolation="nearest")
plt.title("House Price Feature Correlation Matrix")

plt.colorbar()

plt.xticks(
    range(len(correlation_matrix.columns)),
    correlation_matrix.columns,
    rotation=90
)

plt.yticks(
    range(len(correlation_matrix.columns)),
    correlation_matrix.columns
)

plt.tight_layout()
plt.show()


features = [
    "bhk",
    "bathrooms",
    "balconies",
    "built_up_area",
    "carpet_area",
    "property_age",
    "parking_spaces",
    "security_score",
    "gym_available",
    "swimming_pool",
    "power_backup",
    "lift_available",
    "maintenance_fee_monthly",
    "distance_to_city_center_km",
    "distance_to_metro_km",
    "nearby_schools",
    "nearby_hospitals"
]

X = df[features]
y = df["price_in_lakhs"]

print("\nSelected Features:")
print(features)

print("\nInput Data Shape:")
print(X.shape)

print("\nTarget Data Shape:")
print(y.shape)
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\nTraining Data Size:")
print(X_train.shape)

print("\nTesting Data Size:")
print(X_test.shape)

model = LinearRegression()
model.fit(
    X_train,
    y_train
)

print("\nLinear Regression Model Trained Successfully")

y_pred = model.predict(
    X_test
)

print("\nPredictions Generated Successfully")

prediction_result = pd.DataFrame({
    "Actual Price": y_test.values,
    "Predicted Price": y_pred
})

print("\nFirst 10 Predictions:")
print(prediction_result.head(10))

mse = mean_squared_error(
    y_test,
    y_pred
)

rmse = np.sqrt(mse)

r2 = r2_score(
    y_test,
    y_pred
)
print("\nMODEL PERFORMANCE")
print("\nRMSE:")
print(rmse)
print("\nR2 Score:")
print(r2)

print("\nMODEL PERFORMANCE INTERPRETATION")

print("RMSE represents the average prediction error","of the model in lakhs.")

print("R2 Score represents how much variation","in house prices is explained by the model.")

coefficient_df = pd.DataFrame({
    "Feature": features,
    "Coefficient": model.coef_
})
coefficient_df = coefficient_df.sort_values(
    by="Coefficient",
    ascending=False
)

print("\nMODEL COEFFICIENTS")
print(coefficient_df)

print("\nCOEFFICIENT INTERPRETATION")

for index, row in coefficient_df.iterrows():
    feature = row["Feature"]

    coefficient = row["Coefficient"]

    if coefficient > 0:
        print(feature,"has a POSITIVE effect on house price.")
    else:
        print(feature,"has a NEGATIVE effect on house price.")

plt.figure(figsize=(8, 5))

plt.scatter(y_test,y_pred,alpha=0.5)
plt.xlabel("Actual House Price in Lakhs")

plt.ylabel("Predicted House Price in Lakhs")

plt.title("Actual Price vs Predicted Price")

plt.show()
joblib.dump(
    model,
    "house_price_model.pkl"
)

print("\nModel Saved Successfully")

print("Model File: house_price_model.pkl")

example_house = pd.DataFrame({
    "bhk": [3],
    "bathrooms": [2],
    "balconies": [2],
    "built_up_area": [1500],
    "carpet_area": [1200],
    "property_age": [5],
    "parking_spaces": [1],
    "security_score": [8.5],
    "gym_available": [1],
    "swimming_pool": [1],
    "power_backup": [1],
    "lift_available": [1],
    "maintenance_fee_monthly": [5000],
    "distance_to_city_center_km": [5.0],
    "distance_to_metro_km": [2.0],
    "nearby_schools": [4],
    "nearby_hospitals": [3]
})


predicted_price = model.predict(
    example_house
)

print("\nEXAMPLE HOUSE DETAILS")

print(example_house)

print("\nPREDICTED HOUSE PRICE")
print(round(predicted_price[0], 2),"Lakhs")

loaded_model = joblib.load(
    "house_price_model.pkl"
)
loaded_prediction = loaded_model.predict(
    example_house
)

print("\nPREDICTION USING SAVED MODEL")
print(round(loaded_prediction[0], 2), "Lakhs")
print("\nHouse Price Prediction Project Completed Successfully")