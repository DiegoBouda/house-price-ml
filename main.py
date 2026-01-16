from sklearn.datasets import fetch_california_housing
import pandas as pd

# Load dataset
housing = fetch_california_housing(as_frame=True)

# Features (inputs)
X = housing.data

# Target (what we predict)
y = housing.target

# Convert to DataFrame for easier viewing
df = pd.concat([X, y.rename("price")], axis=1)

print(df.head())
print(df.describe())