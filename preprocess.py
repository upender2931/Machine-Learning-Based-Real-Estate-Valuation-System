import pandas as pd
import pickle

print("Loading dataset...")

df = pd.read_csv("real_estate.csv", sep="\t")
df.columns = df.columns.str.strip()

# Separate features and target
X = df.drop("Price", axis=1)
y = df["Price"]

# Save cleaned dataset
pickle.dump(X, open("X_raw.pkl", "wb"))
pickle.dump(y, open("y_raw.pkl", "wb"))

print("Preprocessing completed successfully.")
