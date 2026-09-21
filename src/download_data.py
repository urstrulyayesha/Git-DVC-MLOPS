from sklearn.datasets import load_iris
import pandas as pd

# Load Iris dataset
iris = load_iris()

# Create DataFrame
df = pd.DataFrame(
    iris.data,
    columns=iris.feature_names
)

# Add target column
df["target"] = iris.target

# Save dataset
df.to_csv("data/iris.csv", index=False)

print("Dataset saved successfully.")
print(f"Dataset shape: {df.shape}")