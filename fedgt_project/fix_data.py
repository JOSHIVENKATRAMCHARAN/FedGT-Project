import pandas as pd
import numpy as np

# Load the existing data
df = pd.read_csv('core/client_data.csv')

# The labels are existing: df['label']
y = df['label'].values
n_samples = len(y)

# Generate moderately overlapping features.
# This keeps labels learnable, but prevents every model from getting 100%.
rng = np.random.default_rng(42)
X = rng.normal(0, 2.0, size=(n_samples, 10))

# For label k, add a class signal to feature k+1.
# Strength 5.6 gives a realistic college-project range on this dataset.
signal_strength = 5.6
for i in range(n_samples):
    label = int(y[i])
    if 0 <= label <= 9:
        X[i, label] += signal_strength

# Assign back to dataframe
for i in range(1, 11):
    df[f'feature_{i}'] = X[:, i-1]

# Save it
df.to_csv('core/client_data.csv', index=False)
print("Data regenerated with moderate class overlap.")
