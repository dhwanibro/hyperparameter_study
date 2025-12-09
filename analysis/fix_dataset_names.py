"""
Fix inconsistent dataset naming in the raw CSV file
Consolidates har_smartphone, smartphone, and smartphone_har into a single name
"""

import pandas as pd
from pathlib import Path

# Load the raw data
csv_path = Path('results/all_experiments_raw.csv')
df = pd.read_csv(csv_path)

print("Before fixing:")
print(df['dataset'].value_counts())
print()

# Standardize the smartphone HAR dataset names
# Replace all variations with 'smartphone_har'
df['dataset'] = df['dataset'].replace({
    'har_smartphone': 'smartphone_har',
    'smartphone': 'smartphone_har'
})

print("After fixing:")
print(df['dataset'].value_counts())
print()

# Save the fixed data
df.to_csv(csv_path, index=False)
print(f"✓ Fixed dataset names saved to {csv_path}")
print()
print("Now re-run analyze_all_results.py to regenerate all visualizations with corrected data!")
