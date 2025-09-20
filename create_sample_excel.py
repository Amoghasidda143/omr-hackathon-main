"""
Create sample Excel file for answer key upload demonstration.
"""

import pandas as pd

# Create sample data
data = {
    'Subject': ['Mathematics'] * 10 + ['Physics'] * 10 + ['Chemistry'] * 10 + ['Biology'] * 10 + ['General_Knowledge'] * 10,
    'Question': list(range(1, 51)),
    'Answer': ['A', 'B', 'C', 'D', 'A', 'B', 'C', 'D', 'A', 'B'] * 5
}

# Create DataFrame
df = pd.DataFrame(data)

# Save as Excel file
df.to_excel('sample_answer_key.xlsx', index=False)

print("Sample Excel file created: sample_answer_key.xlsx")
print("Columns:", df.columns.tolist())
print("Shape:", df.shape)
print("\nFirst 10 rows:")
print(df.head(10))
