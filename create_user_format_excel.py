"""
Create sample Excel file in the user's format for testing.
"""

import pandas as pd

# Create sample data in the user's format
data = {
    'Python': [
        '1 - a', '2 - c', '3 - c', '4 - c', '5 - c',
        '6 - a', '7 - c', '8 - c', '9 - b', '10 - c',
        '11 - a', '12 - a', '13 - d', '14 - a', '15 - b',
        '16 - a,b,c,d', '17 - c', '18 - d', '19 - a', '20 - b'
    ],
    'EDA': [
        '21 - a', '22 - d', '23 - b', '24 - a', '25 - c',
        '26 - b', '27 - a', '28 - b', '29 - d', '30 - c',
        '31 - c', '32 - a', '33 - b', '34 - c', '35 - a',
        '36 - b', '37 - d', '38 - b', '39 - a', '40 - b'
    ],
    'SQL': [
        '41 - c', '42 - c', '43 - c', '44 - b', '45 - b',
        '46 - a', '47 - c', '48 - b', '49 - d', '50 - a',
        '51 - c', '52 - b', '53 - c', '54 - c', '55 - a',
        '56 - b', '57 - b', '58 - a', '59 - a,b', '60 - b'
    ],
    'POWER BI': [
        '61 - b', '62 - c', '63 - a', '64 - b', '65 - c',
        '66 - b', '67 - b', '68 - c', '69 - c', '70 - b',
        '71 - b', '72 - b', '73 - d', '74 - b', '75 - a',
        '76 - b', '77 - b', '78 - b', '79 - b', '80 - b'
    ],
    'Statistics': [
        '81 - a', '82 - b', '83 - c', '84 - b', '85 - c',
        '86 - b', '87 - b', '88 - b', '89 - a', '90 - b',
        '91 - c', '92 - b', '93 - c', '94 - b', '95 - b',
        '96 - b', '97 - c', '98 - a', '99 - b', '100 - c'
    ]
}

# Create DataFrame
df = pd.DataFrame(data)

# Save as Excel file
df.to_excel('sample_answer_key_user_format.xlsx', index=False)

print("Sample Excel file created: sample_answer_key_user_format.xlsx")
print("Format: Subject columns with 'Question - Answer' format")
print("Columns:", df.columns.tolist())
print("Shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())
