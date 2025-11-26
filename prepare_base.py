# prepases base dataset with no missing and imputed values
import pandas as pd
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('alzheimers_disease_data.csv')
df = df.drop(columns=['PatientID', 'DoctorInCharge'])

# --- Encode Ethnicity ---
mapping = {
    0: 'Caucasian',
    1: 'African American',
    2: 'Asian',
    3: 'Other'
}

df['Ethnicity'] = df['Ethnicity'].map(mapping)
df_encoded = pd.get_dummies(df, columns=['Ethnicity'], dtype=int)

# Move Ethnicity_* columns so they appear immediately before 'Diagnosis'
eth_cols = [c for c in df_encoded.columns if c.startswith('Ethnicity_')]
if 'Diagnosis' in df_encoded.columns and eth_cols:
    other_cols = [c for c in df_encoded.columns if c not in (eth_cols + ['Diagnosis'])]
    new_order = other_cols + eth_cols + ['Diagnosis']
    df_encoded = df_encoded[new_order]

# --- Scale numerical features ---
continuous_features = [
    'Age', 'BMI', 'AlcoholConsumption', 'PhysicalActivity',
    'DietQuality', 'SleepQuality', 'SystolicBP', 'DiastolicBP',
    'CholesterolTotal', 'CholesterolLDL', 'CholesterolHDL', 'CholesterolTriglycerides',
    'MMSE', 'FunctionalAssessment', 'ADL',
]

scaler = StandardScaler()
df_encoded[continuous_features] = scaler.fit_transform(df_encoded[continuous_features])

df_encoded.to_csv('scaled_data/base.csv', index=False)
