import os
import numpy as np
import pandas as pd

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

# --- Generate Missing Data (for testing) ---
np.random.seed(42)
df_with_missing = df_encoded.copy()

# use column names (safer than positional indices) and exclude the target
valid_cols = [col for col in df_with_missing.columns if col != 'Diagnosis']

n_rows, n_cols = df_with_missing.shape
total_values = n_rows * n_cols
n_missing = int(total_values * 0.10)

row_indices = np.random.randint(0, n_rows, n_missing)
col_choices = np.random.choice(valid_cols, n_missing, replace=True)

# inject NaNs using .loc (row index and column name)
for r, c in zip(row_indices, col_choices):
    df_with_missing.loc[r, c] = np.nan

print("Liczba wartości NaN:", df_with_missing.isna().sum().sum())

# --- Missing Data Imputation ---
from pycaret.classification import setup, get_config
from sklearn.impute import KNNImputer

# ensure later imputers receive numeric input where required (we coerce non-numeric to NaN when needed)
def impute_with_pycaret(df, numeric_method):
    print(f"\n--- Imputacja PyCaret ({numeric_method}) ---")
    print("Braki przed imputacją:", df.isna().sum().sum())

    s = setup(
        data=df,
        target='Diagnosis',
        imputation_type='simple',
        numeric_imputation=numeric_method,
        categorical_imputation='mode',
        session_id=42,
        verbose=False
    )

    pipeline = get_config('pipeline')
    X_no_target = df.drop(columns=['Diagnosis'])
    imputed_array = pipeline.transform(X_no_target)

    imputed_df = pd.DataFrame(imputed_array, columns=X_no_target.columns)
    imputed_df['Diagnosis'] = df['Diagnosis'].values

    print("Braki po imputacji:", imputed_df.isna().sum().sum())
    return imputed_df

# 1) PyCaret – imputacja średnią
data_pycaret_mean = impute_with_pycaret(df_with_missing, 'mean')

# 2) Prosta imputacja – wartość stała (-1)
print("\n--- Imputacja prosta: stała (-1) ---")
print("Braki przed imputacją:", df_with_missing.isna().sum().sum())
data_constant = df_with_missing.fillna(-1)
print("Braki po imputacji:", data_constant.isna().sum().sum())

# 3) Prosta imputacja – interpolacja liniowa
print("\n--- Imputacja prosta: interpolacja liniowa ---")
print("Braki przed imputacją:", df_with_missing.isna().sum().sum())
data_interpolated = df_with_missing.copy()
data_interpolated = data_interpolated.interpolate(method='linear', limit_direction='both')
print("Braki po imputacji:", data_interpolated.isna().sum().sum())

# 4) Zaawansowana imputacja – KNNImputer
print("\n--- Imputacja zaawansowana: KNNImputer ---")
print("Braki przed imputacją:", df_with_missing.isna().sum().sum())

# ensure numeric input for KNNImputer (coerce non-numeric -> NaN)
X = df_with_missing.drop(columns=['Diagnosis']).apply(pd.to_numeric, errors='coerce')
y = df_with_missing['Diagnosis']

imputer = KNNImputer(n_neighbors=5, weights='uniform')
X_imputed = imputer.fit_transform(X)

data_knn = pd.DataFrame(X_imputed, columns=X.columns)
data_knn['Diagnosis'] = y.values

print("Braki po imputacji:", data_knn.isna().sum().sum())

print("\n✅ Wszystkie 4 warianty danych utworzone:")
print("1. data_pycaret_mean     - PyCaret (średnia)")
print("2. data_constant         - Prosta (stała -1)")
print("3. data_interpolated     - Prosta (interpolacja)")
print("4. data_knn              - KNNImputer (zaawansowana)")



# --- Own implementation of Min-Max Scaling and Z-Score Scaling ---
def min_max_scale(df):
    df_scaled = df.copy()
    # scale only numeric columns and handle constant columns
    numeric_cols = df_scaled.select_dtypes(include=[np.number]).columns
    numeric_cols = [c for c in numeric_cols if c != 'Diagnosis']
    for col in numeric_cols:
        min_val = df_scaled[col].min()
        max_val = df_scaled[col].max()
        if pd.isna(min_val) or pd.isna(max_val) or max_val == min_val:
            # constant or all-NaN column -> set to 0 to avoid division by zero
            df_scaled[col] = 0.0
        else:
            df_scaled[col] = (df_scaled[col] - min_val) / (max_val - min_val)
    return df_scaled

def z_score_scale(df):
    df_scaled = df.copy()
    numeric_cols = df_scaled.select_dtypes(include=[np.number]).columns
    numeric_cols = [c for c in numeric_cols if c != 'Diagnosis']
    for col in numeric_cols:
        mean_val = df_scaled[col].mean()
        std_val = df_scaled[col].std()
        if pd.isna(std_val) or std_val == 0:
            df_scaled[col] = 0.0
        else:
            df_scaled[col] = (df_scaled[col] - mean_val) / std_val
    return df_scaled

datasets = [data_pycaret_mean, data_constant, data_interpolated, data_knn]
dataset_names = ['pycaret_mean', 'constant', 'interpolated', 'knn']

scaled_datasets = {}

for df_item, name in zip(datasets, dataset_names):
    scaled_datasets[f'{name}_minmax'] = min_max_scale(df_item)
    scaled_datasets[f'{name}_zscore'] = z_score_scale(df_item)

# --- Save scaled datasets to CSV ---
save_folder = "scaled_data"
os.makedirs(save_folder, exist_ok=True)

for key, df_item in scaled_datasets.items():
    df_item.to_csv(f"{save_folder}/{key}.csv", index=False)
