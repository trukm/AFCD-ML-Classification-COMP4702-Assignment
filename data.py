import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# Configuration flag
print_steps = True

def process_ausnut_excel(file_path):
    if print_steps: print("--- Starting Data Preprocessing Trace ---")
    
    try:
        # 1. Load and immediately defragment
        df = pd.read_excel(file_path, sheet_name='All solids & liquids per 100 g', skiprows=2, engine='openpyxl')
        df = df.copy() 
        if print_steps: print(f"[STATUS] Step 1: Successfully loaded {len(df)} items.")
    except Exception as e:
        print(f"[ERROR] Critical failure loading file: {e}")
        return None, None

    # 2. Extract Categories
    df['Category'] = df['Classification'].astype(str).str[:2]
    if print_steps:
        print("\n[ANALYSIS] Step 2: Category Distribution (Imbalance Matrix):")
        print(df['Category'].value_counts().sort_index())

    # 3. Define Features
    feature_cols = [
        'Protein \n(g)', 'Fat, total \n(g)', 
        'Available carbohydrate, with sugar alcohols \n(g)', 
        'Total dietary fibre \n(g)', 'Moisture (water) \n(g)', 
        'Nitrogen \n(g)', 'Sodium (Na) \n(mg)', 
        'Energy with dietary fibre, equated \n(kJ)'
    ]
    
    # 4. Feature Cleaning
    if print_steps: print("\n[STATUS] Step 4: Cleaning features and handling 'trace' values...")
    X = df[feature_cols].copy()

    def clean_nutrient_values(val):
        if isinstance(val, str):
            val = val.lower().strip()
            if 'trace' in val: return 0.0001
        try:
            return float(val)
        except (ValueError, TypeError):
            return np.nan

    for col in feature_cols:
        X[col] = X[col].apply(clean_nutrient_values)

    # 5. Imputation
    if print_steps: print("[STATUS] Step 5: Applying median imputation...")
    imputer = SimpleImputer(strategy='median')
    X_imputed = imputer.fit_transform(X)

    # 6. Scaling
    if print_steps: print("[STATUS] Step 6: Normalizing features with StandardScaler...")
    scaler = StandardScaler()
    X_final = scaler.fit_transform(X_imputed)

    if print_steps: print("\n--- Preprocessing Trace Complete ---")
    
    # Final Output Summary (Always prints)
    print("\n==========================================")
    print("PREPROCESSING COMPLETE")
    print(f"Total Samples: {X_final.shape[0]}")
    print(f"Total Features: {X_final.shape[1]}")
    print(f"Unique Categories: {df['Category'].nunique()}")
    print("==========================================\n")
    
    return X_final, df['Category']

# Main execution
file_name = "AFCD Release 3 - Nutrient profiles (2).xlsx"
X, y = process_ausnut_excel(file_name)