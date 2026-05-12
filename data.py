import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.pipeline import Pipeline

# Configuration flag
print_steps = True

def run_full_ausnut_analysis(file_path):
    if print_steps: print("--- Starting Data Preprocessing Trace ---")
    
    # 1. Load Data
    try:
        df = pd.read_excel(file_path, sheet_name='All solids & liquids per 100 g', skiprows=2, engine='openpyxl')
        df = df.copy() 
        if print_steps: print(f"[STATUS] Step 1: Successfully loaded {len(df)} items.")
    except Exception as e:
        print(f"[ERROR] Critical failure loading file: {e}")
        return

    # 2. Extract Categories
    df['Category'] = df['Classification'].astype(str).str[:2]

    # 3. Define Features & Cleaning
    feature_cols = [
        'Protein \n(g)', 'Fat, total \n(g)', 
        'Available carbohydrate, with sugar alcohols \n(g)', 
        'Total dietary fibre \n(g)', 'Moisture (water) \n(g)', 
        'Nitrogen \n(g)', 'Sodium (Na) \n(mg)', 
        'Energy with dietary fibre, equated \n(kJ)'
    ]
    
    if print_steps: print("[STATUS] Step 3: Cleaning features and handling 'trace' values...")
    def clean_val(val):
        if isinstance(val, str):
            val = val.lower().strip()
            if 'trace' in val: return 0.0001
        try: return float(val)
        except: return np.nan

    for col in feature_cols:
        df[col] = df[col].apply(clean_val)

    # 4. Imputation & Engineering
    if print_steps: print("[STATUS] Step 4: Applying median imputation...")
    imputer = SimpleImputer(strategy='median')
    df[feature_cols] = imputer.fit_transform(df[feature_cols])

    if print_steps: print("[STATUS] Step 5: Calculating Energy-to-Water Ratio...")
    df['Energy_Water_Ratio'] = df['Energy with dietary fibre, equated \n(kJ)'] / (df['Moisture (water) \n(g)'] + 0.1)
    
    X_features = feature_cols + ['Energy_Water_Ratio']
    X = df[X_features]
    y = df['Category']
    derivations = df['Derivation']

    # --- YOUR REQUESTED SUMMARY BLOCK ---
    print("\n==========================================")
    print("PREPROCESSING COMPLETE")
    print(f"Total Samples: {len(df)}")
    print(f"Total Features (Dimensions): {len(X_features)}")
    print(f"Unique Categories: {y.nunique()}")
    print("==========================================\n")

    # 5. Data Splitting (Stratified)
    mask = y.isin(y.value_counts()[y.value_counts() >= 2].index)
    X, y, derivations = X[mask], y[mask], derivations[mask]
    X_train, X_test, y_train, y_test, _, deriv_test = train_test_split(
        X, y, derivations, test_size=0.20, stratify=y, random_state=42
    )

    # 6. Model 1: k-NN (Baseline)
    knn_pipe = Pipeline([('scaler', StandardScaler()), ('knn', KNeighborsClassifier(n_neighbors=5))])
    knn_pipe.fit(X_train, y_train)
    knn_preds = knn_pipe.predict(X_test)

    # 7. Model 2: Random Forest (Primary)
    rf_model = RandomForestClassifier(n_estimators=200, class_weight='balanced', oob_score=True, random_state=42)
    rf_pipe = Pipeline([('scaler', StandardScaler()), ('rf', rf_model)])
    rf_pipe.fit(X_train, y_train)
    rf_preds = rf_pipe.predict(X_test)

    # 8. Visualizations
    # KNN Confusion Matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(confusion_matrix(y_test, knn_preds), annot=True, fmt='d', cmap='Blues', 
                xticklabels=knn_pipe.classes_, yticklabels=knn_pipe.classes_)
    plt.title('Baseline: k-NN Confusion Matrix')
    plt.show()

    # Random Forest Confusion Matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(confusion_matrix(y_test, rf_preds), annot=True, fmt='d', cmap='Greens', 
                xticklabels=rf_pipe.classes_, yticklabels=rf_pipe.classes_)
    plt.title('Primary: Random Forest Confusion Matrix')
    plt.show()

    # Random Forest Learning Curve (OOB Error Rate)
    oob_errors = []
    tree_range = range(10, 201, 10)
    for i in tree_range:
        rf_test = RandomForestClassifier(n_estimators=i, oob_score=True, random_state=42, warm_start=True)
        rf_test.fit(X_train, y_train)
        oob_errors.append(1 - rf_test.oob_score_)

    plt.figure(figsize=(10, 6))
    plt.plot(tree_range, oob_errors, marker='o', color='red')
    plt.title('Random Forest Learning Curve (OOB Error Rate)')
    plt.xlabel('Number of Trees')
    plt.ylabel('OOB Error Rate')
    plt.grid(True)
    plt.show()

    # 9. Final Results Comparison
    print("--- Final Model Comparison ---")
    print(f"k-NN Overall Accuracy: {accuracy_score(y_test, knn_preds):.4f}")
    print(f"Random Forest Overall Accuracy: {accuracy_score(y_test, rf_preds):.4f}")
    
    # Audit for Random Forest
    results_df = pd.DataFrame({'Actual': y_test, 'Pred': rf_preds, 'Deriv': deriv_test})
    analysed_acc = accuracy_score(results_df[results_df['Deriv'] == 'Analysed']['Actual'], results_df[results_df['Deriv'] == 'Analysed']['Pred'])
    recipe_acc = accuracy_score(results_df[results_df['Deriv'] == 'Recipe']['Actual'], results_df[results_df['Deriv'] == 'Recipe']['Pred'])
    print(f"RF Analysed Accuracy: {analysed_acc:.4f}")
    print(f"RF Recipe Accuracy: {recipe_acc:.4f}")

# Main execution
file_name = "AFCD Release 3 - Nutrient profiles (2).xlsx"
run_full_ausnut_analysis(file_name)