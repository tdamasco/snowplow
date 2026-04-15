#THIS IS THE BEST MODEL AS OF JUNE 26 


import os
import pandas as pd

import matplotlib.pyplot as plt
import numpy as np
import warnings
warnings.filterwarnings('ignore')

import joblib
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.impute import SimpleImputer
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

print("Working directory:", os.getcwd())
data_path = "data/NewFull.csv"  # <- Updated to your new file

def load_and_clean(path):
    """Load and perform initial cleaning of the data."""
    df = pd.read_csv(path, encoding='utf-8-sig')
    
    # Convert target to float
    df["SubContractor Price Per Acre"] = (
        df["SubContractor Price Per Acre"]
        .astype(str)
        .str.replace(r"[\$,]", "", regex=True)
        .astype(float)
    )
    
    # Create binary property type feature
    df["is_retail"] = (df["Property Type"] == "Retail").astype(int)
    
    # --- NEW: parse Sidewalk Acreage ---
    df["Sidewalk Acreage"] = pd.to_numeric(df.get("Sidewalk Acreage", 0), errors="coerce").fillna(0)
    
    # Clean column names
    df.columns = df.columns.str.strip()
    df.columns = df.columns.str.replace(r"\s+", " ", regex=True)
    
    # Handle infinite values
    df = df.replace([np.inf, -np.inf], np.nan)
    
    # Handle remaining NaN values
    if df.isnull().any().any():
        print(f"Warning: Found {df.isnull().sum().sum()} NaN values after feature engineering")
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    
    return df

def handle_outliers(df, target_col='SubContractor Price Per Acre', method='cap', iqr_multiplier=2.0):
    """Handle outliers using IQR method."""
    Q1 = df[target_col].quantile(0.25)
    Q3 = df[target_col].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - iqr_multiplier * IQR
    upper_bound = Q3 + iqr_multiplier * IQR
    
    print(f"Outlier bounds with {iqr_multiplier}x IQR: [{lower_bound:.2f}, {upper_bound:.2f}]")
    outliers = df[(df[target_col] < lower_bound) | (df[target_col] > upper_bound)]
    print(f"Found {len(outliers)} outliers out of {len(df)} records")
    
    if method == 'cap':
        df[target_col] = df[target_col].clip(lower=lower_bound, upper=upper_bound)
        print("Capped outliers at bounds")
    
    return df

def create_core_features(df):
    """Create core features INCLUDING sidewalk acreage features."""
    df = df.copy()
    
    # === SIZE FEATURES ===
    df['log_acreage'] = np.log1p(df['Total Acreage'])
    df['sqrt_acreage'] = np.sqrt(df['Total Acreage'])
    df['log_acreage_squared'] = df['log_acreage'] ** 2
    
    # Size categories
    df['acreage_bin'] = pd.qcut(df['Total Acreage'], q=5, labels=False, duplicates='drop')
    acreage_33 = df['Total Acreage'].quantile(0.33)
    acreage_67 = df['Total Acreage'].quantile(0.67)
    df['is_small_property'] = (df['Total Acreage'] < acreage_33).astype(int)
    df['is_medium_property'] = ((df['Total Acreage'] >= acreage_33) & (df['Total Acreage'] < acreage_67)).astype(int)
    df['is_large_property'] = (df['Total Acreage'] > acreage_67).astype(int)
    
    # === WEATHER FEATURES ===
    df['snowfall_per_acre']      = df['Avg Snowfall (3-Year)'] / (df['Total Acreage'] + 1)
    df['snowfall_x_log_acreage'] = df['Avg Snowfall (3-Year)'] * df['log_acreage']
    df['snowfall_squared']       = df['Avg Snowfall (3-Year)'] ** 2
    df['log_snowfall']           = np.log1p(df['Avg Snowfall (3-Year)'])
    
    # === NEW SIDEWALK FEATURES ===
    df['sidewalk_acreage']       = df['Sidewalk Acreage']
    df['sidewalk_pct']           = df['Sidewalk Acreage'] / (df['Total Acreage'] + 1)
    df['log_sidewalk_acreage']   = np.log1p(df['Sidewalk Acreage'])
    
    total_new = len([c for c in df.columns if c not in [
        'Customer','Property','Address','City','State','Zip'
    ]])
    print(f"Created {total_new} features (including sidewalk features)")
    return df

def create_regional_features(df):
    """Create simplified regional features."""
    if 'Property Region (When Bid)' not in df.columns:
        return df
    
    df = df.copy()
    regional_stats = df.groupby('Property Region (When Bid)').agg({
        'SubContractor Price Per Acre': 'mean',
        'Total Acreage':               'mean',
        'Avg Snowfall (3-Year)':       'mean'
    })
    regional_stats.columns = [f'region_{col}_mean' for col in regional_stats.columns]
    
    df = df.merge(regional_stats, left_on='Property Region (When Bid)',
                  right_index=True, how='left')
    for col in regional_stats.columns:
        original = col.replace('region_','').replace('_mean','')
        df[col] = df[col].fillna(df[original].mean())
    return df

def build_simple_preprocessor(df):
    """Build a simple, memory-efficient preprocessor including sidewalk features."""
    core_features = [
        'log_acreage_squared',
        'snowfall_per_acre',
        'acreage_bin',
        'is_retail',
        'is_small_property',
        'is_medium_property',
        'is_large_property',
        'snowfall_x_log_acreage',
        'log_snowfall',
        'sidewalk_acreage',       # <<< new
        'sidewalk_pct',           # <<< new
        'log_sidewalk_acreage'    # <<< new
    ]
    regional_features = [
        'region_SubContractor Price Per Acre_mean',
        'region_Total Acreage_mean',
        'region_Avg Snowfall (3-Year)_mean'
    ]
    all_features   = core_features + regional_features
    numeric_feats  = [f for f in all_features if f in df.columns]
    log_feats      = ['Total Acreage', 'Avg Snowfall (3-Year)', 'Sidewalk Acreage']
    
    print(f"Using {len(numeric_feats)} core+sidewalk features")
    
    num_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scale',   StandardScaler())
    ])
    log_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('log',     FunctionTransformer(np.log1p, validate=False)),
        ('scale',   StandardScaler())
    ])
    transformers = [
        ('num', num_pipe, numeric_feats),
        ('log', log_pipe, log_feats)
    ]
    return ColumnTransformer(transformers, remainder='drop')

def create_simple_ensemble(preprocessor):
    """Create a simple, memory-efficient ensemble."""
    base_models = [
        ('xgb', XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.1,
                             subsample=0.8, colsample_bytree=0.8,
                             random_state=42, n_jobs=1)),
        ('lgb', LGBMRegressor(n_estimators=100, max_depth=3, learning_rate=0.1,
                              subsample=0.8, colsample_bytree=0.8,
                              random_state=42, verbosity=-1, n_jobs=1)),
        ('rf',  RandomForestRegressor(n_estimators=50, max_depth=4,
                                      min_samples_split=10, min_samples_leaf=5,
                                      random_state=42, n_jobs=1))
    ]
    stacking = StackingRegressor(
        estimators=base_models,
        final_estimator=Ridge(alpha=1.0),
        cv=3, passthrough=False, n_jobs=1
    )
    return Pipeline([('prep', preprocessor), ('stack', stacking)])

def predict_base_price(model, input_row, feature_cols):
    """Generate base prediction (without complexity multiplier)."""
    df = input_row.copy()
    if 'Complexity (1-5)' in df.columns:
        df = df.drop('Complexity (1-5)', axis=1)
    df["is_retail"] = (df["Property Type"] == "Retail").astype(int)
    df = create_regional_features(df)
    df = create_core_features(df)
    return model.predict(df[feature_cols])[0]

def apply_complexity_multiplier(base_price, complexity, multiplier_per_level=0.03):
    adj = (complexity - 1) * multiplier_per_level
    return base_price * (1 + adj)

def predict_final_price(model, input_row, feature_cols,
                        multiplier_per_level=0.03, verbose=False):
    complexity = int(input_row['Complexity (1-5)'].iloc[0])
    base = predict_base_price(model, input_row, feature_cols)
    final = apply_complexity_multiplier(base, complexity, multiplier_per_level)
    if verbose:
        print(f"Base: ${base:,.2f}, Complexity {complexity}, Final: ${final:,.2f}")
    return final

def generate_complexity_pricing_table(model, sample_property, feature_cols, multiplier_per_level=0.03):
    """Generate a table showing how price increases with complexity using the multiplier."""
    print("\n" + "="*70)
    print("COMPLEXITY PRICING TABLE (Hardcoded 3% per Level)")
    print("="*70)
    
    # Get base price (complexity 1 equivalent)
    base_price = predict_base_price(model, sample_property, feature_cols)
    
    print(f"{'Complexity':<12} {'Price/Acre':<14} {'$ Increase':<14} {'% Increase':<14}")
    print("-" * 70)
    
    results = []
    for complexity in [1, 2, 3, 4, 5]:
        final_price = apply_complexity_multiplier(base_price, complexity, multiplier_per_level)
        
        if complexity == 1:
            increase_amount = 0
            increase_percent = 0
            increase_str = "-"
            percent_str = "-"
        else:
            previous_price = apply_complexity_multiplier(base_price, complexity-1, multiplier_per_level)
            increase_amount = final_price - previous_price
            increase_percent = (increase_amount / previous_price) * 100
            increase_str = f"${increase_amount:,.0f}"
            percent_str = f"{increase_percent:.1f}%"
        
        print(f"{complexity:<12} ${final_price:,.0f}      {increase_str:<14} {percent_str:<14}")
        
        results.append({
            'Complexity': complexity,
            'Price_per_Acre': final_price,
            'Increase_Amount': increase_amount,
            'Increase_Percent': increase_percent
        })
    
    print(f"\nBase price (before complexity): ${base_price:,.2f}")
    print(f"Multiplier per complexity level: {multiplier_per_level:.1%}")
    return results

def evaluate_model_with_complexity(model, df, feature_cols, multiplier_per_level=0.03, sample_size=100):
    """Evaluate the model performance when complexity multiplier is applied."""
    print("\n" + "="*70)
    print("MODEL EVALUATION WITH COMPLEXITY MULTIPLIER")
    print("="*70)
    
    # Use a sample to avoid memory issues
    sample_df = df.sample(n=min(sample_size, len(df)), random_state=42)
    
    predictions = []
    actuals = []
    
    for idx, row in sample_df.iterrows():
        row_df = pd.DataFrame([row])
        
        try:
            # Get prediction with complexity multiplier
            pred = predict_final_price(model, row_df, feature_cols, multiplier_per_level)
            actual = row['SubContractor Price Per Acre']
            
            predictions.append(pred)
            actuals.append(actual)
        except Exception as e:
            print(f"Error predicting row {idx}: {e}")
            continue
    
    if len(predictions) == 0:
        print("❌ No successful predictions made")
        return None, None, None
    
    predictions = np.array(predictions)
    actuals = np.array(actuals)
    
    # Calculate metrics
    rmse = root_mean_squared_error(actuals, predictions)
    mae = mean_absolute_error(actuals, predictions)
    r2 = r2_score(actuals, predictions)
    
    print(f"Model Performance WITH Complexity Multiplier (sample of {len(predictions)}):")
    print(f"  RMSE: ${rmse:.0f}")
    print(f"  MAE:  ${mae:.0f}")
    print(f"  R²:   {r2:.3f}")
    print(f"  Complexity Multiplier: {multiplier_per_level:.1%} per level")
    
    return rmse, mae, r2

def plot_simple_diagnostics(model, X_test, y_test, model_name="Model"):
    """Plot simplified diagnostic charts to save memory."""
    y_pred = model.predict(X_test)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # 1. Predicted vs Actual
    ax1 = axes[0]
    ax1.scatter(y_test, y_pred, alpha=0.6)
    ax1.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    ax1.set_xlabel('Actual Price per Acre')
    ax1.set_ylabel('Predicted Price per Acre')
    ax1.set_title(f'{model_name}: Predicted vs Actual')
    
    r2 = r2_score(y_test, y_pred)
    ax1.text(0.05, 0.95, f'R² = {r2:.3f}', transform=ax1.transAxes, 
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # 2. Residuals
    ax2 = axes[1]
    residuals = y_test - y_pred
    ax2.scatter(y_pred, residuals, alpha=0.6)
    ax2.axhline(y=0, color='r', linestyle='--')
    ax2.set_xlabel('Predicted Price per Acre')
    ax2.set_ylabel('Residuals')
    ax2.set_title('Residual Plot')
    
    plt.tight_layout()
    plt.show()
    
    # Print metrics
    rmse = root_mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    print(f"\n{model_name} Performance:")
    print(f"  RMSE: ${rmse:.0f}")
    print(f"  MAE:  ${mae:.0f}")
    print(f"  R²:   {r2:.3f}")

def main():
    print("="*60)
    print("MEMORY-EFFICIENT SNOW PLOWING MODEL WITH COMPLEXITY MULTIPLIER")
    print("="*60)

    # Load and prepare data
    df = load_and_clean(data_path)
    print(f"Loaded {len(df)} records")

    df = handle_outliers(df, method='cap', iqr_multiplier=2.0)
    df = create_regional_features(df)
    df = create_core_features(df)

    # Feature selection - exclude complexity AND non-predictive columns
    feature_cols = [col for col in df.columns if col not in [
        'SubContractor Price Per Acre', 'Customer', 'Property', 'Address', 'City', 'State', 'Zip',
        'HQ Region', 'Property Type', 'Property Region (When Bid)', 'log_acreage',
        'Complexity (1-5)',  # EXCLUDE COMPLEXITY FROM TRAINING
        'complexity_squared', 'complexity_x_log_acreage', 'complexity_x_snowfall',
        'complexity_snowfall_ratio', 'acreage_complexity_ratio', 
        'small_x_complexity', 'large_x_complexity'  # Remove any complexity features
    ]]

    X = df[feature_cols]
    y = df['SubContractor Price Per Acre']

    print(f"Using {X.shape[1]} features for {X.shape[0]} samples (NO COMPLEXITY FEATURES)")
    print(f"Sample-to-feature ratio: {X.shape[0]/X.shape[1]:.1f}:1")

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create simple model (no hyperparameter tuning to save memory)
    print("\nCREATING SIMPLE STACKING MODEL (NO HYPERPARAMETER TUNING)...")
    preprocessor = build_simple_preprocessor(df)
    model = create_simple_ensemble(preprocessor)
    
    print("Training model...")
    model.fit(X_train, y_train)

    # Evaluate on test set (base model without complexity)
    y_pred = model.predict(X_test)
    rmse = root_mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"\nBase Model (without complexity) RMSE: ${rmse:.0f}, R²: {r2:.3f}")

    plot_simple_diagnostics(model, X_test, y_test, "Base Model (No Complexity)")

    # Evaluate model WITH complexity multiplier on sample
    evaluate_model_with_complexity(model, df, feature_cols, multiplier_per_level=0.03, sample_size=50)

    # === Example Predictions ===
    print("\n" + "="*70)
    print("EXAMPLE PREDICTIONS WITH COMPLEXITY MULTIPLIER")
    print("="*70)
    
    example_property = pd.DataFrame([{
        "Total Acreage": 7.72,
        "Complexity (1-5)": 3,
        "Avg Snowfall (3-Year)": 9.1,
        "Property Type": "Office",
        "Property Region (When Bid)": "Philly Metro",
        "SubContractor Price Per Acre": 0,  # Placeholder
        "Sidewalk Acreage": 0.1
    }])

    # Base prediction (complexity 1 equivalent)
    base_bid = predict_base_price(model, example_property, feature_cols)
    
    # Final prediction with complexity multiplier
    final_bid = predict_final_price(model, example_property, feature_cols, multiplier_per_level=0.03, verbose=True)

    print(f"\nExample Property Details:")
    for col in ['Total Acreage', 'Complexity (1-5)', 'Avg Snowfall (3-Year)', 'Property Type']:
        if col in example_property.columns:
            print(f"  {col}: {example_property[col].iloc[0]}")

    # Generate complexity comparison table
    generate_complexity_pricing_table(model, example_property, feature_cols, multiplier_per_level=0.03)
    
    # === Demonstrate Different Complexity Levels ===
    print("\n" + "="*70)
    print("COMPLEXITY LEVEL DEMONSTRATION")
    print("="*70)
    
    demo_complexities = [1, 2, 3, 4, 5]
    print(f"{'Complexity':<12} {'Final Price':<14} {'vs Complexity 1':<16}")
    print("-" * 50)
    
    for complexity in demo_complexities:
        test_prop = example_property.copy()
        test_prop['Complexity (1-5)'] = complexity
        
        final_price = predict_final_price(model, test_prop, feature_cols, multiplier_per_level=0.03)
        
        if complexity == 1:
            base_for_comparison = final_price
            vs_base = "-"
        else:
            increase = final_price - base_for_comparison
            vs_base = f"+${increase:,.0f}"
        
        print(f"{complexity:<12} ${final_price:,.0f}      {vs_base:<16}")
    
    # Save the trained model and configuration
    print("\nSaving model...")
    joblib.dump(model, "models/sidewalkModel.pkl")
    
    # Save configuration
    model_config = {
        'model': model,
        'feature_cols': feature_cols,
        'multiplier_per_level': 0.03,
        'predict_base_function': predict_base_price,
        'predict_final_function': predict_final_price
    }
    
    joblib.dump(model_config, "models/sidewalkModel.pkl")
    
    print("✅ Model saved to models/sidewalkModel.pkl")
    print("✅ Configuration saved to models/sidewalkMdel.pkl")
    
    print("\n" + "="*70)
    print("USAGE INSTRUCTIONS")
    print("="*70)
    print("For final predictions (with complexity multiplier):")
    print("  price = predict_final_price(model, property_data, feature_cols)")
    print("\nFor base predictions (complexity 1 equivalent):")
    print("  price = predict_base_price(model, property_data, feature_cols)")
    print("\nComplexity multiplier: 3% per level above 1")
    print("  Complexity 1: Base price × 1.00")
    print("  Complexity 2: Base price × 1.03")
    print("  Complexity 3: Base price × 1.06")
    print("  Complexity 4: Base price × 1.09")
    print("  Complexity 5: Base price × 1.12")
    
    return model, feature_cols

if __name__ == "__main__":
    model, feature_cols = main()
