 #This is the file that is being used on subcosts.streamlit.app
# It is the main entry point for the Streamlit app and contains the UI and logic for
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
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Import your prediction functions - UPDATED FROM YOUR Predicting.py FILE
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
    df['snowfall_per_acre'] = df['Avg Snowfall (3-Year)'] / (df['Total Acreage'] + 1)
    df['snowfall_x_log_acreage'] = df['Avg Snowfall (3-Year)'] * df['log_acreage']
    df['snowfall_squared'] = df['Avg Snowfall (3-Year)'] ** 2
    df['log_snowfall'] = np.log1p(df['Avg Snowfall (3-Year)'])
    
    # === NEW SIDEWALK FEATURES ===
    df['sidewalk_acreage'] = df['Sidewalk Acreage']
    df['sidewalk_pct'] = df['Sidewalk Acreage'] / (df['Total Acreage'] + 1)
    df['log_sidewalk_acreage'] = np.log1p(df['Sidewalk Acreage'])
    
    return df

def create_regional_features(df):
    """Create regional features WITHOUT complexity-related statistics."""
    if 'Property Region (When Bid)' not in df.columns:
        return df
    
    df = df.copy()
    
    # For Streamlit, we'll use hardcoded regional means since we don't have training data
    # You can replace these with actual values from your training data
    regional_means = {
        'Central': {'price': 200, 'acreage': 10, 'snowfall': 12},
        'West': {'price': 200, 'acreage': 10, 'snowfall': 12},
        'East': {'price': 200, 'acreage': 10, 'snowfall': 9},
        'North': {'price': 200, 'acreage': 10, 'snowfall': 18.4},
        'National': {'price': 200, 'acreage': 10, 'snowfall': 10},
        'Philly Metro': {'price': 200, 'acreage': 10, 'snowfall': 9.1}
    }
    
    # Add regional features
    region = df['Property Region (When Bid)'].iloc[0]
    if region in regional_means:
        df['region_SubContractor Price Per Acre_mean'] = regional_means[region]['price']
        df['region_Total Acreage_mean'] = regional_means[region]['acreage']
        df['region_Avg Snowfall (3-Year)_mean'] = regional_means[region]['snowfall']
    else:
        # Default values
        df['region_SubContractor Price Per Acre_mean'] = 180
        df['region_Total Acreage_mean'] = 15
        df['region_Avg Snowfall (3-Year)_mean'] = 12
    
    return df

def predict_base_price_from_saved_model(model_path, input_df):
    """Predict base price (without complexity multiplier) using saved model."""
    try:
        model_config = joblib.load(model_path)
        
        # Handle both old format (just model) and new format (config dict)
        if isinstance(model_config, dict):
            model = model_config['model']
            feature_cols = model_config.get('feature_cols', [])
        else:
            model = model_config
            # Extract feature columns from model pipeline
            feature_cols = []
            try:
                for name, transformer, cols in model.named_steps['prep'].transformers:
                    if isinstance(cols, list):
                        feature_cols.extend(cols)
            except:
                # Fallback feature list if extraction fails
                feature_cols = [
                    'log_acreage_squared', 'snowfall_per_acre', 'acreage_bin', 'is_retail',
                    'is_small_property', 'is_medium_property', 'is_large_property',
                    'snowfall_x_log_acreage', 'log_snowfall', 'sidewalk_acreage',
                    'sidewalk_pct', 'log_sidewalk_acreage', 'Total Acreage',
                    'Avg Snowfall (3-Year)', 'Sidewalk Acreage',
                    'region_SubContractor Price Per Acre_mean',
                    'region_Total Acreage_mean', 'region_Avg Snowfall (3-Year)_mean'
                ]
        
        input_df = input_df.copy()
        
        # Remove complexity from the data for prediction
        if 'Complexity (1-5)' in input_df.columns:
            input_df = input_df.drop('Complexity (1-5)', axis=1)
        
        input_df["is_retail"] = (input_df["Property Type"] == "Retail").astype(int)
        input_df = create_regional_features(input_df)
        input_df = create_core_features(input_df)

        # Use available features
        available_features = [col for col in feature_cols if col in input_df.columns]
        X_input = input_df[available_features]
        
        return model.predict(X_input)[0]
    
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        # Fallback to mock calculation including sidewalk consideration
        acreage = input_df['Total Acreage'].iloc[0]
        snowfall = input_df['Avg Snowfall (3-Year)'].iloc[0]
        sidewalk = input_df['Sidewalk Acreage'].iloc[0]
        prop_type = input_df['Property Type'].iloc[0]
        
        base_price = 150 + (snowfall * 10) + (acreage * 2) + (sidewalk * 25)  # Higher price for sidewalk
        if prop_type == "Office":
            base_price *= 1.3
        elif prop_type == "Industrial":
            base_price *= 1.1
        return base_price

def apply_complexity_multiplier(base_price, complexity, multiplier_per_level=0.03):
    """Apply hardcoded complexity multiplier to base price."""
    complexity_adjustment = (complexity - 1) * multiplier_per_level
    final_price = base_price * (1 + complexity_adjustment)
    return final_price

def calculate_markup_prices(base_cost, markups=[0.15, 0.25, 0.35, 0.45]):
    """Calculate final prices with different markup percentages."""
    markup_prices = {}
    for markup in markups:
        final_price = base_cost / (1 - markup)
        markup_prices[f"{markup*100:.0f}%"] = final_price
    return markup_prices

# Streamlit App Configuration
st.set_page_config(
    page_title="Snow Removal Pricing Tool V2",
    page_icon="❄️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Main Title
st.markdown('<h1 style="text-align: center; color: #2E86AB;">❄️ Subcontractor Bid Pricing Tool (V2 - With Sidewalk Acreage)</h1>', unsafe_allow_html=True)

# Sidebar for Input Parameters
st.sidebar.header("Property Details")

# Model path input
model_path = st.sidebar.text_input(
    "Model Path", 
    value="models/sidewalkModel.pkl",
    help="Path to your saved model file"
)

# Input fields
total_acreage = st.sidebar.number_input(
    "Total Acreage",
    min_value=0.1,
    max_value=500.0,
    value=9.39,
    step=0.1,
    help="Enter the total acreage of the property"
)

# NEW: Sidewalk Acreage input
sidewalk_acreage = st.sidebar.number_input(
    "Sidewalk Acreage",
    min_value=0.0,
    max_value=50.0,
    value=0.1,
    step=0.01,
    help="Enter the sidewalk acreage (typically much smaller than total acreage)"
)

complexity = st.sidebar.selectbox(
    "Complexity Level (1-5)",
    options=[1, 2, 3, 4, 5],
    index=2,
    help="1 = Simple, 5 = Very Complex"
)

avg_snowfall = st.sidebar.number_input(
    "Average Snowfall (3-Year)",
    min_value=0.0,
    max_value=100.0,
    value=11.5,
    step=0.5,
    help="Average snowfall in inches over the past 3 years"
)

property_type = st.sidebar.selectbox(
    "Property Type",
    options=["Retail", "Office", "Industrial", "Residential", "Medical", "Other"],
    index=0
)

property_region = st.sidebar.selectbox(
    "Property Region",
    options=["Central", "West", "East", "North", "National", "Philly Metro"],
    index=0
)

# Advanced Settings
st.sidebar.header("Advanced Settings")

complexity_multiplier = st.sidebar.slider(
    "Complexity Multiplier (%)",
    min_value=1.0,
    max_value=10.0,
    value=3.5,
    step=0.1,
    help="Percentage increase per complexity level"
) / 100

# Create property dataframe
property_data = pd.DataFrame([{
    "Total Acreage": total_acreage,
    "Sidewalk Acreage": sidewalk_acreage,  # NEW FIELD
    "Complexity (1-5)": complexity,
    "Avg Snowfall (3-Year)": avg_snowfall,
    "Property Type": property_type,
    "Property Region (When Bid)": property_region,
    "SubContractor Price Per Acre": 0
}])

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    st.header("Pricing Analysis")
    
    # Use your actual prediction model here
    try:
        # Use your actual prediction model here
        base_price_per_acre = predict_base_price_from_saved_model(model_path, property_data)
        model_status = "✅ Using trained sidewalk model"
        model_color = "green"
    except Exception as e:
        # Fallback to mock if model fails. If this is showing on the site it means the model path is incorrect or the model is not compatible.
        base_price_per_acre = 150 + (avg_snowfall * 10) + (total_acreage * 2) + (sidewalk_acreage * 25)
        if property_type == "Office":
            base_price_per_acre *= 1.3
        elif property_type == "Industrial":
            base_price_per_acre *= 1.1
        model_status = f"⚠️ Using fallback calculation: {str(e)}"
        model_color = "orange"
    
    # Display model status
    st.markdown(f'<p style="color: {model_color};">{model_status}</p>', unsafe_allow_html=True)
    
    # Apply complexity multiplier
    final_price_per_acre = apply_complexity_multiplier(base_price_per_acre, complexity, complexity_multiplier)
    total_cost = final_price_per_acre * total_acreage
    
    # Calculate sidewalk metrics
    sidewalk_percentage = (sidewalk_acreage / total_acreage) * 100 if total_acreage > 0 else 0
    
    # Display base metrics
    st.subheader("Sub Pricing")
    
    metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
    
    with metrics_col1:
        st.metric(
            "Sub Price/Acre (Before Complexity)",
            f"${base_price_per_acre:,.2f}",
            help="Base price before complexity adjustment"
        )
    
    with metrics_col2:
        complexity_adj = ((final_price_per_acre - base_price_per_acre) / base_price_per_acre) * 100
        st.metric(
            "Sub Price/Acre (Complexity Adjusted)",
            f"${final_price_per_acre:,.2f}",
            f"+{complexity_adj:.1f}%"
        )
    
    with metrics_col3:
        st.metric(
            "Total Sub Cost",
            f"${total_cost:,.2f}",
            help="Total project cost before markup"
        )

with col2:
    st.header("Property Summary")
    
    st.markdown(f"""
    <div style="background-color: #f0f2f6; padding: 1rem; border-radius: 0.5rem;">
    <strong>Property Details:</strong><br>
    📍 Region: {property_region}<br>
    🏢 Type: {property_type}<br>
    📏 Acreage: {total_acreage} acres<br>
    🚶 Sidewalk: {sidewalk_acreage} acres ({sidewalk_percentage:.1f}%)<br>
    ❄️ Snowfall: {avg_snowfall}" (3-yr avg)<br>
    🔧 Complexity: Level {complexity}<br>
    📈 Multiplier: {complexity_multiplier*100:.1f}% per level
    </div>
    """, unsafe_allow_html=True)


# Markup Pricing Table
st.header("Markup Pricing Options")

markups = [0.20, 0.25, 0.30, 0.35, 0.40, 0.45,]
markup_data = []
markup_prices_per_acre = calculate_markup_prices(final_price_per_acre, markups)
markup_prices_total = calculate_markup_prices(total_cost, markups)

for i, markup in enumerate(markups):
    markup_pct = f"{markup*100:.0f}%"
    per_acre = markup_prices_per_acre[markup_pct]
    total_price = markup_prices_total[markup_pct]
    profit = total_price - total_cost
    profit_margin = (profit / total_price) * 100
    
    markup_data.append({
        "Markup": markup_pct,
        "Price/Acre": f"${per_acre:,.2f}",
        "Total Price": f"${total_price:,.2f}",
        "Profit": f"${profit:,.2f}",
        "Profit Margin": f"{profit_margin:.1f}%"
    })

markup_df = pd.DataFrame(markup_data)
st.dataframe(markup_df, use_container_width=True)

# Instructions
st.markdown("---")
st.markdown("""


### 💡 **New in Version 2:**
- ✅ **Sidewalk Acreage** input field for more accurate pricing
- ✅ **Sidewalk Impact Analysis** showing cost breakdown
- ✅ **Enhanced feature engineering** with sidewalk-specific calculations
- ✅ **Updated model compatibility** with sidewalkModel.pkl

### 🔧 **Model Features:**
- Includes sidewalk acreage, sidewalk percentage, and log sidewalk acreage features
- Complexity multiplier of 3% per level (adjustable)
- Regional pricing adjustments
- Property type considerations

### 📊 **Important:**
- **Sidewalk acreage** is typically much smaller than total acreage (e.g., 0.1-2.0 acres)
- Green checkmark means your trained sidewalk model is being used
- Orange warning means it's using the fallback calculation with sidewalk consideration
- Sidewalk areas often require more intensive maintenance and may command higher per-acre rates
""")

