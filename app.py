#This is the file that is being used on subcosts.streamlit.app
# It is the main entry point for the Streamlit app and contains the UI and logic for
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import warnings
warnings.filterwarnings('ignore')
import pgeocode

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

def calculate_markup_prices(base_cost, markups=[0.15, 0.25, 0.35, 0.45]):
    """Calculate final prices with different markup percentages."""
    markup_prices = {}
    for markup in markups:
        final_price = base_cost / (1 - markup)
        markup_prices[f"{markup*100:.0f}%"] = final_price
    return markup_prices

# NEW FUNCTIONS FOR MARKET COMPARISON
def load_training_dataset(dataset_path="SubFull.csv"):
    """Load the training dataset containing all properties under contract."""
    try:
        df = pd.read_csv(dataset_path)
        return df, True
    except Exception as e:
        st.error(f"Could not load training dataset: {str(e)}")
        return None, False

def get_similar_properties(df, zip_code, property_type=None, total_acreage=None, max_results=10, radius_miles=25):
    """
    Filter properties by nearby zip codes, similar acreage (±35%), and optionally by property type.
    Uses pgeocode to find properties within a specified radius.
    """
    if df is None:
        return pd.DataFrame()
    
    # Convert zip_code to string and handle different formats
    zip_code = str(zip_code).strip()
    
    # Ensure required columns exist
    if 'Zip' not in df.columns:
        st.warning("'Zip' column not found in dataset.")
        return pd.DataFrame()
    
    # Debug: Show what acreage we're using for matching
    if total_acreage is not None:
        st.info(f"🔍 Searching for properties with acreage between {total_acreage * 0.65:.1f} - {total_acreage * 1.35:.1f} acres (±35% of {total_acreage} acres)")
    else:
        st.info("🔍 Searching for properties without acreage filtering")
    
    try:
        # Initialize pgeocode for US
        nomi = pgeocode.Nominatim('us')
        dist_calc = pgeocode.GeoDistance('us')
        
        # Get coordinates for the target zip code
        target_location = nomi.query_postal_code(zip_code)
        
        if pd.isna(target_location.latitude) or pd.isna(target_location.longitude):
            st.warning(f"Could not find coordinates for zip code {zip_code}. Falling back to exact match.")
            return get_similar_properties_exact_zip(df, zip_code, property_type, max_results)
        
        df_copy = df.copy()
        df_copy['Zip'] = df_copy['Zip'].astype(str).str.strip()
        
        # Remove rows with missing required data
        df_copy = df_copy.dropna(subset=['Zip'])
        
        # Only filter by acreage if total_acreage is provided and acreage filtering is enabled
        if total_acreage is not None:
            df_copy = df_copy.dropna(subset=['Total Acreage'])
            
            # Calculate acreage similarity bounds (±35%)
            acreage_lower = total_acreage * 0.65  # 35% below
            acreage_upper = total_acreage * 1.35  # 35% above
            
            # Filter by acreage similarity first (faster)
            acreage_filtered = df_copy[
                (df_copy['Total Acreage'] >= acreage_lower) & 
                (df_copy['Total Acreage'] <= acreage_upper)
            ].copy()
            
            if acreage_filtered.empty:
                st.info(f"No properties found with similar acreage ({acreage_lower:.1f} - {acreage_upper:.1f} acres) within the dataset.")
                return pd.DataFrame()
        else:
            # No acreage filtering
            acreage_filtered = df_copy.copy()
        
        # Calculate distances for each property
        distances = []
        valid_indices = []
        
        for idx, row in acreage_filtered.iterrows():
            try:
                property_zip = str(row['Zip']).strip()
                if property_zip and property_zip != 'nan':
                    distance = dist_calc.query_postal_code(zip_code, property_zip)
                    if not pd.isna(distance):
                        # Convert km to miles
                        distance_miles = distance * 0.621371
                        if distance_miles <= radius_miles:
                            distances.append(distance_miles)
                            valid_indices.append(idx)
            except Exception as e:
                # Skip properties with invalid zip codes
                continue
        
        if not valid_indices:
            if total_acreage is not None:
                st.info(f"No properties found within {radius_miles} miles with similar acreage ({total_acreage * 0.65:.1f} - {total_acreage * 1.35:.1f} acres).")
            else:
                st.info(f"No properties found within {radius_miles} miles of zip code {zip_code}.")
            return pd.DataFrame()
        
        # Create filtered dataframe with distances
        nearby_filtered = acreage_filtered.loc[valid_indices].copy()
        nearby_filtered['Distance_Miles'] = distances
        
        # Optionally filter by property type
        if property_type and property_type != "Any":
            if 'Property Type' in nearby_filtered.columns:
                nearby_filtered = nearby_filtered[nearby_filtered['Property Type'] == property_type]
        
        if nearby_filtered.empty:
            st.info("No properties found matching all criteria.")
            return pd.DataFrame()
        
        # Sort by distance first, then by price for consistency
        if 'SubContractor Price Per Acre' in nearby_filtered.columns:
            nearby_filtered = nearby_filtered.sort_values(['Distance_Miles', 'SubContractor Price Per Acre'])
        else:
            nearby_filtered = nearby_filtered.sort_values('Distance_Miles')
        
        return nearby_filtered.head(max_results)
        
    except ImportError:
        st.error("pgeocode library not found. Please install it with: pip install pgeocode")
        st.info("Falling back to exact zip code matching...")
        return get_similar_properties_exact_zip(df, zip_code, property_type, max_results)
    
    except Exception as e:
        st.warning(f"Error with geographic search: {str(e)}. Falling back to exact zip code matching.")
        return get_similar_properties_exact_zip(df, zip_code, property_type, max_results)


def get_similar_properties_exact_zip(df, zip_code, property_type=None, max_results=10):
    """
    Fallback function for exact zip code matching (original functionality).
    """
    if df is None:
        return pd.DataFrame()
    
    # Convert zip_code to string and handle different formats
    zip_code = str(zip_code).strip()
    
    # Ensure Zip Code column exists and convert to string
    if 'Zip' not in df.columns:
        st.warning("'Zip' column not found in dataset.")
        return pd.DataFrame()
    
    df_copy = df.copy()
    df_copy['Zip'] = df_copy['Zip'].astype(str).str.strip()
    
    # Filter by zip code
    filtered_df = df_copy[df_copy['Zip'] == zip_code].copy()
    
    # Optionally filter by property type
    if property_type and property_type != "Any":
        if 'Property Type' in filtered_df.columns:
            filtered_df = filtered_df[filtered_df['Property Type'] == property_type]
    
    # Sort by total acreage for consistency
    if 'SubContractor Price Per Acre' in filtered_df.columns:
        filtered_df = filtered_df.sort_values('SubContractor Price Per Acre')
    
    return filtered_df.head(max_results)

def display_similar_properties_table(similar_props_df):
    """Display similar properties in a clean table format with distance information."""
    if similar_props_df.empty:
        st.info("No similar properties found with the current criteria.")
        return
    
    # Select key columns for display
    display_columns = [
        'Customer', 'SubContractor', 'Property', 'Total Acreage', 'Sidewalk Acreage', 
        'SubContractor Price Per Acre', 'Complexity (1-5)', 'Zip'
    ]
    
    # Add distance if available
    if 'Distance_Miles' in similar_props_df.columns:
        display_columns.append('Distance_Miles')
    
    # Filter to available columns
    available_columns = [col for col in display_columns if col in similar_props_df.columns]
    display_df = similar_props_df[available_columns].copy()
    
    # Format the price column
    if 'SubContractor Price Per Acre' in display_df.columns:
        display_df['SubContractor Price Per Acre'] = display_df['SubContractor Price Per Acre'].apply(
            lambda x: f"${x:,.2f}" if pd.notnull(x) else "N/A"
        )
    
    # Format acreage columns
    for col in ['Total Acreage', 'Sidewalk Acreage']:
        if col in display_df.columns:
            display_df[col] = display_df[col].apply(
                lambda x: f"{x:.2f}" if pd.notnull(x) else "N/A"
            )
    
    # Format distance column
    if 'Distance_Miles' in display_df.columns:
        display_df['Distance_Miles'] = display_df['Distance_Miles'].apply(
            lambda x: f"{x:.1f} mi" if pd.notnull(x) else "N/A"
        )
        # Rename for better display
        display_df = display_df.rename(columns={'Distance_Miles': 'Distance'})
    
    st.dataframe(display_df, use_container_width=True)

def create_price_comparison_chart(similar_props_df, predicted_price):
    """Create a chart comparing predicted price with similar properties."""
    if similar_props_df.empty or 'SubContractor Price Per Acre' not in similar_props_df.columns:
        return None
    
    # Remove any NaN values
    clean_df = similar_props_df.dropna(subset=['SubContractor Price Per Acre', 'Total Acreage'])
    
    if clean_df.empty:
        return None
    
    fig = go.Figure()
    
    # Add similar properties as scatter plot
    fig.add_trace(go.Scatter(
        x=clean_df['Total Acreage'],
        y=clean_df['SubContractor Price Per Acre'],
        mode='markers',
        name='Similar Properties',
        marker=dict(size=10, color='lightblue', line=dict(width=2, color='darkblue')),
        hovertemplate='<b>Acreage:</b> %{x:.2f}<br><b>Price/Acre:</b> $%{y:,.2f}<extra></extra>'
    ))
    
    # Add predicted price as a horizontal line
    fig.add_hline(
        y=predicted_price, 
        line_dash="dash", 
        line_color="red",
        annotation_text=f"Predicted: ${predicted_price:,.2f}/acre",
        annotation_position="top right"
    )
    
    fig.update_layout(
        title="Price Comparison: Predicted vs Similar Properties",
        xaxis_title="Total Acreage",
        yaxis_title="SubContractor Price Per Acre ($)",
        showlegend=True,
        height=400
    )
    
    return fig

def calculate_market_statistics(similar_props_df):
    """Calculate market statistics from similar properties."""
    if similar_props_df.empty or 'SubContractor Price Per Acre' not in similar_props_df.columns:
        return None
    
    prices = similar_props_df['SubContractor Price Per Acre'].dropna()
    if len(prices) == 0:
        return None
    
    stats = {
        'count': len(prices),
        'mean': prices.mean(),
        'median': prices.median(),
        'min': prices.min(),
        'max': prices.max(),
        'std': prices.std() if len(prices) > 1 else 0
    }
    
    return stats

# Streamlit App Configuration
st.set_page_config(
    page_title="Snow Removal Pricing Tool V2",
    page_icon="❄️",
    layout="wide",
    initial_sidebar_state="expanded"
)



# Center the logo using columns
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.image("logo.png", width=300)
# Main Title
st.markdown('<h1 style="text-align: center; color: #2E86AB;">❄️ GSC Subcontractor Bid Pricing Tool (V4 - With Market Triangulation)</h1>', unsafe_allow_html=True)

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

# NEW: Market Comparison Section
st.sidebar.header("Market Comparison")

# Add zip code input
zip_code = st.sidebar.text_input(
    "Zip Code",
    value="19464",  # Default zip code
    help="Enter zip code to find similar properties in the area"
)

# Add dataset path input
dataset_path = st.sidebar.text_input(
    "Training Dataset Path",
    value="SubFull.csv",
    help="Path to your training dataset file"
)

# Add filter options
comparison_property_type = st.sidebar.selectbox(
    "Filter by Property Type",
    options=["Any", "Retail", "Office", "Industrial", "Residential", "Medical", "Other"],
    index=0,
    help="Filter similar properties by type (optional)"
)

# Add search radius input
search_radius = st.sidebar.slider(
    "Search Radius (miles)",
    min_value=5,
    max_value=50,
    value=25,
    step=5,
    help="Find properties within this distance from the target zip code"
)

# Add acreage similarity toggle
use_acreage_filter = st.sidebar.checkbox(
    "Filter by Similar Acreage (±35%)",
    value=True,
    help="Only show properties with acreage within 35% of your property size"
)

max_comparisons = st.sidebar.slider(
    "Max Properties to Show",
    min_value=5,
    max_value=20,
    value=10,
    help="Maximum number of similar properties to display"
)

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
    📈 Multiplier: {complexity_multiplier*100:.1f}% per level<br>
    📮 Zip Code: {zip_code}
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

# NEW: Market Comparison Analysis Section
st.header("Market Comparison Analysis")

# Load and filter similar properties
training_data, data_loaded = load_training_dataset(dataset_path)

if data_loaded and zip_code:
    # Use the enhanced geographic search with acreage filtering
    if use_acreage_filter:
        similar_properties = get_similar_properties(
            training_data, 
            zip_code, 
            comparison_property_type if comparison_property_type != "Any" else None,
            total_acreage,  # Pass the acreage for similarity matching
            max_comparisons,
            search_radius
        )
    else:
        # Use geographic search without acreage filtering - pass None for total_acreage
        similar_properties = get_similar_properties(
            training_data, 
            zip_code, 
            comparison_property_type if comparison_property_type != "Any" else None,
            None,  # No acreage filtering
            max_comparisons,
            search_radius
        )
    
    if not similar_properties.empty:
        # Create tabs for different views
        comparison_tab1, comparison_tab2, comparison_tab3 = st.tabs(
            ["📊 Properties Table", "📈 Price Chart", "📋 Market Stats"]
        )
        
        with comparison_tab1:
            # Display search criteria summary
            search_summary = f"**Search Results:** Properties within {search_radius} miles of {zip_code}"
            if use_acreage_filter:
                acreage_range = f"{total_acreage * 0.65:.1f} - {total_acreage * 1.35:.1f} acres"
                search_summary += f" with similar acreage ({acreage_range})"
            if comparison_property_type != "Any":
                search_summary += f" of type: {comparison_property_type}"
            
            st.markdown(search_summary)
            st.subheader(f"Similar Properties Near Zip Code {zip_code}")
            display_similar_properties_table(similar_properties)
            
            # Add download button for the filtered data
            csv_data = similar_properties.to_csv(index=False)
            st.download_button(
                label="📥 Download Similar Properties Data",
                data=csv_data,
                file_name=f"similar_properties_{zip_code}_{search_radius}mi.csv",
                mime="text/csv"
            )
        
        with comparison_tab2:
            st.subheader("Price Comparison Chart")
            price_chart = create_price_comparison_chart(similar_properties, final_price_per_acre)
            if price_chart:
                st.plotly_chart(price_chart, use_container_width=True)
            else:
                st.info("Chart unavailable - missing price data in similar properties.")
        
        with comparison_tab3:
            st.subheader("Market Statistics")
            market_stats = calculate_market_statistics(similar_properties)
            
            if market_stats:
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Properties Found", market_stats['count'])
                    st.metric("Average Price/Acre", f"${market_stats['mean']:,.2f}")
                
                with col2:
                    st.metric("Median Price/Acre", f"${market_stats['median']:,.2f}")
                    st.metric("Price Range", f"${market_stats['min']:,.2f} - ${market_stats['max']:,.2f}")
                
                with col3:
                    deviation = abs(final_price_per_acre - market_stats['mean'])
                    deviation_pct = (deviation / market_stats['mean']) * 100 if market_stats['mean'] > 0 else 0
                    st.metric(
                        "Prediction vs Market Avg", 
                        f"${deviation:,.2f}",
                        f"{'+' if final_price_per_acre > market_stats['mean'] else '-'}{deviation_pct:.1f}%"
                    )
                
                # Market position indicator
                if market_stats['std'] > 0:
                    if final_price_per_acre < market_stats['mean'] - market_stats['std']:
                        position = "🟢 Below Market (Competitive)"
                    elif final_price_per_acre > market_stats['mean'] + market_stats['std']:
                        position = "🔴 Above Market (Premium)"
                    else:
                        position = "🟡 Within Market Range"
                else:
                    if final_price_per_acre < market_stats['mean']:
                        position = "🟢 Below Market (Competitive)"
                    elif final_price_per_acre > market_stats['mean']:
                        position = "🔴 Above Market (Premium)"
                    else:
                        position = "🟡 At Market Average"
                
                st.markdown(f"**Market Position:** {position}")
            else:
                st.info("No price data available for market statistics.")
    
    else:
        st.info(f"No properties found within {search_radius} miles of zip code {zip_code} with the selected criteria.")

else:
    if not zip_code:
        st.info("Enter a zip code to see similar properties in the area.")
    else:
        st.warning("Training dataset could not be loaded. Check the dataset path in the sidebar.")

# Instructions
st.markdown("---")
st.markdown("""

️

### 💡 **Enhanced Geographic Market Comparison:**
- ✅ **Geographic Radius Search** - Find properties within 5-50 miles of your target zip code
- ✅ **Smart Acreage Filtering** - Optional ±35% acreage similarity matching
- ✅ **Distance Display** - See exactly how far each comparable property is located
- ✅ **Fallback Protection** - Automatically falls back to exact zip matching if pgeocode fails
- ✅ **Property Type Filtering** for more relevant comparisons  
- ✅ **Interactive Price Charts** showing your prediction vs market data
- ✅ **Market Statistics** including averages, ranges, and position analysis
- ✅ **Data Export** functionality for further analysis with distance information


### 🔧 **Required Dataset Columns:**
Your training dataset should include:
- `Zip` - For geographic filtering (required)
- `Property Type` - For property type filtering
- `Total Acreage` - For size comparison and similarity matching
- `Sidewalk Acreage` - For sidewalk comparison
- `SubContractor Price Per Acre` - The key pricing data (required)
- `Complexity (1-5)` - For complexity comparison
- `Avg Snowfall (3-Year)` - For weather comparison
- `Property Region (When Bid)` - For regional context

### 📊 **New Search Features:**
- **Search Radius**: Choose 5-50 mile radius around your target zip code
- **Acreage Similarity**: Toggle ±35% acreage filtering on/off
- **Distance Information**: See how far each comparable property is from your location
- **Smart Sorting**: Results sorted by distance first, then by price


### 🎯 **Market Analysis Benefits:**
- **Validate ML predictions** against real market data from your area
- **Identify pricing opportunities** by seeing local competition
- **Adjust estimates** based on actual contract prices in the same zip code
- **Build client confidence** by showing comparable properties and pricing
- **Track market trends** over time as you accumulate more data

### 🔍 **Troubleshooting:**
- If no properties show up, verify the zip code format in your dataset
- Ensure your training dataset has the required columns listed above
- Check that the dataset path is correct and the file is accessible
- Try different property type filters if results are limited
- Consider expanding to nearby zip codes if data is sparse
           """ )
