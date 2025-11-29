"""
Streamlit web application for Solar Panel Installation Suitability Prediction.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from pathlib import Path

from data_processing import load_data, preprocess_data, get_feature_names, prepare_features
from model_trainer import get_or_train_model, predict, load_model

# Page configuration
st.set_page_config(
    page_title="Solar Panel Suitability Predictor",
    page_icon="☀️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #FF6B35;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #004E89;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        background-color: #F0F8FF;
        padding: 1.5rem;
        border-radius: 10px;
        border: 2px solid #FF6B35;
        margin: 1rem 0;
    }
    .metric-card {
        background-color: #E8F4F8;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'model' not in st.session_state:
    st.session_state.model = None
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'df_processed' not in st.session_state:
    st.session_state.df_processed = None


def main():
    # Header
    st.markdown('<div class="main-header">☀️ Solar Panel Installation Suitability Predictor</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Predict optimal solar panel installation locations using machine learning</div>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("📋 Navigation")
        page = st.radio(
            "Choose a page:",
            ["🏠 Home", "📊 Predict", "📈 Visualizations", "ℹ️ About"]
        )
    
    if page == "🏠 Home":
        show_home_page()
    elif page == "📊 Predict":
        show_predict_page()
    elif page == "📈 Visualizations":
        show_visualizations_page()
    elif page == "ℹ️ About":
        show_about_page()


def show_home_page():
    """Display the home page with project information."""
    st.markdown("""
    ## Welcome to the Solar Panel Suitability Predictor
    
    This application uses a machine learning model (XGBoost Regression) to predict the optimal 
    solar panel installation locations based on environmental and geographical factors.
    
    ### 🎯 Project Goal
    
    Given the worsening climate crisis, this tool helps governments and businesses identify 
    optimal locations for solar panel installations, making renewable energy implementation 
    easier and more efficient.
    
    ### 🔬 Model Details
    
    - **Algorithm**: XGBoost Regression
    - **Performance**: R² score of ~0.68
    - **Features**: 12 environmental and geographical factors including:
      - Date and Time
      - Location (Latitude, Longitude, Altitude)
      - Weather conditions (Temperature, Humidity, Pressure, Wind Speed, Visibility, Cloud Ceiling)
      - Month
    
    ### 📊 Features Used
    
    1. **Date** - Days from the first date in the dataset
    2. **Time** - Time of day in decimal hours
    3. **Latitude** - Geographic latitude
    4. **Longitude** - Geographic longitude
    5. **Altitude** - Elevation above sea level
    6. **Month** - Month of the year
    7. **Humidity** - Relative humidity percentage
    8. **AmbientTemp** - Ambient temperature in Celsius
    9. **Wind.Speed** - Wind speed
    10. **Visibility** - Visibility conditions
    11. **Pressure** - Atmospheric pressure
    12. **Cloud.Ceiling** - Cloud ceiling height
    
    ### 🚀 Getting Started
    
    1. Navigate to the **Predict** page to make predictions
    2. Upload a CSV file or manually input values
    3. View **Visualizations** to explore the data and model insights
    4. Check **About** for more information
    
    ### 📁 Data Setup
    
    To use this application, you need to:
    1. Place your training data CSV file in the `data/` directory
    2. The model will automatically train on first use (or you can retrain it)
    3. The trained model will be saved in the `models/` directory
    """)


def show_predict_page():
    """Display the prediction page."""
    st.header("🔮 Make Predictions")
    
    # Load or train model
    with st.spinner("Loading model..."):
        try:
            model = get_or_train_model(force_retrain=False)
            st.session_state.model = model
            st.success("✅ Model loaded successfully!")
        except FileNotFoundError as e:
            st.error(f"❌ {str(e)}")
            st.info("💡 Please place your data file in the `data/` directory and try again.")
            return
    
    # Input method selection
    input_method = st.radio(
        "Choose input method:",
        ["📤 Upload CSV File", "✍️ Manual Input"],
        horizontal=True
    )
    
    if input_method == "📤 Upload CSV File":
        predict_from_file(model)
    else:
        predict_manual_input(model)


def predict_from_file(model):
    """Handle predictions from uploaded CSV file."""
    st.subheader("Upload CSV File")
    
    uploaded_file = st.file_uploader(
        "Upload a CSV file with location/environment data",
        type=['csv'],
        help="The CSV should contain columns matching the feature names"
    )
    
    if uploaded_file is not None:
        try:
            # Read CSV
            df = pd.read_csv(uploaded_file)
            st.success(f"✅ File uploaded successfully! ({len(df)} rows)")
            
            # Display preview
            st.subheader("Data Preview")
            st.dataframe(df.head(), use_container_width=True)
            
            # Check if required columns exist
            feature_names = get_feature_names()
            missing_cols = [col for col in feature_names if col not in df.columns]
            
            if missing_cols:
                st.warning(f"⚠️ Missing columns: {', '.join(missing_cols)}")
                st.info("💡 Please ensure your CSV contains all required feature columns.")
            else:
                # Preprocess data
                try:
                    df_processed = preprocess_data(df)
                    X, _ = prepare_features(df_processed)
                    
                    # Make predictions
                    if st.button("🔮 Predict Solar Suitability", type="primary"):
                        with st.spinner("Making predictions..."):
                            predictions = predict(model, X)
                            
                            # Add predictions to dataframe
                            df_results = df.copy()
                            df_results['Predicted_Solar_Power'] = predictions
                            df_results['Suitability_Score'] = predictions
                            
                            # Display results
                            st.subheader("📊 Prediction Results")
                            st.dataframe(df_results, use_container_width=True)
                            
                            # Summary statistics
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("Mean Prediction", f"{predictions.mean():.2f}")
                            with col2:
                                st.metric("Max Prediction", f"{predictions.max():.2f}")
                            with col3:
                                st.metric("Min Prediction", f"{predictions.min():.2f}")
                            with col4:
                                st.metric("Std Deviation", f"{predictions.std():.2f}")
                            
                            # Download results
                            csv = df_results.to_csv(index=False)
                            st.download_button(
                                label="📥 Download Results as CSV",
                                data=csv,
                                file_name="solar_predictions.csv",
                                mime="text/csv"
                            )
                except Exception as e:
                    st.error(f"❌ Error processing data: {str(e)}")
                    st.info("💡 Make sure your data format matches the expected structure.")
        
        except Exception as e:
            st.error(f"❌ Error reading file: {str(e)}")


def predict_manual_input(model):
    """Handle predictions from manual input."""
    st.subheader("Manual Input")
    
    with st.form("prediction_form"):
        st.markdown("### Enter Location and Environmental Data")
        
        col1, col2 = st.columns(2)
        
        with col1:
            date_input = st.number_input(
                "Date (YYYYMMDD format, e.g., 20171021)",
                min_value=20000101,
                max_value=20991231,
                value=20171021,
                help="Date in YYYYMMDD format"
            )
            time_input = st.number_input(
                "Time (HHMM format, e.g., 1430 for 2:30 PM)",
                min_value=0,
                max_value=2359,
                value=1200,
                help="Time in HHMM format (24-hour)"
            )
            latitude = st.number_input(
                "Latitude",
                min_value=-90.0,
                max_value=90.0,
                value=30.0,
                format="%.4f"
            )
            longitude = st.number_input(
                "Longitude",
                min_value=-180.0,
                max_value=180.0,
                value=-90.0,
                format="%.4f"
            )
            altitude = st.number_input(
                "Altitude (meters above sea level)",
                min_value=0.0,
                max_value=10000.0,
                value=100.0,
                format="%.2f"
            )
            month = st.number_input(
                "Month (1-12)",
                min_value=1,
                max_value=12,
                value=7
            )
        
        with col2:
            humidity = st.number_input(
                "Humidity (%)",
                min_value=0.0,
                max_value=100.0,
                value=50.0,
                format="%.2f"
            )
            ambient_temp = st.number_input(
                "Ambient Temperature (°C)",
                min_value=-50.0,
                max_value=70.0,
                value=25.0,
                format="%.2f"
            )
            wind_speed = st.number_input(
                "Wind Speed",
                min_value=0.0,
                max_value=200.0,
                value=10.0,
                format="%.2f"
            )
            visibility = st.number_input(
                "Visibility",
                min_value=0.0,
                max_value=100.0,
                value=10.0,
                format="%.2f"
            )
            pressure = st.number_input(
                "Pressure (mbar)",
                min_value=700.0,
                max_value=1100.0,
                value=1013.0,
                format="%.2f"
            )
            cloud_ceiling = st.number_input(
                "Cloud Ceiling",
                min_value=0.0,
                max_value=1000.0,
                value=722.0,
                format="%.2f",
                help="Use 722.0 for clear sky conditions"
            )
        
        submitted = st.form_submit_button("🔮 Predict Solar Suitability", type="primary")
        
        if submitted:
            # Preprocess inputs
            from data_processing import convert_time, day_from_zero, convert_date
            
            # Convert time
            time_hours = convert_time(time_input)
            
            # Convert date (using a reference minimum date)
            # For manual input, we'll use the input date as reference
            min_date = date_input
            date_days = day_from_zero(date_input, min_date)
            
            # Create feature array
            features = np.array([[
                date_days, time_hours, latitude, longitude, altitude, month,
                humidity, ambient_temp, wind_speed, visibility, pressure, cloud_ceiling
            ]])
            
            # Make prediction
            with st.spinner("Making prediction..."):
                prediction = predict(model, features)[0]
                
                # Display result
                st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
                st.markdown("### 🌞 Prediction Result")
                st.metric(
                    "Predicted Solar Power Output",
                    f"{prediction:.2f}",
                    help="Higher values indicate better solar panel suitability"
                )
                
                # Interpret result
                if prediction > 20:
                    st.success("✅ Excellent location for solar panel installation!")
                elif prediction > 10:
                    st.info("✅ Good location for solar panel installation")
                elif prediction > 5:
                    st.warning("⚠️ Moderate suitability for solar panel installation")
                else:
                    st.error("❌ Low suitability for solar panel installation")
                
                st.markdown('</div>', unsafe_allow_html=True)


def show_visualizations_page():
    """Display visualizations page."""
    st.header("📈 Data Visualizations")
    
    # Check if data is available
    data_path = os.path.join("data", "solar_data.csv")
    
    if not os.path.exists(data_path):
        st.warning("⚠️ Data file not found. Please place your CSV file in the `data/` directory to view visualizations.")
        st.info("💡 You can still view model feature importance if a trained model exists.")
        
        # Try to load model for feature importance
        try:
            model = load_model()
            show_feature_importance(model)
        except:
            pass
        return
    
    # Load data
    with st.spinner("Loading data..."):
        try:
            df = load_data(data_path)
            st.session_state.df_processed = df
            st.session_state.data_loaded = True
        except Exception as e:
            st.error(f"❌ Error loading data: {str(e)}")
            return
    
    st.success(f"✅ Data loaded successfully! ({len(df)} rows)")
    
    # Visualization options
    viz_option = st.selectbox(
        "Choose visualization:",
        [
            "📊 Correlation Heatmap",
            "📈 Feature Importance",
            "📉 Target Distribution",
            "🌡️ Feature vs Target Scatter Plots"
        ]
    )
    
    if viz_option == "📊 Correlation Heatmap":
        show_correlation_heatmap(df)
    elif viz_option == "📈 Feature Importance":
        show_feature_importance_plot(df)
    elif viz_option == "📉 Target Distribution":
        show_target_distribution(df)
    elif viz_option == "🌡️ Feature vs Target Scatter Plots":
        show_feature_scatter_plots(df)


def show_correlation_heatmap(df):
    """Display correlation heatmap."""
    st.subheader("Correlation Heatmap")
    
    # Prepare data for correlation
    corr_df = df.drop(columns=['PolyPwr_class', 'PolyPwr_class_num'], errors='ignore')
    corr = corr_df.corr()
    
    # Create mask for upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool))
    
    # Plot
    fig, ax = plt.subplots(figsize=(12, 10))
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    sns.heatmap(
        corr, mask=mask, cmap=cmap, vmax=0.3, center=0,
        square=True, linewidths=0.5, cbar_kws={"shrink": 0.5},
        annot=True, fmt=".2f", ax=ax
    )
    ax.set_title("Feature Correlation Heatmap", fontsize=16, pad=20)
    plt.tight_layout()
    st.pyplot(fig)


def show_feature_importance_plot(df):
    """Display feature importance plot."""
    st.subheader("Feature Importance")
    
    # Train or load model
    try:
        model = get_or_train_model(df=df, force_retrain=False)
    except:
        st.error("❌ Could not load or train model for feature importance.")
        return
    
    # Get feature importances
    importances = model.feature_importances_
    feature_names = get_feature_names()
    
    # Create DataFrame
    feat_imp_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.barplot(x='Importance', y='Feature', data=feat_imp_df, ax=ax, palette='viridis')
    ax.set_title('XGBoost: Feature Importance for Predicting Solar Power', fontsize=14, pad=20)
    ax.set_xlabel('Importance Score', fontsize=12)
    ax.set_ylabel('Feature', fontsize=12)
    plt.tight_layout()
    st.pyplot(fig)
    
    # Display table
    st.dataframe(feat_imp_df, use_container_width=True)


def show_feature_importance(model):
    """Display feature importance from existing model."""
    st.subheader("Feature Importance")
    
    # Get feature importances
    importances = model.feature_importances_
    feature_names = get_feature_names()
    
    # Create DataFrame
    feat_imp_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.barplot(x='Importance', y='Feature', data=feat_imp_df, ax=ax, palette='viridis')
    ax.set_title('XGBoost: Feature Importance for Predicting Solar Power', fontsize=14, pad=20)
    ax.set_xlabel('Importance Score', fontsize=12)
    ax.set_ylabel('Feature', fontsize=12)
    plt.tight_layout()
    st.pyplot(fig)


def show_target_distribution(df):
    """Display target variable distribution."""
    st.subheader("Solar Power Output Distribution")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(df['PolyPwr'], bins=50, edgecolor='black', alpha=0.7)
    ax.set_xlabel('Solar Power Output (PolyPwr)', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Distribution of Solar Power Output', fontsize=14, pad=20)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig)
    
    # Statistics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Mean", f"{df['PolyPwr'].mean():.2f}")
    with col2:
        st.metric("Median", f"{df['PolyPwr'].median():.2f}")
    with col3:
        st.metric("Std Dev", f"{df['PolyPwr'].std():.2f}")
    with col4:
        st.metric("Range", f"{df['PolyPwr'].max() - df['PolyPwr'].min():.2f}")


def show_feature_scatter_plots(df):
    """Display scatter plots of features vs target."""
    st.subheader("Feature vs Target Scatter Plots")
    
    feature_names = get_feature_names()
    selected_features = st.multiselect(
        "Select features to plot:",
        feature_names,
        default=feature_names[:6]  # Show first 6 by default
    )
    
    if selected_features:
        n_features = len(selected_features)
        n_cols = 3
        n_rows = (n_features + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
        axes = axes.flatten() if n_features > 1 else [axes]
        
        for i, feature in enumerate(selected_features):
            ax = axes[i]
            ax.scatter(df[feature], df['PolyPwr'], alpha=0.5, s=10)
            ax.set_xlabel(feature, fontsize=10)
            ax.set_ylabel('Solar Power Output', fontsize=10)
            ax.set_title(f'{feature} vs Solar Power', fontsize=12)
            ax.grid(True, alpha=0.3)
        
        # Hide unused subplots
        for i in range(n_features, len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        st.pyplot(fig)


def show_about_page():
    """Display the about page."""
    st.header("ℹ️ About This Project")
    
    st.markdown("""
    ### Project Information
    
    This project was developed as part of the **AI4ALL Ignite program**, focusing on 
    machine learning applications for environmental sustainability.
    
    ### Problem Statement
    
    Given the worsening climate crisis, we wanted to create a tool that could help 
    increase the ease of renewable energy implementation. This project helps governments 
    and businesses identify optimal locations for solar panel installations, limiting 
    greenhouse gas emissions.
    
    ### Methodology
    
    We compared different models, parameters, and combinations of features using accuracy 
    scores, correlation graphs, and feature importances to maximize accuracy in the real world. 
    We ultimately selected the **XGBoost Regression** algorithm for its superior performance.
    
    ### Model Performance
    
    - **Algorithm**: XGBoost Regression
    - **R² Score**: ~0.68
    - **Cross-Validation**: 5-fold CV with mean score of ~0.68
    
    ### Data Source
    
    Kaggle dataset: [Northern Hemisphere Horizontal Photovoltaic](https://www.kaggle.com/datasets/saurabhshahane/northern-hemisphere-horizontal-photovoltaic)
    
    ### Technologies Used
    
    - Python
    - pandas
    - numpy
    - matplotlib
    - seaborn
    - scikit-learn
    - XGBoost
    - Streamlit
    
    ### Authors
    
    This project was completed by:
    - Bart Scheer
    - Linda Chen
    - Tiffany Atta
    
    ### How to Use
    
    1. **Setup**: Place your training data CSV file in the `data/` directory
    2. **Run**: Execute `streamlit run app.py` in your terminal
    3. **Predict**: Use the Predict page to make predictions
    4. **Visualize**: Explore data insights in the Visualizations page
    
    ### Model Training
    
    The model will automatically train on first use if no saved model exists. 
    You can also retrain the model by deleting the saved model file in the `models/` directory.
    """)


if __name__ == "__main__":
    main()

