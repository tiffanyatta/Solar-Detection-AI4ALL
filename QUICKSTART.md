# Quick Start Guide

## 🚀 Running the App in Cursor

### Step 1: Install Dependencies
Open the terminal in Cursor (Ctrl+` or View → Terminal) and run:
```bash
pip install -r requirements.txt
```

### Step 2: Prepare Your Data
1. Download the dataset from [Kaggle](https://www.kaggle.com/datasets/saurabhshahane/northern-hemisphere-horizontal-photovoltaic)
2. Place the CSV file in the `data/` directory
3. Name it `solar_data.csv`

**Note**: If your file has a different name, you can:
- Rename it to `solar_data.csv`, OR
- Update the `DATA_PATH` variable in `model_trainer.py`

### Step 3: Run the App
In the terminal, run:
```bash
streamlit run app.py
```

The app will automatically:
- Open in your default web browser
- Load or train the model (if no saved model exists)
- Be ready to make predictions!

### Step 4: Use the App
1. **Home**: Read about the project
2. **Predict**: 
   - Upload a CSV file OR
   - Enter values manually
   - Get solar suitability predictions
3. **Visualizations**: Explore data insights
4. **About**: Learn more about the project

## 📋 Data File Requirements

Your CSV file should contain these columns:
- `Date` (YYYYMMDD format, e.g., 20171021)
- `Time` (HHMM format, e.g., 1430 for 2:30 PM)
- `Latitude` (decimal degrees)
- `Longitude` (decimal degrees)
- `Altitude` (meters above sea level)
- `Month` (1-12)
- `Humidity` (percentage)
- `AmbientTemp` (Celsius)
- `Wind.Speed`
- `Visibility`
- `Pressure` (mbar)
- `Cloud.Ceiling` (use 722.0 for clear sky)
- `PolyPwr` (target - solar power output)

## 🔧 Troubleshooting

**Problem**: "Model not found" error
- **Solution**: Make sure `data/solar_data.csv` exists

**Problem**: Import errors
- **Solution**: Run `pip install -r requirements.txt` again

**Problem**: Port already in use
- **Solution**: Streamlit will automatically use the next available port (8502, 8503, etc.)

**Problem**: Data format errors
- **Solution**: Check that your CSV has all required columns and correct formats

## 📁 Directory Structure

```
Solar-Detection-AI4ALL/
├── app.py              ← Main Streamlit app
├── data_processing.py  ← Data preprocessing
├── model_trainer.py    ← Model training
├── data/               ← Place your CSV here
│   └── solar_data.csv
└── models/             ← Saved models (auto-created)
    └── solar_model.pkl
```

## 💡 Tips

- The model trains automatically on first use
- Trained models are saved for faster loading next time
- You can retrain by deleting `models/solar_model.pkl`
- The app works offline once the model is trained


