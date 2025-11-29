# CSV Upload Guide

## Two Different CSV Files

### 1. Training Data (Already Downloaded) ✅
**File:** `data/solar_data.csv`
- **Purpose:** Used to train the model
- **Location:** Already in your `data/` folder (downloaded via `download_data.py`)
- **DO NOT upload this in the app** - it's already being used automatically
- **Contains:** All features + `PolyPwr` (the target variable we're predicting)

### 2. Prediction Data (What You Upload in the App) 📤
**File:** Your own CSV file with new locations to predict
- **Purpose:** Get predictions for new locations/environments
- **Upload location:** In the app's "Predict" page → "Upload CSV File"
- **Contains:** Only the feature columns (NO `PolyPwr` column)

## What Columns Should Your Upload CSV Have?

Your uploaded CSV should contain these **12 feature columns**:

1. `Date` - Date in YYYYMMDD format (e.g., 20171021)
2. `Time` - Time in HHMM format (e.g., 1430 for 2:30 PM)
3. `Latitude` - Geographic latitude (decimal degrees)
4. `Longitude` - Geographic longitude (decimal degrees)
5. `Altitude` - Elevation above sea level (meters)
6. `Month` - Month of year (1-12)
7. `Humidity` - Relative humidity (percentage)
8. `AmbientTemp` - Ambient temperature (°C)
9. `Wind.Speed` - Wind speed
10. `Visibility` - Visibility conditions
11. `Pressure` - Atmospheric pressure (mbar)
12. `Cloud.Ceiling` - Cloud ceiling height (use 722.0 for clear sky)

**Important:** 
- ❌ Do NOT include `PolyPwr` (that's what we're predicting!)
- ✅ Include all 12 feature columns listed above

## Example CSV Structure

```csv
Date,Time,Latitude,Longitude,Altitude,Month,Humidity,AmbientTemp,Wind.Speed,Visibility,Pressure,Cloud.Ceiling
20171021,1200,30.0,-90.0,100,7,50,25,10,10,1013,722
20171022,1400,35.5,-95.2,200,7,45,28,12,15,1015,722
20171023,1000,40.0,-100.0,500,7,55,22,8,8,1010,300
```

## Creating Your Own CSV

You can create a CSV file with:
- **Excel** - Save as CSV format
- **Google Sheets** - Download as CSV
- **Text editor** - Create with comma-separated values
- **Python** - Use pandas to create a DataFrame and save as CSV

## Quick Test

Want to test the upload feature? You can:
1. Take the training data (`data/solar_data.csv`)
2. Remove the `PolyPwr` column
3. Take just a few rows (e.g., first 10 rows)
4. Save as a new file (e.g., `test_predictions.csv`)
5. Upload that in the app to see predictions

## Summary

- **Training data:** Already in `data/solar_data.csv` ✅
- **Upload for predictions:** Your own CSV with 12 feature columns (no PolyPwr) 📤
- **Or use:** Manual input form in the app (no CSV needed) ✍️

