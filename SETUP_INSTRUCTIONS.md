# Setup Instructions

## Step-by-Step Setup Guide

### Step 1: Install Python
If you haven't already:
1. Download from: https://www.python.org/downloads/
2. **Important**: Check "Add Python to PATH" during installation
3. Restart Cursor after installation

### Step 2: Install Dependencies
Open terminal in Cursor and run:
```bash
pip install -r requirements.txt
```

This installs all required packages including:
- Streamlit (for the web app)
- XGBoost (for the ML model)
- pandas, numpy, matplotlib, seaborn (for data processing)
- kagglehub (for downloading the dataset)

### Step 3: Download the Dataset

**Option A: Automatic Download (Recommended)**
```bash
python download_data.py
```

This will:
- Download the dataset from Kaggle
- Find the CSV file(s)
- Copy it to `data/solar_data.csv`

**Option B: Manual Download**
1. Go to: https://www.kaggle.com/datasets/saurabhshahane/northern-hemisphere-horizontal-photovoltaic
2. Download the dataset
3. Extract and find the CSV file
4. Save it as `data/solar_data.csv`

**Option C: Using Python Terminal**
You can also paste this into a Python terminal or create a script:
```python
import kagglehub

# Download latest version
path = kagglehub.dataset_download("saurabhshahane/northern-hemisphere-horizontal-photovoltaic")

print("Path to dataset files:", path)
```

Then manually copy the CSV file to `data/solar_data.csv`.

### Step 4: Run the App
```bash
streamlit run app.py
```

The app will:
- Open automatically in your browser
- Train the model on first use (if no saved model exists)
- Be ready for predictions!

## Troubleshooting

**Python not found:**
- Make sure Python is installed and added to PATH
- Restart Cursor after installing Python

**Kaggle authentication:**
- If kagglehub asks for authentication, you may need a Kaggle API token
- See: https://www.kaggle.com/docs/api
- Or use manual download (Option B)

**Port already in use:**
- Streamlit will automatically use the next available port (8502, 8503, etc.)

**Model training takes time:**
- First run will train the model (may take 1-2 minutes)
- Subsequent runs will load the saved model (much faster)

## Quick Commands Summary

```bash
# Install dependencies
pip install -r requirements.txt

# Download data (automatic)
python download_data.py

# Run the app
streamlit run app.py
```

