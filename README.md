# Solar Detection

As part of the AI4ALL Ignite program, we learned about different AI and machine learning topics, as well as responsible AI.
In a team of 3, we worked to create a supervised regression machine learning model that would detect the optimal environment/location for solar panel placement. Using Python, including pandas, numpy, matplotlib, and seaborn libraries, I created feature importance and feature correlation charts to help decide what variables to keep, cleaned the data by simplifying variables and removing other variables. 

## Problem Statement <!--- do not change this line -->

Given the worsening climate, we wanted to create something that could help increase the ease of renewable energy implementation. This project would hopefully make it easier for governments and businesses to implement solar panel installations, limiting greenhouse gas emissions.

## Key Results <!--- do not change this line -->

Created a supervised regression machine learning model to detect optimal solar panel installation locations.

## Methodologies <!--- do not change this line -->

Compared different models, parameters, and combinations of features using accuracy scores, correlation graphs, and feature importances to maximize accuracy in the real world.
We landed on XGBoost Regression algorithm.

## Data Sources <!--- do not change this line -->

Kaggle dataset: [https://www.kaggle.com/datasets/saurabhshahane/northern-hemisphere-horizontal-photovoltaic](url)

## Technologies Used <!--- do not change this line -->

- Python
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- XGBoost
- Streamlit

## Authors <!--- do not change this line -->

This project was completed by:
- Bart Scheer
- Linda Chen
- Tiffany Atta

## 🚀 How To Use the Web App

### Prerequisites

1. Python 3.8 or higher
2. pip (Python package manager)

### Installation Steps

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd Solar-Detection-AI4ALL
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Prepare your data:**
   - Download the dataset from [Kaggle](https://www.kaggle.com/datasets/saurabhshahane/northern-hemisphere-horizontal-photovoltaic)
   - Place the CSV file in the `data/` directory
   - Name it `solar_data.csv` (or update the path in `model_trainer.py`)

4. **Run the Streamlit app:**
   ```bash
   streamlit run app.py
   ```
   
   The app will automatically open in your web browser at `http://localhost:8501`

### Using the App

1. **Home Page**: Overview of the project and model details
2. **Predict Page**: 
   - Upload a CSV file with location/environment data, OR
   - Manually input values for all features
   - Get predictions for solar panel suitability
3. **Visualizations Page**: 
   - View correlation heatmaps
   - See feature importance charts
   - Explore data distributions
4. **About Page**: Project information and methodology

### Data File Location

Place your training data CSV file in:
```
data/solar_data.csv
```

The CSV should contain the following columns:
- `Date` (YYYYMMDD format, e.g., 20171021)
- `Time` (HHMM format, e.g., 1430)
- `Latitude`
- `Longitude`
- `Altitude`
- `Month` (1-12)
- `Humidity`
- `AmbientTemp`
- `Wind.Speed`
- `Visibility`
- `Pressure`
- `Cloud.Ceiling`
- `PolyPwr` (target variable - solar power output)

### Model Training

- The model will automatically train on first use if no saved model exists
- Trained models are saved in `models/solar_model.pkl`
- To retrain the model, delete the saved model file and restart the app

### Running in Cursor

1. Open the project in Cursor
2. Open the integrated terminal (Ctrl+` or View → Terminal)
3. Run: `streamlit run app.py`
4. The app will open automatically in your default browser

## 📁 Project Structure

```
Solar-Detection-AI4ALL/
├── app.py                 # Streamlit web application
├── data_processing.py     # Data preprocessing functions
├── model_trainer.py       # Model training and persistence
├── ai4all_blt_project.py # Original project script
├── AI4ALL_BLT_Project.ipynb # Original notebook
├── requirements.txt       # Python dependencies
├── data/                  # Data directory (place CSV here)
│   └── solar_data.csv     # Your training data
└── models/                # Saved models directory
    └── solar_model.pkl    # Trained model (auto-generated)
```

## 🔧 Troubleshooting

- **Model not found error**: Make sure your data file is in the `data/` directory
- **Import errors**: Run `pip install -r requirements.txt` again
- **Port already in use**: Streamlit will automatically use the next available port
- **Data format issues**: Ensure your CSV matches the expected column structure
