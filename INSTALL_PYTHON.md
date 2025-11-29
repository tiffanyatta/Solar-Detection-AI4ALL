# Installing Python on Windows

## Quick Install Guide

### Option 1: Install from python.org (Recommended)

1. **Download Python:**
   - Go to: https://www.python.org/downloads/
   - Click the big yellow "Download Python" button (downloads the latest version)

2. **During Installation:**
   - ✅ **IMPORTANT**: Check the box that says **"Add Python to PATH"**
   - Click "Install Now"
   - Wait for installation to complete

3. **Verify Installation:**
   - Close and reopen your terminal/Cursor
   - Run: `python --version`
   - You should see something like: `Python 3.11.x`

4. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

### Option 2: Install from Microsoft Store

1. Open Microsoft Store
2. Search for "Python 3.11" or "Python 3.12"
3. Click "Install"
4. After installation, close and reopen your terminal
5. Run: `pip install -r requirements.txt`

### Option 3: Use Anaconda/Miniconda (If you have it)

If you already have Anaconda or Miniconda installed:

```bash
conda install --file requirements.txt
```

Or create a new environment:
```bash
conda create -n solar-app python=3.11
conda activate solar-app
pip install -r requirements.txt
```

## After Installing Python

Once Python is installed:

1. **Close and reopen Cursor** (or your terminal)
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Get your data file:**
   - Download from: https://www.kaggle.com/datasets/saurabhshahane/northern-hemisphere-horizontal-photovoltaic
   - Save as `data/solar_data.csv`

4. **Run the app:**
   ```bash
   streamlit run app.py
   ```

## Troubleshooting

**If `pip` is not recognized:**
- Try: `python -m pip install -r requirements.txt`

**If you get permission errors:**
- Try: `pip install --user -r requirements.txt`

**If installation is slow:**
- This is normal, it may take a few minutes


