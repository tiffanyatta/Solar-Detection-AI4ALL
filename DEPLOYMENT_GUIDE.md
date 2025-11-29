# Deployment Guide for Solar Detection App

## 🌐 Accessing the App Locally

The app is already running! When you ran `streamlit run app.py`, you should see:

```
Local URL: http://localhost:8501
Network URL: http://192.168.12.241:8501
```

### To Access:
1. **Local access:** Open your browser and go to `http://localhost:8501`
2. **Network access:** Others on your network can use `http://192.168.12.241:8501` (your IP address)

### To Restart the App:
If you closed the terminal, just run:
```bash
streamlit run app.py
```

---

## 🚀 Deploying to the Web (Free Options)

### Option 1: Streamlit Cloud (Recommended - Easiest & Free)

**Best for:** Quick deployment, free hosting, automatic updates

#### Steps:

1. **Push your code to GitHub:**
   ```bash
   git init
   git add .
   git commit -m "Initial commit - Solar Detection App"
   git branch -M main
   git remote add origin <your-github-repo-url>
   git push -u origin main
   ```

2. **Go to Streamlit Cloud:**
   - Visit: https://streamlit.io/cloud
   - Click "Sign up" (free)
   - Sign in with your GitHub account

3. **Deploy:**
   - Click "New app"
   - Select your repository
   - Set main file path: `app.py`
   - Click "Deploy"
   - Wait 2-3 minutes for deployment

4. **Your app will be live at:**
   - `https://your-app-name.streamlit.app`

**Note:** Make sure your `data/solar_data.csv` is in the repository, or the app will need to download it.

---

### Option 2: Heroku (Free Tier Discontinued, but Paid Options Available)

**Best for:** More control, custom domains

#### Steps:

1. **Install Heroku CLI:**
   - Download from: https://devcenter.heroku.com/articles/heroku-cli

2. **Create files:**

   **`Procfile`** (create in root directory):
   ```
   web: streamlit run app.py --server.port=$PORT --server.address=0.0.0.0
   ```

   **`setup.sh`** (create in root directory):
   ```bash
   mkdir -p ~/.streamlit/
   echo "\
   [server]\n\
   headless = true\n\
   port = $PORT\n\
   enableCORS = false\n\
   \n\
   " > ~/.streamlit/config.toml
   ```

3. **Deploy:**
   ```bash
   heroku login
   heroku create your-app-name
   git push heroku main
   ```

---

### Option 3: Render.com (Free Tier Available)

**Best for:** Easy deployment, free tier

#### Steps:

1. **Sign up:** https://render.com
2. **Create new Web Service**
3. **Connect your GitHub repository**
4. **Settings:**
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `streamlit run app.py --server.port=$PORT --server.address=0.0.0.0`
5. **Deploy**

---

### Option 4: Railway (Free Trial)

**Best for:** Simple deployment

1. **Sign up:** https://railway.app
2. **New Project** → **Deploy from GitHub**
3. **Select repository**
4. **Add environment variable:** `PORT=8501`
5. **Deploy**

---

## 📋 Pre-Deployment Checklist

Before deploying, make sure:

- [ ] All dependencies are in `requirements.txt`
- [ ] `data/solar_data.csv` is in the repository (or update code to download it)
- [ ] No hardcoded paths (use relative paths)
- [ ] Test the app locally first
- [ ] Code is pushed to GitHub

---

## 🔧 Important Files for Deployment

### For Streamlit Cloud:
- ✅ `app.py` (main file)
- ✅ `requirements.txt` (dependencies)
- ✅ `data/solar_data.csv` (training data)
- ✅ `data_processing.py` (data functions)
- ✅ `model_trainer.py` (model functions)

### Optional but Recommended:
- `.gitignore` (exclude `models/` folder - models will be retrained)
- `README.md` (project documentation)

---

## 🎯 Recommended: Streamlit Cloud

**Why Streamlit Cloud is best:**
- ✅ Completely free
- ✅ No credit card required
- ✅ Automatic deployments from GitHub
- ✅ Easy to update (just push to GitHub)
- ✅ Built specifically for Streamlit apps
- ✅ HTTPS by default
- ✅ Custom subdomain

### Quick Streamlit Cloud Setup:

1. **Create GitHub repo** (if you haven't):
   ```bash
   git init
   git add .
   git commit -m "Solar Detection App"
   # Create repo on GitHub, then:
   git remote add origin https://github.com/yourusername/solar-detection-app.git
   git push -u origin main
   ```

2. **Deploy on Streamlit Cloud:**
   - Go to: https://share.streamlit.io
   - Sign in with GitHub
   - Click "New app"
   - Select repo and branch
   - Main file: `app.py`
   - Click "Deploy"

3. **Done!** Your app will be live in 2-3 minutes.

---

## 🔒 Security Notes

- Don't commit API keys or secrets
- The app is public by default on Streamlit Cloud
- For private apps, you may need paid plans

---

## 📱 Sharing Your App

Once deployed, you can:
- Share the URL with anyone
- Embed it in websites
- Use it on mobile devices
- No installation needed for users!

---

## 🆘 Troubleshooting Deployment

**App won't start:**
- Check `requirements.txt` has all dependencies
- Verify `app.py` is the main file
- Check logs in deployment platform

**Model not found:**
- Make sure `data/solar_data.csv` is in the repo
- Or update code to download data automatically

**Import errors:**
- Verify all files are in the repository
- Check `requirements.txt` versions

---

## 💡 Quick Start: Deploy to Streamlit Cloud Now

1. **Push to GitHub:**
   ```bash
   git add .
   git commit -m "Ready for deployment"
   git push
   ```

2. **Deploy:**
   - Visit: https://share.streamlit.io
   - Sign in with GitHub
   - Click "New app"
   - Select your repo
   - Main file: `app.py`
   - Deploy!

3. **Share the URL!** 🎉

