# 🚀 Deploying to Streamlit Community Cloud

This project is configured and ready for 1-click deployment on [Streamlit Community Cloud](https://share.streamlit.io).

---

## 📋 Prerequisites
1. A free [GitHub](https://github.com) account.
2. A free [Streamlit Community Cloud](https://share.streamlit.io) account (login with GitHub).

---

## 🛠️ Step 1: Push Project to GitHub

1. Create a **New Repository** on [GitHub](https://github.com/new) (e.g. `diabetic-retinopathy-detection`).
2. Run the following commands in your terminal:

```bash
cd /Users/rimalisaac/Downloads/Diabetic_Retinopathy_Detection-main

# Commit the files
git commit -m "feat: setup project for Streamlit Cloud deployment"

# Link to your new GitHub repository (replace USERNAME and REPO_NAME)
git branch -M main
git remote add origin https://github.com/YOUR_GITHUB_USERNAME/YOUR_REPO_NAME.git
git push -u origin main
```

> **Note on Model Weights (`fold-4.h5`)**:
> `fold-4.h5` is ~71 MB. GitHub supports files up to 100 MB directly. If you don't wish to push the 71MB file to GitHub, the app includes automatic fallback logic that downloads the weights from Google Drive on first boot.

---

## ☁️ Step 2: Deploy on Streamlit Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io/) and click **"Create app"** or **"New app"**.
2. Select **"Deploy a public app from GitHub"**.
3. Fill in the fields:
   - **Repository**: `YOUR_GITHUB_USERNAME/YOUR_REPO_NAME`
   - **Branch**: `main`
   - **Main file path**: `app.py`
   - **App URL** (optional): Choose a custom subdomain (e.g. `retinopathy-ai.streamlit.app`)
4. Click **"Deploy!"** 🚀

Streamlit Cloud will automatically read `packages.txt`, install dependencies from `requirements.txt`, and launch your live application.

---

## 📦 Project Deployment Structure
```
├── app.py                  # Main Streamlit web application
├── utils.py                # Model architecture & preprocessing functions
├── requirements.txt        # Python dependencies (headless OpenCV, TensorFlow, Streamlit)
├── packages.txt            # Debian system libraries for Streamlit Cloud
├── .streamlit/
│   └── config.toml         # UI theme & server configuration
├── .gitignore              # Ignores virtualenv and cache artifacts
├── fold-4.h5               # Model weight checkpoint (71 MB)
└── DEPLOYMENT.md           # This deployment guide
```
