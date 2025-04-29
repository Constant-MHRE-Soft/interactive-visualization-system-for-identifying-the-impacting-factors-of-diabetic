import subprocess
import sys
import os

def install_requirements():
    required = [
        "streamlit", "pandas", "numpy", "seaborn", "matplotlib",
        "scikit-learn", "openpyxl", "xlrd"
    ]
    for package in required:
        subprocess.call([sys.executable, "-m", "pip", "install", package])

def run_app():
    app_file = "interactive_diabetes_app.py"
    if os.path.exists(app_file):
        subprocess.call(["streamlit", "run", app_file])
    else:
        print(f"❌ Could not find {app_file}")

if __name__ == "__main__":
    print("🔧 Installing dependencies...")
    install_requirements()
    print("🚀 Launching the app...")
    run_app()
