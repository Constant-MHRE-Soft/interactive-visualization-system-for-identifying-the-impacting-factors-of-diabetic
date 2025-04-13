@echo off
echo.
echo 🚀 Launching Diabetes Visualization System...
echo.

REM Check if pip is available
where pip >nul 2>nul
IF %ERRORLEVEL% NEQ 0 (
    echo ❌ pip is not installed. Please install Python and pip first.
    pause
    exit /b
)

REM Install required packages
echo 🔧 Installing dependencies...
pip install streamlit pandas numpy matplotlib seaborn scikit-learn

REM Run the Streamlit app
echo.
echo ✅ Starting app in your browser...
streamlit run diabetes_visualizer.py

pause
