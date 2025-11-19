@echo off
echo ===============================
echo Creating virtual environment...
echo ===============================

python -m venv football_env

echo.
echo ===============================
echo Activating virtual environment...
echo ===============================

call football_env\Scripts\activate

echo.
echo ===============================
echo Upgrading pip...
echo ===============================

pip install --upgrade pip

echo.
echo ===============================
echo Installing required packages...
echo ===============================

pip install ^
    ultralytics ^
    supervision ^
    opencv-python ^
    numpy ^
    matplotlib ^
    pandas ^
    scikit-learn

echo.
echo ===============================
echo Setup completed!
echo Virtual environment: football_env
echo To activate later, run:
echo     football_env\Scripts\activate
echo ===============================
pause
