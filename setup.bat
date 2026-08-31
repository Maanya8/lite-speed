@echo off
setlocal

set "SPEED_DATASET_URL=https://zenodo.org/records/6327547/files/speed.zip?download=1"
set "SPEED_DATASET_ZIP=%CD%\speed.zip"

if not exist images\train if not exist speed\images\train (
    echo Downloading SPEED dataset...
    powershell -NoProfile -ExecutionPolicy Bypass -Command "$ErrorActionPreference = 'Stop'; Invoke-WebRequest -Uri '%SPEED_DATASET_URL%' -OutFile '%SPEED_DATASET_ZIP%'; Expand-Archive -LiteralPath '%SPEED_DATASET_ZIP%' -DestinationPath '%CD%' -Force"
    if errorlevel 1 (
        echo Failed to download or extract the SPEED dataset.
        if exist "%SPEED_DATASET_ZIP%" del /q "%SPEED_DATASET_ZIP%"
        exit /b 1
    )
    del /q "%SPEED_DATASET_ZIP%"
) else (
    echo SPEED dataset already exists. Skipping download.
)

if not exist .venv (
    py -3 -m venv .venv
)

call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

echo.
echo Environment ready.
echo Activate it later with: call .venv\Scripts\activate.bat

endlocal