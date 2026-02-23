@echo off
setlocal EnableExtensions

set "ROOT=%~dp0"
cd /d "%ROOT%"
if errorlevel 1 exit /b 1

set "PYTHON="
if exist ".venv\Scripts\python.exe" (
    set "PYTHON=.venv\Scripts\python.exe"
) else (
    where python >nul 2>&1
    if not errorlevel 1 (
        set "PYTHON=python"
    ) else (
        py -3 -c "import sys" >nul 2>&1
        if not errorlevel 1 (
            set "PYTHON=py -3"
        ) else (
            exit /b 1
        )
    )
)

if not exist ".venv\Scripts\python.exe" (
    %PYTHON% -m venv ".venv"
    if errorlevel 1 exit /b 1
)

call ".venv\Scripts\activate.bat"
if errorlevel 1 exit /b 1

if exist "requirements.txt" (
    python -m pip install -r requirements.txt --quiet >nul 2>&1
)

if "%STRATEGY_FILE%"=="" set "STRATEGY_FILE=strategy\spread_strategy.json"
set "CHECK_ARGS=--strategy-file \"%STRATEGY_FILE%\""
if defined DATASET_A_DIR set "CHECK_ARGS=%CHECK_ARGS% --dataset-a-dir \"%DATASET_A_DIR%\""
if defined DATASET_B_DIR set "CHECK_ARGS=%CHECK_ARGS% --dataset-b-dir \"%DATASET_B_DIR%\""
if defined MAPPINGS_PATH set "CHECK_ARGS=%CHECK_ARGS% --mappings-path \"%MAPPINGS_PATH%\""

python -m src.strategy_check %CHECK_ARGS%
set "CHECK_EXIT=%errorlevel%"

if "%CHECK_EXIT%"=="10" (
    echo.
    echo Strategy check found required/recommended action.
    echo Launching Streamlit app in a new window...
    start "DEGIRO Trading Analysis" cmd /c "cd /d \"%ROOT%\" && .venv\Scripts\python.exe -m streamlit run app.py"
    echo.
    echo Streamlit is running in a new window.
    echo Press any key to close this strategy-check window...
    pause >nul
    exit /b 10
)

if "%CHECK_EXIT%"=="0" (
    exit /b 0
)

exit /b %CHECK_EXIT%
