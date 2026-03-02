@echo off
setlocal EnableExtensions

if "%DEGIRO_STARTUP_DEBUG%"=="" set "DEGIRO_STARTUP_DEBUG=1"
set "EXIT_CODE=0"
set "PAUSED_ALREADY=0"

set "ROOT=%~dp0"
cd /d "%ROOT%"
if errorlevel 1 (
    echo [ERROR] Could not change directory to "%ROOT%".
    set "EXIT_CODE=1"
    goto :finalize
)
echo [INFO] Startup script path: %~f0
echo [INFO] Working directory : %CD%

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
            echo [ERROR] Could not detect Python 3.
            set "EXIT_CODE=1"
            goto :finalize
        )
    )
)

if not exist ".venv\Scripts\python.exe" (
    %PYTHON% -m venv ".venv"
    if errorlevel 1 (
        echo [ERROR] Failed to create virtual environment.
        set "EXIT_CODE=1"
        goto :finalize
    )
)

call ".venv\Scripts\activate.bat"
if errorlevel 1 (
    echo [ERROR] Failed to activate virtual environment.
    set "EXIT_CODE=1"
    goto :finalize
)

if exist "requirements.txt" (
    python -m pip install -r requirements.txt --quiet >nul 2>&1
    if errorlevel 1 (
        echo [WARN] requirements install failed; continuing with current environment.
    )
)

if "%STRATEGY_FILE%"=="" set "STRATEGY_FILE=strategy\spread_strategy.json"
set "STRATEGY_FILE=%STRATEGY_FILE:"=%"
if defined DATASET_A_DIR set "DATASET_A_DIR=%DATASET_A_DIR:"=%"
if defined DATASET_B_DIR set "DATASET_B_DIR=%DATASET_B_DIR:"=%"
if defined CLASSIFICATION_PATH set "CLASSIFICATION_PATH=%CLASSIFICATION_PATH:"=%"

echo [INFO] Running strategy check with:
echo [INFO]   STRATEGY_FILE=%STRATEGY_FILE%
if defined DATASET_A_DIR echo [INFO]   DATASET_A_DIR=%DATASET_A_DIR%
if defined DATASET_B_DIR echo [INFO]   DATASET_B_DIR=%DATASET_B_DIR%
if defined CLASSIFICATION_PATH echo [INFO]   CLASSIFICATION_PATH=%CLASSIFICATION_PATH%

python -m src.strategy_check --strategy-file "%STRATEGY_FILE%" ^
  --dataset-a-dir "%DATASET_A_DIR%" ^
  --dataset-b-dir "%DATASET_B_DIR%" ^
  --classification-path "%CLASSIFICATION_PATH%"
set "CHECK_EXIT=%errorlevel%"
echo [INFO] Strategy check exit code: %CHECK_EXIT%

if "%CHECK_EXIT%"=="10" (
    echo.
    echo Strategy check found required/recommended action.
    echo Launching Streamlit app in a new window...
    set "DEGIRO_STARTUP_AUTORUN=1"
    set "DEGIRO_STARTUP_STRATEGY_FILE=%STRATEGY_FILE%"
    if defined DATASET_A_DIR set "DEGIRO_STARTUP_DATASET_A_DIR=%DATASET_A_DIR%"
    if defined DATASET_B_DIR set "DEGIRO_STARTUP_DATASET_B_DIR=%DATASET_B_DIR%"
    if defined CLASSIFICATION_PATH set "DEGIRO_STARTUP_CLASSIFICATION_PATH=%CLASSIFICATION_PATH%"
    if exist "%ROOT%run_app.bat" (
        start "DEGIRO Trading Analysis" "%ROOT%run_app.bat"
    ) else (
        start "DEGIRO Trading Analysis" cmd /k "cd /d ""%ROOT%"" && call "".venv\Scripts\activate.bat"" && python -m streamlit run src/app.py"
    )
    echo If Streamlit does not open, check the new window for errors.
    echo Press any key to close this strategy-check window...
    pause < CON >nul
    if errorlevel 1 timeout /t 30 /nobreak >nul
    set "PAUSED_ALREADY=1"
    set "EXIT_CODE=10"
    goto :finalize
)

if "%CHECK_EXIT%"=="11" (
    echo.
    echo [WARN] Data quality warning detected in source portfolio export(s).
    echo [WARN] See the warning lines above for the exact Portfolio.csv file path(s).
    echo [WARN] Recommendation: re-download the listed Portfolio.csv file(s), then rerun this script.
    echo Press any key to close this strategy-check window...
    pause < CON >nul
    if errorlevel 1 timeout /t 30 /nobreak >nul
    set "PAUSED_ALREADY=1"
    set "EXIT_CODE=11"
    goto :finalize
)

if "%CHECK_EXIT%"=="0" (
    echo [INFO] No action required; Streamlit app not launched.
    set "EXIT_CODE=0"
    goto :finalize
)

echo [WARN] Strategy check did not return launch code 10; Streamlit app not launched.
if "%CHECK_EXIT%"=="2" echo [WARN] Exit 2: no dataset could be loaded.
if "%CHECK_EXIT%"=="3" echo [WARN] Exit 3: user-facing validation/input error.
if "%CHECK_EXIT%"=="4" echo [WARN] Exit 4: unexpected runtime error.
set "EXIT_CODE=%CHECK_EXIT%"

:finalize
if "%DEGIRO_STARTUP_DEBUG%"=="1" (
    if not "%PAUSED_ALREADY%"=="1" (
        echo.
        echo [DEBUG] Exiting run_strategy_check_startup.bat with code %EXIT_CODE%.
        echo [DEBUG] Set DEGIRO_STARTUP_DEBUG=0 to disable this pause.
        echo Press any key to close this window...
        pause < CON >nul
        if errorlevel 1 timeout /t 30 /nobreak >nul
    )
)

exit /b %EXIT_CODE%
