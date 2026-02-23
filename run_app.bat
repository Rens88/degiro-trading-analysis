@echo off
setlocal EnableExtensions

title DEGIRO Trading Analysis Launcher

set "FAIL_STEP="
set "APP_EXIT=0"
set "ROOT=%~dp0"
set "SCRIPT=%~f0"

echo ============================================================
echo DEGIRO Trading Analysis launcher
echo Script : "%SCRIPT%"
echo Folder : "%ROOT%"
echo Started: %DATE% %TIME%
echo ============================================================
echo.

echo [STEP] Change directory to app folder...
cd /d "%ROOT%"
if errorlevel 1 goto :cd_error
echo [OK] Working directory: "%CD%"
echo.

echo [STEP] Detect Python 3...
set "PYTHON="
if exist ".venv\Scripts\python.exe" (
    set "PYTHON=.venv\Scripts\python.exe"
    echo [OK] Found project venv Python: ".venv\Scripts\python.exe"
) else (
    echo [INFO] No existing .venv detected. Looking for Python on PATH...
    where python >nul 2>&1
    if not errorlevel 1 (
        set "PYTHON=python"
        echo [OK] Using Python from PATH: python
    ) else (
        py -3 -c "import sys" >nul 2>&1
        if not errorlevel 1 (
            set "PYTHON=py -3"
            echo [OK] Using Python launcher: py -3
        ) else (
            set "FAIL_STEP=detect_python"
            echo [ERROR] Could not find Python 3.
            echo [ERROR] Install Python 3 and verify that either "python" or "py -3" works.
            goto :fail
        )
    )
)
if not defined PYTHON (
    set "FAIL_STEP=detect_python"
    echo [ERROR] Could not resolve a usable Python interpreter.
    goto :fail
)
%PYTHON% -c "import sys; print('[DEBUG] Executable:', sys.executable); print('[DEBUG] Version   :', sys.version.splitlines()[0])"
if errorlevel 1 (
    set "FAIL_STEP=python_details"
    goto :fail
)
echo.

echo [STEP] Ensure virtual environment exists...
if not exist ".venv\Scripts\python.exe" (
    echo [INFO] .venv not found, creating it...
    %PYTHON% -m venv ".venv"
    if errorlevel 1 (
        set "FAIL_STEP=create_venv"
        goto :fail
    )
) else (
    echo [OK] Existing venv found at ".venv\Scripts\python.exe"
)
echo.

echo [STEP] Activate virtual environment...
call ".venv\Scripts\activate.bat"
if errorlevel 1 (
    set "FAIL_STEP=activate_venv"
    goto :fail
)
where python
python --version
if errorlevel 1 (
    set "FAIL_STEP=verify_active_python"
    goto :fail
)
python -c "import sys; print('[DEBUG] Active exe :', sys.executable)"
if errorlevel 1 (
    set "FAIL_STEP=active_python_details"
    goto :fail
)
echo.

echo [STEP] Install requirements...
if not exist "requirements.txt" (
    set "FAIL_STEP=requirements_missing"
    echo [ERROR] requirements.txt not found in "%CD%".
    goto :fail
)
python -m pip --version
if errorlevel 1 (
    set "FAIL_STEP=pip_missing"
    goto :fail
)
python -m pip install -r requirements.txt --quiet
if errorlevel 1 (
    set "FAIL_STEP=pip_install"
    goto :fail
)
echo.

echo [STEP] Verify Streamlit import...
python -c "import streamlit as st; print('[DEBUG] streamlit :', st.__version__)"
if errorlevel 1 (
    set "FAIL_STEP=streamlit_import"
    goto :fail
)
echo.

echo [STEP] Start app...
echo [INFO] Command: python -m streamlit run app.py
python -m streamlit run app.py
set "APP_EXIT=%errorlevel%"
echo.
echo [INFO] Streamlit process exited with code: %APP_EXIT%
if not "%APP_EXIT%"=="0" (
    set "FAIL_STEP=run_streamlit"
    goto :fail_with_code
)

echo [OK] Launcher completed without reported errors.
goto :final_pause

:cd_error
set "FAIL_STEP=cd_project_root"
echo [ERROR] Could not change directory to "%ROOT%".
goto :fail

:fail_with_code
echo [ERROR] Step "%FAIL_STEP%" failed with exit code %APP_EXIT%.
goto :fail_footer

:fail
echo [ERROR] Step "%FAIL_STEP%" failed.
:fail_footer
echo.
echo Troubleshooting tips:
echo  1. Delete ".venv" and run this script again.
echo  2. Ensure internet access is available for pip install.
echo  3. Ensure Python 3 is installed and accessible as "python" or "py -3".
echo  4. Run this script from Command Prompt to copy full logs.

:final_pause
echo.
echo Finished: %DATE% %TIME%
echo Press any key to close this window...
pause >nul

endlocal
exit /b
