@echo off
rem KeyCall viewer launcher (Windows). Path-safe and interpreter-safe.
cd /d "%~dp0"

rem Resolve the interpreter explicitly.
python --version >nul 2>&1
if errorlevel 1 (
    echo error: Python not found. Install Python 3.10+ from https://www.python.org/downloads/ ^(check "Add python.exe to PATH" during install^), then run this script again.
    exit /b 1
)

rem python exists but may be older than KeyCall's floor.
python -c "import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)" >nul 2>&1
if errorlevel 1 (
    echo error: found an older Python, but KeyCall needs Python 3.10+. Install a newer Python from https://www.python.org/downloads/, then run this script again.
    exit /b 1
)

rem Validate an existing venv by running its own interpreter.
if exist .venv\Scripts\python.exe (
    .venv\Scripts\python.exe --version >nul 2>&1
    if errorlevel 1 (
        echo stale virtual environment, rebuilding...
        rmdir /s /q .venv
    )
)
if not exist .venv\Scripts\python.exe (
    echo creating virtual environment...
    python -m venv .venv
    .venv\Scripts\python.exe -m ensurepip --upgrade >nul 2>&1
)

.venv\Scripts\python.exe -m pip install -q -e .
if errorlevel 1 (
    echo error: install failed
    exit /b 1
)

set "SOURCE=%~1"
if "%SOURCE%"=="" (
    if exist keycall-keys.toml set "SOURCE=keycall-keys.toml"
)
if "%SOURCE%"=="" (
    if exist keycall-test-keys.toml set "SOURCE=keycall-test-keys.toml"
)
if "%SOURCE%"=="" (
    if exist internal\keycall-test-keys.toml set "SOURCE=internal\keycall-test-keys.toml"
)
if "%SOURCE%"=="" (
    rem No key file found - the viewer opens with a prompt to load one.
    .venv\Scripts\python.exe -m keycall._cli view
    exit /b %errorlevel%
)

.venv\Scripts\python.exe -m keycall._cli view --source "%SOURCE%"
