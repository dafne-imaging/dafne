@echo off
setlocal

:: Create the Dafne Windows installer using Inno Setup.
:: The installer script (dafne_win.iss) handles downloading Python,
:: creating a virtual environment, and installing Dafne via pip.
:: No PyInstaller step is needed.

:: Update the version number in the .iss file
python update_version.py

:: Compile the installer
"C:\Program Files (x86)\Inno Setup 6\Compil32.exe" /cc dafne_win.iss

if %ERRORLEVEL% neq 0 (
    echo.
    echo ERROR: Inno Setup compilation failed.
    exit /b %ERRORLEVEL%
)

echo.
echo Installer built successfully. Output is in ..\dist\
endlocal