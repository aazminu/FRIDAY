@echo off
TITLE FRIDAY
rem Black background, light green text for the whole session. The FRIDAY art is drawn in red by src\friday.py.
color 0A
cd /d "%~dp0"
chcp 65001 >nul

rem The intro art, the Soul/Brain progress bars and the chat all run in src\friday.py.
rem (The art used to be echoed here with a new PowerShell per line for the pause, which froze
rem  the window for 20-30 seconds before anything loaded.)  Your art lives in src\friday_art.txt.
"venv\Scripts\python.exe" -u "src\friday.py" %*

echo.
echo Friday has stopped. Press any key to close this window.
pause >nul
