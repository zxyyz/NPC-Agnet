@echo off
chcp 65001 >nul 2>nul

set "ROOT=%~dp0.."
for %%I in ("%ROOT%") do set "ROOT=%%~fI"

if /i "%~1" neq "--hidden" (
    powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%ROOT%\scripts\check_environment.ps1" -Root "%ROOT%"
    if errorlevel 1 (
        echo.
        pause
        exit /b 1
    )
    powershell.exe -NoProfile -WindowStyle Hidden -Command "Start-Process -FilePath '%~f0' -ArgumentList '--hidden' -WindowStyle Hidden"
    exit /b
)

where pythonw.exe >nul 2>nul
if errorlevel 1 (
    set "PYTHONW=python.exe"
) else (
    set "PYTHONW=pythonw.exe"
)
set "PANEL=%ROOT%\src\agent_control_panel.py"

cd /d "%ROOT%"

if not exist "%PANEL%" exit /b 1

start "" "%PYTHONW%" "%PANEL%"

for /l %%I in (1,1,20) do (
    curl.exe -sf -o nul http://127.0.0.1:8090/api/status >nul 2>nul
    if not errorlevel 1 goto OPEN_UI
    timeout /t 1 /nobreak >nul
)

:OPEN_UI
start "" "http://127.0.0.1:8090/"
exit /b 0
