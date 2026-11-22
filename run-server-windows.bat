@echo off
cd /d "%~dp0"
powershell.exe -NoProfile -NonInteractive -File "%~dp0ops\start-backend.ps1"
if errorlevel 1 exit /b 1
echo Local UI: http://127.0.0.1:18765/
echo Phone UI: use the HTTPS Tailscale URL reported by ops\enable-tailnet.ps1.
