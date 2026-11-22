@echo off
setlocal
cd /d "%~dp0"

where uv >nul 2>&1
if errorlevel 1 (
  echo Installing uv for this user...
  powershell.exe -NoProfile -ExecutionPolicy Bypass -Command "irm https://astral.sh/uv/install.ps1 | iex"
  set "PATH=%USERPROFILE%\.local\bin;%USERPROFILE%\.cargo\bin;%PATH%"
)
where uv >nul 2>&1 || goto error

if not exist ".venv-desktop\Scripts\pythonw.exe" (
  echo Creating Windows desktop environment...
  uv venv --python 3.12 .venv-desktop || goto error
)

".venv-desktop\Scripts\python.exe" -c "import sounddevice,websockets,pynput,pystray,PIL" >nul 2>&1
if errorlevel 1 (
  echo Installing Windows desktop dependencies...
  uv pip install --python .venv-desktop\Scripts\python.exe -r requirements-desktop.txt || goto error
)

where ffmpeg >nul 2>&1
if errorlevel 1 (
  where winget >nul 2>&1
  if errorlevel 1 (
    echo ffmpeg is required for file transcription. Install ffmpeg and rerun this launcher.
    goto error
  )
  echo Installing ffmpeg...
  winget install --id Gyan.FFmpeg -e --accept-package-agreements --accept-source-agreements || goto error
  set "PATH=%LOCALAPPDATA%\Microsoft\WinGet\Links;%PATH%"
)

echo Preparing SODA backend and language models...
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0ops\start-backend.ps1"
if errorlevel 1 goto error

start "" ".venv-desktop\Scripts\pythonw.exe" -m desktop.app
exit /b 0

:error
echo.
echo SODA Voice Dictation setup failed. Review the error above.
pause
exit /b 1
