@echo off
setlocal
cd /d "%~dp0"

echo Creating Desktop shortcut...

set "TARGET=%~dp0start_app.bat"
set "SHORTCUT=%USERPROFILE%\Desktop\AI Music Detector Visualizer.lnk"
set "ICON=%SystemRoot%\System32\shell32.dll,168"

powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "$ws = New-Object -ComObject WScript.Shell; $s = $ws.CreateShortcut('%SHORTCUT%'); $s.TargetPath = '%TARGET%'; $s.WorkingDirectory = '%~dp0'; $s.IconLocation = '%ICON%'; $s.Save()"

if exist "%SHORTCUT%" (
    echo.
    echo [SUCCESS] Desktop shortcut created:
    echo "%SHORTCUT%"
    echo.
) else (
    echo.
    echo [NOTE] Could not create desktop shortcut automatically. You can launch start_app.bat directly.
    echo.
)

pause
