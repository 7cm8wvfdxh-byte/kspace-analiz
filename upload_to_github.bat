@echo off
cd /d "%~dp0"
echo Pushing project to GitHub...
git push -u origin main
echo.
echo Process complete.
pause
