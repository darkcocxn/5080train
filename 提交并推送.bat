@echo off
cd /d "%~dp0"

git add .
git commit -m "Initial commit"
git push

pause
