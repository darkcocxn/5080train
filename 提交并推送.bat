@echo off
cd /d "%~dp0"

git add .
git commit -m "5/30.1"
git push origin master

pause
