@echo off
cd /d "%~dp0"

git add .
git commit -m "Initial commit"
git push 5080train master

pause
