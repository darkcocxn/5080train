@echo off
cd /d "%~dp0"

git add .
git commit -m "5/30训练结果"
git push 5080train master

pause
