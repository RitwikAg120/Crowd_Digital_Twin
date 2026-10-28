@echo off
rem Commit everything that is not ignored and push the current branch.
rem   scripts\push.bat "What changed"
rem Weights, venvs, datasets, logs and results are in .gitignore, so they are never pushed.
if "%~1"=="" (
    echo Usage: scripts\push.bat "commit message"
    exit /b 1
)
git add -A
echo.
echo Files in this commit:
git status --short
echo.
git commit -m "%~1" || exit /b 1
for /f %%b in ('git branch --show-current') do git push -u origin %%b
