@echo off
if "%GEMINI_API_KEY%"=="" (
  echo GEMINI_API_KEY is not set.
  echo Please set it in your shell before running this script.
  exit /b 1
)
echo Starting AGNN with Gemini API key from environment...
python -m agnn --dashboard
