@echo off
if "%GROQ_API_KEY%"=="" (
  echo GROQ_API_KEY is not set.
  echo Please set it in your shell before running this script.
  exit /b 1
)
echo Starting AGNN with Groq API key from environment...
python -m agnn --dashboard
