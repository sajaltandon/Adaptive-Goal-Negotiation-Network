@echo off
:: ============================================================
::  AGNN 3.0 — Unified Launcher
::  Loads Groq + Gemini + LM Studio all at once.
::  Keys are read from .env in this folder automatically.
::  Just run:  .\run_agnn.bat
:: ============================================================

:: Load API keys from .env if not already in environment
if exist "%~dp0.env" (
    for /f "usebackq tokens=1,* delims==" %%A in ("%~dp0.env") do (
        set "line=%%A"
        if not "!line:~0,1!"=="#" (
            if not "%%A"=="" if not "%%B"=="" (
                if "!%%A!"=="" set "%%A=%%B"
            )
        )
    )
    setlocal enabledelayedexpansion
    for /f "usebackq tokens=1,* delims==" %%A in ("%~dp0.env") do (
        set "line=%%A"
        if not "!line:~0,1!"=="#" (
            if not "%%A"=="" if not "%%B"=="" (
                set "%%A=%%B"
            )
        )
    )
)

:: Print which providers are active
echo.
echo  ╔══════════════════════════════════════════════════╗
echo  ║       AGNN 3.0  —  Hybrid AI Engine             ║
echo  ╚══════════════════════════════════════════════════╝
echo.

if not "%GROQ_API_KEY%"=="" (
    echo  [+] Groq API        ACTIVE
) else (
    echo  [-] Groq API        NOT SET  (add GROQ_API_KEY to .env^)
)

if not "%GEMINI_API_KEY%"=="" (
    echo  [+] Gemini API      ACTIVE
) else (
    echo  [-] Gemini API      NOT SET  (add GEMINI_API_KEY to .env^)
)

echo  [?] LM Studio       Auto-detect at runtime
echo.
echo  Usage flags (optional^):
echo    .\run_agnn.bat                  — all providers
echo    python -m agnn --no-gemini      — skip Gemini (faster startup^)
echo    python -m agnn --no-groq        — local models only
echo.

:: Require at least one cloud key OR let it run with just LM Studio
if "%GROQ_API_KEY%"=="" if "%GEMINI_API_KEY%"=="" (
    echo  [!] WARNING: No cloud API keys found.
    echo      AGNN will run with LM Studio only.
    echo      Add GROQ_API_KEY or GEMINI_API_KEY to .env for cloud models.
    echo.
)

python -m agnn --dashboard %*
