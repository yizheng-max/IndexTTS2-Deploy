@echo off
chcp 65001 >nul
title IndexTTS2.0 Launcher

cd /d "D:\AI_TTS_Workspace\IndexTTS2_Debug"

if not exist outputs mkdir outputs

set HF_HOME=%CD%\checkpoints
set TRANSFORMERS_CACHE=%CD%\tf_download
set HF_HUB_CACHE=%CD%\checkpoints\hub
set TORCH_HOME=D:\AI_TTS_Workspace\cache\torch
set XDG_CACHE_HOME=D:\AI_TTS_Workspace\cache

set DS_BUILD_AIO=0
set DS_BUILD_SPARSE_ATTN=0
set XFORMERS_FORCE_DISABLE_TRITON=1

set PYTHON_PATH=%CD%\py312
set PATH=%PYTHON_PATH%;%PYTHON_PATH%\Scripts;%PATH%

echo ============================================
echo   IndexTTS2.0 WebUI
echo ============================================
echo.
echo Starting WebUI... please wait 20-30 seconds
echo Browser: http://127.0.0.1:7860
echo.
"%PYTHON_PATH%\python.exe" -s app.py
echo.
echo Service stopped.
pause
