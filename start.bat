@echo off
title IMC Trading Strategy
echo.
echo  ================================
echo   IMC Trading Strategy Runner
echo  ================================
echo.
cd /d "%~dp0"

set PYTHONPATH=F:\envs\pip;%PYTHONPATH%

echo  Running trading strategy...
echo  Press Ctrl+C to stop
echo.
python imc_trading_strategy.py
pause
