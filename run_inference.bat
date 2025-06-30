@echo off
echo ========================================
echo    PaliGemma Inference Launcher
echo ========================================
echo.

REM Activate virtual environment and set Python path
call venv\Scripts\activate.bat
set PYTHON_PATH=venv\Scripts\python.exe

REM Set default values
set MODEL_PATH=weights\paligemma-3b-pt-224
set PROMPT=What objects are present? Where are they located? What are they doing? What is their appearance? What is the setting or background?
set IMAGE_FILE_PATH=test_images\sample2.jpg
set MAX_TOKENS=100
set TEMPERATURE=0.8
set TOP_P=0.9
set DO_SAMPLE=True
set ONLY_CPU=False

echo Current settings:
echo Model path: %MODEL_PATH%
echo Image file: %IMAGE_FILE_PATH%
echo Max tokens: %MAX_TOKENS%
echo Temperature: %TEMPERATURE%
echo Top-p: %TOP_P%
echo Do sample: %DO_SAMPLE%
echo Only CPU: %ONLY_CPU%
echo.
echo Prompt: %PROMPT%
echo.

set /p choice="Do you want to change any settings? (y/n): "
if /i "%choice%"=="y" (
    echo.
    set /p MODEL_PATH="Model path [%MODEL_PATH%]: "
    set /p IMAGE_FILE_PATH="Image file path [%IMAGE_FILE_PATH%]: "
    set /p MAX_TOKENS="Max tokens to generate [%MAX_TOKENS%]: "
    set /p TEMPERATURE="Temperature [%TEMPERATURE%]: "
    set /p TOP_P="Top-p [%TOP_P%]: "
    set /p DO_SAMPLE="Do sample (True/False) [%DO_SAMPLE%]: "
    set /p ONLY_CPU="Only CPU (True/False) [%ONLY_CPU%]: "
    echo.
    set /p PROMPT="Prompt [%PROMPT%]: "
)

echo.
echo Starting inference...
echo.

"%PYTHON_PATH%" inference.py ^
    --model_path "%MODEL_PATH%" ^
    --prompt "%PROMPT%" ^
    --image_file_path "%IMAGE_FILE_PATH%" ^
    --max_tokens_to_generate %MAX_TOKENS% ^
    --temperature %TEMPERATURE% ^
    --top_p %TOP_P% ^
    --do_sample %DO_SAMPLE% ^
    --only_cpu %ONLY_CPU%

echo.
echo Inference completed!
pause 