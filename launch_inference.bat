@echo off
REM Windows batch file for launching PaliGemma inference
REM Based on launch_inference.sh

REM Set model path (relative to current directory)
set MODEL_PATH=weights\paligemma_model.pt

REM Set default parameters
set PROMPT=What objects are present? Where are they located? What are they doing? What is their appearance? What is the setting or background?
set IMAGE_FILE_PATH=test_images\sample2.jpg
set MAX_TOKENS_TO_GENERATE=100
set TEMPERATURE=0.8
set TOP_P=0.9
set DO_SAMPLE=True
set ONLY_CPU=False
set USE_SAVED_MODEL=True

REM Activate virtual environment and set Python path
call venv\Scripts\activate.bat
set PYTHON_PATH=venv\Scripts\python.exe

REM Run inference
"%PYTHON_PATH%" inference.py ^
    --model_path "%MODEL_PATH%" ^
    --prompt "%PROMPT%" ^
    --image_file_path "%IMAGE_FILE_PATH%" ^
    --max_tokens_to_generate %MAX_TOKENS_TO_GENERATE% ^
    --temperature %TEMPERATURE% ^
    --top_p %TOP_P% ^
    --do_sample %DO_SAMPLE% ^
    --only_cpu %ONLY_CPU% ^
    --use_saved_model %USE_SAVED_MODEL%

REM Pause to see the output
pause 