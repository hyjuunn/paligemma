@echo off
call venv\Scripts\activate.bat

set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python inference.py ^
    --model_path="weights/paligemma_model.pt" ^
    --prompt="What objects are present? Where are they located? What are they doing? What is their appearance? What is the setting or background?" ^
    --image_file_path="test_images" ^
    --process_all=True ^
    --max_tokens_to_generate=50 ^
    --temperature=0.8 ^
    --top_p=0.9 ^
    --do_sample=False

pause 