#!/bin/bash

# Set model path to use saved PT model
MODEL_PATH="paligemma-weights/paligemma_model2.pt"
PROMPT="What objects are present? Where are they located? What are they doing? What is their appearance? What is the setting or background?"
IMAGE_FILE_PATH="test_images/sample2.jpg"
MAX_TOKENS_TO_GENERATE=100
TEMPERATURE=0.8
TOP_P=0.9
DO_SAMPLE="True"
ONLY_CPU="False"

python inference.py \
    --model_path "$MODEL_PATH" \
    --prompt "$PROMPT" \
    --image_file_path "$IMAGE_FILE_PATH" \
    --max_tokens_to_generate $MAX_TOKENS_TO_GENERATE \
    --temperature $TEMPERATURE \
    --top_p $TOP_P \
    --do_sample $DO_SAMPLE \
    --only_cpu $ONLY_CPU \
    --use_saved_model True