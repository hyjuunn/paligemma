from PIL import Image
import torch
import fire
import os

from processing_paligemma import PaliGemmaProcessor
from modeling_gemma import KVCache, PaliGemmaForConditionalGeneration
from utils import load_hf_model


def move_inputs_to_device(model_inputs: dict, device: str):
    model_inputs = {k: v.to(device) for k, v in model_inputs.items()}
    return model_inputs


def get_model_inputs(
    processor: PaliGemmaProcessor, prompt: str, image_file_path: str, device: str
):
    image = Image.open(image_file_path)
    images = [image]
    prompts = [prompt]
    model_inputs = processor(text=prompts, images=images)
    model_inputs = move_inputs_to_device(model_inputs, device)
    return model_inputs


def test_inference(
    model: PaliGemmaForConditionalGeneration,
    processor: PaliGemmaProcessor,
    device: str,
    prompt: str,
    image_file_path: str,
    max_tokens_to_generate: int,
    temperature: float,
    top_p: float,
    do_sample: bool,
):
    model_inputs = get_model_inputs(processor, prompt, image_file_path, device)
    input_ids = model_inputs["input_ids"]
    attention_mask = model_inputs["attention_mask"]
    pixel_values = model_inputs["pixel_values"]

    # 디버깅 정보 출력
    print(f"Input shape: {input_ids.shape}")
    print(f"Attention mask shape: {attention_mask.shape}")
    print(f"Pixel values shape: {pixel_values.shape}")
    
    # 입력 토큰 확인
    input_tokens = processor.tokenizer.convert_ids_to_tokens(input_ids[0])
    print(f"Input tokens (first 10): {input_tokens[:10]}")

    kv_cache = KVCache()

    # Generate tokens until you see the stop token
    stop_token = processor.tokenizer.eos_token_id
    generated_tokens = []
    
    # 최소/최대 토큰 수 설정
    min_tokens = 5
    max_tokens = min(max_tokens_to_generate, 100)  # 최대 100토큰으로 제한
    
    # 반복 감지를 위한 변수
    last_tokens = []
    repetition_threshold = 3  # 이 개수만큼 같은 토큰이 반복되면 중단
    
    # 문장 종료 토큰 목록
    sentence_end_tokens = [
        processor.tokenizer.encode('.')[0],
        processor.tokenizer.encode('!')[0],
        processor.tokenizer.encode('?')[0],
        stop_token
    ]
    
    # 콤마 카운터 (너무 많은 콤마는 중단)
    comma_token = processor.tokenizer.encode(',')[0]
    comma_count = 0
    max_commas = 5
    
    for i in range(max_tokens):
        # Get the model outputs
        outputs = model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            attention_mask=attention_mask,
            kv_cache=kv_cache,
        )
        kv_cache = outputs["kv_cache"]
        next_token_logits = outputs["logits"][:, -1, :]
        
        # EOS 토큰 확률 조정
        if len(generated_tokens) < min_tokens:
            next_token_logits[0, stop_token] = -float('inf')  # 최소 토큰 수 전에는 EOS 불가
        elif len(generated_tokens) > max_tokens // 2:
            next_token_logits[0, stop_token] *= 1.2  # 중반 이후에는 EOS 확률 증가
            
        # Sample the next token
        if do_sample:
            # Apply temperature
            next_token_logits = torch.softmax(next_token_logits / temperature, dim=-1)
            next_token = _sample_top_p(next_token_logits, top_p)
        else:
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            
        next_token = next_token.squeeze(0)  # Remove batch dimension
        token_id = next_token.item()
        generated_tokens.append(next_token)
        
        # 디버깅: 생성된 토큰 출력
        token_str = processor.tokenizer.convert_ids_to_tokens(token_id)
        print(f"Generated token {i}: {token_id} -> '{token_str}'")
        
        # 종료 조건 체크
        
        # 1. 반복 감지
        last_tokens.append(token_id)
        if len(last_tokens) > repetition_threshold:
            last_tokens.pop(0)
            if all(t == last_tokens[0] for t in last_tokens):
                print(f"Repetition detected! Stopping generation at position {i}")
                break
        
        # 2. 콤마 개수 체크
        if token_id == comma_token:
            comma_count += 1
            if comma_count >= max_commas:
                print(f"Too many commas ({comma_count})! Stopping generation.")
                break
        
        # 3. 문장 종료 토큰 체크 (최소 토큰 수 이후)
        if len(generated_tokens) >= min_tokens and token_id in sentence_end_tokens:
            print(f"Sentence end token generated at position {i}")
            break
        
        # 4. 최대 토큰 수 도달
        if i >= max_tokens - 1:
            print(f"Maximum tokens ({max_tokens}) reached!")
            break
            
        # Append the next token to the input
        input_ids = next_token.unsqueeze(-1)
        attention_mask = torch.cat(
            [attention_mask, torch.ones((1, 1), device=input_ids.device)], dim=-1
        )

    generated_tokens = torch.cat(generated_tokens, dim=-1)
    # Decode the generated tokens
    decoded = processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)

    print("\nPrompt: ", prompt)
    print("Model output: ", decoded)
    print(f"Generated {len(generated_tokens)} tokens")
    
    # 디버깅: 생성된 모든 토큰 확인
    all_tokens = processor.tokenizer.convert_ids_to_tokens(generated_tokens)
    print(f"All generated tokens: {all_tokens}")
    
    return decoded


def _sample_top_p(probs: torch.Tensor, p: float):
    # (B, vocab_size)
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    # (B, vocab_size)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    # (B, vocab_size)
    # (Substracting "probs_sort" shifts the cumulative sum by 1 position to the right before masking)
    mask = probs_sum - probs_sort > p
    # Zero out all the probabilities of tokens that are not selected by the Top P
    probs_sort[mask] = 0.0
    # Redistribute the probabilities so that they sum up to 1.
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    # Sample a token (its index) from the top p distribution
    next_token = torch.multinomial(probs_sort, num_samples=1)
    # Get the token position in the vocabulary corresponding to the sampled index
    next_token = torch.gather(probs_idx, -1, next_token)
    return next_token


def load_saved_model(model_path: str, device: str):
    print(f"Loading saved model from {model_path}...")
    # 먼저 CPU에 로드
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    
    # 모델 초기화 (CPU에서)
    model = PaliGemmaForConditionalGeneration(checkpoint['config'])
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # 메모리 정리
    del checkpoint['model_state_dict']
    torch.cuda.empty_cache()
    
    # GPU로 이동 (필요한 경우)
    if device != 'cpu':
        print("Moving model to GPU/MPS...")
        model = model.to(device)
    
    model = model.eval()
    return model, checkpoint['tokenizer']


def process_all_images(
    model: PaliGemmaForConditionalGeneration,
    processor: PaliGemmaProcessor,
    device: str,
    prompt: str,
    images_dir: str,
    max_tokens_to_generate: int,
    temperature: float,
    top_p: float,
    do_sample: bool,
):
    # 이미지 파일 목록 가져오기
    image_files = [f for f in os.listdir(images_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    image_files.sort()  # 파일 이름 순으로 정렬
    
    print("\n=== Processing all images in directory ===\n")
    
    for idx, image_file in enumerate(image_files, 1):
        image_path = os.path.join(images_dir, image_file)
        print(f"\n{idx}. Processing {image_file}")
        print("-" * 50)
        
        with torch.no_grad():
            model_inputs = get_model_inputs(processor, prompt, image_path, device)
            input_ids = model_inputs["input_ids"]
            attention_mask = model_inputs["attention_mask"]
            pixel_values = model_inputs["pixel_values"]
            
            kv_cache = KVCache()
            stop_token = processor.tokenizer.eos_token_id
            generated_tokens = []
            min_tokens = 10
            last_tokens = []
            repetition_threshold = 5
            
            for i in range(max_tokens_to_generate):
                outputs = model(
                    input_ids=input_ids,
                    pixel_values=pixel_values,
                    attention_mask=attention_mask,
                    kv_cache=kv_cache,
                )
                kv_cache = outputs["kv_cache"]
                next_token_logits = outputs["logits"][:, -1, :]
                
                if len(generated_tokens) < min_tokens:
                    next_token_logits[0, stop_token] = -float('inf')
                    
                if do_sample:
                    next_token_logits = torch.softmax(next_token_logits / temperature, dim=-1)
                    next_token = _sample_top_p(next_token_logits, top_p)
                else:
                    next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                    
                next_token = next_token.squeeze(0)
                generated_tokens.append(next_token)
                
                last_tokens.append(next_token.item())
                if len(last_tokens) > repetition_threshold:
                    last_tokens.pop(0)
                    if all(t == last_tokens[0] for t in last_tokens) and len(last_tokens) >= repetition_threshold:
                        break
                
                if next_token.item() == stop_token:
                    break
                    
                input_ids = next_token.unsqueeze(-1)
                attention_mask = torch.cat(
                    [attention_mask, torch.ones((1, 1), device=input_ids.device)], dim=-1
                )

            generated_tokens = torch.cat(generated_tokens, dim=-1)
            decoded = processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)
            print(f"Description: {decoded}\n")


def main(
    model_path: str = None,
    prompt: str = None,
    image_file_path: str = None,
    max_tokens_to_generate: int = 100,
    temperature: float = 0.8,
    top_p: float = 0.9,
    do_sample: bool = False,
    only_cpu: bool = False,
    use_saved_model: bool = True,
    process_all: bool = False,  # 모든 이미지 처리 여부
):
    device = "cpu"

    if not only_cpu:
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"

    print("Device in use: ", device)

    print(f"Loading model")
    if use_saved_model and model_path.endswith('.pt'):
        model, tokenizer = load_saved_model(model_path, device)
    else:
        model, tokenizer = load_hf_model(model_path, device)
        model = model.to(device).eval()

    num_image_tokens = model.config.vision_config.num_image_tokens
    image_size = model.config.vision_config.image_size
    processor = PaliGemmaProcessor(tokenizer, num_image_tokens, image_size)

    print("Running inference")
    with torch.no_grad():
        if process_all:
            # image_file_path를 디렉토리로 처리
            process_all_images(
                model,
                processor,
                device,
                prompt,
                image_file_path,  # 이 경우 디렉토리 경로로 사용
                max_tokens_to_generate,
                temperature,
                top_p,
                do_sample,
            )
        else:
            test_inference(
                model,
                processor,
                device,
                prompt,
                image_file_path,
                max_tokens_to_generate,
                temperature,
                top_p,
                do_sample,
            )


if __name__ == "__main__":
    fire.Fire(main)