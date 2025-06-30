import torch
from modeling_gemma import PaliGemmaForConditionalGeneration
from utils import load_hf_model

def main():
    print("Loading model from HuggingFace weights...")
    model_path = "weights/paligemma-3b-pt-224"
    
    # GPU 사용 가능하면 GPU로
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # 모델 로드
    model, tokenizer = load_hf_model(model_path, device)
    model = model.to(device).eval()
    
    # 모델 상태 저장
    save_path = "weights/paligemma_model.pt"
    print(f"Saving model to {save_path}...")
    
    # 모델 상태 딕셔너리와 설정 저장
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': model.config,
        'tokenizer': tokenizer
    }, save_path)
    
    print("Model saved successfully!")

if __name__ == "__main__":
    main() 