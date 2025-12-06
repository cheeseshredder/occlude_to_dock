"""
Test if Grounding DINO can load and run inference.
"""

import torch
import numpy as np
from PIL import Image

print("Testing Grounding DINO...")
print(f"CUDA Available: {torch.cuda.is_available()}")

try:
    from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
    
    model_name = "IDEA-Research/grounding-dino-base"
    print(f"\nLoading model: {model_name}")
    
    processor = AutoProcessor.from_pretrained(model_name)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(model_name)
    model.eval()
    
    if torch.cuda.is_available():
        model = model.to('cuda')
        print("✓ Model loaded on GPU")
    else:
        print("⚠ Model loaded on CPU (will be slow)")
    
    # Test inference on dummy image
    dummy_image = Image.fromarray(np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8))
    text_prompt = "a charging dock"
    
    print(f"\nTesting inference with prompt: '{text_prompt}'")
    
    inputs = processor(images=dummy_image, text=text_prompt, return_tensors="pt")
    if torch.cuda.is_available():
        inputs = {k: v.to('cuda') for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    print("✓ Inference successful!")
    print(f"  Output keys: {outputs.keys()}")
    print(f"  Logits shape: {outputs.logits.shape}")
    
    print("\n✅ Grounding DINO is working correctly!")
    
except ImportError as e:
    print(f"\n❌ Import error: {e}")
    print("Install transformers: pip install transformers>=4.30.0")
    
except Exception as e:
    print(f"\n❌ Error: {e}")
    print("Check CUDA installation and model download")
