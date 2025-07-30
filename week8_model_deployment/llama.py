import modal
from modal import App, Volume, Image
import os

# Setup

app = modal.App("llama")
#image = Image.debian_slim().pip_install("torch", "transformers", "bitsandbytes", "accelerate")
image = (
    Image.debian_slim()
    .pip_install("torch", "transformers", "bitsandbytes", "accelerate")
    .add_local_python_source("llama")  # 👈 Explicitly add your llama.py
)

secrets = [modal.Secret.from_name("hf-secret-2")] # sergio-sanz-rod
GPU = "T4"
MODEL_NAME = "meta-llama/Meta-Llama-3.1-8B" # "google/gemma-2-2b"

from dotenv import load_dotenv
load_dotenv(override=True)
token = os.environ["HF_TOKEN"]


@app.function(image=image, secrets=secrets, gpu=GPU, timeout=1800)
def generate(prompt: str) -> str:
    import os
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, set_seed

    # Quant Config
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4"
    )

    # Load model and tokenizer
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, 
        quantization_config=quant_config,
        device_map="auto",
        token=token,
    )

    set_seed(42)
    inputs = tokenizer.encode(prompt, return_tensors="pt").to("cuda")
    attention_mask = torch.ones(inputs.shape, device="cuda")
    outputs = model.generate(inputs, attention_mask=attention_mask, max_new_tokens=5, num_return_sequences=1)
    return tokenizer.decode(outputs[0])
