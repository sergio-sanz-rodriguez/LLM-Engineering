import os
import modal
from modal import App, Volume, Image

app = modal.App("pricer-service")

# Persistent volume for huggingface model cache
volume = modal.Volume.from_name("pricer-model-cache")

# Constants
GPU = "T4"
BASE_MODEL = "meta-llama/Meta-Llama-3.1-8B"
PROJECT_NAME = "pricer"
HF_USER = "ed-donner"
RUN_NAME = "2024-09-13_13.04.39"
PROJECT_RUN_NAME = f"{PROJECT_NAME}-{RUN_NAME}"
REVISION = "e8d637df551603dc86cd7a1598a8f44af4d7ae36"
FINETUNED_MODEL = f"{HF_USER}/{PROJECT_RUN_NAME}"

MODEL_DIR = "hf-cache/"
BASE_DIR = os.path.join(MODEL_DIR, BASE_MODEL)
FINETUNED_DIR = os.path.join(MODEL_DIR, FINETUNED_MODEL)

QUESTION = "How much does this cost to the nearest dollar?"
PREFIX = "Price is $"

# Build the image, install dependencies, and run model download
image = (
    Image.debian_slim()
    .pip_install("torch", "transformers", "bitsandbytes", "accelerate", "peft", "python-dotenv")
    #.run_function(download_model_to_volume, mounts={volume: "/cache"})
)

secrets = [modal.Secret.from_name("hf-secret-2")]

@app.cls(image=image, secrets=secrets, gpu=GPU, timeout=1800, volumes={"/cache": volume})
class Pricer:

    # Function to check out files inside a folder
    def dir_has_files(self, path):
        return os.path.exists(path) and len(os.listdir(path)) > 0

    # Function to download models into the persistent volume at build time
    def download_model_to_volume(self):
        from huggingface_hub import snapshot_download

        os.makedirs(BASE_DIR, exist_ok=True)
        os.makedirs(FINETUNED_DIR, exist_ok=True)

        print(f"Downloading base model to {BASE_DIR}...")
        snapshot_download(BASE_MODEL, local_dir=BASE_DIR)

        print(f"Downloading finetuned model to {FINETUNED_DIR}...")
        snapshot_download(FINETUNED_MODEL, revision=REVISION, local_dir=FINETUNED_DIR)

        print("Download complete.")

    @modal.enter()
    def setup(self):
        import os
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
        from peft import PeftModel
        from dotenv import load_dotenv

        load_dotenv(override=True)
        token = os.environ["HF_TOKEN"]

        # Download models if not already present in the mounted volume
        if not self.dir_has_files(BASE_DIR) or not self.dir_has_files(FINETUNED_DIR):
            print("Model files not found in volume. Downloading now...")
            self.download_model_to_volume()
        else:
            print("Model files found in volume. Skipping download.")

        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4"
        )

        self.tokenizer = AutoTokenizer.from_pretrained(BASE_DIR)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"

        self.base_model = AutoModelForCausalLM.from_pretrained(
            BASE_DIR,
            quantization_config=quant_config,
            device_map="auto",
            token=token,
        )

        self.fine_tuned_model = PeftModel.from_pretrained(
            self.base_model, FINETUNED_DIR, revision=REVISION
        )

    @modal.method()
    def price(self, description: str) -> float:
        import re
        import torch
        from transformers import set_seed

        set_seed(42)
        prompt = f"{QUESTION}\n\n{description}\n\n{PREFIX}"
        inputs = self.tokenizer.encode(prompt, return_tensors="pt").to("cuda")
        attention_mask = torch.ones(inputs.shape, device="cuda")
        outputs = self.fine_tuned_model.generate(
            inputs, attention_mask=attention_mask, max_new_tokens=5, num_return_sequences=1
        )
        result = self.tokenizer.decode(outputs[0])

        contents = result.split("Price is $")[1]
        contents = contents.replace(",", "")
        match = re.search(r"[-+]?\d*\.\d+|\d+", contents)
        return float(match.group()) if match else 0

    @modal.method()
    def wake_up(self) -> str:
        return "ok"
