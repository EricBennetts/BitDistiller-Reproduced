from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Define the path to your LLaMA model folder
model_path = "llama"  # <--- CHANGE THIS to the actual path of your "llama" folder

# --- 1. Load the Tokenizer ---
try:
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    print("Tokenizer loaded successfully.")
except Exception as e:
    print(f"Error loading tokenizer: {e}")
    exit()

# --- 2. Load the Model ---
try:
    # Check if CUDA is available and use it, otherwise use CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Determine recommended torch_dtype based on CUDA bfloat16 support
    if device.type == 'cuda' and torch.cuda.is_bf16_supported():
        torch_dtype = torch.bfloat16
        print("CUDA supports bfloat16. Using torch.bfloat16 for model loading.")
    else:
        torch_dtype = torch.float16 # Fallback for GPUs not supporting bfloat16 or for CPU
        print("CUDA does not support bfloat16 or using CPU. Using torch.float16 for model loading.")

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch_dtype,  # Load in bfloat16 if available for better precision, or float16
        low_cpu_mem_usage=True,   # Optimizes memory usage during loading, especially for large models
        # device_map="auto" # Optional: if you have multiple GPUs and want to automatically distribute
    )
    model.to(device) # Ensure the model is on the desired device
    model.eval()     # Set the model to evaluation mode (important for consistent PPL results)
    print(f"Model loaded successfully and moved to {device}.")
    print(f"Model memory footprint: {model.get_memory_footprint() / 1e9:.2f} GB")

except Exception as e:
    print(f"Error loading model: {e}")
    print("Make sure you have enough VRAM/RAM for the model.")
    print("For a 1B model in float16/bfloat16, you'll need at least ~2GB VRAM/RAM for weights, plus more for activations.")
    exit()

# Now you have 'model' and 'tokenizer' loaded and ready.
# You can proceed with the rest of your script:

# --- Your PPL Evaluation Function (Placeholder - you need to implement this) ---
def get_wikitext2_ppl(model, tokenizer, dataset_name="wikitext", dataset_config_name="wikitext-2-raw-v1", stride=512, max_length=1024):
    """
    Calculates perplexity on the WikiText-2 dataset.
    This is a basic implementation. For rigorous results, consider existing PPL evaluation scripts.
    """
    from datasets import load_dataset
    import torch
    from tqdm import tqdm

    print(f"Loading {dataset_name} - {dataset_config_name} for PPL evaluation...")
    try:
        # test_dataset = load_dataset(dataset_name, dataset_config_name, split="test") # usually for wikitext-2
        # For wikitext-2-raw-v1, the 'test' split is what you want for perplexity
        test_dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    except Exception as e:
        print(f"Failed to load wikitext-2 dataset: {e}")
        print("Make sure you have an internet connection or the dataset is cached.")
        print("You might need to install the 'datasets' library: pip install datasets")
        return float('inf')

    print("Tokenizing dataset...")
    encodings = tokenizer("\n\n".join(test_dataset["text"]), return_tensors="pt")

    nlls = []
    seq_len = encodings.input_ids.size(1)

    print(f"Calculating perplexity with max_length={max_length}, stride={stride}...")
    for i in tqdm(range(0, seq_len - 1, stride)):
        begin_loc = max(i + stride - max_length, 0)
        end_loc = min(i + stride, seq_len)
        trg_len = end_loc - i  # may be different from stride on last loop
        
        input_ids = encodings.input_ids[:, begin_loc:end_loc].to(model.device)
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100 # We only care about the last trg_len tokens

        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)
            neg_log_likelihood = outputs.loss * trg_len # loss is already averaged

        nlls.append(neg_log_likelihood)
        if i > 0 and i % 10 == 0: # print intermediate ppl to see progress
            current_ppl = torch.exp(torch.stack(nlls).sum() / end_loc)
            print(f"Intermediate PPL at token {end_loc}: {current_ppl.item()}")


    ppl = torch.exp(torch.stack(nlls).sum() / (seq_len-1)) # (seq_len-1) because we predict seq_len-1 tokens
    print(f"Final PPL: {ppl.item()}")
    return ppl.item()


# --- (The rest of your script from the previous message) ---
# ... (store original weights, apply pseudo-quantization, evaluate PPL, restore weights) ...
# Make sure `quantizer.py` is in your Python path or the same directory.
import sys
sys.path.append('quantization/')
from quantizer import pseudo_quantize_tensor # Assuming quantizer.py is accessible

# --- Store original weights (optional, but good practice) ---
original_weights = {}
for name, module in model.named_modules():
    if isinstance(module, torch.nn.Linear): # Use torch.nn.Linear for type checking
        original_weights[name] = module.weight.data.clone()

# --- Apply pseudo-quantization ---
q_config = {
    "n_bit": 2,
    "zero_point": True, # Asymmetric
    "q_group_size": 128 # Or your chosen group size
}

print("\nStarting pseudo-quantization...")
for name, module in model.named_modules():
    if isinstance(module, torch.nn.Linear):
        # print(f"Pseudo-quantizing weights for: {name}") # Can be very verbose
        quantized_weight = pseudo_quantize_tensor(
            module.weight.data.clone(), # Work on a clone
            n_bit=q_config["n_bit"],
            zero_point=q_config["zero_point"],
            q_group_size=q_config["q_group_size"]
        )
        module.weight.data = quantized_weight # Temporarily assign
print("Pseudo-quantization finished.")

# --- Evaluate PPL ---
print("\nEvaluating PPL after naive 2-bit quantization...")
ppl_after_naive_2bit_quant = get_wikitext2_ppl(model, tokenizer)
print(f"PPL after naive 2-bit asymmetric quantization: {ppl_after_naive_2bit_quant}")

# --- Restore original weights (if needed for subsequent steps) ---
print("\nRestoring original weights...")
for name, module in model.named_modules():
    if isinstance(module, torch.nn.Linear) and name in original_weights:
        module.weight.data = original_weights[name]
print("Original weights restored.")