from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import time
import os
import sys

# --- Configuration ---
# Path to your LLaMA model folder (already QAT-trained, as per your description)
MODEL_PATH = "train/ckpts/hf-llama-1B/int2-g128/checkpoint-12" # <--- VERIFY THIS PATH

# Generation Speed Test Parameters
PROMPT = "Once upon a time, in a land far, far away,"
NUM_TOKENS_TO_GENERATE = 200  # How many new tokens to generate for the speed test
NUM_WARMUP_RUNS = 2         # Number of untimed runs to warm up the model/GPU
NUM_TEST_RUNS = 5           # Number of timed runs to average for speed

# --- Add quantization directory to sys.path ---
# Ensure 'quantization/quantizer.py' is accessible
try:
    # Assuming 'quantization' is a subdirectory relative to this script
    # or it's already in PYTHONPATH
    quantization_module_path = 'quantization' # Or the full path if it's elsewhere
    if os.path.isdir(quantization_module_path):
         sys.path.append(os.path.abspath(os.path.join(quantization_module_path, '..'))) # Add parent of quantization
         # Or if quantizer.py is directly in 'quantization/': sys.path.append(os.path.abspath(quantization_module_path))
    from quantization.quantizer import pseudo_quantize_tensor
    print("Successfully imported pseudo_quantize_tensor.")
except ImportError as e:
    print(f"Error importing pseudo_quantize_tensor: {e}")
    print(f"Please ensure 'quantizer.py' is in a directory named 'quantization' relative to your script,")
    print(f"or adjust sys.path accordingly. Current sys.path: {sys.path}")
    exit()


def main():
    # --- 1. Load the Tokenizer ---
    print(f"Loading tokenizer from: {MODEL_PATH}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        print("Tokenizer loaded successfully.")
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        exit()

    # --- 2. Load the Model ---
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")

        if device.type == 'cuda' and hasattr(torch.cuda, 'is_bf16_supported') and torch.cuda.is_bf16_supported():
            torch_dtype = torch.bfloat16
            print("CUDA supports bfloat16. Using torch.bfloat16 for model loading.")
        else:
            torch_dtype = torch.float16
            print("CUDA does not support bfloat16 or using CPU. Using torch.float16 for model loading.")

        print(f"Loading model from: {MODEL_PATH}")
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            # device_map="auto" # Consider if you have multiple GPUs
        )
        model.to(device)
        model.eval() # Set to evaluation mode
        print(f"Model loaded successfully to {device}.")
        print(f"Model memory footprint (before quantization): {model.get_memory_footprint() / 1e9:.2f} GB")

    except Exception as e:
        print(f"Error loading model: {e}")
        exit()

    # Ensure pad_token_id is set (often same as eos_token_id if not explicitly set)
    if tokenizer.pad_token_id is None:
        print("tokenizer.pad_token_id is None, setting it to tokenizer.eos_token_id.")
        tokenizer.pad_token_id = tokenizer.eos_token_id
    # Also ensure the model's config has it, as .generate() might look there
    if model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.eos_token_id


    # --- 3. Apply pseudo-quantization (as per your original script) ---
    q_config = {
        "n_bit": 2,
        "zero_point": True, # Asymmetric
        "q_group_size": 128 # Or your chosen group size
    }

    print("\nStarting 2-bit pseudo-quantization on loaded model weights...")
    # original_weights = {} # If you need to restore later
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            # original_weights[name] = module.weight.data.clone() # Store if needed
            quantized_weight = pseudo_quantize_tensor(
                module.weight.data.clone(), # Work on a clone before assigning
                n_bit=q_config["n_bit"],
                zero_point=q_config["zero_point"],
                q_group_size=q_config["q_group_size"]
            )
            module.weight.data = quantized_weight
    print("2-bit pseudo-quantization finished.")
    # Note: get_memory_footprint() might not change significantly here as pseudo-quantized
    # weights are still stored in their original float dtype (e.g., float16/bfloat16).

    # --- 4. Test Token Generation Speed ---
    print(f"\n--- Testing Token Generation Speed with 2-bit Pseudo-Quantized Weights ---")

    # Prepare Inputs
    input_ids = tokenizer.encode(PROMPT, return_tensors="pt").to(model.device)
    num_input_tokens = input_ids.shape[1]
    print(f"\nPrompt: \"{PROMPT}\" ({num_input_tokens} tokens)")
    print(f"Attempting to generate {NUM_TOKENS_TO_GENERATE} new tokens for speed test.")

    # Warm-up Runs
    if NUM_WARMUP_RUNS > 0:
        print(f"\n--- Performing {NUM_WARMUP_RUNS} Warm-up Run(s) ---")
        for i in range(NUM_WARMUP_RUNS):
            print(f"Warm-up run {i+1}...")
            with torch.no_grad():
                _ = model.generate(
                    input_ids,
                    max_new_tokens=NUM_TOKENS_TO_GENERATE,
                    do_sample=False, # Use greedy for consistent speed test
                    pad_token_id=tokenizer.pad_token_id
                )
            if device.type == 'cuda':
                torch.cuda.synchronize()
        print("Warm-up complete.")

    # Test Runs
    print(f"\n--- Performing {NUM_TEST_RUNS} Test Run(s) ---")
    total_time_taken = 0
    total_tokens_generated = 0

    for i in range(NUM_TEST_RUNS):
        print(f"Test run {i+1}...")
        start_time = time.perf_counter()

        with torch.no_grad():
            outputs = model.generate(
                input_ids,
                max_new_tokens=NUM_TOKENS_TO_GENERATE,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id
            )

        if device.type == 'cuda':
            torch.cuda.synchronize() # Wait for GPU operations to complete

        end_time = time.perf_counter()
        time_taken = end_time - start_time

        new_tokens_generated_this_run = outputs.shape[1] - num_input_tokens
        actual_tokens_generated = min(new_tokens_generated_this_run, NUM_TOKENS_TO_GENERATE)
        
        if actual_tokens_generated <= 0:
             print(f"Warning: Run {i+1} generated 0 or fewer new tokens.")
             actual_tokens_generated = 0

        total_time_taken += time_taken
        total_tokens_generated += actual_tokens_generated

        tokens_per_second_this_run = (actual_tokens_generated / time_taken) if time_taken > 0 else 0
        print(f"Run {i+1}: Generated {actual_tokens_generated} new tokens in {time_taken:.3f} seconds ({tokens_per_second_this_run:.2f} tokens/sec).")

    # Calculate and Print Average Results
    if NUM_TEST_RUNS > 0 and total_tokens_generated > 0:
        average_time_per_run = total_time_taken / NUM_TEST_RUNS
        average_tokens_per_run = total_tokens_generated / NUM_TEST_RUNS
        average_tokens_per_second = total_tokens_generated / total_time_taken if total_time_taken > 0 else 0

        print("\n--- Average Results (2-bit Pseudo-Quantized Model) ---")
        print(f"Average new tokens generated per run: {average_tokens_per_run:.2f}")
        print(f"Average time taken per run: {average_time_per_run:.3f} seconds")
        print(f"Average tokens per second: {average_tokens_per_second:.2f} tokens/sec")
    elif NUM_TEST_RUNS == 0:
        print("\nNo test runs performed.")
    else:
        print("\nNo tokens were generated during test runs. Cannot calculate average speed.")

    # --- Optional: Restore original weights if you stored them and need to ---
    # print("\nRestoring original weights (if applicable)...")
    # for name, module in model.named_modules():
    #     if isinstance(module, torch.nn.Linear) and name in original_weights:
    #         module.weight.data = original_weights[name]
    # print("Original weights restored (if applicable).")

if __name__ == "__main__":
    # Verify MODEL_PATH points to a directory containing model files (config.json, etc.)
    # and not just a single .pth or .safetensors file.
    if not os.path.isdir(MODEL_PATH) or not os.path.exists(os.path.join(MODEL_PATH, "config.json")):
        print(f"ERROR: MODEL_PATH '{MODEL_PATH}' does not appear to be a valid Hugging Face model directory.")
        print("It should contain 'config.json', tokenizer files, and model weight files.")
        exit()
    main()