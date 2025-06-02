import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import time
import os

# --- Configuration ---
# IMPORTANT: Replace this with the actual path to your Llama model directory
# It should be the parent directory containing 'config.json', 'model.safetensors', etc.
MODEL_PATH = "llama" # e.g., "/home/user/models/SE_Proj_clip/llama/"

PROMPT = "Once upon a time, in a land far, far away,"
NUM_TOKENS_TO_GENERATE = 200  # How many new tokens to generate for the test
NUM_WARMUP_RUNS = 1         # Number of untimed runs to warm up the model/GPU
NUM_TEST_RUNS = 3           # Number of timed runs to average

# --- Sanity Check for Model Path ---
if not os.path.isdir(MODEL_PATH) or not os.path.exists(os.path.join(MODEL_PATH, "config.json")):
    print(f"ERROR: Model path '{MODEL_PATH}' does not seem to be a valid Hugging Face model directory.")
    print("Please ensure it contains 'config.json' and other model files.")
    print("The image shows a path like '/.../SE_Proj_clip/llama/'. You need to provide the full path.")
    exit()

def test_generation_speed():
    # --- Device Setup ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")

    # --- Load Tokenizer and Model ---
    print(f"\nLoading tokenizer from: {MODEL_PATH}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        return

    print(f"Loading model from: {MODEL_PATH}")
    try:
        # For faster inference and lower memory, try float16 if on GPU
        model_dtype = torch.float16 if device.type == "cuda" else torch.float32
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            torch_dtype=model_dtype,
            device_map="auto" # Automatically distribute model layers if multiple GPUs or offload to CPU
        )
    except Exception as e:
        print(f"Error loading model: {e}")
        print("If you get an out-of-memory error, try a smaller model or ensure you have enough VRAM/RAM.")
        return

    # Ensure pad_token_id is set (often same as eos_token_id if not explicitly set)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
        model.config.pad_token_id = model.config.eos_token_id
    
    print("Model and tokenizer loaded successfully.")

    # --- Prepare Inputs ---
    input_ids = tokenizer.encode(PROMPT, return_tensors="pt").to(model.device) # Move input_ids to model's device
    num_input_tokens = input_ids.shape[1]
    print(f"\nPrompt: \"{PROMPT}\" ({num_input_tokens} tokens)")
    print(f"Attempting to generate {NUM_TOKENS_TO_GENERATE} new tokens.")

    # --- Warm-up Runs ---
    if NUM_WARMUP_RUNS > 0:
        print(f"\n--- Performing {NUM_WARMUP_RUNS} Warm-up Run(s) ---")
        for i in range(NUM_WARMUP_RUNS):
            print(f"Warm-up run {i+1}...")
            _ = model.generate(
                input_ids,
                max_new_tokens=NUM_TOKENS_TO_GENERATE,
                do_sample=False, # Use greedy decoding for speed test consistency
                pad_token_id=tokenizer.pad_token_id
            )
        if torch.cuda.is_available():
            torch.cuda.synchronize() # Wait for GPU operations to complete
        print("Warm-up complete.")

    # --- Test Runs ---
    print(f"\n--- Performing {NUM_TEST_RUNS} Test Run(s) ---")
    total_time_taken = 0
    total_tokens_generated = 0

    for i in range(NUM_TEST_RUNS):
        print(f"Test run {i+1}...")
        start_time = time.perf_counter()

        with torch.no_grad(): # Disable gradient calculations for inference
            outputs = model.generate(
                input_ids,
                max_new_tokens=NUM_TOKENS_TO_GENERATE,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                # You can add other generation parameters here if needed,
                # but for a pure speed test, keep it simple.
                # e.g., temperature=0.7, top_k=50
            )
        
        if torch.cuda.is_available():
            torch.cuda.synchronize() # Wait for GPU operations to complete before stopping timer
        
        end_time = time.perf_counter()
        time_taken = end_time - start_time
        
        # The output includes the prompt tokens, so subtract them
        new_tokens_generated_this_run = outputs.shape[1] - num_input_tokens
        
        # Handle cases where fewer tokens are generated than requested (e.g., EOS token reached)
        actual_tokens_generated = min(new_tokens_generated_this_run, NUM_TOKENS_TO_GENERATE)
        if actual_tokens_generated <= 0 : # Should not happen with max_new_tokens, but good to check
            print(f"Warning: Run {i+1} generated 0 or fewer new tokens. This might indicate an issue or premature EOS.")
            actual_tokens_generated = 0 # avoid division by zero later if time_taken is also zero
        
        total_time_taken += time_taken
        total_tokens_generated += actual_tokens_generated

        tokens_per_second_this_run = (actual_tokens_generated / time_taken) if time_taken > 0 else 0
        
        print(f"Run {i+1}: Generated {actual_tokens_generated} new tokens in {time_taken:.3f} seconds ({tokens_per_second_this_run:.2f} tokens/sec).")
        
        # Optional: Decode and print the first few tokens of the output for verification
        # decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # print(f"Output sample: {decoded_output[:150]}...")


    # --- Calculate and Print Average Results ---
    if NUM_TEST_RUNS > 0 and total_tokens_generated > 0:
        average_time_per_run = total_time_taken / NUM_TEST_RUNS
        average_tokens_per_run = total_tokens_generated / NUM_TEST_RUNS
        average_tokens_per_second = total_tokens_generated / total_time_taken if total_time_taken > 0 else 0

        print("\n--- Average Results ---")
        print(f"Average new tokens generated per run: {average_tokens_per_run:.2f}")
        print(f"Average time taken per run: {average_time_per_run:.3f} seconds")
        print(f"Average tokens per second: {average_tokens_per_second:.2f} tokens/sec")
    elif NUM_TEST_RUNS == 0:
        print("\nNo test runs performed.")
    else:
        print("\nNo tokens were generated during test runs. Cannot calculate average speed.")

    # --- Clean up (optional, good for releasing VRAM if running multiple tests) ---
    # del model
    # del tokenizer
    # if torch.cuda.is_available():
    #     torch.cuda.empty_cache()

if __name__ == "__main__":
    # Example: Update MODEL_PATH before running
    # MODEL_PATH = "/mnt/c/Users/YourUser/Desktop/SE_Proj_clip/llama/" # Example for WSL
    # MODEL_PATH = "/Users/youruser/SE_Proj_clip/llama/" # Example for macOS
    # MODEL_PATH = "C:/Users/YourUser/Desktop/SE_Proj_clip/llama/" # Example for Windows

    if MODEL_PATH == "/path/to/your/.../SE_Proj_clip/llama/":
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print("!!! PLEASE UPDATE THE 'MODEL_PATH' VARIABLE IN THE SCRIPT FIRST !!!")
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    else:
        test_generation_speed()