
import argparse
import os
import subprocess
import sys

def get_user_input(prompt_message, default_value, expected_type=str, choices=None):
    """
    Prompts the user for input, providing a default value.
    If choices are provided, validates against them.
    Converts to expected_type.
    """
    while True:
        choice_str = ""
        if choices:
            choice_str = f" {choices}"
        
        prompt_with_choices = f"{prompt_message}{choice_str} (default: {default_value}): "
        
        user_val_str = input(prompt_with_choices).strip()
        
        if not user_val_str: # User pressed Enter, use default
            # Ensure default_value is already of the expected type or can be converted
            try:
                if isinstance(default_value, expected_type):
                    return default_value
                else: # Try to convert default if it's a string representation
                    return expected_type(default_value)
            except ValueError:
                print(f"Error: Default value '{default_value}' cannot be converted to {expected_type.__name__}.")
                # This should ideally not happen if defaults are set correctly in the script
                return None # Or raise an error

        try:
            # Attempt type conversion for non-empty input
            if expected_type == bool: # Example for boolean, not used here but good practice
                converted_val = user_val_str.lower() in ['true', 't', 'yes', 'y', '1']
            else:
                converted_val = expected_type(user_val_str)

            if choices:
                if converted_val in choices:
                    return converted_val
                else:
                    print(f"Invalid choice. Please choose from {choices}.")
            else: # No choices, converted input is fine
                return converted_val
        except ValueError:
            print(f"Invalid input. Please enter a value of type {expected_type.__name__}.")

def main():
    # Get the directory of the current script (SE_PROJ)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Define default values, using script_dir for paths
    default_base_model_val = os.path.join(script_dir, "llama")
    default_dataset_name_val = "wikitext"
    default_output_path_val = os.path.join(script_dir, "data/generation/datasets", "hf-llama-1B") # Made path absolute from script_dir
    default_max_sample_val = 3000

    parser = argparse.ArgumentParser(
        description="Wrapper script to run generate_vllm.py for VLLM text generation. "
                    "You can override interactive prompts with command-line arguments."
    )

    # Arguments for generate_vllm.py
    parser.add_argument('--base_model', type=str, default=None, help="Path to the Hugging Face model directory.")
    parser.add_argument('--dataset_name', type=str, default=None, help="Name of the dataset to use (e.g., wikitext, alpaca, code, math).")
    parser.add_argument('--out_path', type=str, default=None, help="Path to save the generated outputs.")
    parser.add_argument('--max_sample', type=int, default=None, help="Maximum number of samples to process from the dataset.")
    
    # Other arguments from generate_vllm.py with their typical defaults
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--temperature", type=float, default=0.7, help="Generation temperature.")
    parser.add_argument("--max_new_tokens", type=int, default=1024, help="Maximum new tokens to generate.")

    cli_args = parser.parse_args()

    print("--- VLLM Generation Configuration ---")
    
    base_model_final = cli_args.base_model if cli_args.base_model is not None else \
                       get_user_input("Enter base model path", default_base_model_val)
    
    dataset_name_final = cli_args.dataset_name if cli_args.dataset_name is not None else \
                         get_user_input("Enter dataset name (e.g., wikitext, alpaca)", default_dataset_name_val)

    output_path_final = cli_args.out_path if cli_args.out_path is not None else \
                        get_user_input("Enter output path for generated data", default_output_path_val)

    max_sample_final = cli_args.max_sample if cli_args.max_sample is not None else \
                       get_user_input("Enter maximum number of samples", default_max_sample_val, expected_type=int)

    # Use CLI args or their defaults for other parameters
    seed_final = cli_args.seed
    temperature_final = cli_args.temperature
    max_new_tokens_final = cli_args.max_new_tokens

    # Construct path to the target script
    generate_vllm_script_path = os.path.join(script_dir, "data", "generation", "generate_vllm.py")

    if not os.path.exists(generate_vllm_script_path):
        print(f"Error: generate_vllm.py not found at {generate_vllm_script_path}")
        sys.exit(1)

    # Ensure output directory exists (generate_vllm.py also does this, but good for clarity here too)
    if output_path_final and not os.path.exists(output_path_final):
        try:
            os.makedirs(output_path_final, exist_ok=True)
            print(f"Created output directory: {output_path_final}")
        except OSError as e:
            print(f"Error creating output directory {output_path_final}: {e}")
            # sys.exit(1) # Optionally exit if directory creation fails

    command = [
        sys.executable, # Path to python interpreter
        generate_vllm_script_path,
        "--base_model", base_model_final,
        "--dataset_name", dataset_name_final,
        "--out_path", output_path_final,
        "--max_sample", str(max_sample_final),
        "--seed", str(seed_final),
        "--temperature", str(temperature_final),
        "--max_new_tokens", str(max_new_tokens_final),
    ]

    print("\n" + "-" * 30)
    print("Final configuration for generate_vllm.py:")
    print(f"  Base Model:         {base_model_final}")
    print(f"  Dataset Name:       {dataset_name_final}")
    print(f"  Output Path:        {output_path_final}")
    print(f"  Max Samples:        {max_sample_final}")
    print(f"  Seed:               {seed_final}")
    print(f"  Temperature:        {temperature_final}")
    print(f"  Max New Tokens:     {max_new_tokens_final}")
    print("-" * 30)

    print(f"\nExecuting: {' '.join(command)}")
    print(f"Target script CWD: {os.path.join(script_dir, 'data', 'generation')}")
    print("-" * 30)

    try:
        # Set the current working directory for the subprocess
        process_cwd = os.path.join(script_dir, "data", "generation")
        process = subprocess.Popen(command, cwd=process_cwd)
        process.wait() # Wait for the subprocess to complete
        
        if process.returncode != 0:
            print(f"\n--- generate_vllm.py exited with error code {process.returncode} ---")
        else:
            print(f"\n--- generate_vllm.py completed successfully. ---")
            print(f"Generated data should be in: {output_path_final}")
            
    except FileNotFoundError:
        print(f"Error: The Python interpreter '{sys.executable}' or the script '{generate_vllm_script_path}' was not found.")
    except Exception as e:
        print(f"An error occurred while trying to run generate_vllm.py: {e}")

if __name__ == "__main__":
    main()