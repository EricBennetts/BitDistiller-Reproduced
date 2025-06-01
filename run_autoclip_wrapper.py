import argparse
import os
import subprocess
import sys

def get_user_input(prompt_message, default_value, choices=None):
    """
    Prompts the user for input, providing a default value.
    If choices are provided, validates against them.
    """
    while True:
        if choices:
            prompt_with_choices = f"{prompt_message} {choices} (default: {default_value}): "
        else:
            prompt_with_choices = f"{prompt_message} (default: {default_value}): "
        
        user_val = input(prompt_with_choices).strip()
        if not user_val: # User pressed Enter
            return default_value
        if choices:
            if user_val in choices:
                return user_val
            else:
                print(f"Invalid choice. Please choose from {choices}.")
        else: # No choices, any non-empty input is fine
            return user_val

def main():
    # Get the directory of the current script (SE_PROJ)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    default_model_path_val = os.path.join(script_dir, "MetaMath")
    default_dump_clip_path_val = os.path.join(script_dir, "clipped_results.pt") 

    parser = argparse.ArgumentParser(
        description="Wrapper script to run autoclip.py for BitDistiller. "
                    "You can override interactive prompts with command-line arguments."
    )

    parser.add_argument('--model_path', type=str, default=None, help="Path to the Hugging Face model directory.")
    parser.add_argument('--calib_dataset', type=str, default=None, choices=["pile", "gsm8k", "code"], help="Calibration dataset to use.")
    # quant_type will now determine w_bit implicitly
    parser.add_argument('--quant_type', type=str, default=None, choices=["int", "nf3"], help="Quantization type for simulation (int for 2-bit, nf3 for 3-bit).")
    parser.add_argument('--dump_clip', type=str, default=None, help="Path to save the clipping results.")
    
    # w_bit is no longer a primary user-settable arg if quant_type is chosen, but can be overridden for advanced use
    # We'll still parse it from CLI if provided, but quant_type will take precedence.
    parser.add_argument('--w_bit', type=int, default=None, help='(Advanced) Override bits of weight for simulation. Usually determined by quant_type.')
    
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size.')
    parser.add_argument('--q_group_size', type=int, default=128, help="Group size for quantization.")
    parser.add_argument('--no_zero_point', action='store_true', help="Disable zero_point for simulation.")
    parser.add_argument('--n_samples', type=int, default=128, help="Number of calibration data samples.")
    parser.add_argument('--seqlen', type=int, default=512, help="Length usage of each calibration data sample.")

    cli_args = parser.parse_args()

    print("--- Asymmetric Clipping Configuration ---")
    model_path_final = cli_args.model_path if cli_args.model_path is not None else \
                       get_user_input("Enter model path", default_model_path_val)
    
    calib_dataset_final = cli_args.calib_dataset if cli_args.calib_dataset is not None else \
                          get_user_input("Choose calibration dataset", "pile", choices=["pile", "code", "gsm8k"])

    # Determine quant_type
    default_interactive_quant_type = "int"
    quant_type_final = cli_args.quant_type if cli_args.quant_type is not None else \
                       get_user_input(f"Choose quantization type (int => 2-bit, nf3 => 3-bit)", 
                                      default_interactive_quant_type, choices=["int", "nf3"])

    # Determine w_bit based on quant_type_final, unless overridden by CLI
    w_bit_final = None
    if cli_args.w_bit is not None:
        w_bit_final = cli_args.w_bit
        print(f"Note: w_bit explicitly set to {w_bit_final} via command line, overriding quant_type coupling.")
    elif quant_type_final == "int":
        w_bit_final = 2
    elif quant_type_final == "nf3":
        w_bit_final = 3
    else: # Should not happen due to choices validation
        print(f"Warning: Unknown quant_type '{quant_type_final}'. Defaulting w_bit to 3.")
        w_bit_final = 3


    dump_clip_final = cli_args.dump_clip if cli_args.dump_clip is not None else \
                      get_user_input("Enter path to save clipped results", default_dump_clip_path_val)

    autoclip_script_path = os.path.join(script_dir, "quantization", "autoclip.py")

    if not os.path.exists(autoclip_script_path):
        print(f"Error: autoclip.py not found at {autoclip_script_path}")
        sys.exit(1)

    command = [
        sys.executable,
        autoclip_script_path,
        "--model_path", model_path_final,
        "--calib_dataset", calib_dataset_final,
        "--quant_type", quant_type_final,
        "--run_clip",
        "--dump_clip", dump_clip_final,
        "--w_bit", str(w_bit_final), # Use the determined w_bit
        "--q_group_size", str(cli_args.q_group_size),
        "--n_samples", str(cli_args.n_samples),
        "--seqlen", str(cli_args.seqlen),
    ]

    if cli_args.no_zero_point:
        command.append("--no_zero_point")

    print("\n" + "-" * 30)
    print("Final configuration selected:")
    print(f"  Model Path: {model_path_final}")
    print(f"  Calibration Dataset: {calib_dataset_final}")
    print(f"  Quantization Type (simulation): {quant_type_final}")
    print(f"  Target Weight Bits (simulation): {w_bit_final} (derived from quant_type)")
    print(f"  Save Clipped Results to: {dump_clip_final}")
    print(f"  Quantization Group Size: {cli_args.q_group_size}")
    print(f"  Number of Samples: {cli_args.n_samples}")
    print(f"  Sequence Length: {cli_args.seqlen}")
    if cli_args.no_zero_point:
        print("  Zero Point: Disabled")
    else:
        print("  Zero Point: Enabled")
    print("-" * 30)

    print("\nRunning autoclip.py with the constructed command...")
    print("-" * 30)

    try:
        process = subprocess.Popen(command, cwd=os.path.join(script_dir, "quantization"))
        process.wait()
        if process.returncode != 0:
            print(f"autoclip.py exited with error code {process.returncode}")
        else:
            print(f"autoclip.py completed successfully. Clipped results should be at: {dump_clip_final}")
            
    except Exception as e:
        print(f"An error occurred while trying to run autoclip.py: {e}")

if __name__ == "__main__":
    main()