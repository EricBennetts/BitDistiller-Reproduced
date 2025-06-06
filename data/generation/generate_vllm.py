import os
import argparse
from data_utils import get_gen_dataset
from transformers import GenerationConfig, AutoModelForCausalLM, AutoTokenizer
from vllm import LLM, SamplingParams
import torch
import json

def main(args):
    torch.manual_seed(args.seed)
    world_size = torch.cuda.device_count()
    n_gpus = torch.cuda.device_count()
    print(f"using {n_gpus} GPUs to generate")

    chosen_dtype = "bfloat16" # Default to bfloat16
    if torch.cuda.is_available():
        major_capability, _ = torch.cuda.get_device_capability(0) # For the first GPU
        if major_capability < 8:
            print(f"GPU compute capability {major_capability}.x < 8.0. BFloat16 not supported. Using float16 (half).")
            chosen_dtype = "half"
        else:
            print(f"GPU compute capability {major_capability}.x >= 8.0. Using bfloat16.")
    else:
        print("CUDA not available. LLM initialization might fail or run on CPU if supported by a different VLLM setup.")
        # VLLM primarily targets GPUs, so this path is less common for standard VLLM usage.
        # Defaulting to bfloat16 here, but CPU execution with VLLM is not its primary mode.
    # ----- END: Added logic for dtype selection -----

    # Original LLM initialization, now using chosen_dtype
    model = LLM(
        model=args.base_model,
        tensor_parallel_size=n_gpus, # Correctly becomes 1 if n_gpus is 1
        dtype=chosen_dtype
    )

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=False)

    prompts, _ = get_gen_dataset(args.dataset_name, args.max_sample, tokenizer)

    sampling_params = SamplingParams(temperature=args.temperature, top_p=1, max_tokens=args.max_new_tokens)

    with torch.no_grad():
        outputs = model.generate(prompts, sampling_params)

    all_outputs = []
    for output in outputs:
        all_outputs.append([[output.prompt, output.outputs[0].text]])

    with open(args.out_path + f'/{args.dataset_name}_T{args.temperature}_N{args.max_new_tokens}_S{args.seed}_{args.max_sample}.json', 'w') as f:
        for item in all_outputs[:len(outputs)]:
            f.write(json.dumps(item) + '\n')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Parameters')
    parser.add_argument("--base_model", default="", type=str, help="model path")
    parser.add_argument("--dataset_name", default="", type=str, help="name of the datasets")
    parser.add_argument("--seed", type=int, default=42, help="seed")
    parser.add_argument("--out_path", default="", type=str, help="output datapath")
    parser.add_argument("--max_sample", type=int, default=None, help="max_sample")
    parser.add_argument("--temperature", type=float, default=0.7, help="generation temperature")
    parser.add_argument("--max_new_tokens", type=int, default=1024, help="max new tokens")

    args = parser.parse_args()

    if not os.path.exists(args.out_path):
        try:
            os.makedirs(args.out_path)
            print(f"dir {args.out_path} create successfully")
        except:
            pass
    else:
        print(f"dir {args.out_path} has existed")

    main(args)