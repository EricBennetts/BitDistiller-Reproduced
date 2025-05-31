import torch
import torch.nn as nn
import gc
import argparse
import os
import sys
from clip_utils import *
from quantizer import pseudo_quantize_tensor, pseudo_quantize_n2f3_tensor
from tqdm import tqdm
from collections import defaultdict
import functools

@torch.no_grad()
def auto_2clip_layer(w, input_feat, n_bit, q_config,
                    n_grid=20,
                    max_shrink=0.5,
                    n_sample_token=512):
    assert w.dim() == 2
    org_w_shape = w.shape
    # w           [co, ci]      -> [co, 1, n_group, group size]
    # input_feat  [n_token, ci] -> [1, n_token, n_group, group size]

    group_size = q_config["q_group_size"] if q_config["q_group_size"] > 0 else w.shape[1]

    input_feat = input_feat.view(-1, input_feat.shape[-1])
    input_feat = input_feat.reshape(1, input_feat.shape[0], -1, group_size)
    input_feat = input_feat[:, 0::input_feat.shape[1] // n_sample_token]
    w = w.reshape(w.shape[0], 1, -1, group_size)

    oc_batch_size = 256 if w.shape[0] % 256 == 0 else 64  # prevent OOM
    
    assert w.shape[0] % oc_batch_size == 0
    w_all = w
    best_max_val_all = []
    best_min_val_all = []
    for i_b in range(w.shape[0] // oc_batch_size):
        w = w_all[i_b * oc_batch_size: (i_b + 1) * oc_batch_size]

        org_max_val = w.amax(dim=-1, keepdim=True)  # co, 1, n_group, 1
        org_min_val = w.amin(dim=-1, keepdim=True)

        best_max_val = org_max_val.clone()
        best_min_val = org_min_val.clone()

        min_errs = torch.ones_like(org_max_val) * 1e9
        input_feat = input_feat.to(w.device)
        org_out = (input_feat * w).sum(dim=-1)  # co, n_token, n_group

        for i_s_p in range(int(max_shrink * n_grid)):
            max_val = org_max_val * (1 - i_s_p / n_grid)
            for i_s_n in range(int(max_shrink * n_grid)):
                min_val = org_min_val * (1 - i_s_n / n_grid)
                # min_val = - max_val
                cur_w = torch.clamp(w, min_val, max_val)
                if q_config["quant_type"] == "int":
                    q_w = pseudo_quantize_tensor(cur_w, n_bit=n_bit, zero_point=True, q_group_size=q_config['q_group_size'])
                elif q_config["quant_type"] == "nf3":
                    q_w = pseudo_quantize_n2f3_tensor(cur_w, q_group_size=q_config['q_group_size'])
                else:
                    quant_type = q_config["quant_type"]
                    raise ValueError(f"Has no support {quant_type}. Valid quant_type:[int, nf3]")
                    
                cur_out = (input_feat * q_w).sum(dim=-1)

                err = (cur_out - org_out).pow(2).mean(dim=1).view(min_errs.shape)

                del cur_w
                del cur_out
                cur_best_idx = err < min_errs
                min_errs[cur_best_idx] = err[cur_best_idx]
                best_max_val[cur_best_idx] = max_val[cur_best_idx]
                best_min_val[cur_best_idx] = min_val[cur_best_idx]

        best_max_val_all.append(best_max_val)
        best_min_val_all.append(best_min_val)
    print("loss:", err.mean().item())
    best_max_val = torch.cat(best_max_val_all, dim=0)
    best_min_val = torch.cat(best_min_val_all, dim=0)
    del input_feat
    del org_out
    gc.collect()
    torch.cuda.empty_cache()
    return best_max_val.squeeze(1), best_min_val.squeeze(1)

    
@torch.no_grad()
def auto_clip_block(module,
                    w_bit, q_config,
                    input_feat):

    named_linears = {name: m for name,
                     m in module.named_modules() if isinstance(m, nn.Linear)}

    clip_list = []
    for name in named_linears:
        # due to qk bmm, it is hard to clip precisely
        if any([_ in name for _ in ["q_", "k_", "query", "key", "Wqkv"]]):
            continue
        named_linears[name].cuda()

        max_val, min_val = auto_2clip_layer(
            named_linears[name].weight, input_feat[name], n_bit=w_bit, q_config=q_config)

        clip_list.append((name, max_val, min_val))

        named_linears[name].cpu()
    return clip_list

@torch.no_grad()
def run_clip(
    model, enc,
    w_bit, q_config,
    n_samples=128, seqlen=1024,
    datasets="pile"
):
    print(f"Using {datasets} dataset to do calibation")
    samples_list = get_calib_dataset(
              datasets=datasets, tokenizer=enc, n_samples=n_samples, block_size=seqlen)
    if not samples_list:
        raise ValueError(f"No calibration data samples loaded from {datasets}.")
    
    samples_for_catcher = samples_list[0] # Use the first block for Catcher

    # This will hold the input to the *current* layer being processed
    # It starts as the output of the embedding layer (hidden_states)
    # and is updated to be the output of layer i-1 to become input for layer i
    current_layer_input_on_cpu = None # Will be populated after Catcher

    # This will hold kwargs like attention_mask, position_ids for layers
    # These are established by the first full model pass (via Catcher)
    # and should generally be the same for all layers unless past_key_values are used (not here)
    layer_kwargs_from_catcher = {} 
    
    layers = get_blocks(model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Target device for model operations: {device}")

    model.to(device)

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp_hidden_states, **kwargs_for_layer): # inp_hidden_states is the input to the first layer
            nonlocal current_layer_input_on_cpu # To assign to the outer scope variable
            nonlocal layer_kwargs_from_catcher

            current_layer_input_on_cpu = inp_hidden_states.detach().cpu()
            
            # Store kwargs, making sure any tensors are moved to CPU for storage
            # to free up GPU memory. They'll be moved back to 'device' when used.
            for k, v_kwarg in kwargs_for_layer.items():
                if isinstance(v_kwarg, torch.Tensor):
                    layer_kwargs_from_catcher[k] = v_kwarg.detach().cpu()
                else:
                    layer_kwargs_from_catcher[k] = v_kwarg
            raise ValueError # early exit
    
    original_layer_0 = layers[0]
    layers[0] = Catcher(original_layer_0)
    
    try:
        # The model's forward pass will generate initial position_ids and attention_mask
        # These will be passed as **kwargs_for_layer to Catcher's forward
        model(input_ids=samples_for_catcher.to(device)) # Pass input_ids to trigger full model logic
    except ValueError:
        pass
    
    del samples_for_catcher
    layers[0] = original_layer_0
    
    if current_layer_input_on_cpu is None:
        raise RuntimeError("Catcher did not capture any input. Model forward pass might have failed silently.")

    model.cpu()
    torch.cuda.empty_cache()
    gc.collect()

    clip_results = {"clip": []}
    
    # 'current_layer_input_on_cpu' now holds the hidden_states output from embeddings (input to layer 0)
    # 'layer_kwargs_from_catcher' holds attention_mask, position_ids etc. from the initial model pass, on CPU.

    for i in tqdm(range(len(layers)), desc="Running Asymmetric Clipping..."):
        layer = layers[i]
        layer.to(device)
        
        named_linears = get_named_linears(layer)
        input_feat_for_clip = defaultdict(list) # Renamed to avoid confusion
        handles = []

        def cache_input_hook_for_clip(m, x, y, name, feat_dict):
            x_tensor = x[0]
            feat_dict[name].append(x_tensor.detach().cpu()) # Cache on CPU

        for name_linear in named_linears:
            handles.append(named_linears[name_linear].register_forward_hook(
                functools.partial(cache_input_hook_for_clip, name=name_linear,
                                  feat_dict=input_feat_for_clip)))
        
        # Prepare inputs and kwargs for the current layer, moving them to 'device'
        current_layer_input_on_device = current_layer_input_on_cpu.to(device)
        
        layer_kwargs_for_current_iter = {}
        if layer_kwargs_from_catcher: # If catcher actually caught kwargs
            for k_kwarg, v_kwarg in layer_kwargs_from_catcher.items():
                if isinstance(v_kwarg, torch.Tensor):
                    layer_kwargs_for_current_iter[k_kwarg] = v_kwarg.to(device)
                else:
                    layer_kwargs_for_current_iter[k_kwarg] = v_kwarg
        
        # Ensure 'position_ids' is present if needed by the layer,
        # and it's not explicitly disabled by use_cache=False and no past_key_values
        # For Llama, position_ids are crucial.
        # The LlamaDecoderLayer forward signature is:
        # hidden_states, attention_mask=None, position_ids=None, past_key_value=None, ...
        # The `layer_kwargs_from_catcher` should contain `attention_mask` and `position_ids`
        # as prepared by `LlamaModel` for the first layer.

        # The `LlamaDecoderLayer.forward` expects `attention_mask` (which can be 4D)
        # and `position_ids` (which is 2D, e.g., [batch_size, seq_len])
        # The `attention_mask` from `LlamaModel.forward` should be correctly shaped (4D).
        
        # Critical: The `attention_mask` in `layer_kwargs_for_current_iter` should be the one
        # generated by the main LlamaModel forward pass, which is typically 4D.
        # The `position_ids` should also be from there.
        # The error "IndexError: too many indices for tensor of dimension 2" for causal_mask
        # implies that the `attention_mask` passed to the layer was 2D and the LlamaAttention
        # class's internal logic tried to slice it as if it were 4D after some processing.

        # One key thing: if use_cache=True (default for Llama), layers expect past_key_value.
        # For calibration, we typically don't use past_key_values.
        # The layer forward call might need output_attentions=False, use_cache=False
        # to simplify its signature if these are not in layer_kwargs_from_catcher.
        # However, the error is *before* past_key_values would be a problem.

        # Let's assume layer_kwargs_for_current_iter has the correct attention_mask and position_ids
        # from the Catcher. The shapes should be:
        # current_layer_input_on_device: [bsz, seq_len, hidden_size]
        # layer_kwargs_for_current_iter['attention_mask']: [bsz, 1, seq_len, seq_len] (for causal)
        # layer_kwargs_for_current_iter['position_ids']: [bsz, seq_len]
        
        # Ensure all expected kwargs are present
        expected_decoder_layer_kwargs = {"attention_mask", "position_ids"}
        final_kwargs_for_layer = {}
        for k_expect in expected_decoder_layer_kwargs:
            if k_expect in layer_kwargs_for_current_iter:
                final_kwargs_for_layer[k_expect] = layer_kwargs_for_current_iter[k_expect]
            # else:
                # print(f"Warning: Expected kwarg '{k_expect}' not found in layer_kwargs_from_catcher for layer {i}")
        
        # Call the layer
        # The LlamaDecoderLayer returns a tuple. The first element is hidden_states.
        # If use_cache=True (default), it also returns present_key_value.
        # If output_attentions=True, it also returns self_attn_weights.
        # We only need the hidden_states.
        layer_outputs = layer(current_layer_input_on_device, **final_kwargs_for_layer)
        output_hidden_states_on_device = layer_outputs[0]
        
        for h in handles:
            h.remove()

        # input_feat_for_clip values are on CPU
        input_feat_for_clip = {k_feat: torch.cat(v_feat, dim=0) for k_feat, v_feat in input_feat_for_clip.items()}

        torch.cuda.empty_cache()

        clip_list = auto_clip_block(layer,
                                    w_bit=w_bit, q_config=q_config,
                                    input_feat=input_feat_for_clip)
        
        apply_clip(layer, clip_list)
        
        clip_results["clip"] += append_str_prefix(clip_list, get_op_name(model, layer) + ".")

        layer.cpu()
        current_layer_input_on_cpu = output_hidden_states_on_device.cpu() # Update for next iteration
        
        del input_feat_for_clip
        del current_layer_input_on_device
        del output_hidden_states_on_device
        del layer_outputs
        gc.collect()
        torch.cuda.empty_cache()
        
    return clip_results

def main(args, q_config):
    if args.dump_clip and os.path.exists(args.dump_clip):
        print(f"Found existing AWQ results {args.dump_clip}, exit.")
        exit()

    model, enc = build_model_and_enc(args.model_path)

    if args.run_clip:
        assert args.dump_clip, "Please save the awq results with --dump_awq"

        clip_results = run_clip(
            model, enc,
            w_bit=args.w_bit, q_config=q_config,
            n_samples=args.n_samples, seqlen=args.seqlen, datasets=args.calib_dataset
        )

        if args.dump_clip:
            dirpath = os.path.dirname(args.dump_clip)
            os.makedirs(dirpath, exist_ok=True)
            
            torch.save(clip_results, args.dump_clip)
            print("Clipping results saved at", args.dump_clip)
            
        exit(0)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, help='path of the hf model')
    parser.add_argument('--batch_size', type=int, default=8, help='batch size')
    parser.add_argument('--w_bit', type=int, default=2, help='bits of weight')
    parser.add_argument('--q_group_size', type=int, default=128)
    parser.add_argument('--quant_type', type=str, default="int", help="quant_type", choices=["int", "nf3"])
    parser.add_argument('--no_zero_point', action='store_true',
                        help="disable zero_point")
    parser.add_argument('--run_clip', action='store_true',
                        help="perform asym-clipping search process")
    parser.add_argument('--dump_clip', type=str, default=None,
                        help="save the asym-clipping search results")
    parser.add_argument('--n_samples', type=int, default=128,
                        help="Number of calibration data samples.")
    parser.add_argument('--seqlen', type=int, default=1024,
                        help="Length usage of each calibration data.")
    parser.add_argument("--calib_dataset", type=str, default="pile",
            choices=["pile", "gsm8k","code"],
            help="Where to extract calibration data from.",
        )

    args = parser.parse_args()

    q_config = {
        "zero_point": not args.no_zero_point,   # by default True
        "q_group_size": args.q_group_size,      # whether to use group quantization
        "quant_type": args.quant_type
    }

    print("Quantization config:", q_config)
    main(args, q_config)