#!/usr/bin/env python
"""
bench_end_removal.py

Inference-latency benchmark for END removal of LLM decoder layers in VLA-Adapter
(drop the last k decoder layers of the Qwen2.5-0.5B backbone).

Two ways a config can "drop" the last k layers:

  full_ignore : what the eval code in EfficientVLALayerDrop / EfficientVLAManyLayerDrop does today.
                All L decoder layers still EXECUTE; modeling_prismatic.py merely slices off the last k
                hidden states (`hidden_states[:-k]`) before the action head. No LLM compute is saved.
  truncated   : true removal. `model.layers = model.layers[:L-k]`, `num_hidden_layers = L-k`.
                The dropped layers never run.

For k = 0 (baseline) the only mode is `full`.

Equivalence of the two modes (verified at runtime, see `verify()`):
  HF Qwen2Model (transformers 4.40) returns hidden_states = (emb, out_1, ..., out_{L-1}, norm(out_L)).
  `hidden_states[:-k]` therefore feeds the head (emb, out_1, ..., out_{L-k}), all *un-normed*.
  After truncation to L-k layers the model would return (..., norm(out_{L-k})), so we also set the final
  RMSNorm to Identity. The repo code still does `hidden_states[:-k]`, so we append k zero-cost references
  to the language-model output's hidden_states tuple; that slice then removes exactly those references.
  Net effect: the action head receives bit-identical inputs while L-k layers execute instead of L.
  (lm_head logits become meaningless, but they are unused by the L1-regression head and lm_head runs
  in every mode, so it does not bias the comparison.)

This script is READ-ONLY w.r.t. the experiment repos and checkpoints (jobs are running against them):
  * code is imported from --repo_dir and never modified; bytecode writing is disabled (no __pycache__);
  * it does NOT call experiments.robot.openvla_utils.get_vla(), because get_vla() rewrites
    <checkpoint>/config.json (update_auto_map) and copies modeling_prismatic.py into the checkpoint
    (check_model_logic_mismatch). The model is loaded the same way, minus those two writes;
  * all patches (truncation, timers, hooks) are applied to the in-memory model only.
"""

import sys

sys.dont_write_bytecode = True  # never write __pycache__ into the experiment repos

import argparse
import inspect
import json
import os
import platform
import re
import time
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--repo_dir", required=True,
                   help="Repo whose code matches the checkpoint (EfficientVLA for k=0, EfficientVLALayerDrop for k=1, "
                        "EfficientVLAManyLayerDrop for k=3). Imported read-only.")
    p.add_argument("--pretrained_checkpoint", required=True, help="Checkpoint dir trained with that repo.")
    p.add_argument("--num_end_layers_drop", type=int, required=True, help="k: number of final LLM layers dropped.")
    p.add_argument("--task_suite_name", default="libero_spatial")
    p.add_argument("--task_label", default="pick up the black bowl between the plate and the ramekin and place it on the plate")
    p.add_argument("--num_warmup", type=int, default=20, help="Untimed queries before each timed block.")
    p.add_argument("--num_iters", type=int, default=100, help="Timed queries per mode per repeat.")
    p.add_argument("--repeats", type=int, default=3,
                   help="Alternate modes this many times (order reversed each repeat) to cancel clock/thermal drift.")
    p.add_argument("--no_breakdown", action="store_true",
                   help="Skip the second pass that times predict_action / LLM forward separately.")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--output_json", required=True)
    return p.parse_args()


# ---------------------------------------------------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------------------------------------------------
def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def summarize(values, actions_per_query):
    """Same statistics as LatencyTracker.summarize in run_libero_eval.py."""
    if len(values) == 0:
        return {"n": 0}
    a = np.asarray(values, dtype=np.float64)
    mean = float(a.mean())
    return {
        "n": int(a.size),
        "mean_ms": mean,
        "std_ms": float(a.std()),
        "median_ms": float(np.median(a)),
        "p90_ms": float(np.percentile(a, 90)),
        "p99_ms": float(np.percentile(a, 99)),
        "min_ms": float(a.min()),
        "max_ms": float(a.max()),
        "queries_per_s": 1000.0 / mean,
        "amortized_ms_per_action": mean / actions_per_query,
        "effective_control_hz": 1000.0 * actions_per_query / mean,
    }


def check_repo_slicing(model_cls, k):
    """Make sure the imported repo does pure END removal of exactly k layers (guards against repo/k mix-ups)."""
    src = inspect.getsource(model_cls._regression_or_discrete_prediction)
    code = "\n".join(l.split("#", 1)[0].rstrip() for l in src.splitlines() if l.split("#", 1)[0].strip())
    if k == 0:
        ok = "language_model_output.hidden_states[0:]" in code and "filtered_hidden_states" not in code
        expect = "for item in language_model_output.hidden_states[0:]"
    else:
        ok = re.search(rf"filtered_hidden_states\s*=\s*hidden_states\[\s*:\s*-\s*{k}\s*\]", code) is not None
        expect = f"filtered_hidden_states = hidden_states[:-{k}]"
    if not ok:
        raise RuntimeError(
            f"Repo hidden-state filtering does not match END removal with k={k} (expected `{expect}` in "
            f"_regression_or_discrete_prediction). Wrong --repo_dir for this k?"
        )
    return expect


# ---------------------------------------------------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------------------------------------------------
def main():
    args = parse_args()
    k = args.num_end_layers_drop
    repo_dir = os.path.abspath(args.repo_dir)
    ckpt = os.path.abspath(args.pretrained_checkpoint)
    assert os.path.isdir(ckpt), f"Checkpoint dir not found: {ckpt}"
    assert torch.cuda.is_available(), "CUDA is required for this benchmark"

    # ---- Import repo code (read-only) ------------------------------------------------------------------------------
    sys.path.insert(0, repo_dir)
    import prismatic  # noqa: E402

    assert os.path.abspath(prismatic.__file__).startswith(repo_dir + os.sep), (
        f"`prismatic` resolved to {prismatic.__file__}, not {repo_dir}. Check PYTHONPATH / editable installs."
    )
    import transformers
    from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor

    from experiments.robot.openvla_utils import (
        DEVICE,
        _load_dataset_stats,
        get_action_head,
        get_proprio_projector,
    )
    from experiments.robot.robot_utils import get_action
    from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
    from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
    from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor
    from prismatic.vla.constants import NUM_ACTIONS_CHUNK, PROPRIO_DIM

    slicing_line = check_repo_slicing(OpenVLAForActionPrediction, k)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # ---- Load model exactly like get_vla(), minus the checkpoint-mutating calls -------------------------------------
    AutoConfig.register("openvla", OpenVLAConfig)
    AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
    AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)

    vla = AutoModelForVision2Seq.from_pretrained(
        ckpt, torch_dtype=torch.bfloat16, low_cpu_mem_usage=False, trust_remote_code=False
    )
    vla.vision_backbone.set_num_images_in_input(2)
    vla.eval().to(DEVICE)
    _load_dataset_stats(vla, ckpt)
    vla.set_version("vla-adapter")

    cfg = SimpleNamespace(
        model_family="openvla",
        pretrained_checkpoint=ckpt,
        use_l1_regression=True,
        use_minivlm=True,
        use_film=False,
        num_images_in_input=2,
        use_proprio=True,
        center_crop=True,
        num_open_loop_steps=NUM_ACTIONS_CHUNK,
        load_in_8bit=False,
        load_in_4bit=False,
        use_pro_version=True,
        save_version="vla-adapter",
        task_suite_name=args.task_suite_name,
        unnorm_key=args.task_suite_name,
    )
    if cfg.unnorm_key not in vla.norm_stats and f"{cfg.unnorm_key}_no_noops" in vla.norm_stats:
        cfg.unnorm_key = f"{cfg.unnorm_key}_no_noops"
    assert cfg.unnorm_key in vla.norm_stats, f"unnorm key {cfg.unnorm_key} not in norm_stats"

    proprio_projector = get_proprio_projector(cfg, vla.llm_dim, proprio_dim=PROPRIO_DIM)
    action_head = get_action_head(cfg, vla.llm_dim)  # num_blocks comes from the repo's action_heads.py
    processor = AutoProcessor.from_pretrained(ckpt, trust_remote_code=False)

    lm = vla.language_model            # Qwen2ForCausalLM
    decoder = lm.model                 # Qwen2Model
    L = len(decoder.layers)
    num_blocks = len(action_head.model.mlp_resnet_blocks)
    if num_blocks != L - k:
        raise RuntimeError(f"Action head has {num_blocks} blocks but L-k = {L}-{k} = {L - k}. Wrong repo/checkpoint for k?")

    # ---- Instrumentation (in-memory only) ---------------------------------------------------------------------------
    state = {"pad": 0, "time": False, "llm_ms": [], "model_ms": [], "count": False, "executed": [], "probe": None}

    # (a) language-model forward: timing + hidden-state padding used by the truncated mode
    orig_lm_forward = lm.forward

    def lm_forward(*a, **kw):
        if state["time"]:
            sync(); t0 = time.perf_counter()
        out = orig_lm_forward(*a, **kw)
        if state["time"]:
            sync(); state["llm_ms"].append((time.perf_counter() - t0) * 1e3)
        if state["pad"] and out.hidden_states is not None:
            # k references to an existing tensor: no compute, no memory. The repo's `hidden_states[:-k]` removes them.
            out.hidden_states = tuple(out.hidden_states) + (out.hidden_states[-1],) * state["pad"]
        return out

    lm.forward = lm_forward

    # (b) VLA.predict_action timing (vision + projector + LLM + action head + unnormalize)
    orig_predict = vla.predict_action

    def predict_action(*a, **kw):
        if state["time"]:
            sync(); t0 = time.perf_counter()
        out = orig_predict(*a, **kw)
        if state["time"]:
            sync(); state["model_ms"].append((time.perf_counter() - t0) * 1e3)
        return out

    vla.predict_action = predict_action

    # (c) capture the exact tensor fed to the action head (for the equivalence check)
    orig_head_predict = action_head.predict_action

    def head_predict(multi_layer_hidden_states, *a, **kw):
        if state["probe"] is not None:
            state["probe"]["head_input"] = multi_layer_hidden_states.detach().float().cpu().clone()
        return orig_head_predict(multi_layer_hidden_states, *a, **kw)

    action_head.predict_action = head_predict

    # (d) count which decoder layers actually execute (hooks on the ORIGINAL layer objects)
    for idx, layer in enumerate(decoder.layers):
        layer.register_forward_pre_hook(lambda m, inp, idx=idx: state["executed"].append(idx) if state["count"] else None)

    # ---- Mode switching ---------------------------------------------------------------------------------------------
    orig_layers, orig_norm = decoder.layers, decoder.norm
    cfg_objs = {id(c): c for c in (lm.config, decoder.config, getattr(vla.config, "text_config", None))
                if c is not None and hasattr(c, "num_hidden_layers")}.values()

    def set_mode(mode):
        if mode == "truncated":
            decoder.layers = orig_layers[: L - k]        # ModuleList slice: dropped layers are no longer called
            decoder.norm = nn.Identity()                 # see module docstring
            for c in cfg_objs:
                c.num_hidden_layers = L - k
            state["pad"] = k
        else:  # "full" / "full_ignore": untouched model, repo slicing does the ignoring
            decoder.layers = orig_layers
            decoder.norm = orig_norm
            for c in cfg_objs:
                c.num_hidden_layers = L
            state["pad"] = 0

    modes = ["full"] if k == 0 else ["full_ignore", "truncated"]

    # ---- Fixed synthetic observation (latency does not depend on pixel values) --------------------------------------
    rng = np.random.default_rng(args.seed)
    base_obs = {
        "full_image": rng.integers(0, 256, size=(224, 224, 3), dtype=np.uint8),   # eval passes 224x224 resized images
        "wrist_image": rng.integers(0, 256, size=(224, 224, 3), dtype=np.uint8),
        "state": rng.uniform(-0.5, 0.5, size=(PROPRIO_DIM,)).astype(np.float64),
    }

    def query():
        # get_vla_action normalizes obs["state"] in place, so give it a fresh copy every call.
        obs = {"full_image": base_obs["full_image"], "wrist_image": base_obs["wrist_image"], "state": base_obs["state"].copy()}
        actions = get_action(cfg, vla, obs, args.task_label, processor=processor, action_head=action_head,
                             proprio_projector=proprio_projector, noisy_action_projector=None,
                             use_film=False, use_minivlm=True)
        return np.stack(actions)

    # ---- Verification: which layers run, and are head inputs/actions identical? -------------------------------------
    verification = {}
    for mode in modes:
        set_mode(mode)
        query()  # warm one call so lazy init does not land in the probe
        state["executed"], state["count"], state["probe"] = [], True, {}
        actions = query()
        state["count"] = False
        executed = list(state["executed"])
        head_in = state["probe"]["head_input"]
        state["probe"] = None
        verification[mode] = {
            "layers_executed": len(executed),
            "executed_layer_indices": executed,
            "decoder_layers_in_module": len(decoder.layers),
            "num_hidden_layers_config": lm.config.num_hidden_layers,
            "head_input_num_layer_states": int(head_in.shape[1]),
            "_actions": actions,
            "_head_in": head_in,
        }
        expected_layers = L - k if mode == "truncated" else L
        assert len(executed) == expected_layers and executed == list(range(expected_layers)), (
            f"[{mode}] expected layers 0..{expected_layers - 1} to execute once, got {executed}"
        )
        assert head_in.shape[1] == num_blocks + 1, (
            f"[{mode}] action head got {head_in.shape[1]} layer states, expected {num_blocks + 1}"
        )

    equivalence = None
    if k > 0:
        a, b = verification["full_ignore"], verification["truncated"]
        hd = (a["_head_in"] - b["_head_in"]).abs().max().item()
        hs = a["_head_in"].abs().max().item()
        ad = float(np.abs(a["_actions"] - b["_actions"]).max())
        equivalence = {
            "head_input_max_abs_diff": hd,
            "head_input_max_rel_diff": hd / max(hs, 1e-12),
            "actions_max_abs_diff": ad,
            "pass": (hd / max(hs, 1e-12)) <= 1e-3 and ad <= 1e-2,
        }
    for v in verification.values():
        v.pop("_actions"); v.pop("_head_in")

    # ---- Timing -----------------------------------------------------------------------------------------------------
    raw = {m: {"e2e_ms": [], "model_ms": [], "llm_ms": []} for m in modes}
    for r in range(args.repeats):
        order = modes if r % 2 == 0 else modes[::-1]
        for mode in order:
            set_mode(mode)
            for _ in range(args.num_warmup):
                query()
            # Pass 1: end-to-end get_action() latency, no inner syncs (same definition as the eval's LatencyTracker)
            for _ in range(args.num_iters):
                sync(); t0 = time.perf_counter()
                query()
                sync(); raw[mode]["e2e_ms"].append((time.perf_counter() - t0) * 1e3)
            # Pass 2: breakdown (predict_action and LLM forward, each bracketed by syncs)
            if not args.no_breakdown:
                state["llm_ms"], state["model_ms"], state["time"] = [], [], True
                for _ in range(args.num_iters):
                    query()
                state["time"] = False
                raw[mode]["model_ms"] += state["model_ms"]
                raw[mode]["llm_ms"] += state["llm_ms"]
            print(f"[repeat {r + 1}/{args.repeats}] k={k} {mode:12s} "
                  f"e2e median={np.median(raw[mode]['e2e_ms'][-args.num_iters:]):.2f} ms", flush=True)
    set_mode(modes[0])

    results = {
        mode: {
            "layers_executed": verification[mode]["layers_executed"],
            "e2e": summarize(raw[mode]["e2e_ms"], NUM_ACTIONS_CHUNK),
            "predict_action": summarize(raw[mode]["model_ms"], NUM_ACTIONS_CHUNK),
            "llm_forward": summarize(raw[mode]["llm_ms"], NUM_ACTIONS_CHUNK),
        }
        for mode in modes
    }

    out = {
        "k": k,
        "total_llm_layers": L,
        "action_head_num_blocks": num_blocks,
        "repo_dir": repo_dir,
        "repo_hidden_state_filter": slicing_line,
        "pretrained_checkpoint": ckpt,
        "attn_implementation": getattr(lm.config, "_attn_implementation", None),
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "host": platform.node(),
        "num_warmup": args.num_warmup,
        "num_iters": args.num_iters,
        "repeats": args.repeats,
        "verification": verification,
        "equivalence_full_ignore_vs_truncated": equivalence,
        "results": results,
        "raw_ms": raw,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.output_json)), exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump(out, f, indent=2)

    # ---- Console summary --------------------------------------------------------------------------------------------
    print("\n" + "=" * 100)
    print(f"END removal k={k} | repo={os.path.basename(repo_dir)} | L={L} | head blocks={num_blocks} | {out['gpu']}")
    print(f"repo filter: {slicing_line}")
    for mode in modes:
        res = results[mode]
        print(f"  {mode:12s} layers_run={res['layers_executed']:2d} | "
              f"e2e mean={res['e2e']['mean_ms']:.2f} median={res['e2e']['median_ms']:.2f} p90={res['e2e']['p90_ms']:.2f} ms"
              + ("" if args.no_breakdown else
                 f" | predict_action median={res['predict_action']['median_ms']:.2f} | llm median={res['llm_forward']['median_ms']:.2f} ms"))
    if equivalence is not None:
        print(f"  equivalence full_ignore vs truncated: head-input max rel diff={equivalence['head_input_max_rel_diff']:.2e}, "
              f"actions max abs diff={equivalence['actions_max_abs_diff']:.2e} -> {'PASS' if equivalence['pass'] else 'FAIL'}")
    print(f"wrote {args.output_json}")
    print("=" * 100)

    if equivalence is not None and not equivalence["pass"]:
        sys.exit(2)


if __name__ == "__main__":
    main()
