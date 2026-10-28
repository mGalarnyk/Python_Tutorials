"""Serve a fine-tuned Nemotron FOMC classifier with vLLM.

Training scores the LM-head logits of three label tokens at the last prompt
token (causal_cls.LabelTokenClassifier). vLLM reproduces that exactly: same
prompt token ids, one generated token, restricted to those three label tokens,
greedy. The argmax over three logits is the same prediction.

Two steps, two environments (vLLM pins its own torch, so keep it out of .venv):

    # 1. Merge the LoRA / QLoRA adapter into the BF16 base (training venv)
    .venv/bin/python serve_vllm.py merge --method qlora --seed 5768

    # 2. Score the test split with vLLM (vLLM venv)
    .venv-vllm/bin/python serve_vllm.py eval --method qlora --seed 5768
    .venv-vllm/bin/python serve_vllm.py eval --method qlora --seed 5768 --lora   # unmerged adapter

    # 3. Or serve it behind an OpenAI-compatible API
    .venv-vllm/bin/vllm serve checkpoints/<run>/merged --served-model-name fomc-nemotron

A QLoRA adapter was trained against an NF4 base. Merging it into the BF16 base
is the standard serving path; `eval` reports the merged F1 next to the
training-time F1 so the gap (usually tiny) is visible, not assumed.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import config as cfg

MODEL_KEY = "nemotron-nano-4b"


def _run_dir(method: str, seed: int, split: str) -> Path:
    return cfg.run_dir(MODEL_KEY, method, seed, split)


def merged_dir(method: str, seed: int, split: str = cfg.DEFAULT_SPLIT) -> Path:
    return _run_dir(method, seed, split) / "merged"


def merge_adapter(method: str, seed: int, split: str = cfg.DEFAULT_SPLIT) -> Path:
    """Fold the trained adapter into the BF16 base and save a plain HF checkpoint."""
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    final = _run_dir(method, seed, split) / "final"
    adapter_dir = final / "backbone"
    if not (adapter_dir / "adapter_config.json").exists():
        raise FileNotFoundError(f"No LoRA adapter at {adapter_dir}. Train {method} seed {seed} first.")
    meta = json.loads((final / "causal_cls.json").read_text())
    base_id = json.loads((adapter_dir / "adapter_config.json").read_text())["base_model_name_or_path"]

    out = merged_dir(method, seed, split)
    print(f"[vllm] merging {adapter_dir} into {base_id} (BF16) -> {out}")
    base = AutoModelForCausalLM.from_pretrained(base_id, dtype=torch.bfloat16, trust_remote_code=True)
    merged = PeftModel.from_pretrained(base, str(adapter_dir)).merge_and_unload()
    from causal_cls import _sanitize_generation_config

    _sanitize_generation_config(merged)
    # Nemotron's remote modeling code declares _tied_weights_keys as a list, which
    # transformers 5.x save_pretrained rejects. Write config + safetensors directly.
    from safetensors.torch import save_file

    out.mkdir(parents=True, exist_ok=True)
    merged.config.save_pretrained(out)
    merged.generation_config.save_pretrained(out)
    seen: set[int] = set()
    tensors = {}
    for name, tensor in merged.state_dict().items():
        ptr = tensor.data_ptr()
        tensors[name] = tensor.detach().clone().contiguous() if ptr in seen else tensor.detach().contiguous()
        seen.add(ptr)
    save_file(tensors, out / "model.safetensors", metadata={"format": "pt"})
    AutoTokenizer.from_pretrained(final, trust_remote_code=True).save_pretrained(out)
    (out / "fomc_labels.json").write_text(json.dumps({"label_token_ids": meta["label_token_ids"]}))
    return out


def _prompt_ids(texts: list[str], tokenizer) -> list[list[int]]:
    """Token ids exactly as training / evaluate_split built them."""
    from fomc_prompt import encode_classify

    prompts = [encode_classify(t, tokenizer) for t in texts]
    enc = tokenizer(prompts, truncation=True, max_length=cfg.NEMOTRON_MAX_LENGTH)
    return enc["input_ids"]


def eval_vllm(
    method: str,
    seed: int,
    split: str = cfg.DEFAULT_SPLIT,
    lora: bool = False,
    gpu_memory_utilization: float = 0.6,
) -> dict:
    """Weighted F1 + throughput on the paper's test split, served by vLLM."""
    from sklearn.metrics import accuracy_score, f1_score
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    from load_data import load_splits

    run = _run_dir(method, seed, split)
    final = run / "final"
    meta = json.loads((final / "causal_cls.json").read_text())
    label_ids = [int(x) for x in meta["label_token_ids"]]

    if lora:
        # Unmerged: BF16 base + adapter attached per request (hot-swappable).
        adapter_dir = final / "backbone"
        cfg_json = json.loads((adapter_dir / "adapter_config.json").read_text())
        model_path = cfg_json["base_model_name_or_path"]
        llm_kwargs = {"enable_lora": True, "max_lora_rank": int(cfg_json["r"])}
    else:
        model_path = str(merged_dir(method, seed, split))
        if not Path(model_path).exists():
            raise FileNotFoundError(f"{model_path} missing. Run `serve_vllm.py merge` first.")
        llm_kwargs = {}

    tokenizer = AutoTokenizer.from_pretrained(final, trust_remote_code=True)
    _, test = load_splits(seed=str(seed), split=split)
    prompts = [TokensPrompt(prompt_token_ids=ids) for ids in _prompt_ids(test["text"].tolist(), tokenizer)]

    llm = LLM(
        model=model_path,
        dtype="bfloat16",
        trust_remote_code=True,
        max_model_len=cfg.NEMOTRON_MAX_LENGTH + 8,
        gpu_memory_utilization=gpu_memory_utilization,
        **llm_kwargs,
    )
    params = SamplingParams(max_tokens=1, temperature=0.0, allowed_token_ids=label_ids)
    lora_request = None
    if lora:
        from vllm.lora.request import LoRARequest

        lora_request = LoRARequest(f"fomc-{method}-{seed}", 1, str(final / "backbone"))

    llm.generate(prompts[:8], params, lora_request=lora_request, use_tqdm=False)  # warmup
    start = time.perf_counter()
    outputs = llm.generate(prompts, params, lora_request=lora_request, use_tqdm=False)
    elapsed = time.perf_counter() - start

    id_to_label = {tok: i for i, tok in enumerate(label_ids)}
    preds = [id_to_label[o.outputs[0].token_ids[0]] for o in outputs]
    labels = test["label"].tolist()

    train_metrics_path = run / "metrics.json"
    train_f1 = None
    if train_metrics_path.exists():
        train_f1 = json.loads(train_metrics_path.read_text())["test"]["weighted_f1"]

    result = {
        "model": MODEL_KEY,
        "method": method,
        "seed": seed,
        "serving": "vllm-lora" if lora else "vllm-merged",
        "vllm_weighted_f1": float(f1_score(labels, preds, average="weighted")),
        "vllm_accuracy": float(accuracy_score(labels, preds)),
        "pytorch_weighted_f1": train_f1,
        "n_test": len(labels),
        "seconds": round(elapsed, 3),
        "sentences_per_second": round(len(labels) / elapsed, 1),
    }
    out_path = run / f"vllm_metrics{'_lora' if lora else ''}.json"
    out_path.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("step", choices=["merge", "eval"])
    parser.add_argument("--method", default="qlora", choices=["lora", "qlora"])
    parser.add_argument("--seed", type=int, default=cfg.DEFAULT_SEED)
    parser.add_argument("--split", default=cfg.DEFAULT_SPLIT)
    parser.add_argument("--lora", action="store_true", help="eval: serve the unmerged adapter")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.6)
    args = parser.parse_args()
    if args.step == "merge":
        merge_adapter(args.method, args.seed, args.split)
    else:
        eval_vllm(args.method, args.seed, args.split, args.lora, args.gpu_memory_utilization)


if __name__ == "__main__":
    main()
