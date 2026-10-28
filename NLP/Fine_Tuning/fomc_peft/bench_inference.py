"""Inference benchmark on the FOMC test split: accuracy, throughput, latency, memory.

Same 496 test sentences for every model, one GPU, nothing else running.

    # RoBERTa (fine-tuned classifier, PyTorch, training venv)
    .venv/bin/python bench_inference.py roberta --method full --seed 5768

    # Nemotron 3 Nano 4B served by vLLM (vLLM venv)
    .venv-vllm/bin/python bench_inference.py nemotron --variant merged-bf16 --seed 5768
    .venv-vllm/bin/python bench_inference.py nemotron --variant lora-bf16 --seed 5768
    .venv-vllm/bin/python bench_inference.py nemotron --variant lora-fp8 --seed 5768

Variants: merged-bf16 is the adapter folded into the BF16 base; lora-* keeps the
adapter separate (one base can serve many adapters) on NVIDIA's BF16 or FP8 release.

Throughput: the whole test split as one batch of requests, median of REPEATS passes.
Latency: one sentence at a time, median and p95 over LATENCY_N sentences.
Results go to outputs/inference_bench.jsonl (one line per run).
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import config as cfg

REPEATS = 5
LATENCY_N = 100
OUT = Path(__file__).resolve().parent / "outputs" / "inference_bench.jsonl"
FP8_BASE = "nvidia/NVIDIA-Nemotron-3-Nano-4B-FP8"


def _f1(labels, preds) -> tuple[float, float]:
    from sklearn.metrics import accuracy_score, f1_score

    return float(f1_score(labels, preds, average="weighted")), float(accuracy_score(labels, preds))


def _gpu_name() -> str:
    import torch

    return torch.cuda.get_device_name(0)


def _save(result: dict) -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("a") as f:
        f.write(json.dumps(result) + "\n")
    print(json.dumps(result, indent=2))


def bench_roberta(model_key: str, method: str, seed: int, split: str, batch_size: int) -> dict:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    from load_data import load_splits

    final = cfg.run_dir(model_key, method, seed, split) / "final"
    if not (final / "model.safetensors").exists():
        raise FileNotFoundError(f"No full model at {final}; LoRA/QLoRA RoBERTa runs save adapters only.")
    tok = AutoTokenizer.from_pretrained(final)
    model = AutoModelForSequenceClassification.from_pretrained(final, dtype=torch.bfloat16).cuda().eval()
    _, test = load_splits(seed=str(seed), split=split)
    texts, labels = test["text"].tolist(), test["label"].tolist()

    @torch.inference_mode()
    def predict(batch: list[str]) -> list[int]:
        enc = tok(batch, truncation=True, max_length=cfg.MAX_LENGTH, padding=True, return_tensors="pt").to("cuda")
        return model(**enc).logits.argmax(-1).tolist()

    def run_all() -> list[int]:
        out = []
        for i in range(0, len(texts), batch_size):
            out += predict(texts[i : i + batch_size])
        torch.cuda.synchronize()
        return out

    torch.cuda.reset_peak_memory_stats()
    run_all()  # warmup
    times = []
    for _ in range(REPEATS):
        start = time.perf_counter()
        preds = run_all()
        times.append(time.perf_counter() - start)
    lat = []
    for t in texts[:LATENCY_N]:
        start = time.perf_counter()
        predict([t])
        torch.cuda.synchronize()
        lat.append(time.perf_counter() - start)
    f1, acc = _f1(labels, preds)
    return {
        "model": model_key,
        "method": method,
        "variant": "pytorch-bf16",
        "seed": seed,
        "gpu": _gpu_name(),
        "params_b": round(sum(p.numel() for p in model.parameters()) / 1e9, 3),
        "weighted_f1": f1,
        "accuracy": acc,
        "n_test": len(labels),
        "sentences_per_second": round(len(labels) / statistics.median(times), 1),
        "latency_ms_p50": round(1000 * statistics.median(lat), 2),
        "latency_ms_p95": round(1000 * statistics.quantiles(lat, n=20)[18], 2),
        "peak_gpu_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2),
    }


def bench_nemotron(variant: str, method: str, seed: int, split: str, gpu_memory_utilization: float) -> dict:
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    from load_data import load_splits
    from serve_vllm import MODEL_KEY, _prompt_ids, merged_dir

    run = cfg.run_dir(MODEL_KEY, method, seed, split)
    final = run / "final"
    label_ids = [int(x) for x in json.loads((final / "causal_cls.json").read_text())["label_token_ids"]]
    adapter = final / "backbone"
    adapter_cfg = json.loads((adapter / "adapter_config.json").read_text())

    llm_kwargs: dict = {}
    if variant == "merged-bf16":
        model_path = str(merged_dir(method, seed, split))
    elif variant == "lora-bf16":
        model_path = adapter_cfg["base_model_name_or_path"]
    elif variant == "lora-fp8":
        model_path = FP8_BASE
        # FP8 weights; KV cache in BF16. The checkpoint asks for an FP8 KV cache, whose
        # decode kernel this vLLM/FlashInfer build lacks on sm_120. Only 4 of 42 layers
        # are attention and prompts are ~135 tokens, so the KV cache is tiny either way.
        llm_kwargs["kv_cache_dtype"] = "bfloat16"
    else:
        raise ValueError(variant)
    if variant.startswith("lora"):
        llm_kwargs |= {"enable_lora": True, "max_lora_rank": int(adapter_cfg["r"])}

    tok = AutoTokenizer.from_pretrained(final, trust_remote_code=True)
    _, test = load_splits(seed=str(seed), split=split)
    ids = _prompt_ids(test["text"].tolist(), tok)
    prompts = [TokensPrompt(prompt_token_ids=x) for x in ids]
    labels = test["label"].tolist()

    llm = LLM(
        model=model_path,
        trust_remote_code=True,
        max_model_len=cfg.NEMOTRON_MAX_LENGTH + 8,
        gpu_memory_utilization=gpu_memory_utilization,
        # With prefix caching on, vLLM pads each hybrid Mamba request to a ~1k-token
        # cache block, so only a handful of 135-token prompts fit at once.
        enable_prefix_caching=False,
        **llm_kwargs,
    )
    params = SamplingParams(max_tokens=1, temperature=0.0, allowed_token_ids=label_ids)
    lora_request = None
    if variant.startswith("lora"):
        from vllm.lora.request import LoRARequest

        lora_request = LoRARequest(f"fomc-{method}-{seed}", 1, str(adapter))

    def gen(ps):
        return llm.generate(ps, params, lora_request=lora_request, use_tqdm=False)

    gen(prompts)  # warmup (CUDA graphs, LoRA load)
    times = []
    for _ in range(REPEATS):
        start = time.perf_counter()
        outputs = gen(prompts)
        times.append(time.perf_counter() - start)
    lat = []
    for p in prompts[:LATENCY_N]:
        start = time.perf_counter()
        gen([p])
        lat.append(time.perf_counter() - start)

    id_to_label = {t: i for i, t in enumerate(label_ids)}
    preds = [id_to_label[o.outputs[0].token_ids[0]] for o in outputs]
    f1, acc = _f1(labels, preds)
    return {
        "model": MODEL_KEY,
        "method": method,
        "variant": variant,
        "base": model_path if variant != "merged-bf16" else "merged",
        "seed": seed,
        "gpu": _gpu_name(),
        "weighted_f1": f1,
        "accuracy": acc,
        "n_test": len(labels),
        "mean_prompt_tokens": round(sum(map(len, ids)) / len(ids), 1),
        "sentences_per_second": round(len(labels) / statistics.median(times), 1),
        "latency_ms_p50": round(1000 * statistics.median(lat), 2),
        "latency_ms_p95": round(1000 * statistics.quantiles(lat, n=20)[18], 2),
        "gpu_memory_utilization": gpu_memory_utilization,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("family", choices=["roberta", "nemotron"])
    parser.add_argument("--model", default="roberta-large", help="roberta only")
    parser.add_argument("--method", default=None)
    parser.add_argument("--variant", default="merged-bf16", choices=["merged-bf16", "lora-bf16", "lora-fp8"])
    parser.add_argument("--seed", type=int, default=cfg.DEFAULT_SEED)
    parser.add_argument("--split", default=cfg.DEFAULT_SPLIT)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    args = parser.parse_args()
    if args.family == "roberta":
        result = bench_roberta(args.model, args.method or "full", args.seed, args.split, args.batch_size)
    else:
        result = bench_nemotron(args.variant, args.method or "qlora", args.seed, args.split, args.gpu_memory_utilization)
    _save(result)


if __name__ == "__main__":
    main()
