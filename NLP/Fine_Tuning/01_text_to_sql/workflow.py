"""
LLM Fine-Tuning Pipeline — Full, LoRA, and QLoRA in one workflow.

One pipeline, three fine-tuning methods. The `method` flag switches between:
  - full:  train all parameters (best quality, most compute)
  - lora:  freeze base model, train low-rank adapters (good balance)
  - qlora: 4-bit quantized base + LoRA adapters (least memory)

Usage:
    # LoRA fine-tuning (default)
    flyte run --local --tui workflow.py pipeline

    # QLoRA (requires CUDA)
    flyte run --local --tui workflow.py pipeline --method qlora

    # Full fine-tuning with a small model
    flyte run --local --tui workflow.py pipeline --method full

    # Quick local test (few samples)
    flyte run --local --tui workflow.py pipeline --max_train_samples 100 --max_eval_samples 20 --epochs 1

    # Remote (GPU cluster)
    flyte run workflow.py pipeline --method lora --epochs 3
"""

import asyncio
import json
import logging
import os
import tempfile
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import flyte
import flyte.io
import flyte.report
from config import cpu_env, gpu_env, HF_TOKEN
from report_helpers import make_bar_chart, make_line_chart, pipeline_step_indicator, wrap_report

logging.basicConfig(level=logging.WARNING, format="%(message)s", force=True)
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

OUTPUTS_DIR = Path(__file__).resolve().parent / "outputs"


def _save_train_stats(
    method: str,
    model_name: str,
    trainable_params: int,
    total_params: int,
    device: str,
) -> None:
    """Sidecar so the notebook comparison table can show trainable % without Flyte UI."""
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "method": method,
        "model_name": model_name,
        "trainable_params": trainable_params,
        "total_params": total_params,
        "trainable_pct": round(100 * trainable_params / max(total_params, 1), 3),
        "device": device,
    }
    (OUTPUTS_DIR / f"train_{method}.json").write_text(json.dumps(payload, indent=2))


def _save_run_metrics(
    method: str,
    model_name: str,
    metrics: dict,
    extra: dict,
) -> None:
    """Write exact-match SQL scores for the notebook results section."""
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    payload = {"method": method, "model_name": model_name, **extra, **metrics}
    train_path = OUTPUTS_DIR / f"train_{method}.json"
    if train_path.exists():
        train = json.loads(train_path.read_text())
        payload = {**train, **payload}
    (OUTPUTS_DIR / f"run_{method}.json").write_text(
        json.dumps(payload, indent=2, default=str)
    )


# ------------------------------------------------------------------
# Task 1: Prepare dataset
# ------------------------------------------------------------------

@cpu_env.task(cache="auto")
async def prepare_data(
    dataset_name: str = "b-mc2/sql-create-context",
    max_train_samples: int = 5000,
    max_eval_samples: int = 500,
) -> flyte.io.Dir:
    """Download dataset from HuggingFace and format for instruction fine-tuning."""
    from datasets import DatasetDict, load_dataset

    log.info(f"Loading dataset: {dataset_name}")
    ds = load_dataset(dataset_name, split="train")

    def format_example(ex):
        return {
            "text": (
                "### Task: Generate a SQL query to answer the question.\n"
                f"### Schema:\n{ex['context']}\n"
                f"### Question:\n{ex['question']}\n"
                f"### SQL:\n{ex['answer']}\n<|endoftext|>"
            )
        }

    ds = ds.map(format_example)

    # Split into train and eval
    total = len(ds)
    train_end = min(max_train_samples, total - max_eval_samples)
    eval_start = train_end
    eval_end = min(eval_start + max_eval_samples, total)

    processed = DatasetDict({
        "train": ds.select(range(train_end)),
        "eval": ds.select(range(eval_start, eval_end)),
    })

    output_dir = os.path.join(tempfile.mkdtemp(), "dataset")
    processed.save_to_disk(output_dir)
    log.info(f"Dataset ready: {len(processed['train'])} train, {len(processed['eval'])} eval")

    return await flyte.io.Dir.from_local(output_dir)


# ------------------------------------------------------------------
# Task 2: Train
# ------------------------------------------------------------------

@gpu_env.task(report=True)
async def train(
    model_name: str,
    data_dir: flyte.io.Dir,
    method: str = "lora",
    epochs: int = 3,
    lr: float = 2e-4,
    batch_size: int = 4,
    lora_r: int = 16,
    lora_alpha: int = 32,
) -> flyte.io.Dir:
    """Fine-tune a model using full, LoRA, or QLoRA method."""
    import torch
    from datasets import load_from_disk
    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
    from trl import SFTConfig, SFTTrainer
    from device_utils import (
        describe as describe_device,
        from_pretrained_kwargs,
        model_dtype,
        place_model,
        require_qlora,
        trainer_precision,
    )

    log.info(f"Training: model={model_name}, method={method}, device={describe_device()}")

    # -- Load data --
    data_path = await data_dir.download()
    dataset = load_from_disk(data_path)

    # -- Load tokenizer --
    token_kwargs = {"token": HF_TOKEN} if HF_TOKEN else {}
    tokenizer = AutoTokenizer.from_pretrained(model_name, **token_kwargs)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # -- Initial report: loading model --
    await flyte.report.replace.aio(
        wrap_report(
            f"<h2>Loading Model...</h2>"
            f"<h3>{model_name}</h3>"
            f'<div class="card">'
            f"<p><b>Method:</b> <span class=\"badge badge-info\">{method.upper()}</span></p>"
            f"<p><b>Dataset:</b> {len(dataset['train']):,} train / {len(dataset['eval']):,} eval</p>"
            f"</div>"
        ),
        do_flush=True,
    )

    dtype = model_dtype()
    precision = trainer_precision()

    if method == "qlora":
        require_qlora()
        from transformers import BitsAndBytesConfig

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            **token_kwargs,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=dtype,
                bnb_4bit_use_double_quant=True,
            ),
            **from_pretrained_kwargs(dtype, quantized=True),
        )
        model = place_model(model, quantized=True)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            **token_kwargs,
            **from_pretrained_kwargs(dtype),
        )
        model = place_model(model)

    # -- Apply LoRA adapters --
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())

    if method in ("lora", "qlora"):
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

        if method == "qlora":
            model = prepare_model_for_kbit_training(model)

        lora_config = LoraConfig(
            r=lora_r,              # Rank — size of the low-rank matrices. Higher = more capacity but more params
            lora_alpha=lora_alpha,  # Scaling factor — controls adapter impact. Effective scale = alpha/r
            # Attention layers — LoRA adapters inject low-rank updates here:
            #   q_proj (Query)     — what to look for in context
            #   k_proj (Key)       — what each token offers to match against
            #   v_proj (Value)     — what information to extract once matched
            #   o_proj (Output)    — combines multi-head attention results
            # MLP layers — LoRA adapters also update the feed-forward network:
            #   gate_proj (Gate)   — controls how much information flows through (SwiGLU activation)
            #   up_proj (Up)       — projects to a higher dimension for richer representations
            #   down_proj (Down)   — projects back down to the model's hidden size
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_dropout=0.05,     # Dropout on adapter weights — light regularization to prevent overfitting
            bias="none",           # Don't train bias terms — keeps adapter small and stable
            task_type="CAUSAL_LM", # Tells PEFT this is a text generation model (vs classification, etc.)
        )
        model = get_peft_model(model, lora_config)
        trainable_params, total_params = model.get_nb_trainable_parameters()
        log.info(f"Trainable params: {trainable_params:,} / {total_params:,} ({trainable_params / total_params * 100:.1f}%)")

    _save_train_stats(method, model_name, trainable_params, total_params, describe_device())

    # -- Live training report state --
    training_log: list[dict] = []
    loop = asyncio.get_running_loop()

    method_badge = f'<span class="badge badge-info">{method.upper()}</span>'
    if method == "qlora":
        method_badge = f'<span class="badge badge-success">QLoRA (4-bit)</span>'
    elif method == "full":
        method_badge = f'<span class="badge badge-danger">Full Fine-Tune</span>'

    def _build_training_report(max_steps: int) -> str:
        """Build the live training report HTML from current training_log."""
        stats_html = f"""
        <h2>Training in Progress...</h2>
        <h3>{model_name}</h3>
        <div class="stat-grid">
          <div class="stat"><div class="value">{method.upper()}</div><div class="label">Method</div></div>
          <div class="stat"><div class="value">{len(dataset['train']):,}</div><div class="label">Train Examples</div></div>
          <div class="stat"><div class="value">{epochs}</div><div class="label">Epochs</div></div>
          <div class="stat"><div class="value">{lr}</div><div class="label">Learning Rate</div></div>
          <div class="stat"><div class="value">{batch_size}</div><div class="label">Batch Size</div></div>
          <div class="stat"><div class="value">{trainable_params / total_params * 100:.1f}%</div><div class="label">Trainable</div></div>
        </div>
        <p>Method: {method_badge} | Total params: {total_params:,} | Trainable: {trainable_params:,}</p>
        """

        charts_html = ""
        if training_log:
            current = training_log[-1]
            progress_pct = current["step"] / max_steps * 100 if max_steps else 0
            charts_html += f"""
            <div class="card">
              <b>Step {current['step']}/{max_steps}</b>
              ({progress_pct:.0f}%) |
              Epoch {current['epoch']:.2f}/{epochs} |
              Loss: <span class="highlight">{current['loss']:.4f}</span>
              <div style="background:#e9ecef;border-radius:4px;height:8px;margin-top:8px;">
                <div style="background:#0f3460;width:{progress_pct:.1f}%;height:100%;border-radius:4px;"></div>
              </div>
            </div>
            """

            loss_chart = make_line_chart(
                data=training_log,
                x_key="epoch",
                y_keys=["loss"],
                title="Training Loss",
                x_label="Epoch",
                y_label="Loss",
                colors=["#5a7db5"],
            )
            charts_html += f'<div class="chart-container">{loss_chart}</div>'

            if "lr" in training_log[0]:
                lr_chart = make_line_chart(
                    data=training_log,
                    x_key="epoch",
                    y_keys=["lr"],
                    title="Learning Rate Schedule",
                    x_label="Epoch",
                    y_label="LR",
                    colors=["#0f3460"],
                )
                charts_html += f'<div class="chart-container">{lr_chart}</div>'

            if "grad_norm" in training_log[0]:
                grad_chart = make_line_chart(
                    data=training_log,
                    x_key="epoch",
                    y_keys=["grad_norm"],
                    title="Gradient Norm",
                    x_label="Epoch",
                    y_label="Grad Norm",
                    colors=["#06d6a0"],
                )
                charts_html += f'<div class="chart-container">{grad_chart}</div>'

        return wrap_report(stats_html + charts_html)

    # -- Metrics callback with live report updates --
    class MetricsCallback(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            if not logs or "loss" not in logs:
                return
            entry = {
                "step": state.global_step,
                "epoch": round(logs.get("epoch", 0), 2),
                "loss": round(logs["loss"], 4),
            }
            if "learning_rate" in logs:
                entry["lr"] = logs["learning_rate"]
            if "grad_norm" in logs:
                entry["grad_norm"] = round(float(logs["grad_norm"]), 4)
            training_log.append(entry)
            log.info(
                f"step={state.global_step}/{state.max_steps} "
                f"epoch={entry['epoch']:.2f} "
                f"loss={entry['loss']:.4f}"
            )

            asyncio.run_coroutine_threadsafe(
                flyte.report.replace.aio(
                    _build_training_report(state.max_steps),
                    do_flush=True,
                ),
                loop,
            )

    # -- Train --
    output_dir = os.path.join(tempfile.mkdtemp(), "checkpoints")
    training_args = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        learning_rate=lr,
        logging_steps=10,
        save_strategy="epoch",
        bf16=precision["bf16"],
        fp16=precision["fp16"],
        gradient_accumulation_steps=4,
        warmup_steps=10,
        report_to="none",
        dataloader_num_workers=0,
        dataloader_pin_memory=False,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["eval"],
        processing_class=tokenizer,
        callbacks=[MetricsCallback()],
    )

    log.info("Starting training...")
    await asyncio.to_thread(trainer.train)
    log.info("Training complete.")

    # -- Merge LoRA weights and save --
    save_dir = os.path.join(tempfile.mkdtemp(), "finetuned_model")

    if method in ("lora", "qlora"):
        log.info("Merging LoRA weights into base model...")
        model = model.merge_and_unload()

    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    log.info(f"Model saved to {save_dir}")

    # -- Final training report --
    final_loss = training_log[-1]["loss"] if training_log else "N/A"

    loss_chart = make_line_chart(
        data=training_log,
        x_key="epoch",
        y_keys=["loss"],
        title="Training Loss",
        x_label="Epoch",
        y_label="Loss",
        colors=["#5a7db5"],
    ) if training_log else ""

    lr_chart = ""
    if training_log and "lr" in training_log[0]:
        lr_chart = make_line_chart(
            data=training_log,
            x_key="epoch",
            y_keys=["lr"],
            title="Learning Rate Schedule",
            x_label="Epoch",
            y_label="LR",
            colors=["#0f3460"],
        )

    await flyte.report.replace.aio(
        wrap_report(
            f"<h2>Training Complete</h2>"
            f"<h3>{model_name}</h3>"
            f'<div class="stat-grid">'
            f'  <div class="stat"><div class="value">{method.upper()}</div><div class="label">Method</div></div>'
            f'  <div class="stat"><div class="value">{final_loss}</div><div class="label">Final Loss</div></div>'
            f'  <div class="stat"><div class="value">{epochs}</div><div class="label">Epochs</div></div>'
            f'  <div class="stat"><div class="value">{total_params:,}</div><div class="label">Total Params</div></div>'
            f'  <div class="stat"><div class="value">{trainable_params:,}</div><div class="label">Trainable Params</div></div>'
            f'  <div class="stat"><div class="value">{trainable_params / total_params * 100:.1f}%</div><div class="label">% Trainable</div></div>'
            f'</div>'
            f'<div class="chart-container">{loss_chart}</div>'
            f'{f"""<div class="chart-container">{lr_chart}</div>""" if lr_chart else ""}'
        ),
        do_flush=True,
    )

    return await flyte.io.Dir.from_local(save_dir)


# ------------------------------------------------------------------
# Task 3: Evaluate — before/after comparison
# ------------------------------------------------------------------

@gpu_env.task(report=True)
async def evaluate(
    model_name: str,
    finetuned_dir: flyte.io.Dir,
    data_dir: flyte.io.Dir,
    num_examples: int = 50,
) -> str:
    """Compare base model vs fine-tuned model on test examples."""
    import torch
    from datasets import load_from_disk
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from device_utils import empty_cache, from_pretrained_kwargs, model_dtype, place_model

    log.info("Starting evaluation...")
    await flyte.report.replace.aio(
        wrap_report(
            "<h2>Evaluation</h2>"
            '<div class="card"><p>Loading models and running inference...</p></div>'
        ),
        do_flush=True,
    )

    dtype = model_dtype()

    # Load eval data
    data_path = await data_dir.download()
    dataset = load_from_disk(data_path)
    eval_ds = dataset["eval"].select(range(min(num_examples, len(dataset["eval"]))))

    # Load tokenizer
    token_kwargs = {"token": HF_TOKEN} if HF_TOKEN else {}
    tokenizer = AutoTokenizer.from_pretrained(model_name, **token_kwargs)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def generate_sql(model, prompt, max_new_tokens=128):
        device = getattr(model, "device", None) or next(model.parameters()).device
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        return tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()

    def normalize_sql(sql):
        """Extract the first SQL statement and normalize for comparison."""
        # Truncate at first ### or newline to isolate the SQL
        for stop in ["###", "\n"]:
            if stop in sql:
                sql = sql[:sql.index(stop)]
        return " ".join(sql.lower().split()).strip().rstrip(";")

    def build_prompt(example):
        return (
            "### Task: Generate a SQL query to answer the question.\n"
            f"### Schema:\n{example['context']}\n"
            f"### Question:\n{example['question']}\n"
            "### SQL:\n"
        )

    # -- Run base model --
    log.info(f"Loading base model: {model_name}")
    await flyte.report.replace.aio(
        wrap_report(
            f"<h2>Evaluation</h2>"
            f'<div class="stat-grid">'
            f'  <div class="stat"><div class="value">{len(eval_ds)}</div><div class="label">Eval Examples</div></div>'
            f'  <div class="stat"><div class="value">1/2</div><div class="label">Phase</div></div>'
            f'</div>'
            f'<div class="card"><p>Running <b>base model</b> inference...</p>'
            f'<div style="background:#e9ecef;border-radius:4px;height:8px;margin-top:8px;">'
            f'<div style="background:#adb5bd;width:25%;height:100%;border-radius:4px;"></div>'
            f'</div></div>'
        ),
        do_flush=True,
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        model_name, **token_kwargs, **from_pretrained_kwargs(dtype),
    )
    base_model = place_model(base_model)

    base_results = []
    for i, example in enumerate(eval_ds):
        prompt = build_prompt(example)
        generated = generate_sql(base_model, prompt)
        base_results.append(generated)
        if (i + 1) % 10 == 0:
            log.info(f"Base model: {i + 1}/{len(eval_ds)}")
            pct = (i + 1) / len(eval_ds) * 50
            await flyte.report.replace.aio(
                wrap_report(
                    f"<h2>Evaluation</h2>"
                    f'<div class="card"><p>Running <b>base model</b> inference... {i + 1}/{len(eval_ds)}</p>'
                    f'<div style="background:#e9ecef;border-radius:4px;height:8px;margin-top:8px;">'
                    f'<div style="background:#adb5bd;width:{pct:.0f}%;height:100%;border-radius:4px;"></div>'
                    f'</div></div>'
                ),
                do_flush=True,
            )

    del base_model
    empty_cache()

    # -- Run fine-tuned model --
    log.info("Loading fine-tuned model...")
    await flyte.report.replace.aio(
        wrap_report(
            f"<h2>Evaluation</h2>"
            f'<div class="card"><p>Running <b>fine-tuned model</b> inference...</p>'
            f'<div style="background:#e9ecef;border-radius:4px;height:8px;margin-top:8px;">'
            f'<div style="background:#0f3460;width:50%;height:100%;border-radius:4px;"></div>'
            f'</div></div>'
        ),
        do_flush=True,
    )

    ft_path = await finetuned_dir.download()
    ft_model = AutoModelForCausalLM.from_pretrained(
        ft_path, **from_pretrained_kwargs(dtype),
    )
    ft_model = place_model(ft_model)

    ft_results = []
    for i, example in enumerate(eval_ds):
        prompt = build_prompt(example)
        generated = generate_sql(ft_model, prompt)
        ft_results.append(generated)
        if (i + 1) % 10 == 0:
            log.info(f"Fine-tuned model: {i + 1}/{len(eval_ds)}")
            pct = 50 + (i + 1) / len(eval_ds) * 50
            await flyte.report.replace.aio(
                wrap_report(
                    f"<h2>Evaluation</h2>"
                    f'<div class="card"><p>Running <b>fine-tuned model</b> inference... {i + 1}/{len(eval_ds)}</p>'
                    f'<div style="background:#e9ecef;border-radius:4px;height:8px;margin-top:8px;">'
                    f'<div style="background:#0f3460;width:{pct:.0f}%;height:100%;border-radius:4px;"></div>'
                    f'</div></div>'
                ),
                do_flush=True,
            )

    del ft_model
    empty_cache()

    # -- Score --
    base_correct = 0
    ft_correct = 0
    comparisons = []

    for i, example in enumerate(eval_ds):
        expected = example["answer"]
        base_gen = base_results[i]
        ft_gen = ft_results[i]

        base_match = normalize_sql(base_gen) == normalize_sql(expected)
        ft_match = normalize_sql(ft_gen) == normalize_sql(expected)

        if base_match:
            base_correct += 1
        if ft_match:
            ft_correct += 1

        comparisons.append({
            "question": example["question"],
            "schema": example["context"],
            "expected": expected,
            "base": base_gen,
            "finetuned": ft_gen,
            "base_correct": base_match,
            "ft_correct": ft_match,
        })

    total = len(eval_ds)
    base_acc = base_correct / total * 100
    ft_acc = ft_correct / total * 100
    improvement = ft_acc - base_acc

    log.info(f"Base model accuracy: {base_acc:.1f}% ({base_correct}/{total})")
    log.info(f"Fine-tuned accuracy: {ft_acc:.1f}% ({ft_correct}/{total})")

    # -- Build final eval report --
    improvement_badge = (
        f'<span class="badge badge-success">+{improvement:.1f}pp</span>'
        if improvement > 0
        else f'<span class="badge badge-danger">{improvement:.1f}pp</span>'
    )

    bar_chart = make_bar_chart(
        labels=["Exact Match Accuracy"],
        series={
            "Base Model": [base_acc],
            "Fine-Tuned": [ft_acc],
        },
        title="Base vs Fine-Tuned Accuracy",
        colors=["#adb5bd", "#0f3460"],
        y_max_cap=100.0,
    )

    examples_html = ""
    for c in comparisons[:10]:
        base_badge = '<span class="badge badge-success">correct</span>' if c["base_correct"] else '<span class="badge badge-danger">wrong</span>'
        ft_badge = '<span class="badge badge-success">correct</span>' if c["ft_correct"] else '<span class="badge badge-danger">wrong</span>'
        examples_html += f"""
        <div class="card">
          <p><b>Q:</b> {c['question']}</p>
          <p style="font-size:0.85em; color:#6c757d;"><b>Schema:</b> {c['schema'][:200]}...</p>
          <table>
            <tr><th>Source</th><th>SQL</th><th>Result</th></tr>
            <tr><td>Expected</td><td><code>{c['expected']}</code></td><td></td></tr>
            <tr><td>Base</td><td><code>{c['base'][:200]}</code></td><td>{base_badge}</td></tr>
            <tr><td>Fine-tuned</td><td><code>{c['finetuned'][:200]}</code></td><td>{ft_badge}</td></tr>
          </table>
        </div>"""

    await flyte.report.replace.aio(
        wrap_report(
            f"<h2>Evaluation Results</h2>"
            f'<div class="stat-grid">'
            f'  <div class="stat"><div class="value">{base_acc:.1f}%</div><div class="label">Base Accuracy</div></div>'
            f'  <div class="stat"><div class="value">{ft_acc:.1f}%</div><div class="label">Fine-Tuned Accuracy</div></div>'
            f'  <div class="stat"><div class="value">{improvement:+.1f}pp</div><div class="label">Improvement</div></div>'
            f'  <div class="stat"><div class="value">{total}</div><div class="label">Eval Examples</div></div>'
            f'</div>'
            f'<div class="chart-container">{bar_chart}</div>'
            f'<h3>Example Comparisons {improvement_badge}</h3>'
            f'{examples_html}'
            f'<div class="note">'
            f'<b>Note:</b> Exact match accuracy compares normalized SQL output. '
            f'The fine-tuned model may generate semantically correct queries that differ in formatting.'
            f'</div>'
        ),
        do_flush=True,
    )

    return json.dumps({
        "base_accuracy": round(base_acc, 1),
        "finetuned_accuracy": round(ft_acc, 1),
        "improvement": round(ft_acc - base_acc, 1),
        "num_examples": total,
        "comparisons": comparisons[:10],
    })


# ------------------------------------------------------------------
# Pipeline: orchestrate everything
# ------------------------------------------------------------------


@cpu_env.task(report=True)
async def pipeline(
    model_name: str = "HuggingFaceTB/SmolLM2-135M",
    dataset_name: str = "b-mc2/sql-create-context",
    method: str = "lora",
    epochs: int = 3,
    lr: float = 2e-4,
    batch_size: int = 4,
    max_train_samples: int = 5000,
    max_eval_samples: int = 500,
    num_eval_examples: int = 50,
    lora_r: int = 16,
    lora_alpha: int = 32,
) -> flyte.io.Dir:
    """
    End-to-end LLM fine-tuning pipeline.

    1. Download and format dataset
    2. Fine-tune model (full / LoRA / QLoRA)
    3. Evaluate: before/after comparison on test set

    Returns the fine-tuned model directory so it can be served directly.
    """
    log.info(f"Pipeline: {model_name} | method={method} | dataset={dataset_name}")
    steps = ["Prepare Data", "Train", "Evaluate"]

    method_badge = f'<span class="badge badge-info">{method.upper()}</span>'

    # Step 1: Prepare data
    await flyte.report.replace.aio(
        wrap_report(
            f"<h2>LLM Fine-Tuning Pipeline</h2>"
            f"<h3>{model_name} {method_badge}</h3>"
            f'{pipeline_step_indicator(0, steps)}'
            f'<div class="card"><p>Downloading and formatting dataset: <b>{dataset_name}</b>...</p></div>'
        ),
        do_flush=True,
    )

    data_dir = await prepare_data(dataset_name, max_train_samples, max_eval_samples)

    # Step 2: Train
    await flyte.report.replace.aio(
        wrap_report(
            f"<h2>LLM Fine-Tuning Pipeline</h2>"
            f"<h3>{model_name} {method_badge}</h3>"
            f'{pipeline_step_indicator(1, steps)}'
            f'<div class="card"><p>Training in progress... check the <b>train</b> task report for live charts.</p></div>'
        ),
        do_flush=True,
    )

    finetuned_dir = await train(
        model_name, data_dir, method, epochs, lr, batch_size, lora_r, lora_alpha,
    )

    # Step 3: Evaluate
    await flyte.report.replace.aio(
        wrap_report(
            f"<h2>LLM Fine-Tuning Pipeline</h2>"
            f"<h3>{model_name} {method_badge}</h3>"
            f'{pipeline_step_indicator(2, steps)}'
            f'<div class="card"><p>Evaluating base vs fine-tuned model...</p></div>'
        ),
        do_flush=True,
    )

    result = await evaluate(model_name, finetuned_dir, data_dir, num_eval_examples)
    metrics = json.loads(result)
    _save_run_metrics(
        method=method,
        model_name=model_name,
        metrics=metrics,
        extra={
            "epochs": epochs,
            "lr": lr,
            "batch_size": batch_size,
            "max_train_samples": max_train_samples,
            "max_eval_samples": max_eval_samples,
            "num_eval_examples": num_eval_examples,
            "lora_r": lora_r,
            "lora_alpha": lora_alpha,
        },
    )

    # Final pipeline report
    improvement = metrics["improvement"]
    improvement_badge = (
        f'<span class="badge badge-success">+{improvement:.1f}pp</span>'
        if improvement > 0
        else f'<span class="badge badge-danger">{improvement:.1f}pp</span>'
    )

    await flyte.report.replace.aio(
        wrap_report(
            f"<h2>Pipeline Complete</h2>"
            f"<h3>{model_name} {method_badge}</h3>"
            f'{pipeline_step_indicator(3, steps)}'
            f'<div class="stat-grid">'
            f'  <div class="stat"><div class="value">{metrics["base_accuracy"]}%</div><div class="label">Base Accuracy</div></div>'
            f'  <div class="stat"><div class="value">{metrics["finetuned_accuracy"]}%</div><div class="label">Fine-Tuned Accuracy</div></div>'
            f'  <div class="stat"><div class="value">{improvement:+.1f}pp</div><div class="label">Improvement {improvement_badge}</div></div>'
            f'  <div class="stat"><div class="value">{method.upper()}</div><div class="label">Method</div></div>'
            f'  <div class="stat"><div class="value">{epochs}</div><div class="label">Epochs</div></div>'
            f'  <div class="stat"><div class="value">{metrics["num_examples"]}</div><div class="label">Eval Examples</div></div>'
            f'</div>'
            f'<div class="note">'
            f'Check the <b>train</b> task report for training loss/LR charts, '
            f'and the <b>evaluate</b> task report for detailed example comparisons.'
            f'</div>'
        ),
        do_flush=True,
    )

    log.info(f"Pipeline complete. Improvement: {metrics['improvement']:+.1f}pp")
    return finetuned_dir
