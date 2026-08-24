"""
Deploy the fine-tuned model as a FastAPI endpoint.

Usage:
    # Deploy the latest trained model
    python serve.py

    # Deploy from a specific pipeline or train task run
    python serve.py --run-name <run-name>

    # Test the endpoint
    curl -X POST https://your-app-url/generate \
      -H "Content-Type: application/json" \
      -d '{"schema": "CREATE TABLE employees (id INT, name VARCHAR, department VARCHAR, salary INT)",
           "question": "What is the average salary by department?"}'
"""

import argparse
import logging
from contextlib import asynccontextmanager
from pathlib import Path

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from transformers import AutoModelForCausalLM, AutoTokenizer

import flyte
from flyte.app import Parameter
from flyte.app.extras import FastAPIAppEnvironment

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_MOUNT_PATH = "/tmp/finetuned_model"


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting up: Loading model...")
    model_path = Path(MODEL_MOUNT_PATH)

    if model_path.exists():
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        app.state.tokenizer = AutoTokenizer.from_pretrained(str(model_path))
        app.state.model = AutoModelForCausalLM.from_pretrained(
            str(model_path), dtype=dtype, device_map="auto",
        )
        app.state.model.eval()
        logger.info("Model loaded successfully")
    else:
        logger.warning(f"Model not found at {model_path}")
        app.state.model = None
        app.state.tokenizer = None

    yield
    logger.info("Shutting down...")


app = FastAPI(
    title="Fine-Tuned SQL Generator",
    description="Generate SQL queries from natural language using a fine-tuned LLM",
    version="1.0.0",
    lifespan=lifespan,
)

env = FastAPIAppEnvironment(
    name="finetuned-sql-api",
    app=app,
    description="Fine-tuned text-to-SQL model serving",
    image=flyte.Image.from_debian_base().with_pip_packages(
        "fastapi", "uvicorn", "torch", "transformers", "accelerate",
    ),
    resources=flyte.Resources(cpu=2, memory="8Gi", gpu=1),
    scaling=flyte.app.Scaling(
        replicas=(0, 1),
        scaledown_after=1800,  # 30 minutes
    ),
    requires_auth=False,
    parameters=[
        Parameter(
            name="model",
            value=flyte.app.RunOutput(
                task_name="llm-finetune-cpu.pipeline",
                type="directory",
            ),
            mount=MODEL_MOUNT_PATH,
        ),
    ],
)


class SQLRequest(BaseModel):
    model_config = {"populate_by_name": True}

    schema_: str | None = Field(None, alias="schema")
    question: str

    @property
    def context(self) -> str:
        return self.schema_ or ""


class SQLResponse(BaseModel):
    sql: str
    raw_output: str


@app.get("/health")
async def health():
    return {
        "status": "healthy" if app.state.model is not None else "not_ready",
        "model_loaded": app.state.model is not None,
    }


@app.post("/generate", response_model=SQLResponse)
async def generate_sql(request: SQLRequest):
    if app.state.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    prompt = (
        "### Task: Generate a SQL query to answer the question.\n"
        f"### Schema:\n{request.context}\n"
        f"### Question:\n{request.question}\n"
        "### SQL:\n"
    )

    tokenizer = app.state.tokenizer
    inputs = tokenizer(prompt, return_tensors="pt").to(app.state.model.device)

    with torch.no_grad():
        outputs = app.state.model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    raw = tokenizer.decode(
        outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True
    ).strip()

    # Extract just the SQL (first line before ### or newline)
    sql = raw
    for stop in ["###", "\n"]:
        if stop in sql:
            sql = sql[:sql.index(stop)]
    sql = sql.strip()

    return SQLResponse(sql=sql, raw_output=raw)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deploy fine-tuned SQL model")
    parser.add_argument(
        "--run-name",
        help="Run name of a specific pipeline or train task run (from Flyte UI). "
             "Default: latest pipeline output.",
    )
    args = parser.parse_args()

    flyte.init_from_config()

    # Override the parameter with a specific run if provided
    if args.run_name:
        for p in env.parameters:
            if p.name == "model":
                p.value = flyte.app.RunOutput(
                    run_name=args.run_name,
                    type="directory",
                )
                break

    deployed = flyte.serve(env)
    print(f"Deployed: {deployed.url}")
