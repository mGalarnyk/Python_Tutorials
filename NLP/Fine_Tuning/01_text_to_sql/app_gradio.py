"""
Text-to-SQL — Gradio frontend.

Type a question and schema, get a SQL query from the fine-tuned model
served by serve.py.

Usage:
    # Deploy to cluster
    python app_gradio.py

    # Local test (requires serve.py running)
    SERVER_URL=https://your-app-url python app_gradio.py
"""

import logging
import os
import pathlib

import flyte
import flyte.app

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Image & environment
# ------------------------------------------------------------------

app_image = flyte.Image.from_debian_base(
    name="sql-generator-gradio",
).with_pip_packages(
    "flyte[tui]>=2.0",
    "gradio>=5.0.0",
    "httpx",
)

SERVER_APP_NAME = "finetuned-sql-api"

gradio_env = flyte.app.AppEnvironment(
    name="sql-generator-ui",
    image=app_image,
    resources=flyte.Resources(cpu=1, memory="2Gi"),
    requires_auth=False,
    port=7860,
    parameters=[
        flyte.app.Parameter(
            name="server_url",
            value="",
            env_var="SERVER_URL",
        ),
    ],
    scaling=flyte.app.Scaling(
        replicas=(0, 1),
        scaledown_after=1800,
    ),
)


# ------------------------------------------------------------------
# Gradio app
# ------------------------------------------------------------------

EXAMPLE_SCHEMAS = [
    "CREATE TABLE employees (id INT, name VARCHAR, department VARCHAR, salary INT)",
    "CREATE TABLE orders (order_id INT, customer_id INT, product VARCHAR, quantity INT, price DECIMAL, order_date DATE)",
    "CREATE TABLE students (student_id INT, name VARCHAR, grade VARCHAR, gpa FLOAT, major VARCHAR)",
]

EXAMPLE_QUESTIONS = [
    ("What is the average salary by department?", EXAMPLE_SCHEMAS[0]),
    ("Which product had the most orders?", EXAMPLE_SCHEMAS[1]),
    ("List students with a GPA above 3.5", EXAMPLE_SCHEMAS[2]),
    ("What is the total revenue per product?", EXAMPLE_SCHEMAS[1]),
    ("How many employees are in each department?", EXAMPLE_SCHEMAS[0]),
]


@gradio_env.server
def launch_app(server_url: str):
    import gradio as gr
    import httpx

    api_url = server_url.rstrip("/")
    log.info(f"Connecting to SQL server: {api_url}")

    def generate_sql(schema: str, question: str):
        if not question.strip():
            return "", "", "Please enter a question."

        try:
            response = httpx.post(
                f"{api_url}/generate",
                json={"schema": schema, "question": question},
                timeout=30.0,
            )
            response.raise_for_status()
            result = response.json()
        except httpx.ConnectError:
            return "", "", "Could not connect to the model server. Is it running?"
        except Exception as e:
            return "", "", f"Error: {e}"

        return result["sql"], result["raw_output"], ""

    with gr.Blocks(title="Text-to-SQL Generator") as demo:
        gr.Markdown(
            "# Text-to-SQL Generator\n"
            "Enter a database schema and a natural language question to generate a SQL query.\n\n"
            "Powered by a fine-tuned SmolLM2-135M model trained with LoRA on the "
            "[sql-create-context](https://huggingface.co/datasets/b-mc2/sql-create-context) dataset."
        )

        with gr.Row():
            with gr.Column(scale=1):
                schema_input = gr.Textbox(
                    label="Database Schema",
                    placeholder="CREATE TABLE employees (id INT, name VARCHAR, ...)",
                    lines=4,
                    value=EXAMPLE_SCHEMAS[0],
                )
                question_input = gr.Textbox(
                    label="Question",
                    placeholder="What is the average salary by department?",
                    lines=2,
                )
                generate_btn = gr.Button("Generate SQL", variant="primary", size="lg")

            with gr.Column(scale=1):
                sql_output = gr.Code(
                    label="Generated SQL",
                    language="sql",
                    lines=4,
                )
                raw_output = gr.Textbox(
                    label="Raw Model Output",
                    lines=6,
                    interactive=False,
                )
                error_output = gr.Textbox(
                    label="",
                    visible=False,
                )

        gr.Examples(
            examples=[[s, q] for q, s in EXAMPLE_QUESTIONS],
            inputs=[schema_input, question_input],
            label="Examples — click to load",
        )

        generate_btn.click(
            generate_sql,
            inputs=[schema_input, question_input],
            outputs=[sql_output, raw_output, error_output],
        )

        question_input.submit(
            generate_sql,
            inputs=[schema_input, question_input],
            outputs=[sql_output, raw_output, error_output],
        )

    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=os.environ.get("GRADIO_SHARE", "") == "1",
        theme=gr.themes.Soft(),
    )


# ------------------------------------------------------------------
# Deploy / local dev
# ------------------------------------------------------------------

if __name__ == "__main__":
    server_url = os.environ.get("SERVER_URL", "")

    if server_url:
        launch_app(server_url)
    else:
        flyte.init_from_config(root_dir=pathlib.Path(__file__).parent)

        from flyte.remote._app import App

        try:
            server_app = App.get(SERVER_APP_NAME)
            server_endpoint = server_app.endpoint
            log.info(f"Found SQL server at: {server_endpoint}")
        except Exception:
            raise RuntimeError(
                f"Could not find deployed app '{SERVER_APP_NAME}'. "
                "Deploy the model server first with: python serve.py"
            )

        for p in gradio_env.parameters:
            if p.name == "server_url":
                p.value = server_endpoint
                break

        log.info("Deploying Gradio frontend...")
        deployed = flyte.deploy(gradio_env)
        log.info(f"Gradio app deployed: {deployed}")
