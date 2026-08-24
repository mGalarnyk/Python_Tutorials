# Image sources

The prompt used to generate this figure is in [../images/PROMPT.md](../images/PROMPT.md).

| File | What it shows | Idea credit |
|------|----------------|-------------|
| `flyte_pipeline.png` | One Flyte 2 DAG on a MacBook Pro (M4 Max, 128 GB, 16-core 12P+4E) and on a remote NVIDIA RTX PRO 6000 Blackwell (1 or 4 GPUs on one node; 16-core / 256 GB host) | Flyte 2 task environments (CPU vs GPU). Hardware names are this tutorial’s machines. Pipeline steps match `workflow.py` (`prepare_data` → `train` → `evaluate`, optional serve). |
