# Image sources

The prompt used to generate these figures is in [../images/PROMPT.md](../images/PROMPT.md).

| File | What it shows | Idea credit |
|------|----------------|-------------|
| `flyte_pipeline.png` | MacBook Pro (M4 Max, MPS), 1× RTX PRO 6000 Blackwell (serial), 4× RTX PRO 6000 Blackwell with 1 GPU/job | Flyte 2 task environments (CPU vs GPU). 1 GPU/job is a faster fill of the yellow CUDA rows, not a second Results block. Hardware names are this tutorial’s machines. |
| `hawkish_dovish_task.png` | Labels in paper id order: 0 Dovish, 1 Hawkish, 2 Neutral | [Shah et al. (2023)](https://aclanthology.org/2023.acl-long.368/). Example sentences are illustrative, not quoted from the paper. |
| `full_lora_qlora.png` | Full FT vs LoRA vs QLoRA | LoRA: [Hu et al., 2021](https://arxiv.org/abs/2106.09685). QLoRA: [Dettmers et al., 2023](https://arxiv.org/abs/2305.14314). |
| `ft_memory_budget.png` | Where GPU memory goes: parameters, grads, AdamW, activations on a 96 GB card | Unit-square memory budget used in LoRA / QLoRA talks. Not a copy of any vendor slide. |
| `full_finetune_memory.png` | Memory to fully fine-tune Nemotron 3 Nano 4B in bf16 (~38 GB) vs a 24 GB GPU | Original diagram for the ODSC blog post. |
| `flyte_report_why_finetune.png` | Flyte UI report: test F1 before fine-tuning and after QLoRA | Screenshot of this pipeline's `main` report. |
| `readme/*.png` | Talk slides 15, 27, and 31, rendered from `slides/presentation_peft_talk.pdf` | Our ODSC AI West deck. Slide 15 diagram: [QLoRA](https://arxiv.org/abs/2305.14314). |
