# GPT image prompts

One prompt per figure. For each one: paste the **Shared style** block, then the figure's prompt, into GPT image (landscape, 1536 × 1024). Save the result as `fomc_peft/images/<file name>` using the exact file name below, then tell Claude which ones are in so the README and notebook can be pointed at them.

| File name to save as | Prompt | Replaces | Used in |
|---|---|---|---|
| `setup_steps.png` | [setup_steps.md](setup_steps.md) | slide 26 code screenshot | README card 1 (Setup) |
| `task_hawkish_dovish.png` | [task_hawkish_dovish.md](task_hawkish_dovish.md) | slide 27 screenshot | README card 2, notebook "Task" |
| `full_lora_qlora_memory.png` | [full_lora_qlora_memory.md](full_lora_qlora_memory.md) | slide 15 screenshot | README card 3, notebook "Full fine-tuning vs LoRA vs QLoRA" |
| `flyte_pipeline_one_laptop.png` | [flyte_pipeline_one_laptop.md](flyte_pipeline_one_laptop.md) | slide 30 screenshot | README card 4, notebook "One pipeline" |
| `vllm_serving.png` | [vllm_serving.md](vllm_serving.md) | slide 31 screenshot | README card 5, notebook "Step 5" |

The title slide (README card 0), the 38 GB memory grid (`full_finetune_memory.png`, slide 3), and the results table stay as they are: they are already exact.

Check every generated image for misspellings and wrong numbers before saving. The numbers must be exactly the ones in the prompt.

## Shared style

```
Clean educational diagram for a technical talk, landscape 3:2, pure white background.
Flat 2D vector style. No 3D, no photos, no gradients, no drop shadows, no watermark, no logos.
Every word spelled exactly as given in quotes. Render only the text given in quotes; add no other text.
Title top-left in bold dark charcoal (#202124) sans-serif like Arial or Helvetica, with a short bright blue underline bar (#4285F4) beneath it.
Body text dark gray (#3c4043), large and readable at small sizes. Generous whitespace.
Accent colors used sparingly: blue #4285F4, teal #0F9D9A, orange #F5A524, slate gray #78909C.
Memory colors when memory is shown: parameters yellow #F6C343, gradients blue #4285F4, optimizer state green #34A853, activations purple #8E6CC9.
Rounded rectangles with thin light gray borders (#DADCE0), flat fills.
```
