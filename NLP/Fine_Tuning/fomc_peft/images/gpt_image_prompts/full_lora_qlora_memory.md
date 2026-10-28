# full_lora_qlora_memory.png

Save as: `fomc_peft/images/full_lora_qlora_memory.png`

Numbers are measured peak training memory for Nemotron 3 Nano 4B on the laptop GPU (full fine-tuning is the bf16 estimate).

```
Title: "Full fine-tuning vs LoRA vs QLoRA"
Subtitle in italic blue (#4285F4): "Nemotron 3 Nano 4B, peak training memory on a 24 GB laptop GPU"

Three equal columns, each a tall rounded card.

Each column shows the same simple model diagram: a wide rounded rectangle labeled "base model" at the bottom.
Column 1 header: "Full fine-tuning". The base model rectangle is filled solid orange (#F5A524) to show every weight is trained. Small label under it: "every weight trained".
Column 2 header: "LoRA". The base model rectangle is light gray with a small snowflake icon (frozen). Next to it, two small orange rectangles labeled "B" and "A". Small label: "frozen base, small adapters".
Column 3 header: "QLoRA". The base model rectangle is light gray, drawn half as tall, with a snowflake icon and the tag "4-bit". The same two small orange rectangles "B" and "A". Small label: "4-bit frozen base, small adapters".

At the bottom of each column, a horizontal bar against a dashed vertical line marked "24 GB":
Column 1 bar is red and crosses past the dashed line. Big number: "~38 GB". Caption: "does not fit".
Column 2 bar is blue, well inside the line. Big number: "9.0 GB". Caption: "F1 0.727".
Column 3 bar is teal, shorter still. Big number: "5.4 GB". Caption: "F1 0.716".

All three bars use the same scale, from 0 to 40 GB.
```
