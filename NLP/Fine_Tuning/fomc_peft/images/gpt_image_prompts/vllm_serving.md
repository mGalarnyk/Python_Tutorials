# vllm_serving.png

Save as: `fomc_peft/images/vllm_serving.png`

Numbers: LoRA adapter on NVIDIA's FP8 Nemotron base, 3-seed mean, laptop GPU (`outputs/inference_bench.jsonl`).

```
Title: "Serving the adapter with vLLM"
Subtitle in italic blue (#4285F4): "The adapter trained on the laptop works unchanged on the FP8 base model."

Left half: a serving diagram. On the far left, three small chat-bubble request icons stacked vertically, each labeled "request". Arrows go into one large rounded box labeled "vLLM" with the sub-label "OpenAI-compatible API". Inside the vLLM box, one wide gray block labeled "Nemotron 3 Nano 4B (FP8)" and, plugged into its top edge, three small colored adapter tiles labeled "adapter A", "adapter B", "adapter C"; adapter A is orange and highlighted, the other two are pale. A small caption under the box: "one base, adapter picked per request".

Right half: three stacked metric tiles, each a light gray rounded rectangle with a small uppercase gray label and a big bold number.
Tile 1 label "ACCURACY", number "0.725", small text "weighted F1 (0.727 in training)".
Tile 2 label "GPU MEMORY", number "5.2 GB", small text "FP8 weights + adapter".
Tile 3 label "THROUGHPUT", number "~100", small text "sentences per second".
```
