# flyte_pipeline_one_laptop.png

Save as: `fomc_peft/images/flyte_pipeline_one_laptop.png`

The step names are the real Flyte task names in `workflow.py`.

```
Title: "33 runs, one Flyte pipeline"
Subtitle in italic blue (#4285F4): "Same workflow.py on a laptop, a laptop GPU, or a Flyte cluster."

A horizontal pipeline of five rounded boxes connected by arrows, left to right. Each box has a small badge in its top-right corner, either "CPU" (slate gray) or "GPU" (orange), a bold monospace task name, and one short line of plain text.

Box 1, badge "CPU": "prepare_data". Text: "download splits, cached".
Box 2, badge "GPU", drawn as a stack of overlapping cards to show many runs: "train_one". Text: "3 models x 4 methods x 3 seeds".
Box 3, badge "CPU": "main". Text: "compare report".
Box 4, badge "CPU", with a small green check-mark button labeled "Approve": "release". Text: "waits for approval".
Box 5, badge "GPU": "merge_adapter". Text: "ready for vLLM".

Under boxes 1 to 3, a thin teal bracket with the label: "every stage writes its own report".
Under box 2, a small curved arrow looping back to itself with the label: "retries from the last epoch".

In the bottom-left corner, a small flat laptop icon with the label: "Dell Pro Max 16 Plus, RTX PRO 5000 Blackwell, 24 GB".
```
