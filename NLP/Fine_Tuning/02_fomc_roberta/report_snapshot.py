# Auto-written by refresh_report_snapshot(). Do not edit by hand.
SNAPSHOT = {
  "updated": "2026-08-22 06:24 UTC",
  "runs": [
    {
      "model": "roberta-base",
      "method": "full",
      "seed": 5768,
      "split": "lab-manual-split-combine",
      "device": "mps",
      "status": "complete",
      "weighted_f1": 0.679692011729891,
      "epochs_trained": 11.0,
      "trainable_pct": 100.0,
      "wall_seconds": 396.6,
      "delta_vs_paper": -0.0184,
      "paper_weighted_f1": 0.6981
    },
    {
      "model": "roberta-base",
      "method": "full",
      "seed": 78516,
      "split": "lab-manual-split-combine",
      "device": "mps",
      "status": "complete",
      "weighted_f1": 0.6946314655791667,
      "epochs_trained": 11.0,
      "trainable_pct": 100.0,
      "wall_seconds": 761.1,
      "delta_vs_paper": -0.0035,
      "paper_weighted_f1": 0.6981
    },
    {
      "model": "roberta-base",
      "method": "full",
      "seed": 944601,
      "split": "lab-manual-split-combine",
      "device": "mps",
      "status": "complete",
      "weighted_f1": 0.7157442955283764,
      "epochs_trained": 11.0,
      "trainable_pct": 100.0,
      "wall_seconds": 314.7,
      "delta_vs_paper": 0.0176,
      "paper_weighted_f1": 0.6981
    },
    {
      "model": "roberta-base",
      "method": "lora",
      "seed": 5768,
      "split": "lab-manual-split-combine",
      "device": "mps",
      "status": "complete",
      "weighted_f1": 0.6638024361961042,
      "epochs_trained": 30.0,
      "trainable_pct": 2.539,
      "wall_seconds": 963.2,
      "delta_vs_paper": -0.0343,
      "paper_weighted_f1": 0.6981
    },
    {
      "model": "roberta-base",
      "method": "lora",
      "seed": 78516,
      "split": "lab-manual-split-combine",
      "device": "mps",
      "status": "complete",
      "weighted_f1": 0.6943718456420958,
      "epochs_trained": 30.0,
      "trainable_pct": 2.539,
      "wall_seconds": 3198.8,
      "delta_vs_paper": -0.0037,
      "paper_weighted_f1": 0.6981
    },
    {
      "model": "roberta-base",
      "method": "lora",
      "seed": 944601,
      "split": "lab-manual-split-combine",
      "device": "mps",
      "status": "complete",
      "weighted_f1": 0.6969099714703945,
      "epochs_trained": 23.0,
      "trainable_pct": 2.539,
      "wall_seconds": 548.4,
      "delta_vs_paper": -0.0012,
      "paper_weighted_f1": 0.6981
    },
    {
      "model": "roberta-large",
      "method": "full",
      "seed": 5768,
      "split": "lab-manual-split-combine",
      "device": "mps",
      "status": "complete",
      "weighted_f1": 0.6951745411887585,
      "epochs_trained": 14.0,
      "trainable_pct": 100.0,
      "wall_seconds": 1668.1,
      "delta_vs_paper": -0.0161,
      "paper_weighted_f1": 0.7113
    },
    {
      "model": "roberta-large",
      "method": "full",
      "seed": 78516,
      "split": "lab-manual-split-combine",
      "device": "mps",
      "status": "complete",
      "weighted_f1": 0.7263827585222272,
      "epochs_trained": 12.0,
      "trainable_pct": 100.0,
      "wall_seconds": 4369.3,
      "delta_vs_paper": 0.0151,
      "paper_weighted_f1": 0.7113
    },
    {
      "model": "roberta-large",
      "method": "full",
      "seed": 944601,
      "split": "lab-manual-split-combine",
      "device": "mps",
      "status": "complete",
      "weighted_f1": 0.7168408653118303,
      "epochs_trained": 16.0,
      "trainable_pct": 100.0,
      "wall_seconds": 1296.2,
      "delta_vs_paper": 0.0055,
      "paper_weighted_f1": 0.7113
    },
    {
      "model": "roberta-large",
      "method": "lora",
      "seed": 5768,
      "split": "lab-manual-split-combine",
      "device": "mps",
      "status": "complete",
      "weighted_f1": 0.7154336404829578,
      "epochs_trained": 28.0,
      "trainable_pct": 2.237,
      "wall_seconds": 8765.4,
      "delta_vs_paper": 0.0041,
      "paper_weighted_f1": 0.7113
    },
    {
      "model": "roberta-large",
      "method": "lora",
      "seed": 78516,
      "split": "lab-manual-split-combine",
      "device": "mps",
      "status": "complete",
      "weighted_f1": 0.719044808623278,
      "epochs_trained": 22.0,
      "trainable_pct": 2.237,
      "wall_seconds": 4768.2,
      "delta_vs_paper": 0.0077,
      "paper_weighted_f1": 0.7113
    },
    {
      "model": "roberta-large",
      "method": "lora",
      "seed": 944601,
      "split": "lab-manual-split-combine",
      "device": "mps",
      "status": "complete",
      "weighted_f1": 0.74450515546903,
      "epochs_trained": 22.0,
      "trainable_pct": 2.237,
      "wall_seconds": 1526.9,
      "delta_vs_paper": 0.0332,
      "paper_weighted_f1": 0.7113
    }
  ],
  "talk_reads": [
    {
      "text": "The Committee judged that a further increase in the target range would be appropriate.",
      "pred": "hawkish",
      "for_you": "Rates likely go up. A new mortgage or a refinance gets more expensive."
    },
    {
      "text": "The Committee decided to lower the target range for the federal funds rate.",
      "pred": "dovish",
      "for_you": "Rates likely go down. Cheaper to borrow; savings yields usually follow."
    },
    {
      "text": "Incoming data suggested that economic activity was expanding at a moderate pace.",
      "pred": "hawkish",
      "for_you": "Sounds bland. The model still leans hawkish \u2014 growth without easing can mean no cut coming."
    },
    {
      "text": "Inflation remains elevated, and the Committee is strongly committed to returning it to 2 percent.",
      "pred": "hawkish",
      "for_you": "Rates likely go up. A new mortgage or a refinance gets more expensive."
    }
  ],
  "talk_source": "roberta-large lora, seed 944601"
}
