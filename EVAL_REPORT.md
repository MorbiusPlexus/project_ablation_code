# Evaluation report

This file is produced by `analysis/aggregate.py`.

## What was run here

The local RTX 4050 path and the CPU suite exercise every evaluation module on a tiny random model.
The seven trained checkpoints live on ada (`~/morbius/final_ckpts`). This machine cannot reach ada
(see `results/ckpt_inventory.json`), so the tables below are pipeline checks, not paper numbers.
Phase B is `eval/submit.py` against `eval/manifest.yaml` once a GPU allocation exists.

## Protocol

Engram Table 1 shots, the factual-vs-reading suppression split, LogitLens, CKA with the soft
alignment index, Lngram code utilisation, address-stability metrics, ID-only KNN and the
host-RAM offload model are implemented. Tasks a 400M model scores at chance on are marked
`floor` in `eval/tasks.py` and are not dropped silently.

## Local suite

### general

```json
{
  "induction": 0.0,
  "passkey": 0.0,
  "mqar": 0.0,
  "next": {
    "acc": 0.0,
    "loss": 5.5316587175641745,
    "bpb": 0.9975620722238684
  },
  "note": "tiny random init; not a trained checkpoint"
}
```

### ablation

```json
{
  "zero": {
    "full": 5.5316587175641745,
    "ablated": 5.5316587175641745,
    "retention": 1.0
  },
  "mean": {
    "full": 5.5316587175641745,
    "ablated": 5.5316587175641745,
    "retention": 1.0
  },
  "logitlens": [
    0.02042340114712715,
    0.012739896774291992,
    0.006445665378123522,
    0.0
  ],
  "cka_self_diag": 1.0,
  "soft_align": [
    1.4037035703659058,
    1.472164273262024,
    1.5421898365020752,
    1.5926839113235474
  ],
  "propagation": {
    "per_layer": [
      {
        "layer": 0,
        "R": 0.0,
        "retention": 0.0
      },
      {
        "layer": 1,
        "R": 0.0,
        "retention": 0.0
      },
      {
        "layer": 2,
        "R": 0.0,
        "retention": 0.0
      },
      {
        "layer": 3,
        "R": 0.0,
        "retention": 0.0
      }
    ],
    "js": -7.450580596923828e-09,
    "top1_agreement": 1.0,
    "logit_R": 0.0
  },
  "codes": {
    "1": {
      "sink_mass": 0.49957790970802307,
      "eff_codes_mean": 9.385152354486008,
      "frac_routes_eff_lt4": 0.0,
      "frac_routes_eff_lt2": 0.0,
      "dead_code_frac": 0.359375,
      "top_code_share": 0.1875
    }
  }
}
```

### interp

```json
{
  "buckets": {
    "other": 0.0,
    "entity": 0.0
  },
  "mi": 2.1333819302645445,
  "purity": {
    "rows": 7,
    "mean_purity": 1.0,
    "example": [
      1,
      [
        2,
        3
      ],
      175,
      1.0
    ]
  },
  "direct": {
    "median_rank": 1.0,
    "random_median_rank": 132.0,
    "top1": 0
  },
  "patch_denoising": {
    "1": -0.0
  },
  "locality": [
    0.0,
    0.0,
    0.0,
    0.0
  ],
  "id_ap": 0.41637286324786327,
  "hidden_ap": 0.544150641025641,
  "rho": 0.7651794041132872,
  "perm_p": 0.25
}
```

### finetune

```json
{
  "all": {
    "score": 0.5,
    "trainable": 454624,
    "steps": 2
  },
  "memory": {
    "score": 0.5,
    "trainable": 290208,
    "steps": 2
  },
  "backbone": {
    "score": 1.0,
    "trainable": 164416,
    "steps": 2
  },
  "tables": {
    "score": 0.5,
    "trainable": 278528,
    "steps": 2
  }
}
```

### offload

```json
{
  "decode_max_abs": 1.7881393432617188e-07,
  "offload_max_abs": 0.0,
  "bench": {
    "device": "cpu",
    "batch": 1,
    "seq": 4,
    "layer_ms": 1.9599244988057762,
    "gather_ms": 0.0035919947549700737,
    "h2d_ms": 0.0019409926608204842,
    "table_rows": 1024,
    "table_dim": 16,
    "e2e_gpu_ms": 2.7012229984393343,
    "e2e_sync_ms": 5.714999002520926,
    "e2e_hybrid_ms": 1.938197499839589,
    "e2e_hotcache_ms": 1.88018249173183
  },
  "delay_injects_at": {
    "1": 2
  },
  "delay_ok": true
}
```

### gpu4050

```json
{
  "gpu": "NVIDIA GeForce RTX 4050 Laptop GPU",
  "vram_mib": 6141,
  "decode_max_abs": 2.384185791015625e-07,
  "offload_max_abs": 0.0,
  "e2e_gpu_ms": 2.467,
  "e2e_sync_ms": 2.486,
  "peak_mem_mb": 10.39,
  "note": "Tiny 4-layer model. Confirms incremental decode and host-RAM gather match the full forward on this GPU."
}
```

### perf_model

```json
{
  "constants": {
    "bw_gather_Bps": 2000000000.0,
    "bw_pcie_Bps": 12000000000.0,
    "layer_us": [
      80.0,
      80.0,
      80.0,
      80.0,
      80.0,
      80.0,
      80.0,
      80.0,
      80.0,
      80.0,
      80.0,
      80.0,
      80.0,
      80.0,
      80.0,
      80.0,
      80.0,
      80.0,
      80.0,
      80.0,
      80.0,
      80.0,
      80.0,
      80.0
    ],
    "readout_us": 20.0,
    "t_d2h_us": 10.0,
    "t_pcie_us": 5.0,
    "a_gather_us": 8.0,
    "routes": 184,
    "orders": 1,
    "dim": 128
  },
  "predict": {
    "1": {
      "fetch_us": 50.477333333333334,
      "n_star": 1,
      "step_us": {
        "0": 1990.4773333333333,
        "1": 1940.0,
        "2": 1940.0,
        "3": 1940.0,
        "4": 1940.0,
        "5": 1940.0,
        "6": 1940.0,
        "7": 1940.0,
        "8": 1940.0,
        "9": 1940.0,
        "10": 1940.0,
        "11": 1940.0,
        "12": 1940.0,
        "13": 1940.0,
        "14": 1940.0,
        "15": 1940.0,
        "16": 1940.0,
        "17": 1940.0,
        "18": 1940.0,
        "19": 1940.0,
        "20": 1940.0,
        "21": 1940.0,
        "22": 1940.0,
        "23": 1940.0
      }
    },
    "4": {
      "fetch_us": 132.90933333333334,
      "n_star": 2,
      "step_us": {
        "0": 2072.9093333333335,
        "1": 1992.9093333333333,
        "2": 1940.0,
        "3": 1940.0,
        "4": 1940.0,
        "5": 1940.0,
        "6": 1940.0,
        "7": 1940.0,
        "8": 1940.0,
        "9": 1940.0,
        "10": 1940.0,
        "11": 1940.0,
        "12": 1940.0,
        "13": 1940.0,
        "14": 1940.0,
        "15": 1940.0,
        "16": 1940.0,
        "17": 1940.0,
        "18": 1940.0,
        "19": 1940.0,
        "20": 1940.0,
        "21": 1940.0,
        "22": 1940.0,
        "23": 1940.0
      }
    },
    "16": {
      "fetch_us": 462.63733333333334,
      "n_star": 6,
      "step_us": {
        "0": 2402.637333333333,
        "1": 2322.637333333333,
        "2": 2242.637333333333,
        "3": 2162.637333333333,
        "4": 2082.637333333333,
        "5": 2002.6373333333333,
        "6": 1940.0,
        "7": 1940.0,
        "8": 1940.0,
        "9": 1940.0,
        "10": 1940.0,
        "11": 1940.0,
        "12": 1940.0,
        "13": 1940.0,
        "14": 1940.0,
        "15": 1940.0,
        "16": 1940.0,
        "17": 1940.0,
        "18": 1940.0,
        "19": 1940.0,
        "20": 1940.0,
        "21": 1940.0,
        "22": 1940.0,
        "23": 1940.0
      }
    }
  }
}
```

### stats

```json
{
  "ci": {
    "mean": 0.25,
    "lo": 0.15,
    "hi": 0.35000000000000003
  },
  "mcnemar": 1.0,
  "holm": [
    0.03,
    0.08,
    0.2
  ],
  "cka_identical": -0.4244231581687927,
  "cka_self": 1.0
}
```

### inventory

```json
{
  "host": "lastvaquita",
  "ada_resolves": false,
  "local_final_ckpts": [],
  "remote_expected": {
    "aditya": "/home/aditya.shankar/morbius/final_ckpts",
    "sparsh": "/home2/sparsh.bhartia/morbius/final_ckpts"
  },
  "files": [
    "w1_mem_full_step3815.pt",
    "w1_mem_cut_near_step3815.pt",
    "w1_mem_cut_far_step3815.pt",
    "w1_dense_full_step3815.pt",
    "w1_dense_cut_near_step3815.pt",
    "w2_mem_rank1_step3815.pt",
    "w2_mem_super1_step3815.pt"
  ],
  "note": "Phase B reads these from ada. export_local.sh makes a 4050-sized fp16 copy."
}
```

## Caveats

- Single seed. Differences inside a paired bootstrap interval are ties.
- Zero ablation is off-distribution. Mean ablation is reported beside it.
- Wave-1 run 1 (2080 Ti) and run 4 (3090) still carry the GPU-type confound.
- Fine-tuning hyperparameters are fixed. No learning-rate sweep.
- `w2_mem_delayed4` stopped at step 90 and is not a result row.
- Local numbers use a 4-layer random model. Do not compare them to the val losses in `state_and_results.md`.
