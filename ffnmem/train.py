"""DDP training loop.

    torchrun --nproc_per_node=4 -m ffnmem.train --config configs/w1_mem_full.yaml --resume auto
    python -m ffnmem.train --config configs/tiny_cpu.yaml --device cpu          # single process

Resume paths:
- ``--resume auto|<ckpt>``: restore everything (model, optimizer, schedule,
  scaler, RNG, data position) and continue exactly.  ``auto`` picks the latest
  checkpoint in the run directory, or starts fresh if there is none.
- ``--init_from <ckpt> --new_schedule``: load weights only and start a fresh
  optimizer + schedule (for extension runs).  Data continues after the source
  run's data position unless ``--data_restart``.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import random
import signal
import sys
import time
import uuid
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import yaml
from torch.nn.parallel import DistributedDataParallel as DDP

from . import checkpoint as ckpt
from .config import load_config
from .data.loader import TrainWindows, ValWindows
from .memory.wrapper import param_group_of
from .model import Transformer, count_params


# ------------------------------------------------------------------ distributed
class Dist:
    def __init__(self, device_arg: str = "auto", force_pg: bool = False):
        if os.name == "nt":
            os.environ.setdefault("USE_LIBUV", "0")  # torch TCPStore on Windows
        self.world_size = int(os.environ.get("WORLD_SIZE", "1"))
        self.rank = int(os.environ.get("RANK", "0"))
        self.local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        use_cuda = torch.cuda.is_available() and device_arg != "cpu"
        if device_arg == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("--device cuda requested but CUDA is not available")
        if use_cuda:
            torch.cuda.set_device(self.local_rank)
            self.device = torch.device("cuda", self.local_rank)
        else:
            self.device = torch.device("cpu")
        self.enabled = self.world_size > 1 or force_pg
        if self.enabled and not dist.is_initialized():
            if "MASTER_ADDR" not in os.environ:  # single process without torchrun (ZeRO-1 needs a group)
                os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=os.environ.get("MASTER_PORT", "29517"),
                                  RANK="0", WORLD_SIZE="1", LOCAL_RANK="0")
            backend = "nccl" if use_cuda and os.name != "nt" else "gloo"
            dist.init_process_group(backend=backend)

    @property
    def is_main(self) -> bool:
        return self.rank == 0

    def barrier(self) -> None:
        if self.enabled:
            dist.barrier()

    def all_reduce_sum(self, values: list[float]) -> list[float]:
        t = torch.tensor(values, dtype=torch.float64, device=self.device)
        if self.enabled:
            dist.all_reduce(t)
        return t.tolist()

    def any(self, flag: bool) -> bool:
        return self.all_reduce_sum([1.0 if flag else 0.0])[0] > 0

    def all_gather_object(self, obj):
        if not self.enabled:
            return [obj]
        out = [None] * self.world_size
        dist.all_gather_object(out, obj)
        return out

    def close(self) -> None:
        if self.enabled and dist.is_initialized():
            dist.destroy_process_group()


# ------------------------------------------------------------------- schedule
def schedule_for(tcfg: dict) -> dict:
    tokens_per_step = int(tcfg["global_batch"]) * int(tcfg["seq_len"])
    total = math.ceil(int(tcfg["tokens_total"]) / tokens_per_step)
    warmup = max(int(tcfg["warmup_min_steps"]), round(float(tcfg["warmup_frac"]) * total))
    warmup = min(warmup, total)
    return {
        "total_steps": total,
        "warmup_steps": warmup,
        "peak_lr": float(tcfg["lr"]),
        "min_lr_ratio": float(tcfg["min_lr_ratio"]),
        "tokens_per_step": tokens_per_step,
    }


def lr_at(step: int, sched: dict) -> float:
    peak, warmup, total = sched["peak_lr"], sched["warmup_steps"], sched["total_steps"]
    if step < warmup:
        return peak * (step + 1) / warmup
    progress = min(1.0, (step - warmup) / max(1, total - warmup))
    ratio = sched["min_lr_ratio"]
    return peak * (ratio + (1.0 - ratio) * 0.5 * (1.0 + math.cos(math.pi * progress)))


# ------------------------------------------------------------------- builders
def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_model(cfg: dict, device: torch.device) -> Transformer:
    tcfg = cfg["train"]
    seed_everything(int(tcfg["seed"]))
    model = Transformer(cfg["model"])
    model.set_attention_backend(tcfg.get("attn_backend", "sdpa"))
    model.act_ckpt = bool(tcfg.get("act_ckpt", False))
    return model.to(device)


def build_optimizer(model: torch.nn.Module, tcfg: dict, device: torch.device, zero1: bool):
    buckets: dict[str, list] = {"decay": [], "no_decay": [], "tables": [], "router": []}
    for name, p in model.named_parameters():
        if p.requires_grad:
            buckets[param_group_of(name, p)].append(p)
    wd = float(tcfg["weight_decay"])
    spec = {
        "decay": (1.0, wd),
        "no_decay": (1.0, 0.0),
        "tables": (float(tcfg["table_lr_mult"]), 0.0),
        "router": (1.0, 0.0),
    }
    groups = [
        {"params": ps, "lr_mult": spec[g][0], "weight_decay": spec[g][1], "lr": float(tcfg["lr"]) * spec[g][0], "group": g}
        for g, ps in buckets.items()
        if ps
    ]
    kwargs = dict(betas=tuple(tcfg["betas"]), eps=float(tcfg["eps"]))
    if device.type == "cuda":
        kwargs["fused"] = True
    if zero1:
        from torch.distributed.optim import ZeroRedundancyOptimizer

        opt = ZeroRedundancyOptimizer(groups[0]["params"], optimizer_class=torch.optim.AdamW,
                                      lr=groups[0]["lr"], weight_decay=groups[0]["weight_decay"], **kwargs)
        opt.param_groups[0].update({k: v for k, v in groups[0].items() if k != "params"})
        for g in groups[1:]:
            opt.add_param_group(dict(g))
        return opt
    return torch.optim.AdamW(groups, **kwargs)


def optimizer_state_to_param_device(optimizer) -> None:
    """After loading a CPU checkpoint: fused CUDA AdamW needs every state tensor,
    including ``step``, on the parameter's device (ZeRO-1's load leaves them on CPU)."""
    inner = getattr(optimizer, "optim", optimizer)
    for group in inner.param_groups:
        for p in group["params"]:
            state = inner.state.get(p)
            if not state:
                continue
            for key, value in state.items():
                if torch.is_tensor(value) and value.device != p.device:
                    state[key] = value.to(p.device)


def micro_steps(tcfg: dict, world_size: int) -> int:
    gb, mb = int(tcfg["global_batch"]), int(tcfg["micro_batch"])
    if gb % (mb * world_size):
        raise ValueError(f"global_batch {gb} not divisible by micro_batch*world_size = {mb}*{world_size}")
    return gb // (mb * world_size)


def amp_context(device: torch.device, precision: str):
    if precision == "fp16":
        return torch.autocast(device_type=device.type, dtype=torch.float16)
    if precision == "fp32":
        return contextlib.nullcontext()
    raise ValueError(f"unsupported precision {precision!r} (fp16 | fp32)")


def table_grad_norm(model: Transformer) -> float:
    sq = 0.0
    for m in model.memory_modules():
        for p in m.table_parameters():
            if p.grad is not None:
                sq += float(p.grad.detach().float().pow(2).sum())
    return math.sqrt(sq)


# ------------------------------------------------------------------- logging
class Logger:
    def __init__(self, out_dir: Path, enabled: bool, use_wandb: bool, run_name: str, cfg: dict):
        self.enabled = enabled
        self.tb = None
        self.wandb = None
        if not enabled:
            return
        (out_dir / "logs").mkdir(parents=True, exist_ok=True)
        self.jsonl = open(out_dir / "logs" / "train.jsonl", "a", encoding="utf-8")
        try:
            from torch.utils.tensorboard import SummaryWriter

            self.tb = SummaryWriter(str(out_dir / "logs" / "tb"))
        except ImportError:
            print("[log] tensorboard not installed; JSONL only", flush=True)
        if use_wandb:
            import wandb

            # Stable id stored in the run dir, so a resumed job continues the same wandb run;
            # wandb's local files stay in the run dir (node ssd), not in home.
            id_file = out_dir / "logs" / "wandb_id.txt"
            run_id = id_file.read_text().strip() if id_file.exists() else f"{run_name}-{uuid.uuid4().hex[:8]}"
            id_file.write_text(run_id)
            self.wandb = wandb.init(project=os.environ.get("WANDB_PROJECT", "morbius-wave1"), name=run_name,
                                    id=run_id, resume="allow", config=cfg, dir=str(out_dir / "logs"))

    def log(self, record: dict) -> None:
        if not self.enabled:
            return
        record = {"time": time.time(), **record}
        self.jsonl.write(json.dumps(record) + "\n")
        self.jsonl.flush()
        step = record.get("step", 0)
        for k, v in record.items():
            if isinstance(v, (int, float)) and k not in ("step", "time"):
                if self.tb is not None:
                    self.tb.add_scalar(k, v, step)
        if self.wandb is not None:
            self.wandb.log({k: v for k, v in record.items() if isinstance(v, (int, float))}, step=step)

    def close(self) -> None:
        if not self.enabled:
            return
        self.jsonl.close()
        if self.tb is not None:
            self.tb.close()
        if self.wandb is not None:
            self.wandb.finish()


# -------------------------------------------------------------------- eval
@torch.no_grad()
def evaluate(model: Transformer, val: ValWindows, d: Dist, tcfg: dict) -> dict:
    was_training = model.training
    model.eval()
    mems = model.memory_modules()
    for m in mems:
        m.reset_diagnostics()
    loss_sum, n_tok = 0.0, 0
    diag_batches = int(tcfg.get("diag_batches", 4))
    for i, (x, y) in enumerate(val.batches(int(tcfg["micro_batch"]), tcfg.get("eval_max_batches"))):
        for m in mems:
            m.collect_diagnostics = i < diag_batches
        x, y = x.to(d.device, non_blocking=True), y.to(d.device, non_blocking=True)
        with amp_context(d.device, tcfg["precision"]):
            loss = model(x, y)[1]
        loss_sum += float(loss) * y.numel()
        n_tok += y.numel()
    for m in mems:
        m.collect_diagnostics = False
    loss_sum, n_tok = d.all_reduce_sum([loss_sum, float(n_tok)])
    out = {"val_loss": loss_sum / max(n_tok, 1.0), "val_tokens": int(n_tok)}
    out["val_ppl"] = math.exp(min(out["val_loss"], 50.0))
    for block_pos, block in enumerate(model.blocks):
        if block.memory is not None:
            for k, v in block.memory.diagnostics().items():
                out[f"mem{block_pos}/{k}"] = v
    if was_training:
        model.train()
    return out


# -------------------------------------------------------------------- main
def parse_args(argv: list[str] | None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True)
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="config override, e.g. train.lr=3e-4")
    p.add_argument("--data_dir")
    p.add_argument("--out_dir")
    p.add_argument("--resume", help="'auto' or a checkpoint path")
    p.add_argument("--init_from", help="load weights only from this checkpoint")
    p.add_argument("--new_schedule", action="store_true", help="with --init_from: fresh optimizer + schedule")
    p.add_argument("--data_restart", action="store_true", help="with --init_from: restart data at step 0")
    p.add_argument("--zero1", action="store_true", help="shard optimizer state (ZeroRedundancyOptimizer)")
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--max_steps", type=int, help="stop after this many optimizer steps (schedule unchanged)")
    return p.parse_args(argv)


class _Stop:
    """SIGUSR1/SIGTERM -> checkpoint at the next step boundary, then exit cleanly."""

    requested = False

    @classmethod
    def install(cls) -> None:
        for name in ("SIGUSR1", "SIGTERM"):
            sig = getattr(signal, name, None)
            if sig is not None:
                try:
                    signal.signal(sig, lambda *_: setattr(cls, "requested", True))
                except ValueError:  # not main thread
                    pass


def main(argv: list[str] | None = None) -> dict:
    args = parse_args(argv)
    overrides = list(args.set)
    if args.data_dir:
        overrides.append(f"train.data_dir={args.data_dir}")
    cfg = load_config(args.config, overrides)
    tcfg = cfg["train"]
    tcfg["seq_len"] = cfg["model"]["seq_len"]
    zero1 = bool(args.zero1 or tcfg.get("zero1", False))
    tcfg["zero1"] = zero1
    d = Dist(args.device, force_pg=zero1)
    out_dir = Path(args.out_dir) if args.out_dir else Path(tcfg["out_root"]) / cfg["name"]
    if d.is_main:
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / "config.yaml", "w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)
    _Stop.install()

    # ---- model + parameter check
    model = build_model(cfg, d.device)
    counts = count_params(model)
    expected = cfg.get("params", {}).get("expected_non_embedding")
    if d.is_main:
        print(f"[params] {cfg['name']}: " + ", ".join(f"{k}={v:,}" for k, v in counts.items()), flush=True)
    if expected is not None and int(expected) != counts["non_embedding"]:
        raise SystemExit(f"parameter check failed: non_embedding {counts['non_embedding']:,} != expected {int(expected):,}")

    precision = tcfg["precision"]
    ddp_model = (
        DDP(model, device_ids=[d.local_rank] if d.device.type == "cuda" else None, gradient_as_bucket_view=True)
        if d.world_size > 1
        else model
    )
    optimizer = build_optimizer(model, tcfg, d.device, zero1)
    scaler = torch.amp.GradScaler(d.device.type, enabled=(precision == "fp16"))
    sched = schedule_for(tcfg)
    n_micro = micro_steps(tcfg, d.world_size)

    # ---- resume / init
    step, data_offset, elapsed_prev = 0, 0, 0.0
    resume_path = None
    if args.resume and args.init_from:
        raise SystemExit("use either --resume or --init_from, not both")
    if args.resume:
        resume_path = ckpt.latest_checkpoint(out_dir) if args.resume == "auto" else Path(args.resume)
        if args.resume != "auto" and not resume_path.exists():
            raise SystemExit(f"checkpoint {resume_path} not found")
    if resume_path is not None:
        state = ckpt.load(resume_path)
        if bool(state.get("zero1")) != zero1:
            raise SystemExit(f"checkpoint zero1={state.get('zero1')} but this run zero1={zero1}; keep them equal")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        optimizer_state_to_param_device(optimizer)
        scaler.load_state_dict(state["scaler"])
        sched = state["schedule"]
        step, data_offset = int(state["step"]), int(state["data_offset"])
        elapsed_prev = float(state.get("elapsed", 0.0))
        rngs = state["rng"]
        if len(rngs) == d.world_size:
            ckpt.set_rng_state(rngs[d.rank])
        elif d.is_main:
            print(f"[resume] world size changed {len(rngs)} -> {d.world_size}; RNG not restored", flush=True)
        if d.is_main:
            print(f"[resume] {resume_path} at step {step}", flush=True)
        del state
        if d.device.type == "cuda":
            torch.cuda.empty_cache()  # drop the transient full-state copy made while loading
    elif args.init_from:
        if not args.new_schedule:
            raise SystemExit("--init_from requires --new_schedule (weights only, fresh optimizer and schedule)")
        state = ckpt.load(args.init_from)
        model.load_state_dict(state["model"])
        data_offset = 0 if args.data_restart else int(state["step"]) + int(state["data_offset"])
        if d.is_main:
            print(f"[init_from] {args.init_from}: weights loaded; new schedule {sched}; data offset {data_offset}", flush=True)
        del state
    elif args.resume == "auto" and d.is_main:
        print("[resume] no checkpoint found; starting fresh", flush=True)

    train_data = TrainWindows(tcfg["data_dir"], cfg["model"]["seq_len"], int(tcfg["data_seed"]),
                              int(tcfg["global_batch"]), int(tcfg["micro_batch"]), d.rank, d.world_size,
                              model_vocab=cfg["model"]["vocab"])
    val_data = ValWindows(tcfg["data_dir"], cfg["model"]["seq_len"], int(tcfg["val_tokens"]), d.rank, d.world_size)
    if sched["total_steps"] + data_offset > train_data.max_steps and d.is_main:
        print(f"[warn] schedule needs {sched['total_steps'] + data_offset} steps of data; dataset has "
              f"{train_data.max_steps}. Training will stop when data runs out.", flush=True)

    logger = Logger(out_dir, d.is_main, args.wandb, cfg["name"], cfg)
    if d.is_main:
        logger.log({"event": "start", "step": step, "config": cfg["name"], "git": ckpt.git_hash(),
                    "world_size": d.world_size, "micro_batch": tcfg["micro_batch"], "accum_steps": n_micro,
                    "zero1": zero1, "schedule": sched, "params": counts,
                    "cudnn": {"enabled": torch.backends.cudnn.enabled, "benchmark": torch.backends.cudnn.benchmark,
                              "deterministic": torch.backends.cudnn.deterministic},
                    "device": str(d.device), "torch": torch.__version__,
                    "gpu": torch.cuda.get_device_name(d.device) if d.device.type == "cuda" else None})

    def save(at_step: int) -> None:
        rng_all = d.all_gather_object(ckpt.rng_state())
        if zero1:
            optimizer.consolidate_state_dict(to=0)
        if d.is_main:
            state = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict(),
                "schedule": sched,
                "step": at_step,
                "data_offset": data_offset,
                "rng": rng_all,
                "config": cfg,
                "git": ckpt.git_hash(),
                "zero1": zero1,
                "world_size": d.world_size,
                "elapsed": elapsed_prev + (time.time() - t_start),
                "params": counts,
            }
            path = ckpt.ckpt_path(out_dir, at_step)
            ckpt.save_atomic(state, path)
            removed = ckpt.prune(out_dir, int(tcfg["keep_last"]), int(tcfg["keep_every"]))
            logger.log({"event": "checkpoint", "step": at_step, "path": str(path), "pruned": [p.name for p in removed]})
        d.barrier()

    # ---- loop
    model.train()
    t_start = time.time()
    t_log, tok_log = time.time(), 0
    low_scale, bad_loss = 0, 0
    history: list[dict] = []
    total = sched["total_steps"]
    stop_at = total if args.max_steps is None else min(total, args.max_steps)
    tokens_per_step = sched["tokens_per_step"]
    if d.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(d.device)
    exit_reason = "completed"

    while step < stop_at:
        data_step = step + data_offset
        if data_step >= train_data.max_steps:
            exit_reason = "data_exhausted"
            break
        lr = lr_at(step, sched)
        for g in optimizer.param_groups:
            g["lr"] = lr * g["lr_mult"]
        batches = train_data.micro_batches(data_step)
        assert len(batches) == n_micro
        loss_acc = 0.0
        for i, (x, y) in enumerate(batches):
            x, y = x.to(d.device, non_blocking=True), y.to(d.device, non_blocking=True)
            sync = contextlib.nullcontext() if (d.world_size == 1 or i == n_micro - 1) else ddp_model.no_sync()
            with sync:
                with amp_context(d.device, precision):
                    loss = ddp_model(x, y)[1]  # drop logits before backward
                scaler.scale(loss / n_micro).backward()
            loss_acc += float(loss.detach()) / n_micro
        scaler.unscale_(optimizer)
        grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), float(tcfg["grad_clip"])))
        done = step + 1
        is_eval = done % int(tcfg["eval_every"]) == 0 or done == total
        tgn = table_grad_norm(model) if (is_eval and model.memory_modules()) else None
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        step = done
        tok_log += tokens_per_step

        loss_mean = d.all_reduce_sum([loss_acc])[0] / d.world_size
        scale = scaler.get_scale() if scaler.is_enabled() else 1.0
        history.append({"step": step, "loss": loss_mean, "lr": lr, "grad_norm": grad_norm, "scale": scale})
        low_scale = low_scale + 1 if (scaler.is_enabled() and scale < float(tcfg["scaler_min"])) else 0
        bad_loss = bad_loss + 1 if not math.isfinite(loss_mean) else 0

        if step % int(tcfg["log_every"]) == 0 or step == 1:
            dt = time.time() - t_log
            rec = {"step": step, "loss": loss_mean, "lr": lr, "grad_norm": grad_norm, "scale": scale,
                   "tokens": (step + data_offset) * tokens_per_step, "tok_per_s": tok_log / max(dt, 1e-9),
                   "elapsed_s": elapsed_prev + time.time() - t_start}
            if d.device.type == "cuda":
                rec["peak_mem_gb"] = torch.cuda.max_memory_allocated(d.device) / 2**30
                rec["peak_reserved_gb"] = torch.cuda.max_memory_reserved(d.device) / 2**30
            logger.log(rec)
            if d.is_main:
                print(" ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in rec.items()), flush=True)
            t_log, tok_log = time.time(), 0

        if is_eval:
            ev = evaluate(model, val_data, d, tcfg)
            ev.update({"step": step, "event": "eval", "tokens": (step + data_offset) * tokens_per_step,
                       "elapsed_s": elapsed_prev + time.time() - t_start})
            if tgn is not None:
                ev["table_grad_norm"] = tgn
            if d.device.type == "cuda":
                ev["peak_mem_gb"] = torch.cuda.max_memory_allocated(d.device) / 2**30
            logger.log(ev)
            history[-1].update({k: v for k, v in ev.items() if isinstance(v, float)})
            if d.is_main:
                print(f"[eval] step={step} " + " ".join(f"{k}={v:.4g}" for k, v in ev.items() if isinstance(v, float)), flush=True)

        if low_scale >= int(tcfg["scaler_patience"]):
            exit_reason = f"abort: grad scale < {tcfg['scaler_min']} for {low_scale} steps"
            break
        if bad_loss >= int(tcfg["nan_patience"]):
            exit_reason = f"abort: non-finite loss for {bad_loss} steps"
            break
        stop_requested = d.any(_Stop.requested)
        if step % int(tcfg["ckpt_every"]) == 0 or step == stop_at or stop_requested:
            save(step)
        if stop_requested:
            exit_reason = "signal"
            break

    if d.is_main:
        logger.log({"event": "end", "step": step, "reason": exit_reason})
        print(f"[end] step={step} reason={exit_reason}", flush=True)
    logger.close()
    d.close()
    if exit_reason.startswith("abort"):
        raise SystemExit(exit_reason)
    return {"history": history, "step": step, "reason": exit_reason, "out_dir": str(out_dir), "params": counts}


if __name__ == "__main__":
    main(sys.argv[1:])
