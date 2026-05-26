"""Phase 2: supervised fine-tuning of envelope forecast head (frozen encoder)."""

import argparse
import sys
from pathlib import Path

import torch
from torch.optim import Adam
from tqdm import tqdm

from src_regi.config import artifacts_dir, load_config
from src_regi.data import prepare_data
from src_regi.datasets import build_dataloaders
from src_regi.evaluate import evaluate_predictions, predict_envelopes
from src_regi.forecast_head import ForecastHead
from src_regi.losses import envelope_loss
from src_regi.utils import get_device, load_checkpoint, save_checkpoint, set_seed
from src_regi.world_encoder import WorldEncoder


def build_encoder_from_checkpoint(cfg: dict, ckpt: dict, num_features: int, seq_len: int, device):
  mcfg = cfg["model"]
  encoder = WorldEncoder(
    num_features=num_features,
    seq_len=seq_len,
    d_model=mcfg["d_model"],
    d_latent=mcfg["d_latent"],
    nhead=mcfg["nhead"],
    num_layers=mcfg["num_layers"],
    dropout=mcfg["dropout"],
  ).to(device)
  encoder.load_state_dict(ckpt["world_encoder"])
  return encoder


def train_epoch(encoder, head, loader, optimizer, device, loss_type, freeze_encoder: bool):
  head.train()
  if freeze_encoder:
    encoder.eval()
  else:
    encoder.train()
  total = 0.0
  n = 0
  for batch in tqdm(loader, desc="phase2", leave=False):
    x = batch["x"].to(device)
    week_max = batch["week_max"].to(device)
    week_min = batch["week_min"].to(device)
    optimizer.zero_grad()
    if freeze_encoder:
      with torch.no_grad():
        z = encoder(x)
    else:
      z = encoder(x)
    pred = head(z)
    loss = envelope_loss(pred, week_max, week_min, loss_type=loss_type)
    loss.backward()
    optimizer.step()
    total += loss.item()
    n += 1
  return total / max(n, 1)


@torch.no_grad()
def eval_epoch(encoder, head, loader, device, loss_type, freeze_encoder: bool):
  head.eval()
  encoder.eval()
  total = 0.0
  n = 0
  for batch in loader:
    x = batch["x"].to(device)
    week_max = batch["week_max"].to(device)
    week_min = batch["week_min"].to(device)
    z = encoder(x)
    pred = head(z)
    total += envelope_loss(pred, week_max, week_min, loss_type=loss_type).item()
    n += 1
  return total / max(n, 1)


def main(config_path: str, phase1_ckpt: str | None = None, epochs: int | None = None):
  cfg = load_config(config_path)
  set_seed(42)
  device = get_device()
  prepared = prepare_data(cfg)
  seq_len = cfg["data"]["sequence_length"]
  batch_size = cfg["training"]["batch_size"]
  loaders = build_dataloaders(prepared, seq_len, batch_size, phase="phase2")

  art = artifacts_dir(cfg)
  ckpt_path = Path(phase1_ckpt) if phase1_ckpt else art / "phase1_best.pt"
  if not ckpt_path.is_absolute():
    ckpt_path = art / ckpt_path.name if not ckpt_path.exists() else ckpt_path
  if not ckpt_path.exists():
    ckpt_path = Path(cfg["_project_root"]) / "artifacts_regi" / "phase1_best.pt"
  ckpt = load_checkpoint(ckpt_path, map_location=device)

  num_features = len(prepared.feature_columns)
  encoder = build_encoder_from_checkpoint(cfg, ckpt, num_features, seq_len, device)
  freeze = cfg["training"].get("freeze_encoder", True)
  if freeze:
    for p in encoder.parameters():
      p.requires_grad = False

  head = ForecastHead(d_latent=cfg["model"]["d_latent"], dropout=cfg["model"]["dropout"]).to(device)
  tcfg = cfg["training"]
  params = head.parameters() if freeze else list(encoder.parameters()) + list(head.parameters())
  optimizer = Adam(params, lr=tcfg["lr_phase2"], weight_decay=1e-5)
  loss_type = tcfg.get("loss", "pinball")
  n_epochs = epochs or tcfg["phase2_epochs"]

  best_val = float("inf")
  stale = 0
  patience = tcfg.get("early_stop_patience", 8)

  for epoch in range(1, n_epochs + 1):
    train_loss = train_epoch(
      encoder, head, loaders["train"], optimizer, device, loss_type, freeze
    )
    val_loss = eval_epoch(encoder, head, loaders["val"], device, loss_type, freeze)
    print(f"Epoch {epoch}/{n_epochs} | train={train_loss:.4f} | val={val_loss:.4f}")
    if val_loss < best_val:
      best_val = val_loss
      stale = 0
      save_checkpoint(
        art / "phase2_head.pt",
        {
          "config": cfg,
          "forecast_head": head.state_dict(),
          "phase1_ckpt": str(ckpt_path),
          "freeze_encoder": freeze,
          "envelope_mean": ckpt.get("envelope_mean"),
          "envelope_scale": ckpt.get("envelope_scale"),
        },
      )
    else:
      stale += 1
      if stale >= patience:
        print(f"Early stopping at epoch {epoch}")
        break

  pred_max, pred_min, true_max, true_min = predict_envelopes(
    encoder,
    head,
    loaders["test"],
    device,
    envelope_mean=ckpt.get("envelope_mean"),
    envelope_scale=ckpt.get("envelope_scale"),
  )
  metrics = evaluate_predictions(pred_max, pred_min, true_max, true_min, loss_type=loss_type)
  print("Test metrics:", metrics)
  print(f"Phase 2 complete. Checkpoint: {art / 'phase2_head.pt'}")


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--config", type=str, default="config_regi.json")
  parser.add_argument("--phase1-ckpt", type=str, default=None)
  parser.add_argument("--epochs", type=int, default=None)
  args = parser.parse_args()
  project_root = Path(__file__).resolve().parent.parent
  if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
  main(args.config, args.phase1_ckpt, args.epochs)
