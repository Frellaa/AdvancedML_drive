"""Phase 1: unsupervised representation learning (TimeGAN + latent Cycle-GAN + WorldEncoder)."""

import argparse
import sys
from pathlib import Path

import torch
from torch.optim import Adam
from tqdm import tqdm

from src_regi.config import artifacts_dir, load_config
from src_regi.data import prepare_data
from src_regi.datasets import build_dataloaders
from src_regi.latent_cycle_gan import LatentCycleGAN
from src_regi.losses import Phase1LossBundle, adversarial_loss, reconstruction_loss, supervised_loss
from src_regi.timegan import TimeGAN
from src_regi.utils import get_device, save_checkpoint, set_seed
from src_regi.world_encoder import WorldEncoder


def _clip_grad(models, max_norm: float):
  for m in models:
    if m is not None:
      torch.nn.utils.clip_grad_norm_(m.parameters(), max_norm)


def train_one_epoch(
  world_encoder,
  timegan,
  cycle_gan,
  loader,
  opt_enc,
  opt_tg,
  opt_cycle,
  opt_d,
  loss_bundle,
  device,
  grad_clip,
):
  world_encoder.train()
  timegan.train()
  cycle_gan.train()
  metrics_sum: dict[str, float] = {}
  n_batches = 0

  for batch in tqdm(loader, desc="phase1", leave=False):
    x = batch["x"].to(device)
    regime = batch["regime"].to(device)
    bsz = x.size(0)

    # --- Discriminator step (TimeGAN + regime) ---
    opt_d.zero_grad()
    with torch.no_grad():
      h = timegan.embed(x)
      z_world = world_encoder(x)
      z_pooled = TimeGAN.pool_hidden(h)
      z_fake_latent = cycle_gan.forward(z_world, regime)["z_ab"]
    h_fake = timegan.generate(timegan.sample_noise(bsz, device))
    d_real = timegan.discriminate(h.detach())
    d_fake = timegan.discriminate(h_fake.detach())
    loss_d_tg = adversarial_loss(d_real, True) + adversarial_loss(d_fake, False)

    z_a = z_world[regime == 0]
    z_b = z_world[regime == 1]
    loss_d_reg = torch.tensor(0.0, device=device)
    if z_a.numel() > 0:
      loss_d_reg = loss_d_reg + adversarial_loss(cycle_gan.discriminate_a(z_a.detach()), True)
    if z_b.numel() > 0:
      loss_d_reg = loss_d_reg + adversarial_loss(cycle_gan.discriminate_b(z_b.detach()), True)
    if z_a.numel() > 0 or z_b.numel() > 0:
      z_ab = cycle_gan.g_ab(z_a) if z_a.numel() > 0 else None
      if z_b.numel() > 0:
        z_from_b = cycle_gan.g_ba(z_b)
        loss_d_reg = loss_d_reg + adversarial_loss(cycle_gan.discriminate_a(z_from_b.detach()), False)
      if z_ab is not None and z_ab.numel() > 0:
        loss_d_reg = loss_d_reg + adversarial_loss(cycle_gan.discriminate_b(z_ab.detach()), False)

    loss_d = loss_d_tg + loss_d_reg
    loss_d.backward()
    _clip_grad([timegan, cycle_gan], grad_clip)
    opt_d.step()

    # --- Generator / cycle step ---
    opt_cycle.zero_grad()
    opt_tg.zero_grad()
    h = timegan.embed(x)
    x_hat = timegan.recover(h)
    h_sup = timegan.supervise(h)
    h_fake = timegan.generate(timegan.sample_noise(bsz, device))
    d_fake_g = timegan.discriminate(h_fake)
    z_world = world_encoder(x)
    z_pooled = TimeGAN.pool_hidden(h)
    z_cycle = cycle_gan.cycle(z_world, regime)
    out_cycle = cycle_gan.forward(z_world, regime)
    d_reg_fake = torch.tensor(0.0, device=device)
    mask_a = regime == 0
    mask_b = regime == 1
    if mask_a.any():
      d_reg_fake = d_reg_fake + adversarial_loss(
        cycle_gan.discriminate_b(out_cycle["z_ab"][mask_a]), True
      )
    if mask_b.any():
      d_reg_fake = d_reg_fake + adversarial_loss(
        cycle_gan.discriminate_a(out_cycle["z_ba"][mask_b]), True
      )

    loss_g, batch_metrics = loss_bundle(
      x,
      x_hat,
      h,
      h_sup,
      z_world,
      z_pooled,
      d_fake_g,
      d_fake_g,
      z_cycle,
      d_regime_fake=d_reg_fake if d_reg_fake.item() != 0 else None,
    )
    loss_g.backward()
    _clip_grad([timegan, cycle_gan, world_encoder], grad_clip)
    opt_cycle.step()
    opt_tg.step()

    # --- Embedder / recovery / world encoder ---
    opt_enc.zero_grad()
    h = timegan.embed(x)
    x_hat = timegan.recover(h)
    h_sup = timegan.supervise(h)
    z_world = world_encoder(x)
    z_pooled = TimeGAN.pool_hidden(h)
    l_recon = reconstruction_loss(x, x_hat)
    l_sup = supervised_loss(h, h_sup)
    l_align = torch.nn.functional.mse_loss(z_world, z_pooled)
    loss_e = 10.0 * l_recon + 5.0 * l_sup + 1.0 * l_align
    loss_e.backward()
    _clip_grad([world_encoder, timegan], grad_clip)
    opt_enc.step()

    for k, v in batch_metrics.items():
      metrics_sum[k] = metrics_sum.get(k, 0.0) + v
    n_batches += 1

  return {k: v / max(n_batches, 1) for k, v in metrics_sum.items()}


@torch.no_grad()
def validate(world_encoder, timegan, cycle_gan, loader, loss_bundle, device):
  world_encoder.eval()
  timegan.eval()
  cycle_gan.eval()
  total = 0.0
  n = 0
  for batch in loader:
    x = batch["x"].to(device)
    regime = batch["regime"].to(device)
    h = timegan.embed(x)
    x_hat = timegan.recover(h)
    h_sup = timegan.supervise(h)
    z_world = world_encoder(x)
    z_pooled = TimeGAN.pool_hidden(h)
    z_cycle = cycle_gan.cycle(z_world, regime)
    h_fake = timegan.generate(timegan.sample_noise(x.size(0), device))
    d_fake = timegan.discriminate(h_fake)
    loss, _ = loss_bundle(
      x, x_hat, h, h_sup, z_world, z_pooled, d_fake, d_fake, z_cycle
    )
    total += loss.item()
    n += 1
  return total / max(n, 1)


def build_models(cfg: dict, num_features: int, seq_len: int, device):
  mcfg = cfg["model"]
  world_encoder = WorldEncoder(
    num_features=num_features,
    seq_len=seq_len,
    d_model=mcfg["d_model"],
    d_latent=mcfg["d_latent"],
    nhead=mcfg["nhead"],
    num_layers=mcfg["num_layers"],
    dropout=mcfg["dropout"],
  ).to(device)
  timegan = TimeGAN(
    num_features=num_features,
    seq_len=seq_len,
    hidden=mcfg["timegan_hidden"],
    z_dim=mcfg["timegan_hidden"],
  ).to(device)
  cycle_gan = LatentCycleGAN(d_latent=mcfg["d_latent"], hidden=mcfg["d_model"]).to(device)
  return world_encoder, timegan, cycle_gan


def main(config_path: str, epochs: int | None = None):
  cfg = load_config(config_path)
  set_seed(42)
  device = get_device()
  prepared = prepare_data(cfg)
  seq_len = cfg["data"]["sequence_length"]
  batch_size = cfg["training"]["batch_size"]
  loaders = build_dataloaders(prepared, seq_len, batch_size, phase="phase1")

  num_features = len(prepared.feature_columns)
  world_encoder, timegan, cycle_gan = build_models(cfg, num_features, seq_len, device)

  tcfg = cfg["training"]
  n_epochs = epochs or tcfg["phase1_epochs"]
  loss_bundle = Phase1LossBundle(tcfg.get("loss_weights", {}))

  opt_enc = Adam(
    list(world_encoder.parameters()) + list(timegan.parameters()),
    lr=tcfg["lr_phase1"],
    weight_decay=1e-5,
  )
  opt_tg = Adam(timegan.parameters(), lr=tcfg["lr_phase1"])
  opt_cycle = Adam(cycle_gan.parameters(), lr=tcfg["lr_phase1"])
  opt_d = Adam(
    list(timegan.parameters()) + list(cycle_gan.parameters()),
    lr=tcfg.get("lr_disc", tcfg["lr_phase1"]),
  )

  best_val = float("inf")
  patience = tcfg.get("early_stop_patience", 8)
  stale = 0
  art = artifacts_dir(cfg)

  for epoch in range(1, n_epochs + 1):
    train_metrics = train_one_epoch(
      world_encoder,
      timegan,
      cycle_gan,
      loaders["train"],
      opt_enc,
      opt_tg,
      opt_cycle,
      opt_d,
      loss_bundle,
      device,
      tcfg.get("grad_clip", 1.0),
    )
    val_loss = validate(world_encoder, timegan, cycle_gan, loaders["val"], loss_bundle, device)
    print(
      f"Epoch {epoch}/{n_epochs} | val_loss={val_loss:.4f} | "
      + " ".join(f"{k}={v:.4f}" for k, v in train_metrics.items())
    )
    if val_loss < best_val:
      best_val = val_loss
      stale = 0
      save_checkpoint(
        art / "phase1_best.pt",
        {
          "config": cfg,
          "world_encoder": world_encoder.state_dict(),
          "timegan": timegan.state_dict(),
          "cycle_gan": cycle_gan.state_dict(),
          "scaler_mean": prepared.scaler.mean_.tolist(),
          "scaler_scale": prepared.scaler.scale_.tolist(),
          "envelope_mean": prepared.envelope_scaler.mean_.tolist(),
          "envelope_scale": prepared.envelope_scaler.scale_.tolist(),
          "feature_columns": prepared.feature_columns,
          "train_end": prepared.train_end,
          "val_end": prepared.val_end,
        },
      )
    else:
      stale += 1
      if stale >= patience:
        print(f"Early stopping at epoch {epoch}")
        break

  print(f"Phase 1 complete. Best val loss: {best_val:.4f}. Checkpoint: {art / 'phase1_best.pt'}")


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument(
    "--config",
    type=str,
    default="config_regi.json",
    help="Path to config_regi.json",
  )
  parser.add_argument("--epochs", type=int, default=None)
  args = parser.parse_args()
  project_root = Path(__file__).resolve().parent.parent
  if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
  main(args.config, args.epochs)
