# src/train_one_epoch.py
import torch
from torch.amp import autocast, GradScaler


def _tensor_stats(name: str, x: torch.Tensor, *, max_items: int = 5):
    """
    Return a compact stats string for a tensor (on GPU ok).
    """
    if x is None:
        return f"{name}: None"
    if not torch.is_tensor(x):
        return f"{name}: not a tensor"

    # avoid huge sync: use small ops but still sync for .item()
    finite = torch.isfinite(x)
    n = x.numel()
    n_finite = finite.sum().item()
    n_nan = torch.isnan(x).sum().item()
    n_inf = torch.isinf(x).sum().item()

    if n_finite == 0:
        return f"{name}: shape={tuple(x.shape)} dtype={x.dtype} device={x.device} finite=0/{n} nan={n_nan} inf={n_inf}"

    x_f = x[finite]
    # keep ops minimal
    xmin = x_f.min().item()
    xmax = x_f.max().item()
    mean = x_f.mean().item()
    std = x_f.std(unbiased=False).item() if x_f.numel() > 1 else 0.0

    return (
        f"{name}: shape={tuple(x.shape)} dtype={x.dtype} device={x.device} "
        f"finite={n_finite}/{n} nan={n_nan} inf={n_inf} "
        f"min={xmin:.6g} max={xmax:.6g} mean={mean:.6g} std={std:.6g}"
    )


def train_one_epoch(
    model,
    dataloader,
    optimizer,
    edge_index,
    edge_weight,
    device,
    *,
    use_amp: bool = True,
    accum_steps: int = 1,
    eps: float = 1e-8,

    # ---------------- DEBUG switches ----------------
    debug: bool = False,
    debug_first_k_steps: int = 3,     # print detailed tensors for first K effective batches
    debug_every_n_steps: int = 200,   # thereafter print a compact line every N effective batches
    debug_check_grad_every: int = 200 # check grad/param update frequency
):
    """
    Paper-aligned training loss (NO q_weight, NO peak weighting):

        loss = mean( (pred - target)^2 over valid (mask==True) entries )

    Debug prints:
    - input/output tensor stats
    - mask coverage
    - diff2 stats
    - loss stats
    - step/zero_grad/optimizer.step counts
    - grad norms & param update sanity check
    """

    model.train()

    total_loss = 0.0
    valid_batches = 0
    skipped_batches = 0
    opt_steps = 0
    zero_grads = 0

    edge_index = edge_index.to(device, non_blocking=True)
    edge_weight = edge_weight.to(device, non_blocking=True)

    amp_enabled = bool(use_amp and device.type == "cuda")
    scaler = GradScaler(
        device="cuda" if amp_enabled else "cpu",
        enabled=amp_enabled,
    )

    optimizer.zero_grad(set_to_none=True)
    zero_grads += 1

    # pick one parameter to track update
    tracked_param_name = None
    tracked_param_ref = None
    tracked_param_prev = None
    for n, p in model.named_parameters():
        if p.requires_grad and p.numel() > 0:
            tracked_param_name = n
            tracked_param_ref = p
            tracked_param_prev = p.detach().clone()
            break

    if debug:
        print(f"[DEBUG] amp_enabled={amp_enabled} accum_steps={accum_steps}")
        print(f"[DEBUG] edge_index: {tuple(edge_index.shape)} edge_weight: {tuple(edge_weight.shape) if edge_weight is not None else None}")
        if tracked_param_name is not None:
            print(f"[DEBUG] tracking param: {tracked_param_name}, shape={tuple(tracked_param_ref.shape)}, dtype={tracked_param_ref.dtype}")

    for step, batch in enumerate(dataloader, start=1):
        dynamic = batch["dynamic"].to(device, non_blocking=True)
        static  = batch["static"].to(device, non_blocking=True)
        target  = batch["target"].to(device, non_blocking=True)  # (B, N)
        mask    = batch["mask"].to(device, non_blocking=True)    # bool (B, N)

        # --- FIX 1: enforce mask excludes non-finite target, and zero-out invalid targets ---
        mask = mask & torch.isfinite(target)
        target = torch.where(mask, target, torch.zeros_like(target))

        # ---------------- batch validity ----------------
        mask_sum = mask.sum().item()
        if mask_sum == 0:
            skipped_batches += 1
            if debug and skipped_batches <= 3:
                print(f"[DEBUG] step={step}: skipped batch because mask.sum()==0")
            continue

        # Your current logic: zero_grad based on valid_batches
        if (valid_batches % accum_steps) == 0:
            optimizer.zero_grad(set_to_none=True)
            zero_grads += 1

        # ---------------- forward ----------------
        with autocast(device_type="cuda", enabled=amp_enabled):
            pred = model(
                dynamic_features=dynamic,
                static_features=static,
                edge_index=edge_index,
                edge_weight=edge_weight,
            )  # (B, N)

            diff2 = (pred - target) ** 2
            diff2_valid = diff2[mask]
            loss = diff2_valid.mean()

        # ---------------- debug prints ----------------
        # Use "effective batch id" = valid_batches+1 (because valid_batches increments below)
        eff = valid_batches + 1
        do_detail = debug and (eff <= debug_first_k_steps)
        do_period = debug and (eff > debug_first_k_steps) and (debug_every_n_steps > 0) and (eff % debug_every_n_steps == 0)

        if do_detail:
            print(f"\n[DEBUG] ===== effective_batch={eff} (raw_step={step}) =====")
            print(_tensor_stats("dynamic", dynamic))
            print(_tensor_stats("static", static))
            print(_tensor_stats("target", target))
            print(_tensor_stats("mask(float)", mask.float()))
            print(_tensor_stats("pred", pred))
            print(_tensor_stats("diff2_valid", diff2_valid))
            # mask coverage
            B, N = target.shape
            print(f"[DEBUG] mask.sum={mask_sum} / (B*N={B*N}) => valid_frac={mask_sum/(B*N):.3%}")
            # some quick sanity: target valid range
            t_valid = target[mask]
            print(_tensor_stats("target_valid", t_valid))
            print(f"[DEBUG] loss={loss.detach().float().item():.6g}")

        elif do_period:
            print(
                f"[DEBUG] eff={eff} step={step} "
                f"loss={loss.detach().float().item():.6g} "
                f"mask_frac={(mask_sum/target.numel()):.3%} "
                f"pred_finite={torch.isfinite(pred).float().mean().item():.3%}"
            )

        # ---------------- backward ----------------
        loss_to_backprop = loss / accum_steps
        if amp_enabled:
            scaler.scale(loss_to_backprop).backward()
        else:
            loss_to_backprop.backward()

        total_loss += float(loss.detach().cpu().item())
        valid_batches += 1

        # ---------------- optimizer step (your current logic) ----------------
        if (valid_batches % accum_steps) == 0:
            if amp_enabled:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            opt_steps += 1

            # param update debug
            if debug and tracked_param_ref is not None and (opt_steps <= 3 or (debug_check_grad_every > 0 and opt_steps % debug_check_grad_every == 0)):
                with torch.no_grad():
                    delta = (tracked_param_ref.detach() - tracked_param_prev).abs().mean().item()
                    tracked_param_prev = tracked_param_ref.detach().clone()
                print(f"[DEBUG] optimizer.step #{opt_steps}: tracked_param mean|Δ|={delta:.6g}")

        # grad norm debug (occasionally)
        if debug and tracked_param_ref is not None and (debug_check_grad_every > 0) and (valid_batches % debug_check_grad_every == 0):
            g = tracked_param_ref.grad
            if g is None:
                print(f"[DEBUG] eff={valid_batches}: tracked_param.grad is None (no backward?)")
            else:
                g_f = g.detach()
                g_norm = torch.linalg.vector_norm(g_f).item()
                g_nan = torch.isnan(g_f).any().item()
                g_inf = torch.isinf(g_f).any().item()
                print(f"[DEBUG] eff={valid_batches}: tracked_grad_norm={g_norm:.6g} nan={g_nan} inf={g_inf}")

    if debug:
        print(
            f"\n[DEBUG] epoch summary: valid_batches={valid_batches}, skipped_batches={skipped_batches}, "
            f"zero_grads={zero_grads}, optimizer_steps={opt_steps}"
        )
        if accum_steps > 1 and opt_steps == 0:
            print("[DEBUG][WARN] optimizer_steps==0 with accum_steps>1 => parameters are NOT updating!")

    return total_loss / max(valid_batches, 1)
