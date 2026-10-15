import argparse
import json
import math
import time
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.signal import butter, sosfilt
from torch import nn
from torch.nn import functional as F


def json_ready(value):
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and (not np.isfinite(value)):
        return None
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_ready(v) for v in value]
    return value


class Quality(nn.Module):

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(6, 1)

    def forward(self, x):
        return self.linear(x).squeeze(-1)


class LossModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(10, 32), nn.ReLU(), nn.Linear(32, 32), nn.ReLU(), nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.layers(x).squeeze(-1)


def estimate(values, fs):
    x = np.asarray(values, dtype=float)
    x = x - x.mean()
    if len(x) < 4 or np.std(x) < 1e-10:
        return (np.nan, np.array([0, 0, 0, 0, 0, 0.0]))
    frequency = np.arange(30, 211) / 60
    spectrum = (
        np.abs(
            np.exp(-2j * np.pi * frequency[:, None] * np.arange(len(x))[None] / fs)
            @ (x * np.hanning(len(x)))
        )
        ** 2
    )
    correlation = np.correlate(x, x, mode="full")[len(x) - 1 :]
    correlation = correlation / max(correlation[0], 1e-12)
    lags = np.clip(np.rint(fs / frequency).astype(int), 0, len(x) - 1)
    ac = correlation[lags].clip(0)
    scores = 0.5 * spectrum / max(spectrum.max(), 1e-12) + 0.5 * ac / max(
        ac.max(), 1e-12
    )
    heart_rate = float(frequency[np.argmax(scores)] * 60)
    concentration = float(
        np.sum(spectrum[np.abs(frequency - heart_rate / 60) <= 0.1])
        / max(spectrum.sum(), 1e-12)
    )
    saturated = float(np.mean((x == x.max()) | (x == x.min())))
    return (
        heart_rate,
        np.array(
            [concentration, float(ac.max()), 0.0, saturated, float(np.std(x)), 0.0]
        ),
    )


def read_record(row, cfg):
    frame = pd.read_csv(row["file"])
    t = frame.time.to_numpy()
    optical = frame.ppg.to_numpy()
    if not len(t) or abs(t[0]) > 1e-06:
        raise ValueError("Each record must start at time zero")
    if np.any(np.diff(t) <= 0):
        raise ValueError("Record times must be strictly increasing")
    fs = float(row["sample_rate"])
    expected = 1 / fs
    if np.max(np.abs(np.diff(t) - expected)) > expected * 0.05:
        raise ValueError("Split acquisition gaps into separate manifest records")
    analog = sosfilt(
        butter(4, cfg["analog_bandwidth_hz"], btype="lowpass", fs=fs, output="sos"),
        optical,
    )
    motion = (
        np.sqrt(frame[["ax", "ay", "az"]].pow(2).sum(1).to_numpy())
        if {"ax", "ay", "az"}.issubset(frame)
        else np.zeros(len(frame))
    )
    return (
        t,
        analog,
        motion,
        frame.hr_reference.to_numpy(),
        frame.usable_reference.to_numpy() if "usable_reference" in frame else None,
    )


def fit(args, cfg, manifest):
    features = []
    labels = []
    for row in manifest[manifest.split == "train"].to_dict("records"):
        t, ppg, motion, reference, _ = read_record(row, cfg)
        for end in np.arange(cfg["window_s"], t[-1], cfg["epoch_s"]):
            full = (t > end - cfg["window_s"]) & (t <= end)
            truth = float(np.mean(reference[full]))
            for action, rate in enumerate(cfg["rates_hz"]):
                sample_times = np.arange(end - cfg["window_s"], end, 1 / rate)
                values = np.interp(sample_times, t, ppg)
                filtered = sosfilt(
                    butter(4, [0.5, 4], btype="bandpass", fs=rate, output="sos"), values
                )
                prediction, z = estimate(filtered, rate)
                z[2] = float(np.sqrt(np.mean(motion[full] ** 2)))
                z[5] = cfg["epoch_s"]
                good = (
                    np.isfinite(prediction)
                    and abs(prediction - truth) <= cfg["quality_error_bpm"]
                )
                features.append(z)
                labels.append(float(good))
                short = values[-max(4, int(cfg["probe_s"] * rate)) :]
                _, probe_features = estimate(short, rate)
                probe_features[2] = z[2]
                probe_features[5] = cfg["probe_gap_s"]
                features.append(probe_features)
                labels.append(float(good))
    x = np.asarray(features, dtype=np.float32)
    mean = x.mean(0)
    scale = x.std(0).clip(1e-06)
    x = torch.tensor((x - mean) / scale)
    y = torch.tensor(labels)
    quality = Quality()
    optimizer = torch.optim.Adam(quality.parameters(), lr=0.001)
    for _ in range(100):
        loss = (
            F.binary_cross_entropy_with_logits(quality(x), y)
            + 0.01 * quality.linear.weight.square().sum()
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    action_features = []
    losses = []
    quality.eval()
    for row in manifest[manifest.split.isin(["train", "development"])].to_dict(
        "records"
    ):
        t, ppg, motion, reference, _ = read_record(row, cfg)
        high = cfg["rates_hz"][-1]
        for start in np.arange(cfg["window_s"], t[-1] - cfg["epoch_s"], cfg["epoch_s"]):
            past = np.arange(start - cfg["window_s"], start, 1 / high)
            values = np.interp(past, t, ppg)
            filtered = sosfilt(
                butter(4, [0.5, 4], btype="bandpass", fs=high, output="sos"), values
            )
            _, z = estimate(filtered, high)
            z[2] = float(
                np.sqrt(
                    np.mean(motion[(t >= start - cfg["window_s"]) & (t < start)] ** 2)
                )
            )
            z[5] = cfg["epoch_s"]
            with torch.no_grad():
                belief = float(
                    torch.sigmoid(
                        quality(
                            torch.tensor(
                                ((z - mean) / scale)[None], dtype=torch.float32
                            )
                        )
                    )[0]
                )
            for action in ["sleep", "probe"] + [str(v) for v in cfg["rates_hz"]]:
                ar = (
                    0
                    if action == "sleep"
                    else high if action == "probe" else float(action)
                )
                future_end = start + cfg["epoch_s"]
                mask = (t >= future_end - cfg["window_s"]) & (t < future_end)
                future = (
                    np.arange(future_end - cfg["window_s"], future_end, 1 / ar)
                    if ar
                    else np.array([])
                )
                if action not in ["sleep", "probe"]:
                    acquired = np.interp(future, t, ppg)
                    filtered = sosfilt(
                        butter(4, [0.5, 4], btype="bandpass", fs=ar, output="sos"),
                        acquired,
                    )
                    pred, _ = estimate(filtered, ar)
                    error = abs(pred - float(np.mean(reference[mask])))
                else:
                    error = cfg["missing_penalty_bpm"]
                for buffer_state in [0.0, 1.0]:
                    action_features.append(
                        [
                            belief,
                            buffer_state,
                            0.0,
                            *[float(action == str(v)) for v in cfg["rates_hz"]],
                            float(action == "probe"),
                            float(action == "sleep"),
                            ar / high,
                            1.0,
                        ]
                    )
                    losses.append(
                        min(error, cfg["missing_penalty_bpm"]) / 15
                        if buffer_state
                        else cfg["missing_penalty_bpm"] / 15
                    )
    model = LossModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    af = torch.tensor(action_features, dtype=torch.float32)
    target = torch.tensor(losses, dtype=torch.float32)
    for _ in range(cfg["fit_epochs"]):
        for ix in torch.randperm(len(af)).split(64):
            optimizer.zero_grad()
            loss = F.mse_loss(model(af[ix]), target[ix])
            loss.backward()
            optimizer.step()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "quality": quality.state_dict(),
            "loss": model.state_dict(),
            "mean": mean,
            "scale": scale,
            "prior": float(np.mean(labels)),
            "config": cfg,
        },
        out / "policy.pt",
    )


def replay(row, cfg, bundle, variant):
    t, source, motion, reference, usable = read_record(row, cfg)
    end = float(t[-1])
    epoch = cfg["epoch_s"]
    high = cfg["rates_hz"][-1]
    connection = (
        pd.read_csv(row["file"]).connected.to_numpy()
        if cfg["phone_connection_column"]
        else None
    )
    baseline_per_epoch = cfg["base_mw"] * epoch + cfg["packet_mj"] + cfg["cpu_mj"][0]
    if cfg["cap_mw"] * epoch < baseline_per_epoch:
        raise ValueError("Energy allowance cannot fund the always-on baseline")
    quality = Quality()
    quality.load_state_dict(bundle["quality"])
    quality.eval()
    loss_model = LossModel()
    loss_model.load_state_dict(bundle["loss"])
    loss_model.eval()
    belief = bundle["prior"]
    tau = cfg["probe_gap_s"]
    age = cfg["stale_display_s"] + 1.0
    previous = "sleep"
    rate = None
    buffer = []
    sample_time = []
    filter_state = None
    since_switch = np.inf
    remaining_budget = cfg["cap_mw"] * cfg["block_s"]
    promised = cfg["reserved_recoveries"]
    probe_gap = cfg["probe_gap_s"]
    charged_rate = high if cfg["acquisition_tier"] == "sensor" else row["sample_rate"]
    recovery = 0.0
    last_estimate = None
    block = -1
    rows = []
    optical_count = 0
    cost_total = 0.0
    switches = 0
    runtime = []
    queue = []
    delivered = []
    recovery_delays = []
    recovery_onset = None
    start = 0.0
    while start < end:
        duration = min(epoch, end - start, cfg["block_s"] - start % cfg["block_s"])
        current_block = int(start // cfg["block_s"])
        if current_block != block:
            block = current_block
            remaining_budget = cfg["cap_mw"] * cfg["block_s"]
            promised = cfg["reserved_recoveries"]
            probe_gap = cfg["probe_gap_s"]
        block_remaining = min(end - start, cfg["block_s"] - start % cfg["block_s"])
        previous_features = np.array(
            [
                belief,
                min(len(buffer) / ((rate or high) * cfg["window_s"]), 1),
                min(age / cfg["stale_display_s"], 2),
            ]
        )

        def action_cost(action):
            action_rate = (
                0 if action == "sleep" else high if action == "probe" else float(action)
            )
            acquire = min(duration, cfg["probe_s"]) if action == "probe" else duration
            reads = (
                int(round(action_rate * acquire))
                if cfg["acquisition_tier"] == "sensor"
                else (
                    int(round(row["sample_rate"] * acquire)) if action != "sleep" else 0
                )
            )
            return (
                cfg["base_mw"] * duration
                + reads * cfg["readout_mj"]
                + cfg["cpu_mj"][
                    0 if action == "sleep" else 1 if action == "probe" else 2
                ]
                + (cfg["switch_mj"] if action != previous else 0)
                + cfg["packet_mj"]
            )

        def reserve_after(action):
            left = max(0, block_remaining - duration)
            checked = action != "sleep"
            new_tau = 0 if checked else tau + duration
            probes = (
                max(
                    0,
                    int(math.floor((left - max(0, probe_gap - new_tau)) / probe_gap))
                    + 1,
                )
                if left >= max(0, probe_gap - new_tau)
                else 0
            )
            probe_increment = (
                charged_rate * cfg["probe_s"] * cfg["readout_mj"]
                + cfg["switch_mj"]
                + cfg["cpu_mj"][1]
            )
            burst_increment = (
                charged_rate * cfg["recovery_s"] * cfg["readout_mj"]
                + cfg["switch_mj"]
                + math.ceil(cfg["recovery_s"] / epoch) * cfg["cpu_mj"][2]
            )
            funded = (
                max(0, recovery - duration)
                if recovery and action == str(high)
                else recovery
            )
            return (
                probes * probe_increment
                + promised * burst_increment
                + funded * charged_rate * cfg["readout_mj"]
                + math.ceil(funded / epoch) * cfg["cpu_mj"][2]
                + left * cfg["base_mw"]
                + math.ceil(left / epoch) * (cfg["packet_mj"] + cfg["cpu_mj"][0])
                + (cfg["switch_mj"] if left > 0 else 0)
            )

        minimum = reserve_after("sleep") + action_cost("sleep")
        service = "funded" if remaining_budget >= minimum else "degraded"
        while promised > 0 and minimum > remaining_budget:
            promised -= 1
            minimum = reserve_after("sleep") + action_cost("sleep")
        while minimum > remaining_budget and probe_gap <= 2 * cfg["block_s"]:
            probe_gap *= 2
            minimum = reserve_after("sleep") + action_cost("sleep")
        actions = ["sleep", "probe"] + [str(v) for v in cfg["rates_hz"]]
        baseline_left = (
            max(0, block_remaining - duration) * cfg["base_mw"]
            + math.ceil(max(0, block_remaining - duration) / epoch)
            * (cfg["packet_mj"] + cfg["cpu_mj"][0])
            + cfg["switch_mj"]
        )
        if (
            variant in ["no-reserve", "greedy"]
            or variant.startswith("fixed")
            or variant == "periodic"
        ):
            feasible = [
                a for a in actions if action_cost(a) + baseline_left <= remaining_budget
            ]
        else:
            feasible = [
                a
                for a in actions
                if action_cost(a) + reserve_after(a) <= remaining_budget + 1e-09
            ]
        if variant == "greedy":
            desired = (
                "probe"
                if tau >= probe_gap
                else (
                    str(cfg["rates_hz"][0])
                    if belief >= cfg["quality_threshold"]
                    else "sleep"
                )
            )
            action = desired if desired in feasible else "sleep"
        elif variant.startswith("fixed"):
            desired = str(cfg["rates_hz"][int(variant[-1])])
            action = desired if desired in feasible else "sleep"
        elif recovery > 0 and str(high) in feasible:
            action = str(high)
        elif tau >= probe_gap and "probe" in feasible:
            action = "probe"
        elif variant == "periodic":
            action = (
                str(high)
                if int(start // epoch) % 2 == 0 and str(high) in feasible
                else "sleep"
            )
        elif (
            variant != "no-dwell"
            and since_switch < cfg["dwell_s"]
            and (previous in feasible)
        ):
            action = previous
        else:
            choices = []
            for candidate in feasible:
                ar = (
                    0
                    if candidate == "sleep"
                    else high if candidate == "probe" else float(candidate)
                )
                vector = np.r_[
                    previous_features,
                    [float(candidate == str(v)) for v in cfg["rates_hz"]],
                    float(candidate == "probe"),
                    float(candidate == "sleep"),
                    ar / high,
                    1.0,
                ]
                with torch.no_grad():
                    predicted_loss = float(
                        loss_model(torch.tensor(vector[None], dtype=torch.float32))[0]
                    )
                if candidate == "sleep":
                    predicted_loss = cfg["missing_penalty_bpm"] / 15
                output_probability = (
                    belief * float(previous_features[1] >= 1)
                    if candidate not in ["sleep", "probe"]
                    else 0.0
                )
                age_penalty = (
                    0
                    if variant == "no-age"
                    else 0.2
                    * (1 - output_probability)
                    * min(age + duration, cfg["stale_display_s"])
                    / cfg["stale_display_s"]
                )
                choices.append(
                    (
                        predicted_loss
                        + 0.1 * action_cost(candidate) / (cfg["cap_mw"] * epoch)
                        + age_penalty,
                        candidate,
                    )
                )
            action = min(choices)[1] if choices else "unavailable"
        timer = time.perf_counter()
        changed = action != previous
        if changed:
            switches += 1
            since_switch = 0
        since_switch += duration
        age += duration
        tau += duration
        estimate_value = np.nan
        valid_output = False
        reads = 0
        cpu_energy = cfg["cpu_mj"][0]
        if action not in ["sleep", "unavailable"]:
            ar = high if action == "probe" else float(action)
            acquire = min(duration, cfg["probe_s"]) if action == "probe" else duration
            sampling = np.arange(start, start + acquire, 1 / ar)
            values = np.interp(sampling, t, source)
            reads = (
                len(values)
                if cfg["acquisition_tier"] == "sensor"
                else int(round(row["sample_rate"] * acquire))
            )
            if rate != ar or previous in ["sleep", "unavailable", "probe"]:
                buffer = []
                sample_time = []
                filter_state = None
            sos = butter(4, [0.5, 4], btype="bandpass", fs=ar, output="sos")
            if filter_state is None:
                filter_state = np.zeros((len(sos), 2))
            filtered, filter_state = sosfilt(sos, values, zi=filter_state)
            buffer.extend(filtered)
            sample_time.extend(sampling)
            rate = ar
            keep = int(ar * cfg["window_s"])
            buffer = buffer[-keep:]
            sample_time = sample_time[-keep:]
            estimate_value, z = estimate(np.array(buffer), ar)
            mask = (t >= start) & (t < start + duration)
            z[2] = float(np.sqrt(np.mean(motion[mask] ** 2))) if mask.any() else 0.0
            z[5] = tau
            with torch.no_grad():
                belief = float(
                    torch.sigmoid(
                        quality(
                            torch.tensor(
                                ((z - bundle["mean"]) / bundle["scale"])[None],
                                dtype=torch.float32,
                            )
                        )
                    )[0]
                )
            tau = max(0.0, duration - acquire)
            cpu_energy = cfg["cpu_mj"][1 if action == "probe" else 2]
            if action == "probe":
                if belief >= cfg["quality_threshold"] and promised > 0:
                    recovery = cfg["recovery_s"]
                    promised -= 1
                buffer = []
                sample_time = []
            else:
                if recovery > 0:
                    recovery = max(0, recovery - duration)
                    if belief < cfg["quality_threshold"]:
                        recovery = 0.0
                        buffer = []
                        sample_time = []
                if (
                    len(buffer) >= keep
                    and belief >= cfg["quality_threshold"]
                    and np.isfinite(estimate_value)
                ):
                    valid_output = True
                    last_estimate = estimate_value
                    age = 0.0
        else:
            belief = bundle["prior"] + (belief - bundle["prior"]) * math.exp(
                -duration / cfg["belief_decay_s"]
            )
            buffer = []
            sample_time = []
        charged = (
            cfg["base_mw"] * duration
            + reads * cfg["readout_mj"]
            + cpu_energy
            + (cfg["switch_mj"] if changed else 0)
            + cfg["packet_mj"]
        )
        if charged > remaining_budget + 1e-08:
            raise ValueError("Actual action cost exceeds its admitted energy allowance")
        remaining_budget -= charged
        cost_total += charged
        optical_count += reads
        now = start + duration
        mask = (t >= start) & (t < now)
        truth = float(np.mean(reference[mask])) if mask.any() else np.nan
        error = abs(estimate_value - truth) if valid_output else np.nan
        if usable is not None:
            true_usable = bool(np.all(usable[mask])) if mask.any() else False
            if (
                true_usable
                and recovery_onset is None
                and (not rows or not rows[-1]["reference_usable"])
            ):
                recovery_onset = start
            if valid_output and recovery_onset is not None:
                recovery_delays.append(now - recovery_onset)
                recovery_onset = None
            if not true_usable:
                recovery_onset = None
        else:
            true_usable = None
        queue.append(
            {"timestamp": now, "hr": last_estimate, "age": age, "valid": valid_output}
        )
        if len(queue) > cfg["queue_limit"]:
            queue.pop(0)
        connected = bool(np.all(connection[mask])) if connection is not None else True
        if connected:
            delivered.extend([now - packet["timestamp"] for packet in queue])
            queue = []
        latency = (time.perf_counter() - timer) * 1000
        runtime.append(latency)
        rows.append(
            {
                "time": now,
                "action": action,
                "belief": belief,
                "tau": tau,
                "age": age,
                "fresh": valid_output,
                "hr": last_estimate,
                "error": float(error) if np.isfinite(error) else None,
                "energy_mj": charged,
                "budget_mj": remaining_budget,
                "service": service,
                "display_available": age <= cfg["stale_display_s"],
                "reference_usable": true_usable,
                "duration_s": duration,
            }
        )
        previous = action
        start = now
    fresh = [r for r in rows if r["fresh"]]
    valid_seconds = sum((r["duration_s"] for r in fresh))
    full_loss = (
        sum(
            (
                (r["error"] if r["fresh"] else cfg["missing_penalty_bpm"])
                * r["duration_s"]
                for r in rows
            )
        )
        / end
    )
    summary = {
        "variant": variant,
        "record": row["record"],
        "mean_power_mw": cost_total / end,
        "readout_fraction": optical_count / (high * end),
        "fresh_availability": valid_seconds / end,
        "mae_bpm": float(np.mean([r["error"] for r in fresh])) if fresh else None,
        "full_timeline_loss": full_loss,
        "recovery_p95_s": (
            float(np.quantile(recovery_delays, 0.95)) if recovery_delays else None
        ),
        "switches_per_minute": switches * 60 / end,
        "controller_estimator_p95_ms": float(np.quantile(runtime, 0.95)),
        "buffer_bytes": int(high * cfg["window_s"] * 16),
        "queue_storage_bytes": cfg["queue_limit"] * 32,
        "remote_delivery_p95_s": (
            float(np.quantile(delivered, 0.95)) if delivered else None
        ),
        "display_unavailable_fraction": float(
            np.mean([not r["display_available"] for r in rows])
        ),
    }
    return (summary, rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("command", choices=["prepare", "fit", "run"])
    p.add_argument("--config", default="config.json")
    p.add_argument("--manifest")
    p.add_argument("--data", default="data/manifest.csv")
    p.add_argument("--output", default="outputs")
    args = p.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    torch.manual_seed(cfg["seed"])
    if args.command == "prepare":
        manifest = pd.read_csv(args.manifest, dtype={"subject": str, "record": str})
        for subject, group in manifest.groupby("subject"):
            if group.split.nunique() > 1:
                raise ValueError(f"Subject {subject} crosses partitions")
        target = Path(args.data)
        target.parent.mkdir(parents=True, exist_ok=True)
        manifest.to_csv(target, index=False)
        return
    manifest = pd.read_csv(args.data, dtype={"subject": str, "record": str})
    if args.command == "fit":
        fit(args, cfg, manifest)
        return
    out = Path(args.output)
    bundle = torch.load(out / "policy.pt", map_location="cpu", weights_only=False)
    summaries = []
    for row in manifest[manifest.split == "test"].to_dict("records"):
        for variant in [
            "full",
            "fixed0",
            "fixed1",
            "fixed2",
            "periodic",
            "greedy",
            "no-reserve",
            "no-age",
            "no-dwell",
        ]:
            summary, ledger = replay(row, cfg, bundle, variant)
            summaries.append(summary)
            (out / f"{row['record']}_{variant}.json").write_text(
                json.dumps(json_ready(ledger), indent=2, allow_nan=False)
            )
    (out / "metrics.json").write_text(
        json.dumps(json_ready(summaries), indent=2, allow_nan=False)
    )
    for row in summaries:
        plt.scatter(
            row["mean_power_mw"], row["fresh_availability"], label=row["variant"]
        )
    plt.xlabel("Modeled power (mW)")
    plt.ylabel("Fresh availability")
    plt.tight_layout()
    plt.savefig(out / "power_service.png")
    plt.close()


if __name__ == "__main__":
    main()
