# ContactSample

Implementation of contact-aware adaptive sampling with funded recovery.

```bash
python -m pip install -e .
python -m contactsample prepare --manifest recordings.csv
python -m contactsample fit
python -m contactsample run
```

Use [PPG-DaLiA](https://archive.ics.uci.edu/dataset/495/ppg+dalia) or [BIDMC](https://physionet.org/content/bidmc/1.0.0/) recording exports. A waveform CSV contains `time,ppg,hr_reference` and optional `ax,ay,az,usable_reference,connected`. Times are seconds at the recorded rate; split real acquisition gaps into separate records. The manifest has `subject,record,file,sample_rate,split`; preserve subject IDs 1–9/10–12/13–15 for the source train/development/test recipe. BIDMC patient linkage must be verified separately. The quality target uses the training reference only; online decisions never read reference pulse rate or `usable_reference`.

`fit` learns a logistic usability predictor from acquired spectral/autocorrelation features and a two-layer width-32 action-loss predictor. It fits training normalization and keeps all reference-derived targets out of replay state. The spectral/autocorrelation estimator uses a causal fourth-order 0.5–4-Hz filter, an 8-second acquired window and the 30–210-bpm grid. State carries across compatible rate epochs; sleep, probes and incompatible rate changes invalidate history. A probe updates quality age but cannot emit a fresh pulse estimate.

Replay tracks belief, time since optical check, output age, energy balance, valid sample history, action dwell and promised recovery bursts. Mandatory probes, active recovery and energy feasibility precede discretionary action selection. The reserve includes remaining checks, promised recoveries, baseline and packet costs. A successful probe starts a funded eight-second acquisition rather than declaring immediate recovery. Unfunded service is marked degraded/unavailable. Freshness, display status and remote delivery remain separate.

`config.json` controls rates, epoch/window/probe durations, quality threshold, energy costs, 60-second blocks, probe gap, two reserved recoveries, dwell and the belief-decay prior. Use 16/32/64 Hz for PPG-DaLiA and 31.25/62.5/125 Hz for BIDMC. `acquisition_tier` is `sensor` or `digital`: the sensor tier simulates a declared 4-Hz analog front-end bandwidth before revealing selected samples; the digital tier charges all source readouts during active epochs. Modeled mJ costs are distinct from measured hardware energy. Refit/recalibrate for a new front-end or source before interpreting energy/service trade-offs.

Controls run the same estimator and waveform boundary: fixed rates, periodic duty, quality-greedy and no-reserve/no-age/no-dwell variants. Outputs include the fitted policy, per-epoch ledgers, summary metrics and `power_service.png`. Metrics report modeled power, actual readout fraction, fresh availability, accepted-output MAE, full-timeline missing-output loss, recovery P95 where independently annotated usability transitions exist, switching, replay execution time, sample/queue storage and remote delivery delay. The storage figures account for declared buffers rather than claiming measured total-device peak RAM. Enable `phone_connection_column` to replay the `connected` trace; local decisions do not depend on it. Optical power has a whole-timeline denominator, including sleep and rejected windows.
