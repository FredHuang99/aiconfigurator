# TD Flip Simulation Guide

This guide explains how to run the WAN2.2 Group 1 flip/re-flip simulator, how to read its CSV outputs, and how to add new trace/profile/config inputs.

## Quick Start

Run the smoke test:

```powershell
$env:PYTHONPATH='src'
& 'C:\Users\woshi\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' tools\td_flip_simulation.py --scenario smoke --out-dir td_outputs\flip_simulation_smoke
```

Run one real case:

```powershell
$env:PYTHONPATH='src'
& 'C:\Users\woshi\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' tools\td_flip_simulation.py --request-rates-per-min 60 --monitor-windows-sec 30 --out-dir td_outputs\flip_simulation_one
```

Run the default 12 real cases:

```powershell
$env:PYTHONPATH='src'
& 'C:\Users\woshi\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' tools\td_flip_simulation.py --out-dir td_outputs\flip_simulation_default
```

Run margin comparison for `a=0.05`, `0.10`, and `0.15`:

```powershell
$env:PYTHONPATH='src'
& 'C:\Users\woshi\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' tools\td_flip_simulation.py --compare-threshold-margin-ratios 0.05 0.10 0.15 --out-dir td_outputs\weighted_scheduling_complement_throughput
```

Run the non-margin theoretical upper-bound case with zero cold start and oracle PE output estimates:

```powershell
$env:PYTHONPATH='src'
& 'C:\Users\woshi\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' tools\td_flip_simulation.py --threshold-margin-ratio 0.0 --cold-start-mode zero --pe-output-estimate-mode oracle --out-dir td_outputs\weighted_scheduling_oracle_no_coldstart
```

## Arguments

- `--dominant-intervals-csv`: dominant request type intervals. Default is `C:\Users\woshi\Downloads\AzureLMMInferenceTrace_multimodal\data\new\window1_hour1_dominant_intervals_thr98.csv`.
- `--out-dir`: output directory. Default is `td_outputs\flip_simulation`.
- `--request-rates-per-min`: one or more request rates. Default real run is `48 60 90`; smoke default is `12`.
- `--monitor-windows-sec`: one or more PE monitor windows. Default real run is `10 20 30 60`; smoke default is `10`.
- `--duration-min`: simulation arrival duration. Default real run is `60`; smoke default is `6`.
- `--token-threshold`: PE output token threshold used by the monitor. Default is `1280`.
- `--threshold-margin-ratio`: hysteresis margin ratio `a`. Default is `0.0`.
- `--threshold-margin-mode`: `hysteresis` or `single`. Default is `hysteresis`.
- `--compare-threshold-margin-ratios`: run paired non-margin vs margin comparisons for one or more ratios, for example `0.05 0.10 0.15`.
- `--seed`: reserved for deterministic extensions. Current default scheduling is deterministic.
- `--mode`: `no_flip`, `can_flip`, or `both`. Default is `both`.
- `--pe-model`: PE profile name. Default is `pe7b`.
- `--generator-model`: generator profile name. Default is `wan2.2-ti2v-5b`.
- `--input-tokens`: PE input tokens per request. Default is `128`.
- `--short-output-tokens`: output tokens for Short-dominant bins. Default is `512`.
- `--long-output-tokens`: output tokens for Long-dominant bins. Default is `2048`.
- `--denoising-steps`: DiT denoising step count. Default is `50`.
- `--scenario`: `default` or `smoke`. Smoke uses a tiny built-in Short/Long/Short trace.
- `--deployment-preset`: deployment layout preset. Default real preset is `wan22_group1_fragment`; smoke uses `smoke`.
- `--flip-plan-preset`: flip plan label. Default is `wan22_group1_restricted`.
- `--flip-source-count`: number of source P8 bundles to flip. Default real value is `2`; smoke value is `1`.
- `--cold-start-mode`: `profile` uses PE/generator init-time profiles; `zero` makes launch complete immediately. Default is `profile`.
- `--pe-output-estimate-mode`: `monitor` estimates PE output length from the PE instance monitor; `oracle` uses each request's known output tokens for dispatch and flip-source work estimates. Default is `monitor`.
- `--trace-start`: timestamp corresponding to elapsed time `0`. Default is `2024-10-15T12:00:00+00:00`.
- `--verbose`: reserved for debug logging.

## Threshold Margin

The default threshold is the midpoint between the short and long output lengths:

```text
mid = (512 + 2048) / 2 = 1280
```

With hysteresis margin `a`, the simulator uses two thresholds:

```text
low  = mid - a * (2048 - 512)
high = mid + a * (2048 - 512)
```

The monitor flips from short to long only when `avg_output_tokens > high`, and flips from long to short only when `avg_output_tokens < low`. If the average stays between `low` and `high`, the simulator keeps the current target config. This deadband is what prevents repeated flip/re-flip when the monitor window average jitters near 1280.

The comparison command writes one directory per margin value under `--out-dir`:

- `margin_0p05`
- `margin_0p10`
- `margin_0p15`

Each directory contains detailed CSVs plus a `summary.md` comparing non-margin can-flip against that margin value.
The non-margin baseline is computed once per comparison command and reused in every margin directory.

## Scheduling

New arrivals and migrated requests are scheduled one request at a time. For each request, the
server chooses an active ready instance with the smallest estimated work in seconds, then updates
that instance's estimated work before assigning the next request.

The estimated work is:

```text
running_remaining_time + sum(waiting_request_service_time)
```

This directly represents remaining execution time instead of a normalized request count. H100
DiT/VAE can therefore be preferred over A800/A100 when the same number of requests is queued,
because the service-time estimate is shorter.

PE dispatch estimates do not use the simulator's true output length. Each PE completion sample is
tagged with an instance id. For each PE instance, the simulator looks at the current monitor window,
computes average output tokens, and classifies the instance into an output-token bin. The default
bins are `512` and `2048`, split by midpoint `1280`; this per-instance classification does not use
margin or hysteresis. If an instance has no samples, the estimator falls back to the global PE
window, then to the current target config. The implementation maintains this monitor as an
incremental sliding window, so long comparison runs do not repeatedly rescan all PE completions.
PE waiting and incoming work use
`TTFT(128, bin) + (bin - 1) * TPOT(128, bin)`. PE running work uses the same bin, upgrades to a
larger bin if generated tokens already exceed the current bin, and counts only remaining TPOT after
the first token has been generated.

For theoretical upper-bound runs, `--pe-output-estimate-mode oracle` replaces only the PE work
estimate with each request's known output tokens. The actual PE execution still uses the same
profile tables and request output tokens as normal simulation. `--cold-start-mode zero` removes
only the launch delay after flip drain; boundary drain and migration semantics remain unchanged.

Flip source selection is staged greedy. The default Group 1 restricted plan is still
`2x DiT_VAE_Comb_A800_P8 -> 8x PE_Only_A800_P2`, so the source candidates are still A800 P8
bundles. The simulator first ranks candidates by restart-ready delay. Only if the cutoff is tied
does it compute complement-added work: PE remaining/waiting tokens divided by complement PE
token/s, DiT work units divided by complement DiT req/s, and VAE work units divided by complement
VAE req/s. For comb bundles, DiT and VAE loads are computed separately and combined with `max`
because the two complement pools absorb work concurrently. Only/comb preference and round-robin are
used only as final tie-breakers.

## Output Files

`run_summary.csv`

One row per `(request_rate, monitor_window, mode)`. Key columns:

- `throughput_req_s`: finished requests divided by `last_finish_s - first_arrival_s`.
- `throughput_vs_no_flip_pct`: can-flip throughput lift compared with no-flip for the same rate/window.
- `slo10_pass_ratio`, `slo5_pass_ratio`: fraction of requests whose end-to-end latency is below weighted baseline times 10 or 5.
- `slo10_vs_no_flip_delta`, `slo5_vs_no_flip_delta`: direct pass-ratio improvement, not percentage.
- `num_flip_events`: number of flip/re-flip detections.
- `margin_ratio`, `margin_label`: identify whether the row is non-margin or a margin run.
- `threshold_mid`, `threshold_low`, `threshold_high`: monitor thresholds used by this run.
- `cold_start_mode`, `pe_output_estimate_mode`: identify whether the run used profile/zero cold start and monitor/oracle PE work estimates.

`request_stage_events.csv`

One row per request-stage attempt. A stage may have multiple attempts if it was migrated.

- `enter_time_s`, `start_time_s`, `exit_time_s`: when the request entered the queue, started serving, and left this stage attempt.
- `queue_time_s`: `start_time_s - enter_time_s`.
- `exit_reason`: `completed`, `migrated_waiting`, or `migrated_running`.
- `instance_id`, `template_name`, `node_name`, `gpu_type`, `parallelism`: where the attempt ran or waited.
- `pe_actual_prefill_tokens`, `pe_actual_remaining_output_tokens`: actual re-prefill lookup target after PE migration.
- `pe_matched_prefill_tokens`, `pe_matched_remaining_output_tokens`: nearest profile key used for TTFT/TPOT.
- `slo5_met`, `slo10_met`: request-level SLO result repeated on each attempt row.
- `margin_ratio`, `margin_label`: identify which threshold margin produced this request-stage attempt.

`flip_events.csv`

One row per flip or re-flip.

- `direction`: `short_to_long` or `long_to_short`.
- `margin_ratio`, `margin_label`: identify the threshold margin used by the monitor.
- `threshold_mid`, `threshold_low`, `threshold_high`: thresholds used for that detection.
- `monitor_decision`: the monitor decision that produced the flip event.
- `detect_time_s`: monitor tick time that detected the threshold crossing.
- `trace_change_time_s`: most recent trace dominant-type change that matches this direction.
- `detection_delay_s`: `detect_time_s - trace_change_time_s`.
- `selected_bundle_keys`, `selected_instance_ids`: resources selected for flip.
- `selection_restart_ready_delay_s`: selected sources' maximum boundary/finish delay plus cold start.
- `selection_complement_added_work_s`: estimated extra time added to the complement resource pool.
- `selection_running_added_work_s`, `selection_waiting_added_work_s`: running/waiting portions of that estimate.
- `selection_pe_bin_tokens`: PE bin classification used for PE-source scoring.
- `selection_migrated_request_count`: number of waiting/running requests counted by source selection.
- `selection_template_penalty`: tie-breaker cost; comb sources have a larger penalty than only sources.
- `selection_round_robin_rank`: deterministic final tie-breaker rank.
- `migrated_waiting_requests`, `migrated_running_requests`: migration counts.
- `drain_done_time_s`: when selected running work reached the required boundary.
- `cold_start_time_s`, `cold_start_done_time_s`: cold-start start and ready times.

`dominant_timeline.csv`

The trace intervals used by the simulator after Tie resolution. Use this file to compare request type changes with `flip_events.csv`.

`summary.md`

Compact table for the 12 default cases, focused on throughput and SLO improvements.

For margin comparison directories, `summary.md` has one table per request rate. Each row is a monitor window. For each metric, the `non-margin` column is can-flip with `a=0`, and the `margin` column is can-flip with that directory's margin value. The three metric groups are:

- throughput lift percentage relative to no-flip;
- SLOx10 direct pass-ratio delta relative to no-flip;
- SLOx5 direct pass-ratio delta relative to no-flip.

## Adding New Inputs

New dominant-interval trace:

1. Create a CSV with `start_minute`, `end_minute`, `dominant`, and optional `start_time`/`end_time`.
2. Use `dominant` values `Short`, `Long`, or `Tie`.
3. Pass it with `--dominant-intervals-csv`.

New raw trace-derived interval file:

1. Convert raw timestamps into per-minute bins.
2. Classify requests by output token threshold.
3. Compute the dominant type per bin.
4. Collapse adjacent bins with the same dominant type.
5. Write the same schema as `dominant_timeline.csv`.

New PE profile tables:

1. Add regular TTFT/TPOT tables to `profile_registry.py` if they are used for normal PE latency.
2. Add simulator TTFT/TPOT tables as `sim_ttft_ms` and `sim_tpot_ms` for re-prefill.
3. Add `init_time_s` for each PE parallelism that can cold start.

New generator profile:

1. Add encoder CPU latency, DiT latency, VAE latency, memory, and `init_time_s` to `profile_registry.py`.
2. Add aliases in `profiles.py` if the model needs short names.
3. Use `--generator-model` to select it.

New deployment config or flip plan:

1. Add a deployment builder in `flip_simulation.py` that creates the PE, Encoder, DiT, and VAE instances.
2. Add a preset name handled by `--deployment-preset`.
3. Add or update flip source selection so the new `--flip-plan-preset` maps to the intended source and target templates.

## Troubleshooting

- Missing profile key: check the actual and matched PE lookup columns in `request_stage_events.csv`, then add the missing profile entry or verify nearest-key matching is acceptable.
- No active migration targets: the flip plan selected too many sources or the deployment has no remaining active instances for that stage.
- No PE completions in a monitor window: the monitor skips that window; increase window size or inspect PE queueing in `request_stage_events.csv`.
- Long run time: first run a single `(rate, window)` pair, then run the full 12-case sweep once the output shape looks right.
