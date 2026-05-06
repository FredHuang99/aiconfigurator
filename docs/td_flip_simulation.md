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

## Arguments

- `--dominant-intervals-csv`: dominant request type intervals. Default is `C:\Users\woshi\Downloads\AzureLMMInferenceTrace_multimodal\data\new\window1_hour1_dominant_intervals_thr98.csv`.
- `--out-dir`: output directory. Default is `td_outputs\flip_simulation`.
- `--request-rates-per-min`: one or more request rates. Default real run is `48 60 90`; smoke default is `12`.
- `--monitor-windows-sec`: one or more PE monitor windows. Default real run is `10 20 30 60`; smoke default is `10`.
- `--duration-min`: simulation arrival duration. Default real run is `60`; smoke default is `6`.
- `--token-threshold`: PE output token threshold used by the monitor. Default is `1280`.
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
- `--trace-start`: timestamp corresponding to elapsed time `0`. Default is `2024-10-15T12:00:00+00:00`.
- `--verbose`: reserved for debug logging.

## Output Files

`run_summary.csv`

One row per `(request_rate, monitor_window, mode)`. Key columns:

- `throughput_req_s`: finished requests divided by `last_finish_s - first_arrival_s`.
- `throughput_vs_no_flip_pct`: can-flip throughput lift compared with no-flip for the same rate/window.
- `slo10_pass_ratio`, `slo5_pass_ratio`: fraction of requests whose end-to-end latency is below weighted baseline times 10 or 5.
- `slo10_vs_no_flip_delta`, `slo5_vs_no_flip_delta`: direct pass-ratio improvement, not percentage.
- `num_flip_events`: number of flip/re-flip detections.

`request_stage_events.csv`

One row per request-stage attempt. A stage may have multiple attempts if it was migrated.

- `enter_time_s`, `start_time_s`, `exit_time_s`: when the request entered the queue, started serving, and left this stage attempt.
- `queue_time_s`: `start_time_s - enter_time_s`.
- `exit_reason`: `completed`, `migrated_waiting`, or `migrated_running`.
- `instance_id`, `template_name`, `node_name`, `gpu_type`, `parallelism`: where the attempt ran or waited.
- `pe_actual_prefill_tokens`, `pe_actual_remaining_output_tokens`: actual re-prefill lookup target after PE migration.
- `pe_matched_prefill_tokens`, `pe_matched_remaining_output_tokens`: nearest profile key used for TTFT/TPOT.
- `slo5_met`, `slo10_met`: request-level SLO result repeated on each attempt row.

`flip_events.csv`

One row per flip or re-flip.

- `direction`: `short_to_long` or `long_to_short`.
- `detect_time_s`: monitor tick time that detected the threshold crossing.
- `trace_change_time_s`: most recent trace dominant-type change that matches this direction.
- `detection_delay_s`: `detect_time_s - trace_change_time_s`.
- `selected_bundle_keys`, `selected_instance_ids`: resources selected for flip.
- `migrated_waiting_requests`, `migrated_running_requests`: migration counts.
- `drain_done_time_s`: when selected running work reached the required boundary.
- `cold_start_time_s`, `cold_start_done_time_s`: cold-start start and ready times.

`dominant_timeline.csv`

The trace intervals used by the simulator after Tie resolution. Use this file to compare request type changes with `flip_events.csv`.

`summary.md`

Compact table for the 12 default cases, focused on throughput and SLO improvements.

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
