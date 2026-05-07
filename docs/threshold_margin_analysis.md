# Threshold Margin Analysis

## Problem

The monitor observes average PE output tokens over a sliding window. With short output `S=512` and long output `L=2048`, the natural midpoint threshold is:

```text
mid = (S + L) / 2 = 1280
```

If the observed average is close to `1280`, small changes in request mix or completion timing can move the average slightly above or below the midpoint. A single-threshold policy can therefore alternate between:

```text
avg > 1280  -> flip to long
avg <= 1280 -> flip/re-flip to short
```

That behavior is a threshold-crossing oscillation. It is especially likely when the monitor window contains a near-balanced mixture of 512-token and 2048-token requests.

## Margin As Hysteresis

A margin is effective when it is implemented as two thresholds, also called hysteresis:

```text
range = L - S = 1536
low   = mid - a * range
high  = mid + a * range
```

The monitor policy becomes:

```text
current target is short and avg > high -> flip to long
current target is long  and avg < low  -> re-flip to short
otherwise                              -> keep current target
```

The interval `[low, high]` is a no-action deadband. Once the system flips to long, it will not re-flip merely because the average falls slightly below `mid`; it must fall below `low`. Once the system is short, it will not flip merely because the average rises slightly above `mid`; it must exceed `high`.

## Tested Margins

For `S=512`, `L=2048`, and `mid=1280`:

| margin `a` | low threshold | high threshold | deadband width |
|---:|---:|---:|---:|
| 0.05 | 1203.2 | 1356.8 | 153.6 |
| 0.10 | 1126.4 | 1433.6 | 307.2 |
| 0.15 | 1049.6 | 1510.4 | 460.8 |

## Why This Prevents Flapping

Suppose the true monitor average is near `1280`, and the observed value has noise `e`, so:

```text
observed_avg = 1280 + e
```

With a single threshold, any sign change in `e` can change the decision. With hysteresis margin `a`, the decision changes only when:

```text
e >  a * 1536  for short -> long
e < -a * 1536  for long  -> short
```

For `a=0.10`, the noise must exceed `153.6` tokens in magnitude before the monitor changes target. Small oscillations around `1280` are absorbed by the deadband and cannot trigger repeated flip/re-flip.

## Tradeoff

The margin reduces false flips and repeated flip/re-flip events, but it can delay legitimate transitions:

- a smaller margin such as `0.05` is more responsive but less stable;
- a larger margin such as `0.15` is more stable but can react later;
- `0.10` is a middle point for this trace experiment.

The simulator reports both throughput and SLO deltas for all three values so the stability/latency tradeoff can be inspected from the output tables.
