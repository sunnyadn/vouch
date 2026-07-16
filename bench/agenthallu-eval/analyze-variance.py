#!/usr/bin/env python3
# Analyze a high-reps adapter.ts --clean run to quantify the PRECISION variance the publish-readiness
# doc flagged but never measured ("same-input-repeated-N stability rate").
#
# Input: the stdout of `adapter.ts --clean --limit 30 --reps N` (per-trajectory "block Y/N" lines).
# Output:
#   1. Per-trajectory block-fire rate (k/N) → stability classes (stable-silent / stable-FP / UNSTABLE).
#   2. Bootstrap of the reps=3 block-precision estimate: for each trajectory with p=k/N, resample a
#      reps=3 block-majority ~ Binomial(3,p)>=2, compute precision per simulated run → percentile CI.
#      This turns one high-reps run into a distribution of the reps=3 number, isolating run-to-run
#      variance from the per-trajectory firing probabilities.
import re, sys, random

path = sys.argv[1] if len(sys.argv) > 1 else "/dev/stdin"
text = open(path).read()

# Per-trajectory line: "... (fired X/N, block Y/N ...)" — capture block Y and the denom N.
rows = [(int(b), int(n)) for b, n in re.findall(r"block (\d+)/(\d+)", text)]
if not rows:
    print("no 'block Y/N' lines found — is this an adapter --clean run output?")
    sys.exit(1)
N = max(n for _, n in rows)
rows = [(b, n) for b, n in rows if n == N]  # drop any partial-rep lines
T = len(rows)
print(f"trajectories: {T}  | reps per trajectory: {N}\n")

stable_silent = [r for r in rows if r[0] <= 1]            # ~never fires
stable_fp     = [r for r in rows if r[0] >= N - 1]        # ~always fires (consistent FP)
unstable      = [r for r in rows if 1 < r[0] < N - 1]     # flips → the variance source
print("── per-trajectory block-fire stability ──")
print(f"  stable-silent (0-1/{N}):  {len(stable_silent)}/{T}")
print(f"  stable-FP   ({N-1}-{N}/{N}):  {len(stable_fp)}/{T}")
print(f"  UNSTABLE    (2..{N-2}/{N}):  {len(unstable)}/{T}   ← these flip run-to-run = the precision variance")
if unstable:
    print(f"    unstable block-rates: {sorted(b for b, _ in unstable)} (out of {N})")

# Point estimate at the high-reps majority threshold (block-majority over N):
hr_fp = sum(1 for b, _ in rows if b * 2 > N)
print(f"\n  high-reps ({N}) majority block-precision: {T - hr_fp}/{T} = {100*(T-hr_fp)/T:.0f}%  ({hr_fp} FPs)")

# Bootstrap the reps=3 estimate (what the deployed config / prior runs report).
random.seed(0)  # deterministic (Math.random is unavailable to the JS workflow; fix seed for repro)
def sim_reps3_precision():
    fp = 0
    for b, _ in rows:
        p = b / N
        fires = sum(1 for _ in range(3) if random.random() < p)
        if fires >= 2:  # block-majority of 3
            fp += 1
    return (T - fp) / T
B = 5000
sims = sorted(sim_reps3_precision() for _ in range(B))
mean = sum(sims) / B
lo, hi = sims[int(0.025 * B)], sims[int(0.975 * B)]
print(f"\n── reps=3 block-precision, bootstrapped from the {N}-rep firing rates ({B} sims) ──")
print(f"  mean: {mean*100:.0f}%   95% interval: [{lo*100:.0f}%, {hi*100:.0f}%]")
print(f"  (prior single reps=3 runs observed: 77% and 83% — check they fall in this interval)")
