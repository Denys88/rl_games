# Wuji empirical audit — 2026-09-16

Read-only audit of 17 local TensorBoard runs. No training was started, stopped, or modified. No AGENTS.md found in the Wuji repository or ancestors. All table values below were independently recomputed from raw TFRecord scalar events; campaign notes are supporting context, not proof of causes.

## What the evidence establishes

The best clean single rl_games seed has only a 3.9% final goal-reach advantage over the one RSL reference (17.557 vs 16.892). The identical lowered-LR-floor recipe fails on seed 42. Reward-floor arms score well while their raw-action smoothness becomes severely worse. A large, reliable win has not been demonstrated.

| Run | Recipe / seed | Steps (B) | Reaches final / peak (EMA50) | Terminal storm iteration / hours | Final action delta / raw penalty magnitude | Final sigma geometric mean |
|---|---|---:|---|---|---|---:|
| arbiter | RSL fixed 1e-4; seed unverified in archived logs | 1.638 | 16.892 / 16.906 | none | 0.388 / 9.1 | 0.2002 |
| A | bignet LR [1e-4,2e-4], seed42 | 1.384 | 3.612 / 16.691 | 3749 / 1.666 | 22.791 / 98,870.1 | 0.3892 |
| B | A, seed7 | 1.638 | 0.003 / 0.290 | 1044 / 0.448 | 659.325 / 66,135,912.0 | 0.4071 |
| C | A + reward floor -5, seed42 | 1.638 | 16.979 / 17.215 | 2967 / 1.345 | 5.075 / 198,694.6 | 0.2128 |
| D | C on 2 GPUs; twice data/iteration | 3.277 | 19.511 / 19.567 | 1918 / 0.912 | 15.992 / 1,142,660.4 | 0.2215 |
| E | bignet LR [5e-5,2e-4], seed42 | 1.638 | 12.927 / 16.541 | 4909 / 2.170 | 0.609 / 184.0 | 0.2340 |
| F | E, seed7 | 1.638 | 17.557 / 17.567 | none | 0.358 / 8.6 | 0.2001 |

Storm definition: start of the final uninterrupted interval in which action-delta RMS EMA20 exceeds 0.45. This is an operational threshold, not a causal change point. Initial exploration exceeds this threshold in all runs (RSL iterations 0–46, rl_games roughly 1–50); reports saying the reference “never leaves 0.39” omit this expected initial interval. Post-initial transient excursions: A401–441, B151–155, D1899 and1903–1904, E684–730 and2613–2733, F614–797. A local log stops at4225; the report says killed4264, so local data do not cover the last39 iterations. RSL indexes 0–4999; rl_games indexes1–5000.

## First visible changes: update instability precedes the largest action penalties

**B offers the sharpest event.** With LR pinned at1e-4 and KL target0.01, KL is0.0285 at1038,0.0942 at1039,3.1603 at1040,1.8316 at1041,6.9366 at1042,39.7965 at1043. Action delta remains0.255–0.257 through1041; it rises to0.332 at1042,0.459 at1043,2.319 at1044. Action-rate penalties are only−4.68,−5.35,−6.77,−7.33 through1038–1041, then−50.1,−136,−1631. Central value loss rises0.003→0.010→0.169 before the largest penalty jump. These are once-per-iteration averages, so within-iteration ordering remains unknown. A replay buffer captured around1038 is a better first-kick diagnostic than a post-collapse checkpoint.

**A.** At3660 KL0.073/LR1e-4, action delta0.357, action rate−9.58, entropy−3.778 (sigma geometric mean0.2003). At3740: KL0.0512, rate−35.2, central value loss0.0693, sigma0.2019. At3750: KL0.0811, rate−160, value loss0.5323, sigma0.2126. At3840 sigma0.287 and action rate−1403; at4150 rate−54,237. The late LR increase occurs after destabilization. Success and adaptive-episode curricula are essentially1 across this onset.

**E.** Entropy drifts while actions initially remain smooth: sigma geometric mean0.2017 at4600,0.2045 at4850,0.2106 at4900,0.2137 at4910,0.2340 at5000. The earlier note’s “0.20→0.22 by4850” overstates this inferred shift. KL around4850–4910 is0.023–0.030; action rate worsens−12.7→−46→−102.3, while both curricula remain1. The final storm is4909.

**C/D.** Storm thresholds occur with KL0.0148/0.0124 and LR2e-4, near geometric-mean sigma0.203/0.202. Their large goal-reach counts cannot establish policy quality because the reward floor saturates the cost of large raw action differences.

Entropy only identifies `exp(H/20 − 0.5*log(2*pi*e))`, the geometric mean of sigma across dimensions/samples. It cannot rule out rare huge sigmas. Learner bounds loss (mean sum(mu²)) remains around4–8 in A even while rollout action deltas become tens. This warrants separate rollout-vs-update mu/std and normalization telemetry, not a conclusion that sigma or mean alone caused the storm.

## Earlier screens: useful rejection evidence, not long-run stability proof

| Arm | Reaches at2500 (EMA50) | Interpretation |
|---|---:|---|
| w0_control | 11.088 | current small network control |
| w2_bignet | 12.081 | wider actor+critic; selected for follow-up |
| r2a_bignet_fp32 | 10.620 | FP32; no terminal storm by2500, untested at5000 |
| r2b_bignet_nowarm | 0.004 | normalizer warm start off; terminal storm620 |
| r2c_bignet_geo16k | 0.002 | 16K envs +32K minibatch; terminal storm1044; twice data |
| r2d_bignet_lr3e4 | 11.034 | higher LR cap; no gain |
| w3_compile | 3.708 | 7-reach peak then3.7; no persistent delta storm at endpoint |
| w4_cvep2 | 8.356 | 2 central-value epochs; weaker at2500 |
| w5_ent0 | 10.917 | no entropy bonus; little score change |
| w1_std016 | 0.017 | per-epoch scheduler +KL .016; stopped1573, not single-variable vs control |

## Comparison and causal confounds

- 8192 envs×40 steps×5000=1.6384B steps. DDP arm D doubles envs and data per iteration (3.2768B total), so its19.5 score is not a matched-sample win. Report data and GPU-hours separately from wall time.
- September single-GPU runs used one GPU of a2×RTX PRO6000 workstation; July baseline used RTX4090 and older code. July-versus-September timing is not an algorithm speed comparison. Screens ran two jobs per GPU, unlike long runs.
- RSL reference uses512/256/128 actor,512/512/256/128 critic,32 minibatches/epoch, fixed1e-4, no value normalization or clipped value loss. Bignet uses larger networks,20 minibatches/epoch, adaptive LR, value normalization, central value clipping, bf16, and bounds regularization. Therefore this campaign compares recipes, not isolated implementations. RSL seed and exact launched config are not archived alongside the copied TensorBoard file.
- A single clean seed F and one clean reference do not estimate stability or justify “necessary” fixes. Lowering LR floor helped seed7 but did not prevent seed42 failure. A failed fixed-LR arm would rule out adaptive scheduling as a necessary cause, not exonerate update size.
- Scalar-only logs cannot establish the hypothesized negative-advantage/sigma feedback mechanism. With normalized advantages, claims about a batch being dominated by negative advantages need tail/count/gradient measurements.

## Most informative next experiment

1. Obtain final4_fixedlr and final4_novalnorm logs/configs before duplicating those interventions; they were reported running since16:32 and are absent from all local directories searched. The queued final4_ddp_lr5e5_seed7 is likewise unavailable.
2. Add a first-kick diagnostic around seed7 iterations1000–1060 (and late seed42 window3600–3800): preserve rollout tensors, policy/value/optimizer/RMS state and RNG state before each update. A normal checkpoint lacks live simulator state; a resumed run is not an identical trajectory. Replay the same cached batch to localize a destabilizing minibatch.
3. Record rollout and learner mu/std quantiles+max separately, pre/post-update KL and ratio clip fraction per minibatch, per-head preclip gradient norms, parameter-step norms, value/return/advantage quantiles and explained variance, value RMS mean/std/count, observation RMS change including previous-action features, and raw/clipped/filtered action saturation fractions. Existing logs have none of these distributions; only aggregate iteration averages.
4. Run a matched RSL-shaped rl_games recipe (network widths, fixedLR1e-4,32 minibatches, raw/unclipped value loss,FP32) against the same RSL seeds42,7 and a held-out seed, unchanged task/reward and327680 steps/update. Introduce one confirmed code correction at a time; separate implementation parity from wider-network optimization. Carry every stability arm past5000—E fails only at4909—and evaluate held-out rollouts with full curriculum and fixed episode budgets.
5. Optimize throughput/architecture only after stability holds. Judge clean time-to-target, matched-frame learning curve, failure rate and held-out reaches/drop/smoothness metrics. A large margin requires multiple seeds and equal compute/sample accounting, not reward flooring.

## Artifacts and availability

- Raw log root: `/tmp/claude-1000/-home-viktor-Projects-Research-rl-games/170a2599-6ecd-48cd-b797-4c2724627508/scratchpad/wuji/tb` (17 copied runs).
- Metrics JSON: `/tmp/wuji_empirical_metrics.json`, containing source-note metrics, independently parsed values, all delta excursions, exact onset timestamps, and dense B first-kick samples.
- Verified standalone plot: `/tmp/wuji_empirical_comparison.png` (RSL,F,A,E; goal reaches and raw action-rate cost). Existing full campaign plot copied to `/tmp/wuji_all_runs_existing.png`.
- Reproducible read-only parsing/plot script: `/tmp/wuji_empirical_audit.py`.
- Local nvidia-smi cannot reach the driver and sandbox ps sees its own namespace. This does not establish remote-machine/GPU availability. No SSH config exists here. No remote host or run root was discoverable from the task-local script files.
