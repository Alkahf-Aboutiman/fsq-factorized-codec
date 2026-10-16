# Detailed results

LibriSpeech test-clean, 500 utterances (at most 12 s), Whisper-small, text normalized (lowercase, letters and apostrophes),
95% bootstrap CIs over utterances (2000 resamples). Frame rate 80 Hz, so bitrate = 80 x bits per frame.

## 1. WER versus bitrate

| Bitrate | RVQ (truncated levels) | FSQ (distilled) |
|---|---|---|
| 0.8 kbps | n/a | 72.5% [58.2, 90.0] (config D) |
| 1.2 kbps | n/a | 13.8% [8.1, 24.6] (config C) |
| 1.6 kbps | 9.5% [5.9, 16.2] | 8.1% [7.2, 9.1] (config B) |
| 2.4 kbps | 5.2% [4.5, 5.8] | 5.5% [4.7, 6.2] (config A, no residual) |
| 4.8 kbps | 4.7% [4.0, 5.3] | 6.5% [5.6, 7.3] (config A, with residual) |
| original audio | 4.2% [3.6, 4.9] | |

Paired differences (FSQ minus RVQ, same utterances): 2.4 kbps +0.31 pt [-0.13, +0.74]; 1.6 kbps -1.25 pt [-7.75, +2.08].
At 4.8 kbps, adding the FSQ residual raises WER by about 1.0 pt relative to FSQ without residual (6.5% vs 5.5%), and FSQ is 1.8 pt above the RVQ at the same rate (6.5% vs 4.7%). These are point estimates; see the intervals above.

Errors are heavy-tailed. At 1.6 kbps the per-sentence median WER is 0.0% for RVQ and 5.3% for FSQ, and 1.4% vs 1.8% of sentences
have more than 50% errors, so we read 1.6 kbps as "comparable, no demonstrated advantage".
There is no RVQ reference below 1.6 kbps.

## 2. Fidelity and speaker similarity

Log-mel L1 distance to the original (lower is better) and cosine similarity of WavLM-SV speaker embeddings (higher is better).
Reference similarities from original audio: same speaker, different utterance 0.934; different speakers 0.704.
Absolute similarity values depend on the extractor; only comparisons between systems are meaningful.

| Bitrate | System | log-mel L1 | speaker sim |
|---|---|---|---|
| 0.8 | FSQ D | 1.656 | 0.749 |
| 1.2 | FSQ C | 1.324 | 0.893 |
| 1.6 | RVQ | 1.226 | 0.926 |
| 1.6 | FSQ B | 1.198 | 0.904 |
| 2.4 | RVQ | 0.956 | 0.967 |
| 2.4 | FSQ A (no residual) | 1.090 | 0.955 |
| 4.8 | RVQ | 0.726 | 0.981 |
| 4.8 | FSQ A (with residual) | 1.365 | 0.935 |

Paired differences, FSQ minus RVQ at matched bitrate (95% CI over utterances):

| Bitrate | log-mel L1 | speaker sim |
|---|---|---|
| 1.6 kbps | -0.028 [-0.035, -0.021] | -0.022 [-0.026, -0.017] |
| 2.4 kbps | +0.133 [+0.127, +0.139] | -0.012 [-0.013, -0.010] |
| 4.8 kbps | +0.640 [+0.629, +0.649] | -0.046 [-0.048, -0.043] |

Config A is worse with its residual (mel 1.365) than without it (mel 1.090): the distilled residual factor degrades the output.

## 3. Environment factor (reverberation and noise)

Setup: an encoder `E` maps the FACodec latents of a degraded utterance to a vector `e`; a renderer `R` maps the clean latents of another
utterance plus `e` to the latents of the degraded version. `E` sees a different utterance from the same room, so `e` can only carry
environment information. Rooms: 320 train / 80 test simulated shoebox RIRs (pyroomacoustics, RT60 0.2 to 1.0 s); all test rooms are unseen.

### (a) E recovers RT60 of unseen rooms (160 utterances, 4 s crops; chance RMSE 0.215 s)

| Scenario | RMSE | correlation | slope |
|---|---|---|---|
| reverb only | 0.127 s | 0.87 | 0.74 |
| noise 10-30 dB + reverb | 0.118 s | 0.89 | 0.78 |
| noise 0-20 dB + reverb (training-like) | 0.139 s | 0.80 | 0.66 |

Predictions regress toward the mean (short rooms are overestimated). `e` does not identify the room itself (top-1 retrieval 0% over 80 rooms);
the retrieved room is within 0.1 s of the true RT60 in 54% of cases, which matches the supervision (RT60 only).

### (b) R responds to e, monotonically

Log-mel distance to the true reverberated audio, relative to using the `e` of the same room (reverb only, 300 utterances; negative = closer to the target):

| `e` taken from a room that is... | change in distance | share of cases better than same room |
|---|---|---|
| much shorter (> 0.3 s shorter) | -0.163 [-0.226, -0.102] | 62% |
| shorter (0.1 to 0.3 s) | -0.137 [-0.186, -0.088] | 70% |
| about equal (< 0.1 s) | +0.016 [-0.025, +0.056] | 45% |
| longer (0.1 to 0.3 s) | +0.270 [+0.215, +0.329] | 21% |
| much longer (> 0.3 s) | +0.572 [+0.496, +0.641] | 5% |

A longer `e` makes the rendering worse, but a somewhat shorter `e` makes it better than the true one: the renderer seems to over-reverberate
(hypothesis, not tested). Rendering with a short-room `e` vs a long-room `e` differs by 1.608 log-mel units, comparable to their distance
to the clean codec output (1.971 and 3.212).

### (c) Rendering costs intelligibility (100 utterances, SNR 10-30 dB; codec on clean input: 5.1% WER)

| Scenario | WER true degraded | WER rendered | rendered minus clean codec | wrong-room minus right-room |
|---|---|---|---|---|
| reverb only | 9.3% | 17.5% | +12.4 pt [+9.5, +15.5] | +0.5 [-2.6, +4.0] |
| noise only | 5.1% | 8.4% | +3.4 pt [+1.9, +4.9] | +0.4 [-0.1, +0.9] |
| both | 13.0% | 18.9% | +13.9 pt [+10.9, +17.0] | +0.5 [-2.4, +3.4] |

The criterion "swapping the factor changes acoustics without degrading WER" is not met: the rendered speech is less intelligible than both the
clean codec output and the true degraded audio, and WER does not depend measurably on which room's `e` is used.
An earlier evaluation with harsher degradations (SNR 0-20 dB) gave 119.7% WER even on the true degraded audio, which makes WER uninformative
there; we moved to SNR 10-30 dB. That earlier script is not part of this repository.
