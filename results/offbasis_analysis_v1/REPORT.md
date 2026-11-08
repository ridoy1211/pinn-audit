# Off-basis robustness evaluation

240 completed fits; 120 datasets; 30 independent paired blocks per observation count.
All comparisons are secondary. Intervals are 95% percentile whole-block bootstrap intervals (10,000 resamples), pointwise rather than simultaneous.

## Median absolute errors (%)

| N | Timing | Outcome | CV12 | PINN |
|---|---|---|---:|---:|
| 10 | mixed | diffusivity | 74.6244 | 73.0752 |
| 10 | mixed | initial_field | 74.0456 | 70.4831 |
| 10 | mixed | full_field | 41.8301 | 43.9698 |
| 10 | late | diffusivity | 99.9800 | 67.5259 |
| 10 | late | initial_field | 82.7573 | 79.7393 |
| 10 | late | full_field | 42.8038 | 43.8732 |
| 50 | mixed | diffusivity | 4.4207 | 3.3157 |
| 50 | mixed | initial_field | 30.1281 | 24.8040 |
| 50 | mixed | full_field | 6.9364 | 6.1261 |
| 50 | late | diffusivity | 7.9483 | 7.6752 |
| 50 | late | initial_field | 65.1001 | 58.7115 |
| 50 | late | full_field | 22.2002 | 18.7599 |

## Paired diffusivity contrasts

D = PINN error minus CV12 error. Negative values favour the PINN.

| N | Timing | Median D (pp) | 95% interval | PINN lower error |
|---|---|---:|---|---:|
| 10 | mixed | -0.0234 | [-1.4383, 7.0112] | 18/30 |
| 10 | late | -0.2447 | [-18.7921, -0.0045] | 27/30 |
| 50 | mixed | -0.2792 | [-1.2477, 0.7682] | 16/30 |
| 50 | late | -0.8896 | [-1.5833, 0.2963] | 19/30 |

## Timing interaction

I = D(mixed) - D(late), computed within blocks before taking the median.

N=10: 16.4986 pp; interval [-2.3494, 38.3046].
N=50: 0.5421 pp; interval [-0.8181, 2.2548].

## Limits

This is one localized initial field and two observation schedules. It does not establish universal robustness.
The twelve-mode initial projection error is about 17.86%; initial-field errors must be interpreted with that representation restriction.
The new field has a different initial L2 energy from A/B despite fixed absolute noise. Changes across truth families cannot be attributed solely to basis mismatch.
Do not infer paired effect sizes from differences of marginal medians. This study does not replace the original primary timing test.
Inspect diagnostics.csv before final interpretation. Reported bound flags use the existing tolerances; zero flags do not exclude near-bound estimates.
No settings have been selected using these evaluation results.
