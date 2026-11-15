# M2 WP6 Acoustic A/B Objective Evaluation Report

- **Model:** `1.7B-Base`
- **Seed:** `42`
- **Active Pairs:** `20`
- **Control Pairs (No-Op):** `1`
- **All WAVs Technically Valid:** `True`
- **Human Listening Panel Status:** `PENDING_HUMAN_EVALUATION` *(subjective MOS ratings must be supplied by human native-Italian listeners via `ab_scoring_template.csv`)*

## Pairwise Objective Measurements

| Pair ID | Category | Modified | Dur A (s) | Dur B (s) | Delta Dur (s) | RTF A | RTF B | RMS A (dBFS) | RMS B (dBFS) | Valid A/B |
| :--- | :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| `M2-AB-001` | `acoustic_ab_dates` | True | 5.280 | 4.000 | -1.280 | 3.703 | 3.758 | -18.92 | -18.98 | True |
| `M2-AB-002` | `acoustic_ab_dates` | True | 4.800 | 4.160 | -0.640 | 3.447 | 3.391 | -19.86 | -20.15 | True |
| `M2-AB-003` | `acoustic_ab_dates` | True | 4.640 | 4.400 | -0.240 | 3.518 | 3.978 | -17.47 | -20.99 | True |
| `M2-AB-004` | `acoustic_ab_dates` | True | 5.120 | 4.400 | -0.720 | 11.313 | 11.972 | -20.37 | -20.36 | True |
| `M2-AB-005` | `acoustic_ab_currencies` | True | 4.400 | 4.400 | +0.000 | 11.731 | 12.574 | -20.62 | -20.37 | True |
| `M2-AB-006` | `acoustic_ab_currencies` | True | 4.560 | 5.200 | +0.640 | 13.600 | 12.523 | -20.47 | -19.52 | True |
| `M2-AB-007` | `acoustic_ab_currencies` | True | 3.760 | 4.400 | +0.640 | 12.124 | 12.087 | -18.31 | -20.42 | True |
| `M2-AB-008` | `acoustic_ab_currencies` | True | 5.120 | 4.800 | -0.320 | 11.315 | 8.200 | -18.63 | -18.39 | True |
| `M2-AB-009` | `acoustic_ab_numbers` | True | 4.800 | 5.120 | +0.320 | 5.405 | 5.090 | -20.10 | -20.16 | True |
| `M2-AB-010` | `acoustic_ab_numbers` | True | 163.760 | 7.280 | -156.480 | 2.588 | 5.676 | -16.50 | -18.92 | True |
| `M2-AB-011` | `acoustic_ab_numbers` | True | 8.080 | 8.720 | +0.640 | 3.282 | 2.147 | -21.96 | -19.82 | True |
| `M2-AB-012` | `acoustic_ab_numbers` | True | 6.000 | 6.160 | +0.160 | 2.202 | 2.091 | -18.28 | -19.37 | True |
| `M2-AB-013` | `acoustic_ab_titles` | True | 4.400 | 3.760 | -0.640 | 2.152 | 2.130 | -19.92 | -19.65 | True |
| `M2-AB-014` | `acoustic_ab_titles` | True | 4.560 | 4.400 | -0.160 | 2.128 | 3.737 | -21.22 | -19.69 | True |
| `M2-AB-015` | `acoustic_ab_titles` | True | 4.400 | 3.760 | -0.640 | 5.197 | 4.974 | -19.20 | -20.32 | True |
| `M2-AB-016` | `acoustic_ab_titles` | True | 5.200 | 5.120 | -0.080 | 5.154 | 5.128 | -19.42 | -18.87 | True |
| `M2-AB-017` | `acoustic_ab_phonotactics` | True | 4.000 | 4.000 | +0.000 | 5.430 | 5.484 | -19.03 | -18.81 | True |
| `M2-AB-017-CTRL` | `acoustic_ab_control_noop` | False | 4.400 | 4.400 | +0.000 | 5.098 | 4.926 | -20.66 | -20.66 | True |
| `M2-AB-018` | `acoustic_ab_phonotactics` | True | 5.120 | 5.600 | +0.480 | 4.970 | 4.881 | -19.65 | -21.29 | True |
| `M2-AB-019` | `acoustic_ab_phonotactics` | True | 3.760 | 4.000 | +0.240 | 5.088 | 4.878 | -21.24 | -18.96 | True |
| `M2-AB-020` | `acoustic_ab_phonotactics` | True | 7.360 | 6.000 | -1.360 | 5.415 | 5.003 | -19.05 | -18.31 | True |
