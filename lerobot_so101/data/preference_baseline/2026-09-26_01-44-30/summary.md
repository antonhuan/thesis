# VLA baseline: preference vs base instruction embeddings

- checkpoint: `ant0nh/pi05_500_30k`
- scenes: 12  (pooling: mean, state-conditioned: True)
- excluded pull = sim(pref, excluded base) − mean sim(pref, target bases); **> 0 means the VLA embedding sits closer to the object it was told to avoid**
- collapse margin = sim(pref, nearest base) − mean inter-base sim; > 0 means the preference prompt sits inside the base-instruction cluster

## full_prefix

Inter-base mean similarity: 0.9929 ± 0.0014

| preference | kind | sim→banana | sim→toy | sim→pouch | nearest (scenes) | excluded pull | nearest=excluded | collapse margin |
|---|---|---|---|---|---|---|---|---|
| put everything on the tray except for the banana | exclusion | 0.9986 ✗ | 0.9932 | 0.9932 | banana:10, pouch:1, toy:1 | +0.0054 ± 0.0041 | 83% | +0.0057 |
| put everything on the tray except for the toy | exclusion | 0.9939 | 0.9989 ✗ | 0.9930 | toy:11, banana:1 | +0.0054 ± 0.0029 | 92% | +0.0060 |
| put everything on the tray except for the pouch | exclusion | 0.9941 | 0.9951 | 0.9983 ✗ | pouch:10, toy:2 | +0.0037 ± 0.0025 | 83% | +0.0056 |
| clean up the table but leave the banana | exclusion | 0.9979 ✗ | 0.9924 | 0.9922 | banana:12 | +0.0056 ± 0.0049 | 100% | +0.0049 |
| clean up the table but leave the toy | exclusion | 0.9933 | 0.9982 ✗ | 0.9919 | toy:12 | +0.0055 ± 0.0036 | 100% | +0.0053 |
| don't put the banana on the tray | negation | 0.9997 ✗ | 0.9934 | 0.9925 | banana:12 | +0.0067 ± 0.0032 | 100% | +0.0067 |
| do not put the pouch on the tray | negation | 0.9921 | 0.9920 | 0.9996 ✗ | pouch:12 | +0.0076 ± 0.0027 | 100% | +0.0067 |
| put the toy on the tray first, then the banana | ordering | 0.9981 | 0.9959 | 0.9911 | banana:7, toy:5 | — | — | +0.0056 |
| put the pouch on the tray first, then the toy | ordering | 0.9942 | 0.9981 | 0.9963 | toy:6, pouch:5, banana:1 | — | — | +0.0055 |
| put everything on the tray | all | 0.9951 | 0.9968 | 0.9965 | pouch:5, toy:4, banana:3 | — | — | +0.0051 |

## image_tokens

Inter-base mean similarity: 0.9927 ± 0.0015

| preference | kind | sim→banana | sim→toy | sim→pouch | nearest (scenes) | excluded pull | nearest=excluded | collapse margin |
|---|---|---|---|---|---|---|---|---|
| put everything on the tray except for the banana | exclusion | 0.9985 ✗ | 0.9930 | 0.9929 | banana:11, pouch:1 | +0.0056 ± 0.0042 | 92% | +0.0059 |
| put everything on the tray except for the toy | exclusion | 0.9938 | 0.9989 ✗ | 0.9929 | toy:11, banana:1 | +0.0055 ± 0.0029 | 92% | +0.0062 |
| put everything on the tray except for the pouch | exclusion | 0.9939 | 0.9950 | 0.9983 ✗ | pouch:10, toy:2 | +0.0038 ± 0.0024 | 83% | +0.0058 |
| clean up the table but leave the banana | exclusion | 0.9978 ✗ | 0.9921 | 0.9918 | banana:12 | +0.0059 ± 0.0052 | 100% | +0.0051 |
| clean up the table but leave the toy | exclusion | 0.9932 | 0.9981 ✗ | 0.9915 | toy:12 | +0.0057 ± 0.0037 | 100% | +0.0054 |
| don't put the banana on the tray | negation | 0.9997 ✗ | 0.9933 | 0.9923 | banana:12 | +0.0069 ± 0.0033 | 100% | +0.0070 |
| do not put the pouch on the tray | negation | 0.9919 | 0.9918 | 0.9996 ✗ | pouch:12 | +0.0078 ± 0.0027 | 100% | +0.0069 |
| put the toy on the tray first, then the banana | ordering | 0.9980 | 0.9958 | 0.9908 | toy:6, banana:6 | — | — | +0.0057 |
| put the pouch on the tray first, then the toy | ordering | 0.9940 | 0.9981 | 0.9962 | toy:6, pouch:5, banana:1 | — | — | +0.0057 |
| put everything on the tray | all | 0.9950 | 0.9967 | 0.9965 | pouch:5, toy:4, banana:3 | — | — | +0.0053 |

## language_tokens

Inter-base mean similarity: 0.9855 ± 0.0025

| preference | kind | sim→banana | sim→toy | sim→pouch | nearest (scenes) | excluded pull | nearest=excluded | collapse margin |
|---|---|---|---|---|---|---|---|---|
| put everything on the tray except for the banana | exclusion | 0.9907 ✗ | 0.9854 | 0.9868 | banana:10, toy:2 | +0.0046 ± 0.0033 | 83% | +0.0055 |
| put everything on the tray except for the toy | exclusion | 0.9802 | 0.9906 ✗ | 0.9838 | toy:11, pouch:1 | +0.0086 ± 0.0087 | 92% | +0.0052 |
| put everything on the tray except for the pouch | exclusion | 0.9780 | 0.9844 | 0.9874 ✗ | pouch:8, toy:3, banana:1 | +0.0062 ± 0.0060 | 67% | +0.0036 |
| clean up the table but leave the banana | exclusion | 0.9881 ✗ | 0.9815 | 0.9833 | banana:11, toy:1 | +0.0057 ± 0.0025 | 92% | +0.0027 |
| clean up the table but leave the toy | exclusion | 0.9796 | 0.9883 ✗ | 0.9837 | toy:11, pouch:1 | +0.0067 ± 0.0051 | 92% | +0.0030 |
| don't put the banana on the tray | negation | 0.9932 ✗ | 0.9817 | 0.9848 | banana:12 | +0.0099 ± 0.0027 | 100% | +0.0077 |
| do not put the pouch on the tray | negation | 0.9816 | 0.9831 | 0.9939 ✗ | pouch:12 | +0.0116 ± 0.0043 | 100% | +0.0085 |
| put the toy on the tray first, then the banana | ordering | 0.9838 | 0.9857 | 0.9832 | toy:6, banana:4, pouch:2 | — | — | +0.0042 |
| put the pouch on the tray first, then the toy | ordering | 0.9802 | 0.9861 | 0.9893 | pouch:10, toy:2 | — | — | +0.0039 |
| put everything on the tray | all | 0.9850 | 0.9904 | 0.9890 | toy:7, pouch:5 | — | — | +0.0053 |
