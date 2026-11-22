# VLA baseline: preference vs base instruction embeddings

- checkpoint: `ant0nh/pi05_500_30k`
- scenes: 12  (pooling: mean, state-conditioned: True)
- every prompt is encoded on every scene; all similarities are within-scene
- admissible set A_p: omission G_b\E_p · sequencing <_p-minimal · conditional {x: phi_p} · quantity class c
- violation pull = mean sim(pref, X\A_p) − mean sim(pref, A_p); **> 0 leans toward an inadmissible object** (n/a when A_p = X)

## Scene agreement

Counts of scenes, per region: `viol` = scenes with violation pull > 0, `adm` = scenes whose nearest base instruction is admissible.

| category | preference | A_p | full_prefix viol / adm | image_tokens viol / adm | language_tokens viol / adm | nearest (language) |
|---|---|---|---|---|---|---|
| omission | put everything on the tray except the banana | toy, pouch | 12/12 / 2/12 | 12/12 / 2/12 | 10/12 / 3/12 | banana:9, toy:3 |
| omission | put everything on the tray except the toy | banana, pouch | 12/12 / 1/12 | 12/12 / 1/12 | 12/12 / 1/12 | toy:11, pouch:1 |
| omission | put everything on the tray except the pouch | banana, toy | 12/12 / 2/12 | 12/12 / 2/12 | 10/12 / 2/12 | toy:2, pouch:10 |
| omission | put everything on the tray except the banana and the toy | pouch | 12/12 / 0/12 | 12/12 / 0/12 | 7/12 / 1/12 | banana:6, toy:5, pouch:1 |
| omission | put everything on the tray except the banana and the pouch | toy | 10/12 / 1/12 | 9/12 / 1/12 | 8/12 / 2/12 | banana:4, toy:2, pouch:6 |
| omission | put everything on the tray except the toy and the pouch | banana | 9/12 / 1/12 | 9/12 / 2/12 | 10/12 / 0/12 | toy:4, pouch:8 |
| sequencing | put the banana on the tray first, then the toy, then the pouch | banana | 0/12 / 12/12 | 0/12 / 12/12 | 3/12 / 7/12 | banana:7, toy:3, pouch:2 |
| sequencing | put the toy on the tray first, then the pouch, then the banana | toy | 3/12 / 7/12 | 2/12 / 7/12 | 6/12 / 5/12 | banana:4, toy:5, pouch:3 |
| sequencing | put the pouch on the tray first, then the banana, then the toy | pouch | 7/12 / 3/12 | 7/12 / 3/12 | 1/12 / 7/12 | banana:4, toy:1, pouch:7 |
| sequencing | put the banana on the tray after the toy | toy, pouch | 12/12 / 0/12 | 12/12 / 0/12 | 10/12 / 2/12 | banana:10, toy:2 |
| sequencing | put the pouch on the tray last | banana, toy | 12/12 / 0/12 | 12/12 / 0/12 | 12/12 / 0/12 | pouch:12 |
| conditional | put the yellow object on the tray | banana | 4/12 / 4/12 | 4/12 / 4/12 | 3/12 / 7/12 | banana:7, toy:5 |
| conditional | put the fruit on the tray | banana | 6/12 / 2/12 | 6/12 / 2/12 | 6/12 / 4/12 | banana:4, toy:7, pouch:1 |
| conditional | put the furry object on the tray | toy | 2/12 / 8/12 | 2/12 / 8/12 | 1/12 / 6/12 | banana:2, toy:6, pouch:4 |
| conditional | put the cylindrical object on the tray | pouch | 8/12 / 1/12 | 8/12 / 1/12 | 7/12 / 4/12 | banana:4, toy:4, pouch:4 |
| conditional | put the brown objects on the tray | toy, pouch | 3/12 / 9/12 | 3/12 / 9/12 | 3/12 / 11/12 | banana:1, toy:3, pouch:8 |
| conditional | put everything that is not edible on the tray | toy, pouch | 4/12 / 10/12 | 4/12 / 10/12 | 5/12 / 11/12 | banana:1, toy:3, pouch:8 |
| quantity | put one object on the tray | banana, toy, pouch | n/a / 12/12 | n/a / 12/12 | n/a / 12/12 | banana:1, toy:8, pouch:3 |
| quantity | put two objects on the tray | banana, toy, pouch | n/a / 12/12 | n/a / 12/12 | n/a / 12/12 | banana:1, toy:5, pouch:6 |
| quantity | put one of the brown objects on the tray | toy, pouch | 3/12 / 10/12 | 3/12 / 10/12 | 1/12 / 12/12 | toy:4, pouch:8 |
| quantity | put only one fruit on the tray | banana | 5/12 / 3/12 | 5/12 / 2/12 | 5/12 / 5/12 | banana:5, toy:6, pouch:1 |

## Failure clustering in latent space

Predicted failure = nearest base instruction (within scene) is inadmissible. Scene-centred, L2-normalised 2048-d pooled states; kNN fail purity = mean fraction of a failure's k nearest labelled neighbours that also fail. *cross-prompt* ignores neighbours from the same prompt. p from a 2000-shuffle label permutation.

| region | failures | chance | purity (all) | p | purity (cross-prompt) | p | purity (cross-category) | p |
|---|---|---|---|---|---|---|---|---|
| full_prefix | 152/228 | 0.667 | 0.719 | 0.0230 | 0.693 | 0.1909 | 0.668 | 0.4658 |
| image_tokens | 152/228 | 0.667 | 0.713 | 0.0340 | 0.693 | 0.1819 | 0.658 | 0.5892 |
| language_tokens | 138/228 | 0.605 | 0.764 | 0.0005 | 0.727 | 0.0005 | 0.672 | 0.0315 |

## Scene 001_banana

### full_prefix  (inter-base 0.9930)

| preference | sim→banana | sim→toy | sim→pouch | nearest | violation pull |
|---|---|---|---|---|---|
| omit banana | 0.9987 ✗ | 0.9954 | 0.9954 | banana | +0.0033 |
| omit toy | 0.9905 | 0.9989 ✗ | 0.9858 | toy | +0.0107 |
| omit pouch | 0.9972 | 0.9938 | 0.9981 ✗ | pouch | +0.0026 |
| omit banana+toy | 0.9954 ✗ | 0.9982 ✗ | 0.9916 | toy | +0.0052 |
| omit banana+pouch | 0.9984 ✗ | 0.9928 | 0.9979 ✗ | banana | +0.0053 |
| omit toy+pouch | 0.9945 | 0.9986 ✗ | 0.9916 ✗ | toy | +0.0006 |
| seq banana>toy>pouch | 0.9984 | 0.9947 ✗ | 0.9966 ✗ | banana | -0.0027 |
| seq toy>pouch>banana | 0.9941 ✗ | 0.9986 | 0.9904 ✗ | toy | -0.0063 |
| seq pouch>banana>toy | 0.9980 ✗ | 0.9947 ✗ | 0.9970 | banana | -0.0006 |
| seq banana after toy | 0.9995 ✗ | 0.9936 | 0.9968 | banana | +0.0043 |
| seq pouch last | 0.9971 | 0.9896 | 0.9999 ✗ | pouch | +0.0066 |
| cond yellow | 0.9976 | 0.9937 ✗ | 0.9965 ✗ | banana | -0.0025 |
| cond fruit | 0.9914 | 0.9986 ✗ | 0.9862 ✗ | toy | +0.0010 |
| cond furry | 0.9958 ✗ | 0.9980 | 0.9936 ✗ | toy | -0.0033 |
| cond cylindrical | 0.9974 ✗ | 0.9954 ✗ | 0.9966 | banana | -0.0002 |
| cond brown | 0.9974 ✗ | 0.9958 | 0.9966 | banana | +0.0011 |
| cond not edible | 0.9968 ✗ | 0.9972 | 0.9946 | toy | +0.0008 |
| qty one object | 0.9977 | 0.9936 | 0.9981 | pouch | n/a |
| qty two objects | 0.9979 | 0.9917 | 0.9989 | pouch | n/a |
| qty one brown | 0.9967 ✗ | 0.9974 | 0.9946 | toy | +0.0008 |
| qty one fruit | 0.9914 | 0.9984 ✗ | 0.9862 ✗ | toy | +0.0009 |

### image_tokens  (inter-base 0.9929)

| preference | sim→banana | sim→toy | sim→pouch | nearest | violation pull |
|---|---|---|---|---|---|
| omit banana | 0.9988 ✗ | 0.9952 | 0.9953 | banana | +0.0036 |
| omit toy | 0.9909 | 0.9988 ✗ | 0.9858 | toy | +0.0105 |
| omit pouch | 0.9969 | 0.9933 | 0.9982 ✗ | pouch | +0.0031 |
| omit banana+toy | 0.9955 ✗ | 0.9980 ✗ | 0.9916 | toy | +0.0052 |
| omit banana+pouch | 0.9983 ✗ | 0.9926 | 0.9979 ✗ | banana | +0.0056 |
| omit toy+pouch | 0.9948 | 0.9985 ✗ | 0.9918 ✗ | toy | +0.0003 |
| seq banana>toy>pouch | 0.9983 | 0.9948 ✗ | 0.9964 ✗ | banana | -0.0028 |
| seq toy>pouch>banana | 0.9941 ✗ | 0.9985 | 0.9901 ✗ | toy | -0.0064 |
| seq pouch>banana>toy | 0.9980 ✗ | 0.9948 ✗ | 0.9969 | banana | -0.0005 |
| seq banana after toy | 0.9996 ✗ | 0.9935 | 0.9967 | banana | +0.0045 |
| seq pouch last | 0.9968 | 0.9895 | 0.9999 ✗ | pouch | +0.0068 |
| cond yellow | 0.9975 | 0.9938 ✗ | 0.9963 ✗ | banana | -0.0024 |
| cond fruit | 0.9917 | 0.9985 ✗ | 0.9862 ✗ | toy | +0.0007 |
| cond furry | 0.9957 ✗ | 0.9980 | 0.9934 ✗ | toy | -0.0035 |
| cond cylindrical | 0.9972 ✗ | 0.9954 ✗ | 0.9964 | banana | -0.0001 |
| cond brown | 0.9972 ✗ | 0.9959 | 0.9965 | banana | +0.0010 |
| cond not edible | 0.9968 ✗ | 0.9971 | 0.9946 | toy | +0.0009 |
| qty one object | 0.9974 | 0.9933 | 0.9980 | pouch | n/a |
| qty two objects | 0.9977 | 0.9916 | 0.9989 | pouch | n/a |
| qty one brown | 0.9968 ✗ | 0.9972 | 0.9946 | toy | +0.0009 |
| qty one fruit | 0.9919 | 0.9984 ✗ | 0.9865 ✗ | toy | +0.0006 |

### language_tokens  (inter-base 0.9807)

| preference | sim→banana | sim→toy | sim→pouch | nearest | violation pull |
|---|---|---|---|---|---|
| omit banana | 0.9862 ✗ | 0.9874 | 0.9863 | toy | -0.0006 |
| omit toy | 0.9604 | 0.9915 ✗ | 0.9688 | toy | +0.0270 |
| omit pouch | 0.9716 | 0.9901 | 0.9796 ✗ | toy | -0.0012 |
| omit banana+toy | 0.9642 ✗ | 0.9884 ✗ | 0.9710 | toy | +0.0054 |
| omit banana+pouch | 0.9781 ✗ | 0.9833 | 0.9816 ✗ | toy | -0.0035 |
| omit toy+pouch | 0.9525 | 0.9850 ✗ | 0.9643 ✗ | toy | +0.0221 |
| seq banana>toy>pouch | 0.9784 | 0.9804 ✗ | 0.9793 ✗ | toy | +0.0014 |
| seq toy>pouch>banana | 0.9587 ✗ | 0.9819 | 0.9674 ✗ | toy | -0.0188 |
| seq pouch>banana>toy | 0.9746 ✗ | 0.9824 ✗ | 0.9810 | toy | -0.0025 |
| seq banana after toy | 0.9908 ✗ | 0.9847 | 0.9870 | banana | +0.0050 |
| seq pouch last | 0.9880 | 0.9814 | 0.9972 ✗ | pouch | +0.0125 |
| cond yellow | 0.9916 | 0.9843 ✗ | 0.9894 ✗ | banana | -0.0048 |
| cond fruit | 0.9713 | 0.9933 ✗ | 0.9769 ✗ | toy | +0.0137 |
| cond furry | 0.9814 ✗ | 0.9915 | 0.9850 ✗ | toy | -0.0083 |
| cond cylindrical | 0.9870 ✗ | 0.9885 ✗ | 0.9891 | pouch | -0.0013 |
| cond brown | 0.9875 ✗ | 0.9896 | 0.9896 | pouch | -0.0022 |
| cond not edible | 0.9746 ✗ | 0.9903 | 0.9803 | toy | -0.0107 |
| qty one object | 0.9849 | 0.9918 | 0.9885 | toy | n/a |
| qty two objects | 0.9880 | 0.9871 | 0.9911 | pouch | n/a |
| qty one brown | 0.9693 ✗ | 0.9906 | 0.9768 | toy | -0.0144 |
| qty one fruit | 0.9579 | 0.9893 ✗ | 0.9656 ✗ | toy | +0.0196 |


## Scene 002_banana

### full_prefix  (inter-base 0.9898)

| preference | sim→banana | sim→toy | sim→pouch | nearest | violation pull |
|---|---|---|---|---|---|
| omit banana | 0.9982 ✗ | 0.9869 | 0.9797 | banana | +0.0149 |
| omit toy | 0.9912 | 0.9989 ✗ | 0.9932 | toy | +0.0067 |
| omit pouch | 0.9899 | 0.9988 | 0.9970 ✗ | toy | +0.0027 |
| omit banana+toy | 0.9980 ✗ | 0.9910 ✗ | 0.9841 | banana | +0.0104 |
| omit banana+pouch | 0.9979 ✗ | 0.9903 | 0.9833 ✗ | banana | +0.0003 |
| omit toy+pouch | 0.9884 | 0.9991 ✗ | 0.9974 ✗ | toy | +0.0098 |
| seq banana>toy>pouch | 0.9989 | 0.9915 ✗ | 0.9853 ✗ | banana | -0.0105 |
| seq toy>pouch>banana | 0.9959 ✗ | 0.9952 | 0.9896 ✗ | banana | -0.0024 |
| seq pouch>banana>toy | 0.9964 ✗ | 0.9953 ✗ | 0.9908 | banana | +0.0050 |
| seq banana after toy | 0.9997 ✗ | 0.9893 | 0.9846 | banana | +0.0128 |
| seq pouch last | 0.9844 | 0.9961 | 0.9998 ✗ | pouch | +0.0095 |
| cond yellow | 0.9941 | 0.9977 ✗ | 0.9926 ✗ | toy | +0.0011 |
| cond fruit | 0.9962 | 0.9953 ✗ | 0.9884 ✗ | banana | -0.0044 |
| cond furry | 0.9943 ✗ | 0.9976 | 0.9937 ✗ | toy | -0.0036 |
| cond cylindrical | 0.9920 ✗ | 0.9985 ✗ | 0.9934 | toy | +0.0018 |
| cond brown | 0.9919 ✗ | 0.9986 | 0.9943 | toy | -0.0046 |
| cond not edible | 0.9936 ✗ | 0.9973 | 0.9912 | toy | -0.0006 |
| qty one object | 0.9919 | 0.9988 | 0.9945 | toy | n/a |
| qty two objects | 0.9907 | 0.9992 | 0.9958 | toy | n/a |
| qty one brown | 0.9920 ✗ | 0.9982 | 0.9933 | toy | -0.0038 |
| qty one fruit | 0.9953 | 0.9952 ✗ | 0.9880 ✗ | banana | -0.0038 |

### image_tokens  (inter-base 0.9892)

| preference | sim→banana | sim→toy | sim→pouch | nearest | violation pull |
|---|---|---|---|---|---|
| omit banana | 0.9981 ✗ | 0.9865 | 0.9784 | banana | +0.0156 |
| omit toy | 0.9906 | 0.9989 ✗ | 0.9926 | toy | +0.0073 |
| omit pouch | 0.9894 | 0.9988 | 0.9968 ✗ | toy | +0.0027 |
| omit banana+toy | 0.9978 ✗ | 0.9907 ✗ | 0.9828 | banana | +0.0114 |
| omit banana+pouch | 0.9977 ✗ | 0.9901 | 0.9822 ✗ | banana | -0.0002 |
| omit toy+pouch | 0.9875 | 0.9990 ✗ | 0.9972 ✗ | toy | +0.0106 |
| seq banana>toy>pouch | 0.9988 | 0.9912 ✗ | 0.9843 ✗ | banana | -0.0111 |
| seq toy>pouch>banana | 0.9956 ✗ | 0.9951 | 0.9890 ✗ | banana | -0.0028 |
| seq pouch>banana>toy | 0.9962 ✗ | 0.9952 ✗ | 0.9902 | banana | +0.0055 |
| seq banana after toy | 0.9997 ✗ | 0.9888 | 0.9835 | banana | +0.0136 |
| seq pouch last | 0.9836 | 0.9958 | 0.9998 ✗ | pouch | +0.0101 |
| cond yellow | 0.9938 | 0.9976 ✗ | 0.9918 ✗ | toy | +0.0009 |
| cond fruit | 0.9960 | 0.9952 ✗ | 0.9877 ✗ | banana | -0.0045 |
| cond furry | 0.9942 ✗ | 0.9975 | 0.9932 ✗ | toy | -0.0038 |
| cond cylindrical | 0.9916 ✗ | 0.9984 ✗ | 0.9924 | toy | +0.0026 |
| cond brown | 0.9914 ✗ | 0.9985 | 0.9935 | toy | -0.0046 |
| cond not edible | 0.9931 ✗ | 0.9973 | 0.9905 | toy | -0.0008 |
| qty one object | 0.9914 | 0.9988 | 0.9937 | toy | n/a |
| qty two objects | 0.9904 | 0.9992 | 0.9951 | toy | n/a |
| qty one brown | 0.9914 ✗ | 0.9982 | 0.9925 | toy | -0.0039 |
| qty one fruit | 0.9950 | 0.9952 ✗ | 0.9873 ✗ | toy | -0.0038 |

### language_tokens  (inter-base 0.9875)

| preference | sim→banana | sim→toy | sim→pouch | nearest | violation pull |
|---|---|---|---|---|---|
| omit banana | 0.9928 ✗ | 0.9860 | 0.9860 | banana | +0.0068 |
| omit toy | 0.9868 | 0.9913 ✗ | 0.9889 | toy | +0.0035 |
| omit pouch | 0.9870 | 0.9911 | 0.9923 ✗ | pouch | +0.0032 |
| omit banana+toy | 0.9896 ✗ | 0.9835 ✗ | 0.9831 | banana | +0.0035 |
| omit banana+pouch | 0.9884 ✗ | 0.9809 | 0.9833 ✗ | banana | +0.0050 |
| omit toy+pouch | 0.9841 | 0.9857 ✗ | 0.9886 ✗ | pouch | +0.0030 |
| seq banana>toy>pouch | 0.9863 | 0.9741 ✗ | 0.9777 ✗ | banana | -0.0105 |
| seq toy>pouch>banana | 0.9790 ✗ | 0.9709 | 0.9775 ✗ | banana | +0.0073 |
| seq pouch>banana>toy | 0.9814 ✗ | 0.9725 ✗ | 0.9802 | banana | -0.0032 |
| seq banana after toy | 0.9949 ✗ | 0.9844 | 0.9834 | banana | +0.0110 |
| seq pouch last | 0.9867 | 0.9904 | 0.9974 ✗ | pouch | +0.0088 |
| cond yellow | 0.9897 | 0.9935 ✗ | 0.9882 ✗ | toy | +0.0012 |
| cond fruit | 0.9906 | 0.9893 ✗ | 0.9874 ✗ | banana | -0.0022 |
| cond furry | 0.9888 ✗ | 0.9905 | 0.9868 ✗ | toy | -0.0026 |
| cond cylindrical | 0.9839 ✗ | 0.9914 ✗ | 0.9853 | toy | +0.0024 |
| cond brown | 0.9873 ✗ | 0.9938 | 0.9903 | toy | -0.0048 |
| cond not edible | 0.9876 ✗ | 0.9861 | 0.9880 | pouch | +0.0005 |
| qty one object | 0.9868 | 0.9949 | 0.9898 | toy | n/a |
| qty two objects | 0.9841 | 0.9926 | 0.9882 | toy | n/a |
| qty one brown | 0.9858 ✗ | 0.9886 | 0.9884 | toy | -0.0028 |
| qty one fruit | 0.9881 | 0.9846 ✗ | 0.9853 ✗ | banana | -0.0032 |


## Scene 003_banana

### full_prefix  (inter-base 0.9921)

| preference | sim→banana | sim→toy | sim→pouch | nearest | violation pull |
|---|---|---|---|---|---|
| omit banana | 0.9974 ✗ | 0.9885 | 0.9987 | pouch | +0.0038 |
| omit toy | 0.9962 | 0.9991 ✗ | 0.9892 | toy | +0.0064 |
| omit pouch | 0.9974 | 0.9904 | 0.9988 ✗ | pouch | +0.0049 |
| omit banana+toy | 0.9988 ✗ | 0.9951 ✗ | 0.9956 | banana | +0.0014 |
| omit banana+pouch | 0.9972 ✗ | 0.9886 | 0.9988 ✗ | pouch | +0.0094 |
| omit toy+pouch | 0.9963 | 0.9982 ✗ | 0.9897 ✗ | toy | -0.0024 |
| seq banana>toy>pouch | 0.9985 | 0.9912 ✗ | 0.9977 ✗ | banana | -0.0040 |
| seq toy>pouch>banana | 0.9983 ✗ | 0.9958 | 0.9942 ✗ | banana | +0.0005 |
| seq pouch>banana>toy | 0.9968 ✗ | 0.9877 ✗ | 0.9988 | pouch | -0.0065 |
| seq banana after toy | 0.9996 ✗ | 0.9937 | 0.9961 | banana | +0.0047 |
| seq pouch last | 0.9956 | 0.9859 | 0.9999 ✗ | pouch | +0.0091 |
| cond yellow | 0.9970 | 0.9987 ✗ | 0.9895 ✗ | toy | -0.0029 |
| cond fruit | 0.9977 | 0.9984 ✗ | 0.9914 ✗ | toy | -0.0028 |
| cond furry | 0.9964 ✗ | 0.9982 | 0.9898 ✗ | toy | -0.0051 |
| cond cylindrical | 0.9972 ✗ | 0.9986 ✗ | 0.9908 | toy | +0.0071 |
| cond brown | 0.9983 ✗ | 0.9946 | 0.9967 | banana | +0.0027 |
| cond not edible | 0.9984 ✗ | 0.9953 | 0.9961 | banana | +0.0027 |
| qty one object | 0.9977 | 0.9976 | 0.9925 | banana | n/a |
| qty two objects | 0.9978 | 0.9975 | 0.9931 | banana | n/a |
| qty one brown | 0.9982 ✗ | 0.9946 | 0.9967 | banana | +0.0026 |
| qty one fruit | 0.9978 | 0.9979 ✗ | 0.9920 ✗ | toy | -0.0028 |

### image_tokens  (inter-base 0.9919)

| preference | sim→banana | sim→toy | sim→pouch | nearest | violation pull |
|---|---|---|---|---|---|
| omit banana | 0.9973 ✗ | 0.9881 | 0.9986 | pouch | +0.0039 |
| omit toy | 0.9961 | 0.9992 ✗ | 0.9889 | toy | +0.0067 |
| omit pouch | 0.9974 | 0.9906 | 0.9987 ✗ | pouch | +0.0047 |
| omit banana+toy | 0.9988 ✗ | 0.9949 ✗ | 0.9955 | banana | +0.0013 |
| omit banana+pouch | 0.9972 ✗ | 0.9886 | 0.9987 ✗ | pouch | +0.0093 |
| omit toy+pouch | 0.9962 | 0.9981 ✗ | 0.9892 ✗ | toy | -0.0025 |
| seq banana>toy>pouch | 0.9984 | 0.9909 ✗ | 0.9977 ✗ | banana | -0.0041 |
| seq toy>pouch>banana | 0.9983 ✗ | 0.9959 | 0.9939 ✗ | banana | +0.0002 |
| seq pouch>banana>toy | 0.9967 ✗ | 0.9874 ✗ | 0.9988 | pouch | -0.0067 |
| seq banana after toy | 0.9996 ✗ | 0.9938 | 0.9958 | banana | +0.0048 |
| seq pouch last | 0.9956 | 0.9859 | 0.9999 ✗ | pouch | +0.0091 |
| cond yellow | 0.9972 | 0.9985 ✗ | 0.9899 ✗ | toy | -0.0030 |
| cond fruit | 0.9976 | 0.9984 ✗ | 0.9910 ✗ | toy | -0.0029 |
| cond furry | 0.9966 ✗ | 0.9981 | 0.9902 ✗ | toy | -0.0048 |
| cond cylindrical | 0.9974 ✗ | 0.9984 ✗ | 0.9911 | toy | +0.0068 |
| cond brown | 0.9983 ✗ | 0.9945 | 0.9967 | banana | +0.0027 |
| cond not edible | 0.9983 ✗ | 0.9952 | 0.9960 | banana | +0.0027 |
| qty one object | 0.9978 | 0.9975 | 0.9926 | banana | n/a |
| qty two objects | 0.9977 | 0.9974 | 0.9929 | banana | n/a |
| qty one brown | 0.9982 ✗ | 0.9946 | 0.9967 | banana | +0.0026 |
| qty one fruit | 0.9977 | 0.9980 ✗ | 0.9916 ✗ | toy | -0.0029 |

### language_tokens  (inter-base 0.9860)

| preference | sim→banana | sim→toy | sim→pouch | nearest | violation pull |
|---|---|---|---|---|---|
| omit banana | 0.9911 ✗ | 0.9830 | 0.9909 | banana | +0.0041 |
| omit toy | 0.9879 | 0.9914 ✗ | 0.9841 | toy | +0.0054 |
| omit pouch | 0.9863 | 0.9758 | 0.9922 ✗ | pouch | +0.0112 |
| omit banana+toy | 0.9897 ✗ | 0.9848 ✗ | 0.9866 | banana | +0.0006 |
| omit banana+pouch | 0.9839 ✗ | 0.9709 | 0.9879 ✗ | pouch | +0.0150 |
| omit toy+pouch | 0.9861 | 0.9843 ✗ | 0.9878 ✗ | pouch | -0.0000 |
| seq banana>toy>pouch | 0.9878 | 0.9753 ✗ | 0.9869 ✗ | banana | -0.0067 |
| seq toy>pouch>banana | 0.9817 ✗ | 0.9719 | 0.9841 ✗ | pouch | +0.0110 |
| seq pouch>banana>toy | 0.9820 ✗ | 0.9695 ✗ | 0.9872 | pouch | -0.0114 |
| seq banana after toy | 0.9896 ✗ | 0.9760 | 0.9889 | banana | +0.0072 |
| seq pouch last | 0.9874 | 0.9779 | 0.9972 ✗ | pouch | +0.0146 |
| cond yellow | 0.9810 | 0.9883 ✗ | 0.9693 ✗ | toy | -0.0022 |
| cond fruit | 0.9917 | 0.9907 ✗ | 0.9876 ✗ | banana | -0.0026 |
| cond furry | 0.9833 ✗ | 0.9889 | 0.9763 ✗ | toy | -0.0091 |
| cond cylindrical | 0.9843 ✗ | 0.9907 ✗ | 0.9766 | toy | +0.0109 |
| cond brown | 0.9917 ✗ | 0.9883 | 0.9920 | pouch | +0.0016 |
| cond not edible | 0.9903 ✗ | 0.9842 | 0.9900 | banana | +0.0032 |
| qty one object | 0.9904 | 0.9930 | 0.9864 | toy | n/a |
| qty two objects | 0.9906 | 0.9906 | 0.9895 | toy | n/a |
| qty one brown | 0.9882 ✗ | 0.9812 | 0.9904 | pouch | +0.0024 |
| qty one fruit | 0.9894 | 0.9851 ✗ | 0.9880 ✗ | banana | -0.0029 |


## Scene 016_banana

### full_prefix  (inter-base 0.9960)

| preference | sim→banana | sim→toy | sim→pouch | nearest | violation pull |
|---|---|---|---|---|---|
| omit banana | 0.9988 ✗ | 0.9968 | 0.9968 | banana | +0.0020 |
| omit toy | 0.9982 | 0.9979 ✗ | 0.9968 | banana | +0.0004 |
| omit pouch | 0.9973 | 0.9970 | 0.9984 ✗ | pouch | +0.0012 |
| omit banana+toy | 0.9987 ✗ | 0.9973 ✗ | 0.9965 | banana | +0.0015 |
| omit banana+pouch | 0.9983 ✗ | 0.9972 | 0.9978 ✗ | banana | +0.0009 |
| omit toy+pouch | 0.9977 | 0.9979 ✗ | 0.9980 ✗ | pouch | +0.0002 |
| seq banana>toy>pouch | 0.9993 | 0.9977 ✗ | 0.9960 ✗ | banana | -0.0024 |
| seq toy>pouch>banana | 0.9988 ✗ | 0.9981 | 0.9968 ✗ | banana | -0.0004 |
| seq pouch>banana>toy | 0.9972 ✗ | 0.9968 ✗ | 0.9987 | pouch | -0.0018 |
| seq banana after toy | 0.9996 ✗ | 0.9975 | 0.9953 | banana | +0.0032 |
| seq pouch last | 0.9949 | 0.9957 | 0.9999 ✗ | pouch | +0.0046 |
| cond yellow | 0.9982 | 0.9987 ✗ | 0.9969 ✗ | toy | -0.0004 |
| cond fruit | 0.9976 | 0.9990 ✗ | 0.9968 ✗ | toy | +0.0003 |
| cond furry | 0.9976 ✗ | 0.9962 | 0.9969 ✗ | banana | +0.0010 |
| cond cylindrical | 0.9983 ✗ | 0.9979 ✗ | 0.9973 | banana | +0.0008 |
| cond brown | 0.9965 ✗ | 0.9970 | 0.9991 | pouch | -0.0015 |
| cond not edible | 0.9975 ✗ | 0.9965 | 0.9976 | pouch | +0.0004 |
| qty one object | 0.9975 | 0.9971 | 0.9977 | pouch | n/a |
| qty two objects | 0.9976 | 0.9968 | 0.9974 | banana | n/a |
| qty one brown | 0.9970 ✗ | 0.9975 | 0.9988 | pouch | -0.0012 |
| qty one fruit | 0.9978 | 0.9984 ✗ | 0.9969 ✗ | toy | -0.0002 |

### image_tokens  (inter-base 0.9960)

| preference | sim→banana | sim→toy | sim→pouch | nearest | violation pull |
|---|---|---|---|---|---|
| omit banana | 0.9988 ✗ | 0.9966 | 0.9969 | banana | +0.0021 |
| omit toy | 0.9981 | 0.9979 ✗ | 0.9969 | banana | +0.0004 |
| omit pouch | 0.9973 | 0.9968 | 0.9984 ✗ | pouch | +0.0014 |
| omit banana+toy | 0.9987 ✗ | 0.9972 ✗ | 0.9966 | banana | +0.0013 |
| omit banana+pouch | 0.9983 ✗ | 0.9971 | 0.9978 ✗ | banana | +0.0010 |
| omit toy+pouch | 0.9977 | 0.9978 ✗ | 0.9980 ✗ | pouch | +0.0002 |
| seq banana>toy>pouch | 0.9993 | 0.9975 ✗ | 0.9959 ✗ | banana | -0.0026 |
| seq toy>pouch>banana | 0.9989 ✗ | 0.9979 | 0.9966 ✗ | banana | -0.0002 |
| seq pouch>banana>toy | 0.9973 ✗ | 0.9966 ✗ | 0.9987 | pouch | -0.0018 |
| seq banana after toy | 0.9996 ✗ | 0.9972 | 0.9951 | banana | +0.0035 |
| seq pouch last | 0.9949 | 0.9957 | 0.9999 ✗ | pouch | +0.0046 |
| cond yellow | 0.9981 | 0.9987 ✗ | 0.9970 ✗ | toy | -0.0003 |
| cond fruit | 0.9976 | 0.9990 ✗ | 0.9967 ✗ | toy | +0.0003 |
| cond furry | 0.9975 ✗ | 0.9960 | 0.9970 ✗ | banana | +0.0012 |
| cond cylindrical | 0.9982 ✗ | 0.9979 ✗ | 0.9974 | banana | +0.0007 |
| cond brown | 0.9964 ✗ | 0.9970 | 0.9991 | pouch | -0.0016 |
| cond not edible | 0.9974 ✗ | 0.9963 | 0.9977 | pouch | +0.0003 |
| qty one object | 0.9975 | 0.9970 | 0.9977 | pouch | n/a |
| qty two objects | 0.9974 | 0.9966 | 0.9974 | banana | n/a |
| qty one brown | 0.9969 ✗ | 0.9975 | 0.9988 | pouch | -0.0012 |
| qty one fruit | 0.9978 | 0.9984 ✗ | 0.9969 ✗ | toy | -0.0001 |

### language_tokens  (inter-base 0.9874)

| preference | sim→banana | sim→toy | sim→pouch | nearest | violation pull |
|---|---|---|---|---|---|
| omit banana | 0.9906 ✗ | 0.9887 | 0.9898 | banana | +0.0014 |
| omit toy | 0.9893 | 0.9913 ✗ | 0.9873 | toy | +0.0031 |
| omit pouch | 0.9870 | 0.9870 | 0.9912 ✗ | pouch | +0.0042 |
| omit banana+toy | 0.9896 ✗ | 0.9879 ✗ | 0.9870 | banana | +0.0017 |
| omit banana+pouch | 0.9841 ✗ | 0.9825 | 0.9881 ✗ | pouch | +0.0036 |
| omit toy+pouch | 0.9827 | 0.9841 ✗ | 0.9880 ✗ | pouch | +0.0034 |
| seq banana>toy>pouch | 0.9811 | 0.9793 ✗ | 0.9846 ✗ | pouch | +0.0009 |
| seq toy>pouch>banana | 0.9704 ✗ | 0.9712 | 0.9796 ✗ | pouch | +0.0038 |
| seq pouch>banana>toy | 0.9696 ✗ | 0.9674 ✗ | 0.9819 | pouch | -0.0135 |
| seq banana after toy | 0.9907 ✗ | 0.9877 | 0.9889 | banana | +0.0025 |
| seq pouch last | 0.9850 | 0.9820 | 0.9973 ✗ | pouch | +0.0138 |
| cond yellow | 0.9934 | 0.9920 ✗ | 0.9861 ✗ | banana | -0.0044 |
| cond fruit | 0.9935 | 0.9913 ✗ | 0.9882 ✗ | banana | -0.0038 |
| cond furry | 0.9902 ✗ | 0.9898 | 0.9887 ✗ | banana | -0.0003 |
| cond cylindrical | 0.9922 ✗ | 0.9898 ✗ | 0.9860 | banana | +0.0050 |
| cond brown | 0.9900 ✗ | 0.9877 | 0.9930 | pouch | -0.0004 |
| cond not edible | 0.9891 ✗ | 0.9875 | 0.9905 | pouch | +0.0000 |
| qty one object | 0.9917 | 0.9916 | 0.9904 | banana | n/a |
| qty two objects | 0.9915 | 0.9904 | 0.9921 | pouch | n/a |
| qty one brown | 0.9868 ✗ | 0.9846 | 0.9901 | pouch | -0.0005 |
| qty one fruit | 0.9902 | 0.9870 ✗ | 0.9882 ✗ | banana | -0.0026 |


## Scene 004_pouch

### full_prefix  (inter-base 0.9939)

| preference | sim→banana | sim→toy | sim→pouch | nearest | violation pull |
|---|---|---|---|---|---|
| omit banana | 0.9987 ✗ | 0.9881 | 0.9897 | banana | +0.0098 |
| omit toy | 0.9924 | 0.9990 ✗ | 0.9987 | toy | +0.0034 |
| omit pouch | 0.9946 | 0.9977 | 0.9986 ✗ | pouch | +0.0025 |
| omit banana+toy | 0.9985 ✗ | 0.9926 ✗ | 0.9937 | banana | +0.0019 |
| omit banana+pouch | 0.9985 ✗ | 0.9897 | 0.9913 ✗ | banana | +0.0052 |
| omit toy+pouch | 0.9931 | 0.9987 ✗ | 0.9990 ✗ | pouch | +0.0058 |
| seq banana>toy>pouch | 0.9992 | 0.9929 ✗ | 0.9942 ✗ | banana | -0.0057 |
| seq toy>pouch>banana | 0.9972 ✗ | 0.9963 | 0.9967 ✗ | banana | +0.0007 |
| seq pouch>banana>toy | 0.9979 ✗ | 0.9955 ✗ | 0.9966 | banana | +0.0001 |
| seq banana after toy | 0.9998 ✗ | 0.9915 | 0.9928 | banana | +0.0077 |
| seq pouch last | 0.9925 | 0.9985 | 0.9998 ✗ | pouch | +0.0043 |
| cond yellow | 0.9956 | 0.9970 ✗ | 0.9977 ✗ | pouch | +0.0018 |
| cond fruit | 0.9934 | 0.9986 ✗ | 0.9984 ✗ | toy | +0.0051 |
| cond furry | 0.9955 ✗ | 0.9979 | 0.9989 ✗ | pouch | -0.0007 |
| cond cylindrical | 0.9931 ✗ | 0.9987 ✗ | 0.9989 | pouch | -0.0029 |
| cond brown | 0.9957 ✗ | 0.9975 | 0.9985 | pouch | -0.0023 |
| cond not edible | 0.9962 ✗ | 0.9965 | 0.9975 | pouch | -0.0008 |
| qty one object | 0.9944 | 0.9980 | 0.9986 | pouch | n/a |
| qty two objects | 0.9949 | 0.9975 | 0.9981 | pouch | n/a |
| qty one brown | 0.9950 ✗ | 0.9975 | 0.9983 | pouch | -0.0029 |
| qty one fruit | 0.9941 | 0.9980 ✗ | 0.9980 ✗ | pouch | +0.0039 |

### image_tokens  (inter-base 0.9938)

| preference | sim→banana | sim→toy | sim→pouch | nearest | violation pull |
|---|---|---|---|---|---|
| omit banana | 0.9986 ✗ | 0.9878 | 0.9897 | banana | +0.0099 |
| omit toy | 0.9921 | 0.9990 ✗ | 0.9988 | toy | +0.0035 |
| omit pouch | 0.9943 | 0.9977 | 0.9987 ✗ | pouch | +0.0027 |
| omit banana+toy | 0.9985 ✗ | 0.9924 ✗ | 0.9937 | banana | +0.0017 |
| omit banana+pouch | 0.9985 ✗ | 0.9895 | 0.9914 ✗ | banana | +0.0054 |
| omit toy+pouch | 0.9927 | 0.9987 ✗ | 0.9990 ✗ | pouch | +0.0062 |
| seq banana>toy>pouch | 0.9993 | 0.9925 ✗ | 0.9940 ✗ | banana | -0.0060 |
| seq toy>pouch>banana | 0.9970 ✗ | 0.9963 | 0.9969 ✗ | banana | +0.0007 |
| seq pouch>banana>toy | 0.9980 ✗ | 0.9953 ✗ | 0.9966 | banana | +0.0000 |
| seq banana after toy | 0.9998 ✗ | 0.9912 | 0.9927 | banana | +0.0079 |
| seq pouch last | 0.9925 | 0.9984 | 0.9999 ✗ | pouch | +0.0044 |
| cond yellow | 0.9954 | 0.9968 ✗ | 0.9977 ✗ | pouch | +0.0019 |
| cond fruit | 0.9932 | 0.9985 ✗ | 0.9984 ✗ | toy | +0.0053 |
| cond furry | 0.9955 ✗ | 0.9978 | 0.9989 ✗ | pouch | -0.0006 |
| cond cylindrical | 0.9930 ✗ | 0.9986 ✗ | 0.9988 | pouch | -0.0030 |
| cond brown | 0.9957 ✗ | 0.9973 | 0.9984 | pouch | -0.0022 |
| cond not edible | 0.9960 ✗ | 0.9963 | 0.9975 | pouch | -0.0009 |
| qty one object | 0.9942 | 0.9980 | 0.9986 | pouch | n/a |
| qty two objects | 0.9946 | 0.9974 | 0.9981 | pouch | n/a |
| qty one brown | 0.9948 ✗ | 0.9974 | 0.9983 | pouch | -0.0030 |
| qty one fruit | 0.9938 | 0.9980 ✗ | 0.9981 ✗ | pouch | +0.0042 |

### language_tokens  (inter-base 0.9887)

| preference | sim→banana | sim→toy | sim→pouch | nearest | violation pull |
|---|---|---|---|---|---|
| omit banana | 0.9929 ✗ | 0.9837 | 0.9815 | banana | +0.0102 |
| omit toy | 0.9886 | 0.9925 ✗ | 0.9882 | toy | +0.0042 |
| omit pouch | 0.9890 | 0.9899 | 0.9873 ✗ | toy | -0.0021 |
| omit banana+toy | 0.9912 ✗ | 0.9856 ✗ | 0.9820 | banana | +0.0064 |
| omit banana+pouch | 0.9892 ✗ | 0.9798 | 0.9768 ✗ | banana | +0.0033 |
| omit toy+pouch | 0.9880 | 0.9881 ✗ | 0.9841 ✗ | toy | -0.0019 |
| seq banana>toy>pouch | 0.9876 | 0.9804 ✗ | 0.9768 ✗ | banana | -0.0089 |
| seq toy>pouch>banana | 0.9841 ✗ | 0.9779 | 0.9720 ✗ | banana | +0.0002 |
| seq pouch>banana>toy | 0.9821 ✗ | 0.9800 ✗ | 0.9795 | banana | +0.0015 |
| seq banana after toy | 0.9953 ✗ | 0.9845 | 0.9818 | banana | +0.0121 |
| seq pouch last | 0.9868 | 0.9936 | 0.9976 ✗ | pouch | +0.0074 |
| cond yellow | 0.9914 | 0.9931 ✗ | 0.9906 ✗ | toy | +0.0005 |
| cond fruit | 0.9886 | 0.9929 ✗ | 0.9915 ✗ | toy | +0.0036 |
| cond furry | 0.9885 ✗ | 0.9909 | 0.9921 ✗ | pouch | -0.0006 |
| cond cylindrical | 0.9853 ✗ | 0.9929 ✗ | 0.9935 | pouch | -0.0043 |
| cond brown | 0.9864 ✗ | 0.9930 | 0.9936 | pouch | -0.0069 |
| cond not edible | 0.9894 ✗ | 0.9899 | 0.9881 | toy | +0.0005 |
| qty one object | 0.9881 | 0.9948 | 0.9935 | toy | n/a |
| qty two objects | 0.9890 | 0.9938 | 0.9932 | toy | n/a |
| qty one brown | 0.9892 ✗ | 0.9910 | 0.9899 | toy | -0.0013 |
| qty one fruit | 0.9892 | 0.9891 ✗ | 0.9885 ✗ | banana | -0.0004 |


## Scene 009_pouch

### full_prefix  (inter-base 0.9930)

| preference | sim→banana | sim→toy | sim→pouch | nearest | violation pull |
|---|---|---|---|---|---|
| omit banana | 0.9981 ✗ | 0.9969 | 0.9954 | banana | +0.0020 |
| omit toy | 0.9980 | 0.9983 ✗ | 0.9939 | toy | +0.0023 |
| omit pouch | 0.9904 | 0.9889 | 0.9987 ✗ | pouch | +0.0091 |
| omit banana+toy | 0.9985 ✗ | 0.9981 ✗ | 0.9933 | banana | +0.0050 |
| omit banana+pouch | 0.9970 ✗ | 0.9958 | 0.9968 ✗ | banana | +0.0011 |
| omit toy+pouch | 0.9968 | 0.9966 ✗ | 0.9964 ✗ | banana | -0.0004 |
| seq banana>toy>pouch | 0.9989 | 0.9986 ✗ | 0.9896 ✗ | banana | -0.0048 |
| seq toy>pouch>banana | 0.9986 ✗ | 0.9990 | 0.9897 ✗ | toy | -0.0048 |
| seq pouch>banana>toy | 0.9977 ✗ | 0.9969 ✗ | 0.9947 | banana | +0.0026 |
| seq banana after toy | 0.9997 ✗ | 0.9987 | 0.9912 | banana | +0.0047 |
| seq pouch last | 0.9901 | 0.9886 | 0.9998 ✗ | pouch | +0.0104 |
| cond yellow | 0.9989 | 0.9987 ✗ | 0.9916 ✗ | banana | -0.0038 |
| cond fruit | 0.9990 | 0.9991 ✗ | 0.9917 ✗ | toy | -0.0036 |
| cond furry | 0.9962 ✗ | 0.9954 | 0.9975 ✗ | pouch | +0.0015 |
| cond cylindrical | 0.9987 ✗ | 0.9988 ✗ | 0.9933 | toy | +0.0055 |
| cond brown | 0.9948 ✗ | 0.9938 | 0.9981 | pouch | -0.0011 |
| cond not edible | 0.9919 ✗ | 0.9904 | 0.9984 | pouch | -0.0025 |
| qty one object | 0.9964 | 0.9957 | 0.9971 | pouch | n/a |
| qty two objects | 0.9969 | 0.9965 | 0.9965 | banana | n/a |
| qty one brown | 0.9933 ✗ | 0.9920 | 0.9981 | pouch | -0.0018 |
| qty one fruit | 0.9988 | 0.9987 ✗ | 0.9923 ✗ | banana | -0.0033 |

### image_tokens  (inter-base 0.9928)

| preference | sim→banana | sim→toy | sim→pouch | nearest | violation pull |
|---|---|---|---|---|---|
| omit banana | 0.9981 ✗ | 0.9967 | 0.9954 | banana | +0.0021 |
| omit toy | 0.9981 | 0.9983 ✗ | 0.9937 | toy | +0.0024 |
| omit pouch | 0.9908 | 0.9890 | 0.9988 ✗ | pouch | +0.0089 |
| omit banana+toy | 0.9986 ✗ | 0.9980 ✗ | 0.9930 | banana | +0.0052 |
| omit banana+pouch | 0.9970 ✗ | 0.9957 | 0.9968 ✗ | banana | +0.0012 |
| omit toy+pouch | 0.9969 | 0.9966 ✗ | 0.9962 ✗ | banana | -0.0006 |
| seq banana>toy>pouch | 0.9989 | 0.9986 ✗ | 0.9896 ✗ | banana | -0.0048 |
| seq toy>pouch>banana | 0.9985 ✗ | 0.9989 | 0.9890 ✗ | toy | -0.0052 |
| seq pouch>banana>toy | 0.9976 ✗ | 0.9967 ✗ | 0.9948 | banana | +0.0024 |
| seq banana after toy | 0.9997 ✗ | 0.9987 | 0.9908 | banana | +0.0049 |
| seq pouch last | 0.9903 | 0.9884 | 0.9998 ✗ | pouch | +0.0105 |
| cond yellow | 0.9988 | 0.9985 ✗ | 0.9916 ✗ | banana | -0.0038 |
| cond fruit | 0.9990 | 0.9991 ✗ | 0.9913 ✗ | toy | -0.0038 |
| cond furry | 0.9963 ✗ | 0.9952 | 0.9975 ✗ | pouch | +0.0016 |
| cond cylindrical | 0.9987 ✗ | 0.9987 ✗ | 0.9932 | banana | +0.0055 |
| cond brown | 0.9948 ✗ | 0.9936 | 0.9980 | pouch | -0.0010 |
| cond not edible | 0.9921 ✗ | 0.9903 | 0.9984 | pouch | -0.0022 |
| qty one object | 0.9964 | 0.9956 | 0.9969 | pouch | n/a |
| qty two objects | 0.9969 | 0.9963 | 0.9965 | banana | n/a |
| qty one brown | 0.9935 ✗ | 0.9920 | 0.9980 | pouch | -0.0015 |
| qty one fruit | 0.9988 | 0.9987 ✗ | 0.9920 ✗ | banana | -0.0035 |

### language_tokens  (inter-base 0.9832)

| preference | sim→banana | sim→toy | sim→pouch | nearest | violation pull |
|---|---|---|---|---|---|
| omit banana | 0.9881 ✗ | 0.9901 | 0.9858 | toy | +0.0002 |
| omit toy | 0.9831 | 0.9913 ✗ | 0.9837 | toy | +0.0079 |
| omit pouch | 0.9642 | 0.9767 | 0.9861 ✗ | pouch | +0.0156 |
| omit banana+toy | 0.9818 ✗ | 0.9890 ✗ | 0.9837 | toy | +0.0017 |
| omit banana+pouch | 0.9803 ✗ | 0.9861 | 0.9851 ✗ | toy | -0.0035 |
| omit toy+pouch | 0.9671 | 0.9807 ✗ | 0.9835 ✗ | pouch | +0.0150 |
| seq banana>toy>pouch | 0.9839 | 0.9860 ✗ | 0.9795 ✗ | toy | -0.0011 |
| seq toy>pouch>banana | 0.9640 ✗ | 0.9780 | 0.9764 ✗ | toy | -0.0079 |
| seq pouch>banana>toy | 0.9781 ✗ | 0.9828 ✗ | 0.9834 | pouch | -0.0030 |
| seq banana after toy | 0.9850 ✗ | 0.9905 | 0.9834 | toy | -0.0020 |
| seq pouch last | 0.9699 | 0.9799 | 0.9948 ✗ | pouch | +0.0199 |
| cond yellow | 0.9934 | 0.9903 ✗ | 0.9794 ✗ | banana | -0.0085 |
| cond fruit | 0.9876 | 0.9917 ✗ | 0.9853 ✗ | toy | +0.0009 |
| cond furry | 0.9833 ✗ | 0.9873 | 0.9876 ✗ | pouch | -0.0018 |
| cond cylindrical | 0.9921 ✗ | 0.9907 ✗ | 0.9823 | banana | +0.0091 |
| cond brown | 0.9849 ✗ | 0.9898 | 0.9910 | pouch | -0.0055 |
| cond not edible | 0.9715 ✗ | 0.9799 | 0.9876 | pouch | -0.0122 |
| qty one object | 0.9842 | 0.9907 | 0.9893 | toy | n/a |
| qty two objects | 0.9877 | 0.9901 | 0.9869 | toy | n/a |
| qty one brown | 0.9689 ✗ | 0.9794 | 0.9859 | pouch | -0.0137 |
| qty one fruit | 0.9853 | 0.9891 ✗ | 0.9839 ✗ | toy | +0.0011 |


## Scene 010_pouch

### full_prefix  (inter-base 0.9931)

| preference | sim→banana | sim→toy | sim→pouch | nearest | violation pull |
|---|---|---|---|---|---|
| omit banana | 0.9982 ✗ | 0.9950 | 0.9962 | banana | +0.0026 |
| omit toy | 0.9918 | 0.9990 ✗ | 0.9899 | toy | +0.0082 |
| omit pouch | 0.9979 | 0.9935 | 0.9990 ✗ | pouch | +0.0033 |
| omit banana+toy | 0.9959 ✗ | 0.9976 ✗ | 0.9944 | toy | +0.0023 |
| omit banana+pouch | 0.9987 ✗ | 0.9928 | 0.9988 ✗ | pouch | +0.0060 |
| omit toy+pouch | 0.9961 | 0.9973 ✗ | 0.9961 ✗ | toy | +0.0006 |
| seq banana>toy>pouch | 0.9982 | 0.9939 ✗ | 0.9977 ✗ | banana | -0.0024 |
| seq toy>pouch>banana | 0.9963 ✗ | 0.9972 | 0.9953 ✗ | toy | -0.0014 |
| seq pouch>banana>toy | 0.9978 ✗ | 0.9933 ✗ | 0.9984 | pouch | -0.0029 |
| seq banana after toy | 0.9993 ✗ | 0.9938 | 0.9972 | banana | +0.0038 |
| seq pouch last | 0.9976 | 0.9901 | 0.9999 ✗ | pouch | +0.0061 |
| cond yellow | 0.9981 | 0.9947 ✗ | 0.9975 ✗ | banana | -0.0021 |
| cond fruit | 0.9921 | 0.9986 ✗ | 0.9896 ✗ | toy | +0.0020 |
| cond furry | 0.9965 ✗ | 0.9975 | 0.9961 ✗ | toy | -0.0012 |
| cond cylindrical | 0.9970 ✗ | 0.9969 ✗ | 0.9964 | banana | +0.0006 |
| cond brown | 0.9973 ✗ | 0.9966 | 0.9971 | banana | +0.0004 |
| cond not edible | 0.9970 ✗ | 0.9967 | 0.9965 | banana | +0.0004 |
| qty one object | 0.9978 | 0.9942 | 0.9985 | pouch | n/a |
| qty two objects | 0.9977 | 0.9925 | 0.9991 | pouch | n/a |
| qty one brown | 0.9970 ✗ | 0.9966 | 0.9968 | banana | +0.0003 |
| qty one fruit | 0.9936 | 0.9981 ✗ | 0.9914 ✗ | toy | +0.0011 |

### image_tokens  (inter-base 0.9929)

| preference | sim→banana | sim→toy | sim→pouch | nearest | violation pull |
|---|---|---|---|---|---|
| omit banana | 0.9982 ✗ | 0.9947 | 0.9959 | banana | +0.0029 |
| omit toy | 0.9920 | 0.9989 ✗ | 0.9899 | toy | +0.0080 |
| omit pouch | 0.9978 | 0.9932 | 0.9990 ✗ | pouch | +0.0035 |
| omit banana+toy | 0.9959 ✗ | 0.9975 ✗ | 0.9943 | toy | +0.0024 |
| omit banana+pouch | 0.9987 ✗ | 0.9926 | 0.9987 ✗ | banana | +0.0062 |
| omit toy+pouch | 0.9961 | 0.9972 ✗ | 0.9960 ✗ | toy | +0.0005 |
| seq banana>toy>pouch | 0.9982 | 0.9939 ✗ | 0.9976 ✗ | banana | -0.0024 |
| seq toy>pouch>banana | 0.9962 ✗ | 0.9970 | 0.9952 ✗ | toy | -0.0013 |
| seq pouch>banana>toy | 0.9978 ✗ | 0.9932 ✗ | 0.9985 | pouch | -0.0030 |
| seq banana after toy | 0.9993 ✗ | 0.9934 | 0.9970 | banana | +0.0041 |
| seq pouch last | 0.9974 | 0.9898 | 0.9999 ✗ | pouch | +0.0063 |
| cond yellow | 0.9981 | 0.9946 ✗ | 0.9972 ✗ | banana | -0.0021 |
| cond fruit | 0.9921 | 0.9985 ✗ | 0.9893 ✗ | toy | +0.0018 |
| cond furry | 0.9963 ✗ | 0.9975 | 0.9958 ✗ | toy | -0.0014 |
| cond cylindrical | 0.9968 ✗ | 0.9969 ✗ | 0.9960 | toy | +0.0008 |
| cond brown | 0.9971 ✗ | 0.9966 | 0.9969 | banana | +0.0004 |
| cond not edible | 0.9969 ✗ | 0.9965 | 0.9963 | banana | +0.0005 |
| qty one object | 0.9977 | 0.9941 | 0.9985 | pouch | n/a |
| qty two objects | 0.9976 | 0.9924 | 0.9991 | pouch | n/a |
| qty one brown | 0.9969 ✗ | 0.9965 | 0.9967 | banana | +0.0003 |
| qty one fruit | 0.9937 | 0.9980 ✗ | 0.9913 ✗ | toy | +0.0009 |

### language_tokens  (inter-base 0.9828)

| preference | sim→banana | sim→toy | sim→pouch | nearest | violation pull |
|---|---|---|---|---|---|
| omit banana | 0.9909 ✗ | 0.9822 | 0.9903 | banana | +0.0047 |
| omit toy | 0.9706 | 0.9915 ✗ | 0.9778 | toy | +0.0173 |
| omit pouch | 0.9838 | 0.9886 | 0.9900 ✗ | pouch | +0.0038 |
| omit banana+toy | 0.9778 ✗ | 0.9876 ✗ | 0.9817 | toy | +0.0010 |
| omit banana+pouch | 0.9863 ✗ | 0.9822 | 0.9883 ✗ | pouch | +0.0051 |
| omit toy+pouch | 0.9774 | 0.9876 ✗ | 0.9837 ✗ | toy | +0.0083 |
| seq banana>toy>pouch | 0.9826 | 0.9813 ✗ | 0.9833 ✗ | pouch | -0.0003 |
| seq toy>pouch>banana | 0.9665 ✗ | 0.9868 | 0.9748 ✗ | toy | -0.0161 |
| seq pouch>banana>toy | 0.9767 ✗ | 0.9808 ✗ | 0.9836 | pouch | -0.0049 |
| seq banana after toy | 0.9923 ✗ | 0.9844 | 0.9894 | banana | +0.0054 |
| seq pouch last | 0.9895 | 0.9819 | 0.9976 ✗ | pouch | +0.0119 |
| cond yellow | 0.9922 | 0.9783 ✗ | 0.9894 ✗ | banana | -0.0083 |
| cond fruit | 0.9835 | 0.9923 ✗ | 0.9873 ✗ | toy | +0.0063 |
| cond furry | 0.9871 ✗ | 0.9861 | 0.9908 ✗ | pouch | +0.0028 |
| cond cylindrical | 0.9899 ✗ | 0.9829 ✗ | 0.9906 | pouch | -0.0042 |
| cond brown | 0.9898 ✗ | 0.9850 | 0.9926 | pouch | +0.0010 |
| cond not edible | 0.9850 ✗ | 0.9852 | 0.9876 | pouch | -0.0014 |
| qty one object | 0.9895 | 0.9882 | 0.9929 | pouch | n/a |
| qty two objects | 0.9897 | 0.9820 | 0.9932 | pouch | n/a |
| qty one brown | 0.9842 ✗ | 0.9854 | 0.9880 | pouch | -0.0025 |
| qty one fruit | 0.9785 | 0.9917 ✗ | 0.9848 ✗ | toy | +0.0097 |


## Scene 011_pouch

### full_prefix  (inter-base 0.9926)

| preference | sim→banana | sim→toy | sim→pouch | nearest | violation pull |
|---|---|---|---|---|---|
| omit banana | 0.9982 ✗ | 0.9918 | 0.9875 | banana | +0.0085 |
| omit toy | 0.9916 | 0.9995 ✗ | 0.9970 | toy | +0.0052 |
| omit pouch | 0.9907 | 0.9987 | 0.9988 ✗ | pouch | +0.0041 |
| omit banana+toy | 0.9967 ✗ | 0.9962 ✗ | 0.9922 | banana | +0.0042 |
| omit banana+pouch | 0.9976 ✗ | 0.9948 | 0.9911 ✗ | banana | -0.0005 |
| omit toy+pouch | 0.9908 | 0.9991 ✗ | 0.9986 ✗ | toy | +0.0081 |
| seq banana>toy>pouch | 0.9992 | 0.9928 ✗ | 0.9901 ✗ | banana | -0.0078 |
| seq toy>pouch>banana | 0.9957 ✗ | 0.9968 | 0.9940 ✗ | toy | -0.0019 |
| seq pouch>banana>toy | 0.9974 ✗ | 0.9950 ✗ | 0.9919 | banana | +0.0043 |
| seq banana after toy | 0.9997 ✗ | 0.9912 | 0.9887 | banana | +0.0098 |
| seq pouch last | 0.9898 | 0.9976 | 0.9998 ✗ | pouch | +0.0062 |
| cond yellow | 0.9918 | 0.9978 ✗ | 0.9979 ✗ | pouch | +0.0061 |
| cond fruit | 0.9939 | 0.9981 ✗ | 0.9973 ✗ | toy | +0.0038 |
| cond furry | 0.9914 ✗ | 0.9990 | 0.9986 ✗ | toy | -0.0040 |
| cond cylindrical | 0.9917 ✗ | 0.9994 ✗ | 0.9975 | toy | -0.0019 |
| cond brown | 0.9915 ✗ | 0.9992 | 0.9985 | toy | -0.0073 |
| cond not edible | 0.9924 ✗ | 0.9990 | 0.9980 | toy | -0.0061 |
| qty one object | 0.9905 | 0.9991 | 0.9986 | toy | n/a |
| qty two objects | 0.9900 | 0.9989 | 0.9988 | toy | n/a |
| qty one brown | 0.9914 ✗ | 0.9992 | 0.9979 | toy | -0.0071 |
| qty one fruit | 0.9939 | 0.9984 ✗ | 0.9973 ✗ | toy | +0.0039 |

### image_tokens  (inter-base 0.9921)

| preference | sim→banana | sim→toy | sim→pouch | nearest | violation pull |
|---|---|---|---|---|---|
| omit banana | 0.9981 ✗ | 0.9918 | 0.9870 | banana | +0.0087 |
| omit toy | 0.9911 | 0.9995 ✗ | 0.9969 | toy | +0.0055 |
| omit pouch | 0.9901 | 0.9985 | 0.9988 ✗ | pouch | +0.0044 |
| omit banana+toy | 0.9964 ✗ | 0.9962 ✗ | 0.9917 | banana | +0.0046 |
| omit banana+pouch | 0.9973 ✗ | 0.9949 | 0.9908 ✗ | banana | -0.0009 |
| omit toy+pouch | 0.9901 | 0.9989 ✗ | 0.9987 ✗ | toy | +0.0087 |
| seq banana>toy>pouch | 0.9992 | 0.9925 ✗ | 0.9895 ✗ | banana | -0.0082 |
| seq toy>pouch>banana | 0.9953 ✗ | 0.9968 | 0.9937 ✗ | toy | -0.0023 |
| seq pouch>banana>toy | 0.9973 ✗ | 0.9950 ✗ | 0.9915 | banana | +0.0046 |
| seq banana after toy | 0.9997 ✗ | 0.9910 | 0.9881 | banana | +0.0102 |
| seq pouch last | 0.9891 | 0.9973 | 0.9998 ✗ | pouch | +0.0067 |
| cond yellow | 0.9916 | 0.9978 ✗ | 0.9976 ✗ | toy | +0.0061 |
| cond fruit | 0.9936 | 0.9979 ✗ | 0.9971 ✗ | toy | +0.0039 |
| cond furry | 0.9912 ✗ | 0.9990 | 0.9985 ✗ | toy | -0.0042 |
| cond cylindrical | 0.9914 ✗ | 0.9994 ✗ | 0.9971 | toy | -0.0017 |
| cond brown | 0.9912 ✗ | 0.9992 | 0.9983 | toy | -0.0075 |
| cond not edible | 0.9918 ✗ | 0.9989 | 0.9980 | toy | -0.0066 |
| qty one object | 0.9899 | 0.9990 | 0.9985 | toy | n/a |
| qty two objects | 0.9896 | 0.9988 | 0.9988 | toy | n/a |
| qty one brown | 0.9910 ✗ | 0.9992 | 0.9977 | toy | -0.0075 |
| qty one fruit | 0.9935 | 0.9983 ✗ | 0.9972 ✗ | toy | +0.0043 |

### language_tokens  (inter-base 0.9894)

| preference | sim→banana | sim→toy | sim→pouch | nearest | violation pull |
|---|---|---|---|---|---|
| omit banana | 0.9920 ✗ | 0.9800 | 0.9864 | banana | +0.0088 |
| omit toy | 0.9875 | 0.9913 ✗ | 0.9913 | pouch | +0.0019 |
| omit pouch | 0.9877 | 0.9904 | 0.9928 ✗ | pouch | +0.0038 |
| omit banana+toy | 0.9900 ✗ | 0.9823 ✗ | 0.9868 | banana | -0.0006 |
| omit banana+pouch | 0.9883 ✗ | 0.9763 | 0.9846 ✗ | banana | +0.0101 |
| omit toy+pouch | 0.9837 | 0.9830 ✗ | 0.9874 ✗ | pouch | +0.0015 |
| seq banana>toy>pouch | 0.9873 | 0.9723 ✗ | 0.9802 ✗ | banana | -0.0111 |
| seq toy>pouch>banana | 0.9828 ✗ | 0.9722 | 0.9803 ✗ | banana | +0.0094 |
| seq pouch>banana>toy | 0.9817 ✗ | 0.9695 ✗ | 0.9797 | banana | -0.0042 |
| seq banana after toy | 0.9941 ✗ | 0.9808 | 0.9850 | banana | +0.0113 |
| seq pouch last | 0.9893 | 0.9911 | 0.9978 ✗ | pouch | +0.0076 |
| cond yellow | 0.9879 | 0.9946 ✗ | 0.9925 ✗ | toy | +0.0056 |
| cond fruit | 0.9905 | 0.9917 ✗ | 0.9936 ✗ | pouch | +0.0021 |
| cond furry | 0.9859 ✗ | 0.9928 | 0.9918 ✗ | toy | -0.0040 |
| cond cylindrical | 0.9861 ✗ | 0.9940 ✗ | 0.9928 | toy | -0.0028 |
| cond brown | 0.9869 ✗ | 0.9948 | 0.9940 | toy | -0.0076 |
| cond not edible | 0.9863 ✗ | 0.9840 | 0.9894 | pouch | -0.0004 |
| qty one object | 0.9894 | 0.9945 | 0.9950 | pouch | n/a |
| qty two objects | 0.9867 | 0.9942 | 0.9943 | pouch | n/a |
| qty one brown | 0.9852 ✗ | 0.9899 | 0.9922 | pouch | -0.0059 |
| qty one fruit | 0.9886 | 0.9861 ✗ | 0.9904 ✗ | pouch | -0.0004 |


## Scene 012_toy

### full_prefix  (inter-base 0.9935)

| preference | sim→banana | sim→toy | sim→pouch | nearest | violation pull |
|---|---|---|---|---|---|
| omit banana | 0.9976 ✗ | 0.9962 | 0.9969 | banana | +0.0011 |
| omit toy | 0.9982 | 0.9990 ✗ | 0.9932 | toy | +0.0033 |
| omit pouch | 0.9928 | 0.9915 | 0.9991 ✗ | pouch | +0.0070 |
| omit banana+toy | 0.9983 ✗ | 0.9978 ✗ | 0.9953 | banana | +0.0027 |
| omit banana+pouch | 0.9966 ✗ | 0.9951 | 0.9978 ✗ | pouch | +0.0021 |
| omit toy+pouch | 0.9977 | 0.9977 ✗ | 0.9960 ✗ | toy | -0.0008 |
| seq banana>toy>pouch | 0.9991 | 0.9982 ✗ | 0.9911 ✗ | banana | -0.0045 |
| seq toy>pouch>banana | 0.9984 ✗ | 0.9990 | 0.9907 ✗ | toy | -0.0045 |
| seq pouch>banana>toy | 0.9973 ✗ | 0.9961 ✗ | 0.9968 | banana | -0.0001 |
| seq banana after toy | 0.9996 ✗ | 0.9981 | 0.9922 | banana | +0.0045 |
| seq pouch last | 0.9913 | 0.9897 | 0.9998 ✗ | pouch | +0.0093 |
| cond yellow | 0.9984 | 0.9987 ✗ | 0.9890 ✗ | toy | -0.0045 |
| cond fruit | 0.9985 | 0.9991 ✗ | 0.9920 ✗ | toy | -0.0030 |
| cond furry | 0.9985 ✗ | 0.9978 | 0.9944 ✗ | banana | -0.0013 |
| cond cylindrical | 0.9987 ✗ | 0.9991 ✗ | 0.9923 | toy | +0.0066 |
| cond brown | 0.9962 ✗ | 0.9950 | 0.9983 | pouch | -0.0004 |
| cond not edible | 0.9954 ✗ | 0.9942 | 0.9984 | pouch | -0.0010 |
| qty one object | 0.9982 | 0.9976 | 0.9955 | banana | n/a |
| qty two objects | 0.9984 | 0.9979 | 0.9946 | banana | n/a |
| qty one brown | 0.9964 ✗ | 0.9955 | 0.9978 | pouch | -0.0003 |
| qty one fruit | 0.9984 | 0.9989 ✗ | 0.9914 ✗ | toy | -0.0033 |

### image_tokens  (inter-base 0.9935)

| preference | sim→banana | sim→toy | sim→pouch | nearest | violation pull |
|---|---|---|---|---|---|
| omit banana | 0.9976 ✗ | 0.9960 | 0.9969 | banana | +0.0011 |
| omit toy | 0.9982 | 0.9990 ✗ | 0.9931 | toy | +0.0034 |
| omit pouch | 0.9932 | 0.9917 | 0.9992 ✗ | pouch | +0.0067 |
| omit banana+toy | 0.9983 ✗ | 0.9978 ✗ | 0.9951 | banana | +0.0029 |
| omit banana+pouch | 0.9967 ✗ | 0.9950 | 0.9978 ✗ | pouch | +0.0022 |
| omit toy+pouch | 0.9978 | 0.9978 ✗ | 0.9958 ✗ | banana | -0.0010 |
| seq banana>toy>pouch | 0.9991 | 0.9982 ✗ | 0.9912 ✗ | banana | -0.0044 |
| seq toy>pouch>banana | 0.9983 ✗ | 0.9990 | 0.9902 ✗ | toy | -0.0047 |
| seq pouch>banana>toy | 0.9973 ✗ | 0.9960 ✗ | 0.9968 | banana | -0.0002 |
| seq banana after toy | 0.9996 ✗ | 0.9980 | 0.9918 | banana | +0.0047 |
| seq pouch last | 0.9914 | 0.9896 | 0.9998 ✗ | pouch | +0.0094 |
| cond yellow | 0.9985 | 0.9988 ✗ | 0.9895 ✗ | toy | -0.0043 |
| cond fruit | 0.9985 | 0.9991 ✗ | 0.9918 ✗ | toy | -0.0030 |
| cond furry | 0.9985 ✗ | 0.9977 | 0.9946 ✗ | banana | -0.0011 |
| cond cylindrical | 0.9987 ✗ | 0.9991 ✗ | 0.9926 | toy | +0.0064 |
| cond brown | 0.9960 ✗ | 0.9947 | 0.9983 | pouch | -0.0005 |
| cond not edible | 0.9955 ✗ | 0.9942 | 0.9984 | pouch | -0.0008 |
| qty one object | 0.9982 | 0.9975 | 0.9957 | banana | n/a |
| qty two objects | 0.9984 | 0.9977 | 0.9947 | banana | n/a |
| qty one brown | 0.9965 ✗ | 0.9954 | 0.9978 | pouch | -0.0002 |
| qty one fruit | 0.9984 | 0.9989 ✗ | 0.9912 ✗ | toy | -0.0034 |

### language_tokens  (inter-base 0.9855)

| preference | sim→banana | sim→toy | sim→pouch | nearest | violation pull |
|---|---|---|---|---|---|
| omit banana | 0.9911 ✗ | 0.9892 | 0.9869 | banana | +0.0030 |
| omit toy | 0.9873 | 0.9912 ✗ | 0.9835 | toy | +0.0058 |
| omit pouch | 0.9702 | 0.9741 | 0.9860 ✗ | pouch | +0.0138 |
| omit banana+toy | 0.9838 ✗ | 0.9855 ✗ | 0.9856 | pouch | -0.0009 |
| omit banana+pouch | 0.9822 ✗ | 0.9834 | 0.9870 ✗ | pouch | +0.0013 |
| omit toy+pouch | 0.9756 | 0.9809 ✗ | 0.9853 ✗ | pouch | +0.0075 |
| seq banana>toy>pouch | 0.9869 | 0.9859 ✗ | 0.9834 ✗ | banana | -0.0023 |
| seq toy>pouch>banana | 0.9741 ✗ | 0.9799 | 0.9805 ✗ | pouch | -0.0027 |
| seq pouch>banana>toy | 0.9776 ✗ | 0.9775 ✗ | 0.9854 | pouch | -0.0078 |
| seq banana after toy | 0.9864 ✗ | 0.9860 | 0.9851 | banana | +0.0009 |
| seq pouch last | 0.9777 | 0.9804 | 0.9956 ✗ | pouch | +0.0165 |
| cond yellow | 0.9874 | 0.9853 ✗ | 0.9676 ✗ | banana | -0.0109 |
| cond fruit | 0.9921 | 0.9928 ✗ | 0.9851 ✗ | toy | -0.0031 |
| cond furry | 0.9898 ✗ | 0.9887 | 0.9836 ✗ | banana | -0.0020 |
| cond cylindrical | 0.9901 ✗ | 0.9879 ✗ | 0.9768 | banana | +0.0122 |
| cond brown | 0.9902 ✗ | 0.9895 | 0.9888 | banana | +0.0011 |
| cond not edible | 0.9820 ✗ | 0.9833 | 0.9886 | pouch | -0.0039 |
| qty one object | 0.9908 | 0.9910 | 0.9834 | toy | n/a |
| qty two objects | 0.9898 | 0.9888 | 0.9833 | banana | n/a |
| qty one brown | 0.9835 ✗ | 0.9845 | 0.9883 | pouch | -0.0029 |
| qty one fruit | 0.9888 | 0.9902 ✗ | 0.9853 ✗ | toy | -0.0011 |


## Scene 013_toy

### full_prefix  (inter-base 0.9921)

| preference | sim→banana | sim→toy | sim→pouch | nearest | violation pull |
|---|---|---|---|---|---|
| omit banana | 0.9985 ✗ | 0.9876 | 0.9865 | banana | +0.0115 |
| omit toy | 0.9912 | 0.9992 ✗ | 0.9974 | toy | +0.0049 |
| omit pouch | 0.9903 | 0.9987 | 0.9984 ✗ | toy | +0.0039 |
| omit banana+toy | 0.9984 ✗ | 0.9914 ✗ | 0.9901 | banana | +0.0048 |
| omit banana+pouch | 0.9988 ✗ | 0.9902 | 0.9892 ✗ | banana | +0.0038 |
| omit toy+pouch | 0.9905 | 0.9992 ✗ | 0.9984 ✗ | toy | +0.0083 |
| seq banana>toy>pouch | 0.9992 | 0.9909 ✗ | 0.9899 ✗ | banana | -0.0088 |
| seq toy>pouch>banana | 0.9964 ✗ | 0.9951 | 0.9942 ✗ | banana | +0.0002 |
| seq pouch>banana>toy | 0.9971 ✗ | 0.9943 ✗ | 0.9932 | banana | +0.0025 |
| seq banana after toy | 0.9997 ✗ | 0.9896 | 0.9894 | banana | +0.0102 |
| seq pouch last | 0.9895 | 0.9985 | 0.9998 ✗ | pouch | +0.0059 |
| cond yellow | 0.9949 | 0.9967 ✗ | 0.9956 ✗ | toy | +0.0012 |
| cond fruit | 0.9970 | 0.9952 ✗ | 0.9938 ✗ | banana | -0.0025 |
| cond furry | 0.9912 ✗ | 0.9978 | 0.9976 ✗ | toy | -0.0034 |
| cond cylindrical | 0.9902 ✗ | 0.9994 ✗ | 0.9981 | toy | -0.0033 |
| cond brown | 0.9887 ✗ | 0.9990 | 0.9984 | toy | -0.0099 |
| cond not edible | 0.9953 ✗ | 0.9968 | 0.9955 | toy | -0.0008 |
| qty one object | 0.9902 | 0.9992 | 0.9981 | toy | n/a |
| qty two objects | 0.9897 | 0.9994 | 0.9985 | toy | n/a |
| qty one brown | 0.9903 ✗ | 0.9991 | 0.9981 | toy | -0.0083 |
| qty one fruit | 0.9961 | 0.9958 ✗ | 0.9945 ✗ | banana | -0.0010 |

### image_tokens  (inter-base 0.9917)

| preference | sim→banana | sim→toy | sim→pouch | nearest | violation pull |
|---|---|---|---|---|---|
| omit banana | 0.9985 ✗ | 0.9876 | 0.9859 | banana | +0.0118 |
| omit toy | 0.9908 | 0.9993 ✗ | 0.9973 | toy | +0.0052 |
| omit pouch | 0.9897 | 0.9986 | 0.9985 ✗ | toy | +0.0043 |
| omit banana+toy | 0.9982 ✗ | 0.9912 ✗ | 0.9895 | banana | +0.0053 |
| omit banana+pouch | 0.9987 ✗ | 0.9901 | 0.9886 ✗ | banana | +0.0036 |
| omit toy+pouch | 0.9900 | 0.9992 ✗ | 0.9983 ✗ | toy | +0.0088 |
| seq banana>toy>pouch | 0.9992 | 0.9907 ✗ | 0.9892 ✗ | banana | -0.0092 |
| seq toy>pouch>banana | 0.9961 ✗ | 0.9952 | 0.9940 ✗ | banana | -0.0001 |
| seq pouch>banana>toy | 0.9970 ✗ | 0.9944 ✗ | 0.9929 | banana | +0.0027 |
| seq banana after toy | 0.9998 ✗ | 0.9893 | 0.9887 | banana | +0.0108 |
| seq pouch last | 0.9889 | 0.9984 | 0.9998 ✗ | pouch | +0.0061 |
| cond yellow | 0.9947 | 0.9966 ✗ | 0.9953 ✗ | toy | +0.0012 |
| cond fruit | 0.9968 | 0.9952 ✗ | 0.9936 ✗ | banana | -0.0023 |
| cond furry | 0.9909 ✗ | 0.9976 | 0.9975 ✗ | toy | -0.0034 |
| cond cylindrical | 0.9901 ✗ | 0.9994 ✗ | 0.9978 | toy | -0.0031 |
| cond brown | 0.9887 ✗ | 0.9990 | 0.9983 | toy | -0.0100 |
| cond not edible | 0.9950 ✗ | 0.9968 | 0.9953 | toy | -0.0010 |
| qty one object | 0.9899 | 0.9992 | 0.9980 | toy | n/a |
| qty two objects | 0.9893 | 0.9994 | 0.9984 | toy | n/a |
| qty one brown | 0.9901 ✗ | 0.9992 | 0.9979 | toy | -0.0085 |
| qty one fruit | 0.9959 | 0.9958 ✗ | 0.9944 ✗ | banana | -0.0008 |

### language_tokens  (inter-base 0.9866)

| preference | sim→banana | sim→toy | sim→pouch | nearest | violation pull |
|---|---|---|---|---|---|
| omit banana | 0.9908 ✗ | 0.9774 | 0.9854 | banana | +0.0094 |
| omit toy | 0.9841 | 0.9903 ✗ | 0.9888 | toy | +0.0039 |
| omit pouch | 0.9837 | 0.9841 | 0.9899 ✗ | pouch | +0.0060 |
| omit banana+toy | 0.9888 ✗ | 0.9761 ✗ | 0.9838 | banana | -0.0013 |
| omit banana+pouch | 0.9886 ✗ | 0.9752 | 0.9850 ✗ | banana | +0.0116 |
| omit toy+pouch | 0.9831 | 0.9853 ✗ | 0.9893 ✗ | pouch | +0.0042 |
| seq banana>toy>pouch | 0.9839 | 0.9647 ✗ | 0.9777 ✗ | banana | -0.0127 |
| seq toy>pouch>banana | 0.9764 ✗ | 0.9595 | 0.9733 ✗ | banana | +0.0154 |
| seq pouch>banana>toy | 0.9766 ✗ | 0.9579 ✗ | 0.9747 | banana | -0.0075 |
| seq banana after toy | 0.9938 ✗ | 0.9802 | 0.9825 | banana | +0.0125 |
| seq pouch last | 0.9863 | 0.9888 | 0.9975 ✗ | pouch | +0.0099 |
| cond yellow | 0.9900 | 0.9894 ✗ | 0.9895 ✗ | banana | -0.0005 |
| cond fruit | 0.9892 | 0.9811 ✗ | 0.9880 ✗ | banana | -0.0047 |
| cond furry | 0.9857 ✗ | 0.9891 | 0.9895 ✗ | pouch | -0.0015 |
| cond cylindrical | 0.9778 ✗ | 0.9909 ✗ | 0.9824 | toy | +0.0020 |
| cond brown | 0.9813 ✗ | 0.9927 | 0.9868 | toy | -0.0085 |
| cond not edible | 0.9857 ✗ | 0.9806 | 0.9881 | pouch | +0.0014 |
| qty one object | 0.9870 | 0.9940 | 0.9921 | toy | n/a |
| qty two objects | 0.9844 | 0.9927 | 0.9914 | toy | n/a |
| qty one brown | 0.9797 ✗ | 0.9875 | 0.9848 | toy | -0.0064 |
| qty one fruit | 0.9866 | 0.9773 ✗ | 0.9852 ✗ | banana | -0.0054 |


## Scene 014_toy

### full_prefix  (inter-base 0.9937)

| preference | sim→banana | sim→toy | sim→pouch | nearest | violation pull |
|---|---|---|---|---|---|
| omit banana | 0.9988 ✗ | 0.9950 | 0.9978 | banana | +0.0024 |
| omit toy | 0.9895 | 0.9992 ✗ | 0.9887 | toy | +0.0101 |
| omit pouch | 0.9973 | 0.9961 | 0.9979 ✗ | pouch | +0.0012 |
| omit banana+toy | 0.9952 ✗ | 0.9985 ✗ | 0.9942 | toy | +0.0027 |
| omit banana+pouch | 0.9986 ✗ | 0.9950 | 0.9979 ✗ | banana | +0.0033 |
| omit toy+pouch | 0.9922 | 0.9992 ✗ | 0.9918 ✗ | toy | +0.0033 |
| seq banana>toy>pouch | 0.9982 | 0.9957 ✗ | 0.9973 ✗ | banana | -0.0017 |
| seq toy>pouch>banana | 0.9962 ✗ | 0.9979 | 0.9953 ✗ | toy | -0.0022 |
| seq pouch>banana>toy | 0.9972 ✗ | 0.9969 ✗ | 0.9965 | banana | +0.0005 |
| seq banana after toy | 0.9991 ✗ | 0.9951 | 0.9976 | banana | +0.0028 |
| seq pouch last | 0.9982 | 0.9910 | 0.9999 ✗ | pouch | +0.0053 |
| cond yellow | 0.9978 | 0.9956 ✗ | 0.9971 ✗ | banana | -0.0014 |
| cond fruit | 0.9929 | 0.9990 ✗ | 0.9919 ✗ | toy | +0.0025 |
| cond furry | 0.9948 ✗ | 0.9990 | 0.9939 ✗ | toy | -0.0047 |
| cond cylindrical | 0.9942 ✗ | 0.9989 ✗ | 0.9934 | toy | +0.0031 |
| cond brown | 0.9966 ✗ | 0.9978 | 0.9965 | toy | -0.0005 |
| cond not edible | 0.9929 ✗ | 0.9988 | 0.9923 | toy | -0.0026 |
| qty one object | 0.9958 | 0.9982 | 0.9956 | toy | n/a |
| qty two objects | 0.9969 | 0.9974 | 0.9968 | toy | n/a |
| qty one brown | 0.9954 ✗ | 0.9985 | 0.9949 | toy | -0.0013 |
| qty one fruit | 0.9914 | 0.9985 ✗ | 0.9904 ✗ | toy | +0.0030 |

### image_tokens  (inter-base 0.9935)

| preference | sim→banana | sim→toy | sim→pouch | nearest | violation pull |
|---|---|---|---|---|---|
| omit banana | 0.9988 ✗ | 0.9950 | 0.9976 | banana | +0.0025 |
| omit toy | 0.9899 | 0.9992 ✗ | 0.9884 | toy | +0.0101 |
| omit pouch | 0.9974 | 0.9958 | 0.9978 ✗ | pouch | +0.0012 |
| omit banana+toy | 0.9953 ✗ | 0.9985 ✗ | 0.9938 | toy | +0.0031 |
| omit banana+pouch | 0.9987 ✗ | 0.9948 | 0.9978 ✗ | banana | +0.0035 |
| omit toy+pouch | 0.9923 | 0.9991 ✗ | 0.9913 ✗ | toy | +0.0029 |
| seq banana>toy>pouch | 0.9982 | 0.9956 ✗ | 0.9971 ✗ | banana | -0.0019 |
| seq toy>pouch>banana | 0.9962 ✗ | 0.9979 | 0.9950 ✗ | toy | -0.0023 |
| seq pouch>banana>toy | 0.9972 ✗ | 0.9968 ✗ | 0.9963 | banana | +0.0007 |
| seq banana after toy | 0.9991 ✗ | 0.9950 | 0.9973 | banana | +0.0030 |
| seq pouch last | 0.9980 | 0.9903 | 0.9999 ✗ | pouch | +0.0057 |
| cond yellow | 0.9978 | 0.9957 ✗ | 0.9967 ✗ | banana | -0.0016 |
| cond fruit | 0.9930 | 0.9989 ✗ | 0.9913 ✗ | toy | +0.0021 |
| cond furry | 0.9948 ✗ | 0.9991 | 0.9934 ✗ | toy | -0.0049 |
| cond cylindrical | 0.9940 ✗ | 0.9989 ✗ | 0.9927 | toy | +0.0038 |
| cond brown | 0.9966 ✗ | 0.9978 | 0.9961 | toy | -0.0004 |
| cond not edible | 0.9932 ✗ | 0.9987 | 0.9919 | toy | -0.0021 |
| qty one object | 0.9959 | 0.9982 | 0.9952 | toy | n/a |
| qty two objects | 0.9969 | 0.9974 | 0.9965 | toy | n/a |
| qty one brown | 0.9954 ✗ | 0.9985 | 0.9945 | toy | -0.0010 |
| qty one fruit | 0.9917 | 0.9984 ✗ | 0.9899 ✗ | toy | +0.0025 |

### language_tokens  (inter-base 0.9834)

| preference | sim→banana | sim→toy | sim→pouch | nearest | violation pull |
|---|---|---|---|---|---|
| omit banana | 0.9905 ✗ | 0.9851 | 0.9892 | banana | +0.0033 |
| omit toy | 0.9678 | 0.9921 ✗ | 0.9857 | toy | +0.0154 |
| omit pouch | 0.9754 | 0.9907 | 0.9910 ✗ | pouch | +0.0080 |
| omit banana+toy | 0.9752 ✗ | 0.9903 ✗ | 0.9876 | toy | -0.0049 |
| omit banana+pouch | 0.9795 ✗ | 0.9876 | 0.9892 ✗ | pouch | -0.0033 |
| omit toy+pouch | 0.9691 | 0.9901 ✗ | 0.9860 ✗ | toy | +0.0189 |
| seq banana>toy>pouch | 0.9748 | 0.9853 ✗ | 0.9850 ✗ | toy | +0.0104 |
| seq toy>pouch>banana | 0.9650 ✗ | 0.9859 | 0.9822 ✗ | toy | -0.0123 |
| seq pouch>banana>toy | 0.9681 ✗ | 0.9845 ✗ | 0.9863 | pouch | -0.0100 |
| seq banana after toy | 0.9905 ✗ | 0.9861 | 0.9891 | banana | +0.0029 |
| seq pouch last | 0.9768 | 0.9884 | 0.9966 ✗ | pouch | +0.0140 |
| cond yellow | 0.9911 | 0.9834 ✗ | 0.9877 ✗ | banana | -0.0055 |
| cond fruit | 0.9835 | 0.9904 ✗ | 0.9891 ✗ | toy | +0.0062 |
| cond furry | 0.9828 ✗ | 0.9900 | 0.9884 ✗ | toy | -0.0043 |
| cond cylindrical | 0.9881 ✗ | 0.9870 ✗ | 0.9884 | pouch | -0.0009 |
| cond brown | 0.9881 ✗ | 0.9905 | 0.9919 | pouch | -0.0031 |
| cond not edible | 0.9735 ✗ | 0.9885 | 0.9865 | toy | -0.0140 |
| qty one object | 0.9839 | 0.9931 | 0.9916 | toy | n/a |
| qty two objects | 0.9851 | 0.9907 | 0.9910 | pouch | n/a |
| qty one brown | 0.9796 ✗ | 0.9873 | 0.9879 | pouch | -0.0080 |
| qty one fruit | 0.9757 | 0.9900 ✗ | 0.9879 ✗ | toy | +0.0132 |


## Scene 015_toy

### full_prefix  (inter-base 0.9924)

| preference | sim→banana | sim→toy | sim→pouch | nearest | violation pull |
|---|---|---|---|---|---|
| omit banana | 0.9969 ✗ | 0.9972 | 0.9954 | toy | +0.0006 |
| omit toy | 0.9970 | 0.9988 ✗ | 0.9931 | toy | +0.0037 |
| omit pouch | 0.9891 | 0.9916 | 0.9992 ✗ | pouch | +0.0088 |
| omit banana+toy | 0.9976 ✗ | 0.9986 ✗ | 0.9932 | toy | +0.0049 |
| omit banana+pouch | 0.9960 ✗ | 0.9967 | 0.9964 ✗ | toy | -0.0005 |
| omit toy+pouch | 0.9945 | 0.9968 ✗ | 0.9972 ✗ | pouch | +0.0025 |
| seq banana>toy>pouch | 0.9988 | 0.9982 ✗ | 0.9910 ✗ | banana | -0.0042 |
| seq toy>pouch>banana | 0.9982 ✗ | 0.9987 | 0.9914 ✗ | toy | -0.0039 |
| seq pouch>banana>toy | 0.9956 ✗ | 0.9965 ✗ | 0.9960 | toy | +0.0001 |
| seq banana after toy | 0.9993 ✗ | 0.9982 | 0.9910 | banana | +0.0047 |
| seq pouch last | 0.9884 | 0.9906 | 0.9999 ✗ | pouch | +0.0104 |
| cond yellow | 0.9981 | 0.9990 ✗ | 0.9919 ✗ | toy | -0.0026 |
| cond fruit | 0.9975 | 0.9987 ✗ | 0.9933 ✗ | toy | -0.0015 |
| cond furry | 0.9971 ✗ | 0.9978 | 0.9941 ✗ | toy | -0.0021 |
| cond cylindrical | 0.9965 ✗ | 0.9981 ✗ | 0.9953 | toy | +0.0020 |
| cond brown | 0.9926 ✗ | 0.9947 | 0.9985 | pouch | -0.0040 |
| cond not edible | 0.9924 ✗ | 0.9945 | 0.9985 | pouch | -0.0041 |
| qty one object | 0.9949 | 0.9967 | 0.9972 | pouch | n/a |
| qty two objects | 0.9947 | 0.9965 | 0.9974 | pouch | n/a |
| qty one brown | 0.9912 ✗ | 0.9937 | 0.9986 | pouch | -0.0050 |
| qty one fruit | 0.9974 | 0.9985 ✗ | 0.9936 ✗ | toy | -0.0013 |

### image_tokens  (inter-base 0.9922)

| preference | sim→banana | sim→toy | sim→pouch | nearest | violation pull |
|---|---|---|---|---|---|
| omit banana | 0.9969 ✗ | 0.9971 | 0.9952 | toy | +0.0007 |
| omit toy | 0.9970 | 0.9988 ✗ | 0.9931 | toy | +0.0037 |
| omit pouch | 0.9894 | 0.9916 | 0.9992 ✗ | pouch | +0.0087 |
| omit banana+toy | 0.9976 ✗ | 0.9986 ✗ | 0.9928 | toy | +0.0052 |
| omit banana+pouch | 0.9960 ✗ | 0.9966 | 0.9963 ✗ | toy | -0.0005 |
| omit toy+pouch | 0.9946 | 0.9967 ✗ | 0.9970 ✗ | pouch | +0.0023 |
| seq banana>toy>pouch | 0.9987 | 0.9982 ✗ | 0.9912 ✗ | banana | -0.0040 |
| seq toy>pouch>banana | 0.9982 ✗ | 0.9986 | 0.9910 ✗ | toy | -0.0041 |
| seq pouch>banana>toy | 0.9955 ✗ | 0.9964 ✗ | 0.9959 | toy | +0.0000 |
| seq banana after toy | 0.9994 ✗ | 0.9982 | 0.9904 | banana | +0.0051 |
| seq pouch last | 0.9884 | 0.9904 | 0.9999 ✗ | pouch | +0.0105 |
| cond yellow | 0.9981 | 0.9990 ✗ | 0.9918 ✗ | toy | -0.0027 |
| cond fruit | 0.9974 | 0.9987 ✗ | 0.9931 ✗ | toy | -0.0016 |
| cond furry | 0.9970 ✗ | 0.9977 | 0.9941 ✗ | toy | -0.0021 |
| cond cylindrical | 0.9963 ✗ | 0.9980 ✗ | 0.9953 | toy | +0.0019 |
| cond brown | 0.9926 ✗ | 0.9945 | 0.9984 | pouch | -0.0039 |
| cond not edible | 0.9925 ✗ | 0.9945 | 0.9984 | pouch | -0.0039 |
| qty one object | 0.9949 | 0.9967 | 0.9970 | pouch | n/a |
| qty two objects | 0.9946 | 0.9964 | 0.9973 | pouch | n/a |
| qty one brown | 0.9914 ✗ | 0.9938 | 0.9986 | pouch | -0.0048 |
| qty one fruit | 0.9974 | 0.9985 ✗ | 0.9933 ✗ | toy | -0.0015 |

### language_tokens  (inter-base 0.9841)

| preference | sim→banana | sim→toy | sim→pouch | nearest | violation pull |
|---|---|---|---|---|---|
| omit banana | 0.9866 ✗ | 0.9883 | 0.9876 | toy | -0.0013 |
| omit toy | 0.9857 | 0.9929 ✗ | 0.9843 | toy | +0.0079 |
| omit pouch | 0.9650 | 0.9791 | 0.9866 ✗ | pouch | +0.0145 |
| omit banana+toy | 0.9825 ✗ | 0.9885 ✗ | 0.9862 | toy | -0.0008 |
| omit banana+pouch | 0.9818 ✗ | 0.9874 | 0.9875 ✗ | pouch | -0.0028 |
| omit toy+pouch | 0.9718 | 0.9846 ✗ | 0.9865 ✗ | pouch | +0.0138 |
| seq banana>toy>pouch | 0.9871 | 0.9859 ✗ | 0.9806 ✗ | banana | -0.0038 |
| seq toy>pouch>banana | 0.9754 ✗ | 0.9851 | 0.9821 ✗ | toy | -0.0064 |
| seq pouch>banana>toy | 0.9740 ✗ | 0.9797 ✗ | 0.9845 | pouch | -0.0077 |
| seq banana after toy | 0.9820 ✗ | 0.9881 | 0.9855 | toy | -0.0048 |
| seq pouch last | 0.9711 | 0.9823 | 0.9942 ✗ | pouch | +0.0175 |
| cond yellow | 0.9920 | 0.9921 ✗ | 0.9838 ✗ | toy | -0.0040 |
| cond fruit | 0.9900 | 0.9918 ✗ | 0.9856 ✗ | toy | -0.0013 |
| cond furry | 0.9860 ✗ | 0.9872 | 0.9861 ✗ | toy | -0.0011 |
| cond cylindrical | 0.9904 ✗ | 0.9875 ✗ | 0.9846 | banana | +0.0043 |
| cond brown | 0.9818 ✗ | 0.9875 | 0.9918 | pouch | -0.0078 |
| cond not edible | 0.9737 ✗ | 0.9797 | 0.9875 | pouch | -0.0099 |
| qty one object | 0.9840 | 0.9906 | 0.9909 | pouch | n/a |
| qty two objects | 0.9846 | 0.9883 | 0.9903 | pouch | n/a |
| qty one brown | 0.9691 ✗ | 0.9789 | 0.9873 | pouch | -0.0140 |
| qty one fruit | 0.9855 | 0.9883 ✗ | 0.9862 ✗ | toy | +0.0017 |
