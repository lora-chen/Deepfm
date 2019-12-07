import numpy as np
import csv
import matplotlib.pyplot as plt
import brewer2mpl

PI2=[{'logloss': 0.539, 'auc': 0.8041, 'acc': 0.7558, 'f1': 0.7529, 'spec': 0.7149, 'cm': np.array([[0.834, 0.166],[0.363, 0.637]])},
     {'logloss': 0.5544, 'auc': 0.7876, 'acc': 0.7365, 'f1': 0.734, 'spec': 0.6966, 'cm': np.array([[0.813, 0.187],[0.38 , 0.62 ]])},
     {'logloss': 0.5569, 'auc': 0.7848, 'acc': 0.7393, 'f1': 0.737, 'spec': 0.7011, 'cm': np.array([[0.812, 0.188],[0.372, 0.628]])},
     {'logloss': 0.5548, 'auc': 0.7876, 'acc': 0.736, 'f1': 0.735, 'spec': 0.7057, 'cm': np.array([[0.794, 0.206],[0.352, 0.648]])},
     {'logloss': 0.587, 'auc': 0.7448, 'acc': 0.705, 'f1': 0.7034, 'spec': 0.6687, 'cm': np.array([[0.774, 0.226],[0.401, 0.599]])}
     ]
# ABC= [
#     {'loss': 0.4261},
#
#     {'loss': 0.3662},
#
#     {'loss': 0.3594},
#
#     {'loss': 0.3575},
#
#     {'loss': 0.3569},
#
#     {'loss': 0.3566}
# ,
#     {'loss': 0.3565}
# ,
#     {'loss': 0.3565},
#
#     {'loss': 0.3564},
#
#     {'loss': 0.3564},
#
# ]
ABC = [
    {'loss': 0.4279},
    {'loss': 0.3669},
{'loss': 0.3602 },
{'loss': 0.3580 },
{'loss': 0.3572 },
{'loss': 0.3569},
{'loss': 0.3567},
{'loss': 0.3565 },
{'loss': 0.3564},
{'loss': 0.3563 },
]

sort = [ABC]#FC2,
name = ['DeepFm']#'FC-2'
bmap = brewer2mpl.get_map('Set2', 'qualitative', 3)
colors = bmap.mpl_colors
for j in range(len(sort)):
    (line, n) = (sort[j],name[j])

    x = ['Ep'+ str(ii) for ii in range(10)   ]
    y = []
    for m in [0, 1, 2, 3, 4,5,6,7,8,9]:
        y.append(line[m]['loss'])
    l = plt.plot(x, y, label=n)
plt.legend()
plt.show()