import BreastCancerDeepLearning.breastcancerdata as bcd
import seaborn as sns

import matplotlib.pyplot as plt

sns.pairplot(bcd.train_di, diag_kind="kde")

plt.show()
