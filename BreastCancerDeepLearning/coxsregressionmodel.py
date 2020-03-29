from lifelines import CoxPHFitter

import BreastCancerDeepLearning.breastcancerdata as bcd


cph = CoxPHFitter()
cph.fit(bcd.train_df, duration_col='Age at Diagnosis', event_col='Overall Survival (Months)')

cph.print_summary()
