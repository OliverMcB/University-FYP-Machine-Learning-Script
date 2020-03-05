import pandas as pd

train_df = pd.read_csv('breast_cancer_data.csv')

train_df.head()

train_df = train_df.dropna()

train_df['Type of Breast Surgery'] = train_df['Type of Breast Surgery'].map(lambda x: {'MASTECTOMY': 1,
                                                                                       'BREAST CONSERVING': 2}.get(x))

train_df['Cancer Type'] = train_df['Cancer Type'].map(lambda x: {'Breast Cancer': 1,
                                                                 'Breast Sarcoma': 2}.get(x))

train_df['Cancer Type Detailed'] = train_df['Cancer Type Detailed']\
    .map(lambda x: {'Breast': 1,
                    'Breast Invasive Ductal Carcinoma': 2,
                    'Breast Invasive Lobular Carcinoma': 3,
                    'Breast Invasive Mixed Mucinous Carcinoma': 4,
                    'Breast Mixed Ductal and Lobular Carcinoma': 5,
                    'Metaplastic Breast Cancer': 6}.get(x))

train_df['Cellularity'] = train_df['Cellularity'].map(lambda x: {'High': 1,
                                                                 'Moderate': 2,
                                                                 'Low': 3}.get(x))

train_df['Chemotherapy'] = train_df['Chemotherapy'].map(lambda x: {'YES': 1,
                                                                   'NO': 2}.get(x))

train_df['Pam50 + Claudin-low subtype'] = train_df['Pam50 + Claudin-low subtype'].map(lambda x: {'Basal': 1,
                                                                                                 'claudin-low': 2,
                                                                                                 'Her2': 3,
                                                                                                 'LumA': 4,
                                                                                                 'LumB': 5,
                                                                                                 'NC': 6,
                                                                                                 'Normal': 7}.get(x))

train_df['ER status measured by IHC'] = train_df['ER status measured by IHC'].map(lambda x: {'Positve': 1,
                                                                                             'Negative': 2}.get(x))

train_df['ER Status'] = train_df['ER Status'].map(lambda x: {'Positive': 1,
                                                             'Negative': 2}.get(x))

train_df['HER2 status measured by SNP6'] = train_df['HER2 status measured by SNP6'].map(lambda x: {'GAIN': 1,
                                                                                                   'NEUTRAL': 2,
                                                                                                   'LOSS': 3}.get(x))

train_df['HER2 Status'] = train_df['HER2 Status'].map(lambda x: {'Positive': 1,
                                                                 'Negative': 2}.get(x))

train_df['Tumor Other Histologic Subtype'] = train_df['Tumor Other Histologic Subtype']\
    .map(lambda x: {'Ductal/NST': 1,
                    'Lobular': 2,
                    'Medullary': 3,
                    'Metaplastic': 4,
                    'Mixed': 5,
                    'Mucinous': 6,
                    'Other': 7,
                    'Tubular/ cribriform': 8}.get(x))

train_df['Hormone Therapy'] = train_df['Hormone Therapy'].map(lambda x: {'YES': 1,
                                                                         'NO': 2}.get(x))

train_df['Inferred Menopausal State'] = train_df['Inferred Menopausal State'].map(lambda x: {'Post': 1,
                                                                                             'Pre': 2}.get(x))

train_df['Integrative Cluster'] = train_df['Integrative Cluster'].map(lambda x: {'1': 1,
                                                                                 '2': 2,
                                                                                 '3': 3,
                                                                                 '5': 5,
                                                                                 '6': 6,
                                                                                 '7': 7,
                                                                                 '8': 8,
                                                                                 '9': 9,
                                                                                 '10': 10,
                                                                                 '4ER-': 11,
                                                                                 '4ER+': 12}.get(x))

train_df['Primary Tumor Laterality'] = train_df['Primary Tumor Laterality'].map(lambda x: {'Left': 1,
                                                                                           'Right': 2}.get(x))

train_df['Oncotree Code'] = train_df['Oncotree Code'].map(lambda x: {'BREAST': 1,
                                                                     'IDC': 2,
                                                                     'ILC': 3,
                                                                     'IMMC': 4,
                                                                     'MBC': 5,
                                                                     'MDLC': 6}.get(x))

train_df['PR Status'] = train_df['PR Status'].map(lambda x: {'Positive': 1,
                                                             'Negative': 2}.get(x))

train_df['Radio Therapy'] = train_df['Radio Therapy'].map(lambda x: {'YES': 1,
                                                                     'NO': 2}.get(x))

train_df['Sample Type'] = train_df['Sample Type'].map(lambda x: {'Primary': 1}.get(x))

train_df['3-Gene classifier subtype'] = train_df['3-Gene classifier subtype'].map(lambda x: {'ER-/HER2-': 1,
                                                                                             'ER+/HER2- High Prolif': 2,
                                                                                             'ER+/HER2- Low Prolif': 3,
                                                                                             'HER2+': 4}.get(x))

train_df = train_df.drop(columns=['Study ID', 'Patient ID', 'Sample ID', 'Number of Samples Per Patient',
                                  'Sample Type', 'Cancer Type'])

train_di = train_df[['Overall Survival (Months)', 'Overall Survival Status', 'Patient\'s Vital Status',
                     'Age at Diagnosis', 'Mutation Count']]

train_df = train_df.drop(columns=['Overall Survival Status', 'Patient\'s Vital Status'])

train_df = train_df.dropna()

train_dataset = train_df.sample(frac=0.8, random_state=0)

test_dataset = train_df.drop(train_dataset.index)

train_stats = train_dataset.describe()
train_stats.pop("Overall Survival (Months)")
train_stats = train_stats.transpose()


def norm(x):
    return (x - train_stats['mean']) / train_stats['std']


train_labels = train_dataset.pop("Overall Survival (Months)")
test_labels = test_dataset.pop("Overall Survival (Months)")

normed_train_data = norm(train_dataset)
normed_test_data = norm(test_dataset)

print(normed_train_data.isna().sum())

# train_X = train_dataset.drop(columns=['Overall Survival (Months)'])
#
# train_X.head()
#
# train_Y = train_dataset[['Overall Survival (Months)']]
#
# train_Y.head()
#
# test_X = test_dataset.drop(columns=['Overall Survival (Months)'])
#
# test_Y = test_dataset[['Overall Survival (Months)']]

