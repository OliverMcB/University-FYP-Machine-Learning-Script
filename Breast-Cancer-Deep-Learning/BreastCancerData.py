import pandas as pd

train_df = pd.read_csv('breast_cancer_data.csv')

train_df.head()

train_df = train_df.dropna()

train_df['Type of Breast Surgery'] = train_df['Type of Breast Surgery'].map(lambda x: {1: 'MASTECTOMY',
                                                                                       2: 'BREAST CONSERVING'}.get(x))

train_df['Cancer Type'] = train_df['Cancer Type'].map(lambda x: {1: 'Breast Cancer',
                                                                 2: 'Breast Sarcoma'}.get(x))

train_df['Cancer Type Detailed'] = train_df['Cancer Type Detailed']\
    .map(lambda x: {1: 'Breast',
                    2: 'Breast Invasive Ductal Carcinoma',
                    3: 'Breast Invasive Lobular Carcinoma',
                    4: 'Breast Invasive Mixed Mucinous Carcinoma',
                    5: 'Breast Mixed Ductal and Lobular Carcinoma',
                    6: 'Metaplastic Breast Cancer'}.get(x))

train_df['Cellularity'] = train_df['Cellularity'].map(lambda x: {1: 'High',
                                                                 2: 'Moderate',
                                                                 3: 'Low'}.get(x))

train_df['Chemotherapy'] = train_df['Chemotherapy'].map(lambda x: {1: 'YES',
                                                                   2: 'NO'}.get(x))

train_df['Pam50 + Claudin-low subtype'] = train_df['Pam50 + Claudin-low subtype'].map(lambda x: {1: 'Basal',
                                                                                                 2: 'claudin-low',
                                                                                                 3: 'Her2',
                                                                                                 4: 'LumA',
                                                                                                 5: 'LumB',
                                                                                                 6: 'NC',
                                                                                                 7: 'Normal'}.get(x))

train_df['ER status measured by IHC'] = train_df['ER status measured by IHC'].map(lambda x: {1: 'Positive',
                                                                                             2: 'Negative'}.get(x))

train_df['ER Status'] = train_df['ER Status'].map(lambda x: {1: 'Positive',
                                                             2: 'Negative'}.get(x))

train_df['HER2 status measured by SNP6'] = train_df['HER2 status measured by SNP6'].map(lambda x: {1: 'GAIN',
                                                                                                   2: 'NEUTRAL',
                                                                                                   3: 'LOSS'}.get(x))

train_df['HER2 Status'] = train_df['HER2 Status'].map(lambda x: {1: 'Positive',
                                                                 2: 'Negative'}.get(x))

train_df['Tumor Other Histologic Subtype'] = train_df['Tumor Other Histologic Subtype']\
    .map(lambda x: {1: 'Ductal/NST',
                    2: 'Lobular',
                    3: 'Medullary',
                    4: 'Metaplastic',
                    5: 'Mixed',
                    6: 'Mucinous',
                    7: 'Other',
                    8: 'Tubular/ cribriform'}.get(x))

train_df['Hormone Therapy'] = train_df['Hormone Therapy'].map(lambda x: {1: 'YES',
                                                                         2: 'NO'}.get(x))

train_df['Inferred Menopausal State'] = train_df['Inferred Menopausal State'].map(lambda x: {1: 'Post',
                                                                                             2: 'Pre'}.get(x))

train_df['Integrative Cluster'] = train_df['Integrative Cluster'].map(lambda x: {1: '1',
                                                                                 2: '2',
                                                                                 3: '3',
                                                                                 5: '5',
                                                                                 6: '6',
                                                                                 7: '7',
                                                                                 8: '8',
                                                                                 9: '9',
                                                                                 10: '10',
                                                                                 11: '4ER-',
                                                                                 12: '4ER+'}.get(x))

train_df['Primary Tumor Laterality'] = train_df['Primary Tumor Laterality'].map(lambda x: {1: 'Left',
                                                                                           2: 'Right'}.get(x))

train_df['Oncotree Code'] = train_df['Oncotree Code'].map(lambda x: {1: 'BREAST',
                                                                     2: 'IDC',
                                                                     3: 'ILC',
                                                                     4: 'IMMC',
                                                                     5: 'MBC',
                                                                     6: 'MDLC'}.get(x))

train_df['PR Status'] = train_df['PR Status'].map(lambda x: {1: 'Positive',
                                                             2: 'Negative'}.get(x))

train_df['Radio Therapy'] = train_df['Radio Therapy'].map(lambda x: {1: 'YES',
                                                                     2: 'NO'}.get(x))

train_df['Sample Type'] = train_df['Sample Type'].map(lambda x: {1: 'Primary'}.get(x))

train_df['3-Gene classifier subtype'] = train_df['3-Gene classifier subtype'].map(lambda x: {1: 'ER-/HER2-',
                                                                                             2: 'ER+/HER2- High Prolif',
                                                                                             3: 'ER+/HER2- Low Prolif',
                                                                                             4: 'HER2+'}.get(x))

train_X = train_df.drop(columns=['Study ID', 'Patient ID', 'Sample ID', 'Overall Survival (Months)',
                                 'Overall Survival Status', 'Patient\'s Vital Status'])

train_X.head()

train_Y = train_df[['Overall Survival (Months)']]

train_Y.head()
