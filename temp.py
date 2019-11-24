import pandas as pd
import numpy as np
import os
import cv2


stats = pd.read_csv('./Data/ProbData/Statistics.csv', sep=';', header=0)

print(np.mean(stats['mu']))
print(np.sqrt(np.mean(stats['2mu']) - (np.mean(stats['mu']) ** 2)))


# for file in os.listdir('./Data/ProbData/Images'):
#
#     path = f'./Data/ProbData/Images/{file}'
#     img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
#     if img.shape != (1024, 1024):
#         print(img.shape)
#
#     # break

# submission = pd.read_csv('./Submissions/Submission_v004.csv', sep=',', header=0)
# mapping = pd.read_csv('./RawData/kaggle_to_nih_id.csv', sep=',', header=0)
#
# nih_targets = pd.read_csv('./RawData/Data_Entry_2017.csv', sep=',', header=0)
# nih_targets = nih_targets[['Image Index', 'Finding Labels']]
# nih_targets['IsPneumo'] = nih_targets['Finding Labels'].apply(lambda x: 1 if 'Pneumothorax' in x else 0)
#
#
# data = pd.merge(
#     left=submission,
#     right=mapping,
#     left_on='ImageId',
#     right_on='Kaggle_ID',
#     how='left'
# )
#
# data = pd.merge(
#     left=data,
#     right=nih_targets,
#     left_on='NIH_ID',
#     right_on='Image Index',
#     how='left'
# )
#
# # print(data.head())
# # print(len(data))
# # print(data.columns)
#
# # print(submission.head())
#
# sd = data[['ImageId', 'IsPneumo']][(data['IsPneumo'] == 0) & (data['EncodedPixels'] != '-1')].copy()
# submission = pd.merge(
#     left=submission,
#     right=sd,
#     left_on='ImageId',
#     right_on='ImageId',
#     how='left'
# )
#
# submission['IsPneumo'] = submission['IsPneumo'].fillna(1)
# submission['EncodedPixels'] = np.where(submission['IsPneumo'].values == 0,
#                                        '-1',
#                                        submission['EncodedPixels'].values)
#
# submission[['ImageId', 'EncodedPixels']].to_csv('./Submissions/Submission_v005.csv', sep=',', header=True, index=False)