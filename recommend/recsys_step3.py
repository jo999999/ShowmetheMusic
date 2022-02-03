import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
from recsys_step1 import recsys_step1
from recsys_step2 import recsys_step2

def recsys_step3(emotion_list, text, weights):
    step1_result = recsys_step1(emotion_list)
    print('recsys_step1 완료')
    step1_result.columns = ['song_index', '기쁨', '분노', '슬픔', 'euclidean distance', 'cosine sim']
    step1_result['song_index'].astype('int64')
    step2_result = recsys_step2(text)
    print('recsys_step2 완료')
    step1_result['song_index'] = step1_result['song_index'].astype('int64')
    merged = pd.merge(step2_result,step1_result, how='inner', left_on='song id', right_on='song_index')

    merged['fin_score'] = merged['cosine sim']*weights[0] + merged['sim']*weights[1]
    recys12 = merged.sort_values(by=['fin_score'], axis=0, ascending=False)
    recys12.to_csv('recys_step3_result.csv', encoding='utf-8-sig')
    return recys12


