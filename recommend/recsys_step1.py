import pandas as pd
import numpy as np
from numpy import dot
from numpy.linalg import norm
from math import *

# 코사인 유사도 함수
def cos_sim(A, B):
    return dot(A, B)/(norm(A)*norm(B))
# 유클리드 거리 함수
def euclidean_distance(x, y):
    return sqrt(sum(pow(a - b, 2) for a, b in zip(x, y)))

#### 데이터 불러오는 경로만 바꾸면 됨 ####
matrix = pd.read_csv('/Users/eunsunkim/web_boaz/recsys/fin_3_rating matrix.csv')
matrix.set_index('Unnamed: 0', inplace = True)
matrix = matrix.T

idx = list(matrix.index)
# 기쁨, 분노, 슬픔 순인데 수정함
ems = [[matrix['기쁨'][i], matrix['슬픔'][i], matrix['분노'][i]] for i in idx]

def recsys_step1(emotion_list):
    cos = pd.Series([cos_sim(ems[i], emotion_list) for i in range(len(matrix))])
    cos.index = idx

    ucl = pd.Series([euclidean_distance(ems[i], emotion_list) for i in range(len(matrix))])
    ucl.index = idx

    new = matrix
    new['euclidean distance'] = ucl
    new['cosine sim'] = cos
    new.reset_index(inplace=True, drop=False)
    #new.to_csv('recsys_step1_result.csv')  ### 파일 저장하고 싶으면 넣어도 될 거 같아요
    return new