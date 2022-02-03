import pandas as pd
import numpy as np
from numpy import dot
from numpy.linalg import norm
import warnings
warnings.filterwarnings('ignore')
from recsys_step1 import recsys_step1
from recsys_step2 import recsys_step2
from recsys_step3 import recsys_step3

def cos_sim(A, B):
    return dot(A, B)/(norm(A)*norm(B))

def emt_detect(emotion_list):
    if emotion_list[0] == max(emotion_list): ## 기쁨
        return 0
    elif emotion_list[1] == max(emotion_list): ## 슬픔
        return 1
    else: ## 분노 
        return 2


def reco(max_emt,data,emt_ch):  #emt_ch가 1이면 정방향 감정을 선택한것, 2이면 역방향 감정을 선택한것 
    if max_emt == 0:    #기쁨 
        em1_idx = data[data['emotion max'].str.contains('행복한')].index.to_list()  # 기쁨일 경우 행복한과 경쾌한을 max_tag로 가진 노래를 선택 
        em2_idx = data[data['emotion max'].str.contains('경쾌한')].index.to_list()
        em3_idx = list(set(em1_idx + em2_idx))
        rec_data = data.iloc[em3_idx]
        return rec_data
    elif max_emt ==1:   #슬픔
        if emt_ch == 1:  ## 정방향 감정
            em1_idx = data[data['emotion max'].str.contains('우울한')].index.to_list()  # 감정에 충실한 데이터셋
            em2_idx = data[data['emotion max'].str.contains('울고싶은')].index.to_list()  
            em12_idx = list(set(em1_idx + em2_idx))
            dataset = data.iloc[em12_idx]
            return dataset
        else:    ##역방향 감정
            em3_idx = data[data['emotion max'].str.contains('편안한')].index.to_list()  # 감정과 반전되는 데이터셋
            em4_idx = data[data['emotion max'].str.contains('잔잔한')].index.to_list() 
            em34_idx = list(set(em3_idx + em4_idx))
            dataset = data.iloc[em34_idx]
            return dataset
    else:   #분노    
        if emt_ch == 1:  ## 정방향 감정
            em1_idx = data[data['emotion max'].str.contains('긴장되는')].index.to_list()  # 감정에 충실한 데이터셋 
            dataset = data.iloc[em1_idx]
            return dataset
        else:    ##역방향 감정
            em1_idx = data[data['emotion max'].str.contains('경쾌한')].index.to_list()  # 감정과 반전되는 데이터셋
            dataset = data.iloc[em1_idx]
            return dataset


def get_score(data):
    score=[]
    for i in data['곡 제목'].iloc[:5]:
        score.append(input(f'{i}의 점수를 입력해주세요! (5점 만점) ', ))
    return score



def cal_cos(data,score):
    rec_data = data.iloc[:5]
    idx = list(data.index)
    rec_data['score'] = 0
    rec_data['score']= score
    mx = rec_data[rec_data['score']==max(score)].index.to_list()
    max_ems = [[rec_data['우울한'][i], rec_data['울고싶은'][i], rec_data['긴장되는'][i],rec_data['무서운'][i],rec_data['잔잔한'][i],rec_data['행복한'][i],rec_data['경쾌한'][i],rec_data['편안한'][i]] for i in mx]
    ems = [[data['우울한'][i], data['울고싶은'][i], data['긴장되는'][i],data['무서운'][i],data['잔잔한'][i],data['행복한'][i],data['경쾌한'][i],data['편안한'][i]] for i in idx]

    data['cos']=0
    for n in range(len(mx)):            
        cos = pd.Series([cos_sim(ems[i], max_ems[n]) for i in range(len(data))])  # 사용자가 가장 높은 점수를 준 곡의 8가지 감정을 기준으로 코사인 유사도를 구한다. 
        cos.index = idx
        data['cos']+=cos
        print(f'{n}번째 코사인 유사도 계산완료')

    data = data[5:]  # 현재 추천한 5곡을 제외하여 데이터 다시 설정 
    data = data.sort_values(by='cos', ascending = False)  # 계산한 코사인 유사도를 기준으로 정렬처리 
    return data



def recsys_step4(emotion_list, text, weights):
    df = recsys_step3(emotion_list, text, weights)
    data = pd.read_csv('recys_step3_result.csv')
    data = data.drop_duplicates(['song id'],keep='first')
    data = data.reset_index()
    data = data.drop(['index'],axis = 1)
    rec_data = reco(emotion_list, data)
    return rec_data

def show(result):
    # 추천 노래
    first = [result['곡 제목'][0], result['가수'][0], result['원가사'][0], result['tag max'][0], result['대분류str'][0]]
    second = [result['곡 제목'][1], result['가수'][1], result['원가사'][1], result['tag max'][1], result['대분류str'][1]]
    third = [result['곡 제목'][2], result['가수'][2], result['원가사'][2], result['tag max'][2], result['대분류str'][2]]
    fourth = [result['곡 제목'][3], result['가수'][3], result['원가사'][3], result['tag max'][3], result['대분류str'][3]]
    fifth = [result['곡 제목'][4], result['가수'][4], result['원가사'][4], result['tag max'][4], result['대분류str'][4]]
    df = [first, second, third, fourth, fifth]
    print()
    print()
    print('**Top1')
    print('곡 제목:', df[0][0])
    print('가수:', df[0][1])
    print('원가사:', df[0][2])
    print('tag max:', df[0][3])
    print('대분류str:', df[0][4])
    print()
    print('**Top2')
    print('곡 제목:', df[1][0])
    print('가수:', df[1][1])
    print('원가사:', df[1][2])
    print('tag max:', df[1][3])
    print('대분류str:', df[1][4])
    print()
    print('**Top3')
    print('곡 제목:', df[2][0])
    print('가수:', df[2][1])
    print('원가사:', df[2][2])
    print('tag max:', df[2][3])
    print('대분류str:', df[2][4])
    print()
    print('**Top4')
    print('곡 제목:', df[3][0])
    print('가수:', df[3][1])
    print('원가사:', df[3][2])
    print('tag max:', df[3][3])
    print('대분류str:', df[3][4])
    print()
    print('**Top5')
    print('곡 제목:', df[4][0])
    print('가수:', df[4][1])
    print('원가사:', df[4][2])
    print('tag max:', df[4][3])
    print('대분류str:', df[4][4])

# 통합
def all_rs(emotion_list, text, weights, emt_ch):
    df = recsys_step3(emotion_list, text, weights) # 함수: recsys_step3까지
    max_emt = emt_detect(emotion_list)
    df = reco(max_emt,df,emt_ch)    # (입력:정/양방향 감정에 따른 노래 선택 입력 받기) 함수: 정/양방향 감정에 따른 노래 선택 반영한 1차 추천
    df.reset_index(inplace=True, drop=True)
    show(df) # 1차 추천 결과인 Top5노래 보여주기 (함수인데 만들었음)
    score = get_score(df)    #평점 입력 받기
    a = int(input('추천 받기 = 1, 추천 멈추기 = 0', ))  # 0입력시 추천 종료
    while a:
        print(score, '길이:', len(score))
        df = cal_cos(df,score)  # 함수: 5점 평점 반영한 재추천
        df.reset_index(inplace=True, drop=True)
        show(df) # 2차 추천 결과인 Top5노래 보여주기 (함수인데 만들었음)
        score = get_score(df)
        a = int(input('추천 받기 = 1, 추천 멈추기 = 0', ))  # 0입력시 추천 종료 (버튼)
    return df


# while a:    # 함수: 5점 평점 반영한 재추천
#         show(df) # 2차 추천 결과인 Top5노래 보여주기 (함수인데 만들었음)
        
#         get_score(df)        # (입력: 5점 평점 받기)