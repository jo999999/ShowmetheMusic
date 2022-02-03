#!pip install konlpy
#!pip install sentence_transformers
import re
from konlpy.tag import Okt
import pandas as pd
import numpy as np
import warnings
from sentence_transformers import SentenceTransformer, util
import torch
warnings.filterwarnings('ignore')

# 데이터 읽고 전처리하기
song_all_info = pd.read_csv('/Users/eunsunkim/web_boaz/recsys/fin_selected_song_all_info.csv')
song_all_info_lyrics = song_all_info[song_all_info['1차 가공'] != '없음']
song_all_info_lyrics.reset_index(inplace=True, drop=True)
x = [', '.join(re.findall('[ㄱ-ㅣ가-힣]+', song_all_info_lyrics['tag count'][i])) for i in range(len(song_all_info_lyrics))]
song_all_info_lyrics['tag_ori'] = x

all_encoding_df = pd.read_csv('/Users/eunsunkim/web_boaz/recsys/all_df_embedding.csv')
all_encoding_df.drop('Unnamed: 0', axis=1, inplace=True)

# 모델 부르기
sbert_model = SentenceTransformer("jhgan/ko-sbert-sts")
okt = Okt()

def cosine(u, v):
    return np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))

def recsys_step2(text):
    # 태그 만들기
    okt_pos = okt.pos(text, norm=True, stem=True)
    okt_filtering = [x for x, y in okt_pos if y in ['Noun', 'Adjective', 'Verb']]

    # input에 따른 인코딩
    test = sbert_model.encode([text], convert_to_tensor=True)
    input_tag_test = sbert_model.encode([', '.join(okt_filtering)], convert_to_tensor=True)
    data_all_test = torch.cat([test,input_tag_test], dim=1)

    # 값 비교해서 저장하기
    sim_list = []
    for i in range(len(all_encoding_df)):
        sim = cosine(data_all_test.detach().numpy(), all_encoding_df.iloc[i])
        sim_list.append(sim[0])
    song_all_info_lyrics['sim'] = sim_list
    #song_all_info_lyrics.to_csv('recsys_step2_result.csv', encoding='utf-8-sig', index=False)

    return song_all_info_lyrics
