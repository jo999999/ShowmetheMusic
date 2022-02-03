# eval.py

# -*- coding: utf-8 -*-
import numpy as np
# import matplotlib.pyplot as plt

import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm, tqdm_notebook

from kobert.utils import get_tokenizer
from kobert.pytorch_kobert import get_pytorch_kobert_model
from transformers import AdamW
from transformers.optimization import get_cosine_schedule_with_warmup
from model import BERTDataset, BERTClassifier
import sys

sys.path.insert(0, 'C:\projects\smtm\nlp_step4\recsys')
from recsys_step4 import all_rs


import gluonnlp as nlp

# softmax 구하는 함수
def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


# device cpu로 불러오기 (local에서 돌리기 때문에)
device = torch.device("cpu")
# load model, map_location=device -> gpu로 모델을 저장하기 때문에 map_location을 통해 cpu로 다시 불러와야 오류가 안 남.
model = torch.load('C:\projects\smtm\nlp_step4\boaz_music_nlp\model.pt', map_location=device)

# get_kobert
bertmodel, vocab = get_pytorch_kobert_model()
tokenizer = get_tokenizer()
tok = nlp.data.BERTSPTokenizer(tokenizer, vocab, lower=False)

label_softmax = []

# Setting parameters
max_len = 64
batch_size = 64
warmup_ratio = 0.1
num_epochs = 20
max_grad_norm = 1
log_interval = 200
learning_rate = 5e-5


# model predict
def predict(predict_sentence):
    data = [predict_sentence, '0']
    dataset_another = [data]

    another_test = BERTDataset(dataset_another, 0, 1, tok, max_len, True, False)
    test_dataloader = torch.utils.data.DataLoader(another_test, batch_size=batch_size, num_workers=0)

    model.eval()

    for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(test_dataloader):
        token_ids = token_ids.long().to(device)
        segment_ids = segment_ids.long().to(device)

        valid_length = valid_length
        label = label.long().to(device)
        out = model(token_ids, valid_length, segment_ids)

        test_eval = []
        for i in out:
            logits = i
            logits = logits.detach().cpu().numpy()
            logits = softmax(logits)  # softmax
            print("softmax : ", logits)
            label_softmax = logits
            if np.argmax(logits) == 0:
                test_eval.append("기쁨")
            elif np.argmax(logits) == 1:
                test_eval.append("슬픔")
            elif np.argmax(logits) == 2:
                test_eval.append("분노")
            elif np.argmax(logits) == 3:
                test_eval.append("중립")

                pass

        print(label_softmax, np.argmax(logits))

        return label_softmax, np.argmax(logits)


# 오늘 하루 너무 힘들었다. 과제랑 공부, 할 일들은 많은데 뜻대로 되지 않았다. 공허한 느낌이 들고 어지러웠다.
def main():
    print('글을 입력해주세요')
    sentence = input("x : ")
    nlp_output_to_rs = predict(sentence)
    print('노래 추천을 받을 때 감정과 가사의 중요도를 알려주세요. ex:"0.3 0.7"')
    weights = list(map(float, input().split()))
    print('nlp 결과 출력', nlp_output_to_rs[0][:3])
    print('정/역방향 선택 (1:정, 2:역)')
    direction = int(input())
    result = all_rs(nlp_output_to_rs[0][:3], sentence, weights, direction)
    result.to_csv('fin_result.csv', encoding='utf-8-sig')


if __name__ == '__main__':
    main()
