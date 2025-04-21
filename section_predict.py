import pickle
import pandas as pd
import  numpy as np
from keras.utils import to_categorical
from konlpy.tag import Okt
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
import re

df = pd.read_csv('./crawling_data/shopping_category_titles.csv')
df.drop_duplicates(inplace=True)
df.reset_index(drop=True, inplace=True)
print(df.head()) # 데이터 앞부분 출력
df.info() # 데이터 정보 출력
print(df.category.value_counts()) # 카테고리별로 데이터 몇개 있는지?

# 제목(입력데이터), 카테고리(라벨)
X = df.titles
Y = df.category

with open('./models/encoder.pickle', 'rb') as f:
    encoder = pickle.load(f)
label = encoder.classes_
print(label)

labeled_y = encoder.transform(Y)
onehot_y = to_categorical(labeled_y)
print(onehot_y)

okt = Okt()
for i in range(len(X)):
    X[i] = re.sub('[^가-힣]', ' ', X[i])
    X[i] = okt.morphs(X[i], stem=True)
print(X)

for idx, sentence in enumerate(X):
    words = []
    for word in sentence:
        if len(word) > 1:
            words.append(word)
    X[idx] = ' '.join(words)

print(X[:10])

with open('./models/token_max_14.pickle', 'rb') as f:
    token = pickle.load(f)
tokened_x = token.texts_to_sequences(X)
print(tokened_x[:5])

for i in range(len(tokened_x)):
    if len(tokened_x[i]) > 14:
        tokened_x[i] = tokened_x[i][:14]
x_pad = pad_sequences(tokened_x, 14)
print(x_pad)

model = load_model('./models/shopping_section_classification_model_0.7682043313980103.h5')
preds = model.predict(x_pad)
print(preds)

# predict_section = [] # 첫번째 예측값
# for pred in preds:
#     predict_section.append(label[np.argmax(pred)])
# print(predict_section)

predict_section = []
for pred in preds:
    most = label[np.argmax(pred)]
    pred[np.argmax(pred)] = 0
    second = label[np.argmax(pred)]
    predict_section.append([most, second])
print(predict_section)

df['predict'] = predict_section
print(df[['category', 'predict']].head(30))

score = model.evaluate(x_pad, onehot_y)
print(score[1])

df['OX'] = 0
for i in range(len(df)):
    if df.loc[i, 'category'] in df.loc[i, 'predict']:
        df.loc[i, 'OX'] = 1
print(df.OX.mean())