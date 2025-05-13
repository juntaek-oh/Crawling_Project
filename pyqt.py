import sys
import re
import pickle
import numpy as np
import traceback
from PyQt5.QtWidgets import QApplication, QMainWindow
from PyQt5 import uic
from keras.models import load_model
from konlpy.tag import Okt
from keras.preprocessing.sequence import pad_sequences
import os

# 띄어쓰기 교정용 라이브러리
from pykospacing import Spacing

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

form_class = uic.loadUiType("category.ui")[0]

class CategoryClassifier(QMainWindow, form_class):
    def __init__(self):
        super().__init__()
        self.setupUi(self)

        self.model = load_model('./models/shopping_section_classification_model_0.7758001089096069.h5')
        with open('./models/encoder.pickle', 'rb') as f:
            self.encoder = pickle.load(f)
        with open('./models/token_max_14.pickle', 'rb') as f:
            self.token = pickle.load(f)

        self.max_len = 14

        # 띄어쓰기 보정기 초기화
        self.spacer = Spacing()

        self.pushButton_predict.clicked.connect(self.predict_category)
        self.pushButton_clear.clicked.connect(self.clear_input)

    def predict_category(self):
        original_text = self.lineEdit_input.text().strip()

        if not original_text:
            self.label_result.setText("입력값이 없습니다.")
            return

        try:
            okt = Okt()
            best_result = None
            best_probs = None

            custom_split_dict = {
                '로봇청소기': '로봇 청소기',
                '여성신발': '여성 신발',
                '남성가방': '남성 가방',
                '아기옷': '아기 옷',
                '주방용품': '주방 용품',
            }

            spaced_text = self.spacer(original_text)

            for keyword, spaced in custom_split_dict.items():
                original_text = original_text.replace(keyword, spaced)
                spaced_text = spaced_text.replace(keyword, spaced)

            for text in [original_text, spaced_text]:
                cleaned = re.sub('[^가-힣]', ' ', text)
                morphs = okt.morphs(cleaned, stem=True)
                tokens = [word for word in morphs if len(word) > 1]
                if not tokens:
                    continue

                sequence = self.token.texts_to_sequences([' '.join(tokens)])
                padded = pad_sequences(sequence, maxlen=self.max_len)
                prediction = self.model.predict(padded)[0]

                # 상위 2개 인덱스 추출
                top_indices = prediction.argsort()[-2:][::-1]
                top_categories = self.encoder.inverse_transform(top_indices)
                top_probs = prediction[top_indices]

                if best_result is None or top_probs[0] > best_probs[0]:
                    best_result = top_categories
                    best_probs = top_probs

            if best_result is not None:
                result_text = (
                    f"1위: {best_result[0]} ({best_probs[0] * 100:.2f}%)\n"
                    f"2위: {best_result[1]} ({best_probs[1] * 100:.2f}%)"
                )
                self.label_result.setText(result_text)
            else:
                self.label_result.setText("❌ 예측 실패")

        except Exception as e:
            traceback.print_exc()
            self.label_result.setText("❌ 예측 중 오류 발생!")

    def clear_input(self):
        self.lineEdit_input.clear()
        self.label_result.setText("")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    classifier = CategoryClassifier()
    classifier.show()
    sys.exit(app.exec_())
