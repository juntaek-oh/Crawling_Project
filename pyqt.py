import sys
import re
import pickle
import traceback
import os
import webbrowser
from urllib.parse import quote_plus

from PyQt5.QtWidgets import QApplication, QMainWindow
from PyQt5 import uic
from keras.models import load_model
from konlpy.tag import Okt
from keras.preprocessing.sequence import pad_sequences

from pykospacing import Spacing

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

form_class = uic.loadUiType("category.ui")[0]

# 네이버 라이브쇼핑 카테고리별 URL 매핑
NAVER_LIVE_CATEGORY_URLS = {
    'Food': 'https://shoppinglive.naver.com/categories/dc:2',
    'Fashion': 'https://shoppinglive.naver.com/categories/dc:3',
    'Life': 'https://shoppinglive.naver.com/categories/dc:4',
    'Trip': 'https://shoppinglive.naver.com/categories/dc:5',
    'Kids': 'https://shoppinglive.naver.com/categories/dc:6',
    'Tech': 'https://shoppinglive.naver.com/categories/dc:7',
    'Hobby': 'https://shoppinglive.naver.com/categories/dc:8',
    'Culture': 'https://shoppinglive.naver.com/categories/dc:9',
    'Beauty': 'https://shoppinglive.naver.com/categories/dc:1'
}

class CategoryClassifier(QMainWindow, form_class):
    def __init__(self):
        super().__init__()
        self.setupUi(self)

        self.model = load_model('C:/Users/SAMSUNG/Downloads/Crawling_Project-main/models/shopping_section_classification_model_0.7751677632331848.h5')
        with open('./models/encoder.pickle', 'rb') as f:
            self.encoder = pickle.load(f)
        with open('./models/token_max_14.pickle', 'rb') as f:
            self.token = pickle.load(f)

        self.max_len = 14

        self.spacer = Spacing()

        self.pushButton_predict.clicked.connect(self.predict_category)

        # 초기화 버튼 → 네이버 라이브쇼핑 카테고리 이동 버튼으로 변경
        self.pushButton_clear.setText("라이브쇼핑 열기")
        self.pushButton_clear.clicked.connect(self.open_naver_shopping)

        self.last_best_category = None

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

                self.last_best_category = best_result[0]
            else:
                self.label_result.setText("❌ 예측 실패")
                self.last_best_category = None

        except Exception as e:
            traceback.print_exc()
            self.label_result.setText("❌ 예측 중 오류 발생!")
            self.last_best_category = None

    def open_naver_shopping(self):
        if not self.last_best_category:
            self.label_result.setText("먼저 예측하기 버튼을 눌러주세요.")
            return

        category = self.last_best_category

        if category in NAVER_LIVE_CATEGORY_URLS:
            url = NAVER_LIVE_CATEGORY_URLS[category]
        else:
            query = quote_plus(category)
            url = f"https://search.shopping.naver.com/search/all?query={query}&cat_id=&frm=NVSHATC"

        webbrowser.open_new_tab(url)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    classifier = CategoryClassifier()
    classifier.show()
    sys.exit(app.exec_())
