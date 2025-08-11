# Crawling Project

<img width="627" height="429" alt="image" src="https://github.com/user-attachments/assets/e8436bf2-4d77-47a9-9d16-a3b10ad6c434" />

<img width="622" height="490" alt="image" src="https://github.com/user-attachments/assets/f1110972-c389-43e7-bc24-4ed34491e740" />


**웹 데이터 크롤링 + 전처리 + 분류 모델까지 한 번에 수행하는 데이터 수집·분석 프로젝트**  
다양한 웹사이트 데이터를 수집하고, 자동으로 전처리 및 머신러닝 모델을 학습하여 카테고리 예측까지 가능하게 구성되었습니다.

---

## 📸 프로젝트 구성 흐름
1. **데이터 수집** — Selenium, BeautifulSoup 기반 웹 크롤링  
2. **데이터 전처리** — 텍스트 정제, 중복 제거, 형태소 분석  
3. **모델 학습** — scikit-learn 기반 분류 모델 훈련  
4. **카테고리 예측** — PyQt GUI로 예측 결과 확인  

---

## ✨ 주요 기능
- 🌐 **다중 사이트 크롤링**: HTML 구조 분석 후 자동 데이터 수집  
- 🧹 **전처리 자동화**: 특수문자 제거, 형태소 분석, 중복 제거  
- 🤖 **머신러닝 분류**: 다양한 분류 알고리즘 실험  
- 🖥 **GUI 사용성**: PyQt로 간단하게 카테고리 예측 실행  

---

## 🛠 설치 & 실행
필수 라이브러리 설치
pip install -r requirements.txt

GUI 실행 예시
python pyqt.py
---

## 📂 파일 구조
Crawling_Project/
├── crawling_data/ # 수집된 데이터
├── crawling_kakao_data/ # 카카오 뉴스 데이터
├── models/ # 학습된 모델 저장소
├── 01_classification.py # 분류 실행
├── 02_concat.py # 데이터 병합
├── 03_preprocess.py # 전처리
├── 04_model_learning.py # 모델 학습
├── pyqt.py # GUI 실행
└── requirements.txt

---

## 🚨 트러블 슈팅
| 문제 | 해결 방법 |
|------|-----------|
| 크롤링 코드 실행 오류 | HTML 구조 변경 시 XPath, 태그 재확인 |
| 서버 차단(403, CAPTCHA) | User-Agent 변경, 요청 간격 조절 |
| 데이터 인코딩 깨짐 | UTF-8 강제 설정 및 특수문자 필터링 |
| 모델 예측 정확도 저하 | 데이터 전처리 강화 및 하이퍼파라미터 튜닝 |

---

## 📋 기술 스택
- **언어**: Python  
- **크롤링**: Selenium, BeautifulSoup  
- **분석/모델링**: Pandas, scikit-learn  
- **GUI**: PyQt  
- **기타**: joblib, re, konlpy  
