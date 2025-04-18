import pandas as pd

# CSV 파일 읽기
input_file = './crawling_data/shoping_category_hobby.csv'  # 입력 CSV 파일 경로
output_file = './crawling_data/shoping_category_hobby.csv'  # 출력 CSV 파일 경로
word_to_remove = ',nan'  # 제거하고 싶은 단어

# CSV 파일을 데이터프레임으로 로드
df = pd.read_csv(input_file)

# 모든 열에서 특정 단어 제거
for column in df.columns:
    df[column] = df[column].astype(str).str.replace(word_to_remove, '', regex=False)

# 수정된 데이터를 새 CSV 파일로 저장
df.to_csv(output_file, index=False, encoding='utf-8-sig')

print(f"{output_file}에 수정된 파일이 저장되었습니다.")