import pandas as pd

# CSV 파일 읽기
input_file = './crawling_data/shoping_category_Trip.csv'  # 입력 CSV 파일 경로
output_file = './crawling_data/shoping_category_Trip.csv'  # 출력 CSV 파일 경로
word_to_add = 'Trip'  # 추가할 단어

# CSV 파일을 데이터프레임으로 로드
df = pd.read_csv(input_file)

# 새로운 열 'category' 추가 및 'Culture' 값 설정
df['category'] = word_to_add

# 수정된 데이터를 새 CSV 파일로 저장
df.to_csv(output_file, index=False, encoding='utf-8-sig')

print(f"{output_file}에 수정된 파일이 저장되었습니다.")