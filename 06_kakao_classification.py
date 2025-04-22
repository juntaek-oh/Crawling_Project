from bs4 import BeautifulSoup
import pandas as pd
import time
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.chrome.options import Options as ChromeOptions
from webdriver_manager.chrome import ChromeDriverManager

# 스크롤 함수
def scroll_to_bottom(driver, max_scrolls):
    scroll_count = 0
    last_height = driver.execute_script("return document.body.scrollHeight")

    while scroll_count < max_scrolls:
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(3)
        new_height = driver.execute_script("return document.body.scrollHeight")
        if new_height == last_height:
            print(f"{scroll_count + 1}번 스크롤 후 더 이상 로드할 콘텐츠 없음")
            time.sleep(1)
            new_height = driver.execute_script("return document.body.scrollHeight")
            if new_height == last_height:
                break
        last_height = new_height
        scroll_count += 1
        print(f"스크롤 {scroll_count}/{max_scrolls} 완료")

    if scroll_count >= max_scrolls:
        print(f"최대 스크롤 횟수({max_scrolls}) 도달")

# 카테고리
category_mapping = {
    'Food':    '2'
    # 'Life':    '3',
    # 'Beauty':  '4',
    # 'Fashion': '5',
    # 'Tech':    '6',
    # 'Kids':    '7',
    # 'Trip':    '8',
    # 'Culture':  '10'
}

# 크롤링 시작
for category_name, category_code in category_mapping.items():
    start = time.time()
    print(f"===== {category_name} 카테고리 크롤링 시작 =====")

    # Selenium 설정
    options = ChromeOptions()
    options.add_argument('lang=ko_KR')
    # options.add_argument('--headless')
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')

    service = ChromeService(executable_path=ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=options)

    # 네이버 쇼핑 라이브 페이지
    url = f'https://shoppinglive.kakao.com/categories?t_src=shopping_live&t_ch=home&t_obj=live_shortcut&categoryId={category_code}'
    driver.get(url)
    time.sleep(2)  # 초기 페이지 로드 대기

    # 제목 리스트
    titles = []

    # 스크롤 실행
    try:
        scroll_to_bottom(driver, max_scrolls=5)
    except Exception as e:
        print(f"스크롤 중 에러 발생: {e}")

    print("스크롤 완료. 제목 수집 시작...")

    # 페이지 소스 파싱
    soup = BeautifulSoup(driver.page_source, 'html.parser')
    title_elements = soup.find_all('span', class_='info_name')  # 클래스 이름 확인 필요

    for element in title_elements:
        # 전체 텍스트에서 자식 span의 텍스트("라이브명 :")를 제거
        screen_out = element.find('span', class_='screen_out')
        if screen_out:
            screen_out.extract()  # 자식 태그 제거
        title = element.get_text(separator=' ', strip=True).replace('\n', ' ')

        if title:
            titles.append((title,category_name))
            if len(titles) % 50 == 0:
                print(f"현재 {len(titles)}개 제목 수집됨")

    # 중복 제거
    unique_titles = list(dict.fromkeys(titles))
    print(f"총 {len(unique_titles)}개 제목 수집 완료 (중복 제거 전: {len(titles)}, 후: {len(unique_titles)})")

    # CSV 저장
    df_titles = pd.DataFrame(unique_titles, columns=['titles','category'])
    file_path = f'./crawling_kakao_data/kakao_shopping_category_{category_name}.csv'
    df_titles.to_csv(file_path, index=False)
    print(f"{len(df_titles)}개의 고유 제목을 {file_path}에 저장 완료")

    # 브라우저 종료
    driver.quit()

    end = time.time()
    print(f"{category_name} 실행 시간: {end - start:.2f}초")
    print(f"===== {category_name} 카테고리 크롤링 완료 =====\n")

    time.sleep(5)

print("모든 카테고리 크롤링 완료!")