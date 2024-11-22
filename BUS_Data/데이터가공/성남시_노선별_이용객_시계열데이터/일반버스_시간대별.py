import pandas as pd

# 1번 파일 ('성남시_일반버스_시계열데이터.csv') 읽기
df_1 = pd.read_csv('성남시_일반버스_시계열데이터.csv', encoding='utf-8-sig')

# 2번 파일 ('성남시_노선_시간대별이용_평일.csv') 읽기
df_2 = pd.read_csv('성남시_노선_시간대별이용_주말.csv', encoding='utf-8-sig')

# 1번 파일에서 '노선번호'만 추출하여 유효한 노선번호 목록을 만든다
valid_routes = df_1['노선번호'].unique()

# 결과를 저장할 빈 DataFrame 생성
result = pd.DataFrame(columns=['연도', '월', '노선', '노선아이디', '기점', '종점', '일시', '시간', '이용객수'])

# 2번 파일에서 '노선' 열을 하나씩 가져와서 1번 파일의 '노선번호'와 비교
for _, row_2 in df_2.iterrows():
    # 2번 파일의 '노선'과 1번 파일의 '노선번호' 비교
    if row_2['노선'] in valid_routes:
        # 1번 파일에서 해당 '노선'에 일치하는 데이터를 찾음
        matching_rows = df_1[df_1['노선번호'] == row_2['노선']]

        # 일치하는 데이터가 있을 경우, 필요한 열만 선택하여 새로운 데이터 생성
        for _, row_1 in matching_rows.iterrows():
            # 새로운 행 생성
            new_row = pd.DataFrame([{
                '연도': row_2['연도'],
                '월': row_2['월'],
                '노선': row_2['노선'],
                '노선아이디': row_1['노선아이디'],
                '기점': row_1['기점'],
                '종점': row_1['종점'],
                '일시': row_2['일시'],
                '시간': row_2['시간'],
                '이용객수': row_2['이용객수']
            }])
            # 결과 DataFrame에 새로운 행 추가 (concat 사용)
            result = pd.concat([result, new_row], ignore_index=True)

# 결과를 '성남시_일반노선_시간대_이용객.csv'로 저장 (UTF-8-SIG 인코딩)
result.to_csv('성남시_일반노선_시간대_이용객_주말.csv', index=False, encoding='utf-8-sig')

print("파일이 성공적으로 저장되었습니다.")
