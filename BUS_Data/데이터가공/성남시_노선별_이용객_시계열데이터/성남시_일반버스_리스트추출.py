import pandas as pd

# '노선별_정류소_일반버스만.csv' 파일 읽기
df = pd.read_csv('노선별_정류소_일반버스만.csv')

# '노선번호', '기점', '종점', '노선아이디' 컬럼만 추출
df_route = df[['노선번호', '기점', '종점', '노선아이디']]

# '노선아이디'를 기준으로 중복값 제거
df_route_unique = df_route.drop_duplicates(subset='노선아이디')

# 중복을 제거한 결과를 '노선별_정류소_일반버스_목록.csv'로 저장 (UTF-8-SIG 인코딩)
df_route_unique.to_csv('노선별_정류소_일반버스_목록.csv', index=False, encoding='utf-8-sig')

print("파일이 성공적으로 저장되었습니다.")
