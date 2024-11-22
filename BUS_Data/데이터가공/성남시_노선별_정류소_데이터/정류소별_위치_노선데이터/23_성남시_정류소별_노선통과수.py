import pandas as pd

# 파일 경로 설정
file_1 = '23_성남시_정류소별_일반버스통과수.csv'  # 버스통과수 데이터
file_2 = '성남시정류장_1028.csv'  # 성남시정류장 데이터
output_file = '23_성남시_정류소별_노선통과수.csv'  # 결과 파일

# CSV 파일 읽기
df_bus = pd.read_csv(file_1)
df_station = pd.read_csv(file_2)

# 병합: '정류소ID'와 '정류소아이디' 기준으로
merged_df = pd.merge(df_bus, df_station, how='left', left_on='정류소ID', right_on='정류소아이디')

# 필요한 열만 선택
result_df = merged_df[['연도', '관할지역', '버스유형', '정류소ID', '정류소번호', '정류소명', '통과노선수', 
                       '정류소아이디', '정류소명2', '정류소번호2', 'x좌표', 'y좌표', '행정동', 
                       '정류장명', '상세위치', '시내버스_경유노선번호', '시외버스_경유노선번호']]

# 결과를 새로운 CSV로 저장
result_df.to_csv(output_file, index=False, encoding='utf-8-sig')

print(f"새로운 CSV 파일 '{output_file}'가 생성되었습니다.")
