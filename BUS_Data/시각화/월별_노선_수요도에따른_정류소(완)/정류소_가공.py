import pandas as pd

# 1번 파일과 2번 파일 경로 지정
file1 = '23_성남시_정류소별_노선통과수_널제거.csv'  # 1번 파일
file2 = '성남시_일반노선39개_정류소.csv'  # 2번 파일

# 1번 파일과 2번 파일을 pandas로 읽기
df1 = pd.read_csv(file1)
df2 = pd.read_csv(file2)

# 1번 파일에서 '정류소아이디'를 '정류소아이디2'로 복사
df1['정류소아이디2'] = df1['정류소아이디']

# 1번 파일의 '정류소아이디2'를 기준으로 추가할 데이터만 추출
df1_filtered = df1[['정류소아이디2', '통과노선수', '상세위치', '시내버스_경유노선번호', '시외버스_경유노선번호']]

# 2번 파일을 기준으로 각 행에 대해 정류소아이디가 일치하는 데이터만 추가
result_rows = []

# 2번 파일의 각 행을 반복하며
for _, row2 in df2.iterrows():
    정류소아이디_2 = row2['정류소아이디']
    
    # 1번 파일에서 해당 정류소아이디2에 맞는 데이터 가져오기
    matching_rows = df1_filtered[df1_filtered['정류소아이디2'] == 정류소아이디_2]
    
    if not matching_rows.empty:
        # 해당 정류소아이디에 맞는 데이터가 존재하면, 해당 행을 result_rows에 추가
        for _, row1 in matching_rows.iterrows():
            new_row = row2.to_dict()  # 2번 파일의 행을 복사
            new_row.update(row1.to_dict())  # 1번 파일에서 데이터를 덧붙임
            result_rows.append(new_row)

# 결과 데이터프레임 만들기
result_df = pd.DataFrame(result_rows)

# 새로운 CSV 파일로 저장
output_file = '성남시_일반노선_정류소별_노선통과.csv'
result_df.to_csv(output_file, index=False, encoding='utf-8-sig')

print(f"새로운 CSV 파일이 생성되었습니다: {output_file}")
