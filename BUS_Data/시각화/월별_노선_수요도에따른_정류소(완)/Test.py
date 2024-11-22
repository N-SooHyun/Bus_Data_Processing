import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

# CSV 파일 읽기
df = pd.read_csv('23_성남시_일반노선별_시간대_이용객순위_평일_순위수정.csv', encoding='utf-8-sig')

# 시간대 문자열 가공
df['시간'] = df['시간'].str.replace('시~', ':00 - ').str.replace('시', ':00')
df['시작시간'] = pd.to_datetime(df['시간'].str.split(' - ').str[0], format='%H:%M')

df_stops = pd.read_csv('성남시_일반노선_정류소별_노선통과.csv', encoding='utf-8-sig')

# 1월 데이터를 필터링
df_january = df[df['월'] == 1]

# 시간대별 상위 3개 노선, 하위 3개 노선 추출
top_3_routes_per_time = pd.DataFrame()
bottom_3_routes_per_time = pd.DataFrame()

# 각 시간대에 대해 상위 3개, 하위 3개 노선 추출
for time in df_january['시간'].unique():
    # 해당 시간대 데이터 필터링
    time_data = df_january[df_january['시간'] == time]

    # 상위 3개 노선 (순위 1~3)
    top_3_routes = time_data[time_data['순위'] <= 3]
    top_3_routes_per_time = pd.concat([top_3_routes_per_time, top_3_routes])

    #해당 시간대에 노선 개수(꼴등 구하기)
    num_routes = len(time_data)

    # 하위 3개 순위 구하기: 순위가 (num_routes-1), (num_routes-2), (num_routes-3)인 노선 추출
    bottom_3_routes = time_data[time_data['순위'].isin([num_routes - 2, num_routes - 1, num_routes])]
    bottom_3_routes_per_time = pd.concat([bottom_3_routes_per_time, bottom_3_routes])

    # 상위 3개 노선 번호 출력
    top_3_route_numbers = top_3_routes['노선'].unique()

    # 하위 3개 노선 번호 출력
    bottom_3_route_numbers = bottom_3_routes['노선'].unique()

    # top_3_routes, bottom_3_routes에서 노선번호들을 추출
    top_routes = top_3_routes['노선'].values
    bottom_routes = bottom_3_routes['노선'].values
    top_routes_stops = df_stops[df_stops['노선번호_x'].isin(top_routes)]
    bottom_routes_stops = df_stops[df_stops['노선번호_x'].isin(bottom_routes)]

    #print(top_routes_stops)
    #print(bottom_routes_stops)
    
    # 지도 초기화
    #m = folium.Map(location=[37.420077, 127.126637], zoom_start=12)  # 성남시 중심 좌표를 기준으로 지도 생성

    # 통합 데이터프레임 생성 (상위와 하위 노선 데이터 병합)
    combined_routes_stops = pd.concat([top_routes_stops, bottom_routes_stops]).drop_duplicates()
    #print(combined_routes_stops)
    # '23_성남시_일반노선별_시간대_이용객순위_평일_순위수정.csv' 불러오기
    df_traffic = pd.read_csv('23_성남시_일반노선별_시간대_이용객순위_평일_순위수정.csv', encoding='utf-8-sig')

    # 병합: 노선 번호를 기준으로 combined_routes_stops와 df_traffic 병합
    combined_routes_stops = combined_routes_stops.merge(
        df_traffic[['노선', '이용객수', '배차간격(평일)(분)']].rename(columns={'노선': '노선번호_x'}),
        on='노선번호_x',
        how='left'
    )

    # 이용객수와 배차간격으로 트래픽 계산 (weight 계산)
    # 기본 weight 공식: 이용객수 / 배차간격
    combined_routes_stops['트래픽'] = combined_routes_stops['이용객수'] / combined_routes_stops['배차간격(평일)(분)']
    print(combined_routes_stops)

    
    
    
    
