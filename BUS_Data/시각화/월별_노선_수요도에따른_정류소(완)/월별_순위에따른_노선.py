import pandas as pd

# CSV 파일 읽기
df = pd.read_csv('23_성남시_일반노선별_시간대_이용객순위_평일_순위수정.csv', encoding='utf-8-sig')

# 1월 데이터를 필터링
df_january = df[df['월'] == 1]

# 시간대 문자열 가공
df_january['시간'] = df_january['시간'].str.replace('시~', ':00 - ').str.replace('시', ':00')
df_january['시작시간'] = pd.to_datetime(df_january['시간'].str.split(' - ').str[0], format='%H:%M')

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
    
    # 해당 시간대에 노선 개수(꼴등 구하기)
    num_routes = len(time_data)
    
    # 하위 3개 순위 구하기: 순위가 (num_routes-2), (num_routes-1), (num_routes)인 노선 추출
    bottom_3_routes = time_data[time_data['순위'].isin([num_routes - 2, num_routes - 1, num_routes])]
    bottom_3_routes_per_time = pd.concat([bottom_3_routes_per_time, bottom_3_routes])

# 결과 정렬
top_3_routes_per_time = top_3_routes_per_time.sort_values(by='시작시간')
bottom_3_routes_per_time = bottom_3_routes_per_time.sort_values(by='시작시간')

# 상위 3개 노선 출력
print("시간대별 상위 3개 노선:")
for time in top_3_routes_per_time['시간'].unique():
    print(f"\n{time} 시간대:")
    print(top_3_routes_per_time[top_3_routes_per_time['시간'] == time][['순위', '노선', '시간']])

# 하위 3개 노선 출력
print("\n시간대별 하위 3개 노선:")
for time in bottom_3_routes_per_time['시간'].unique():
    print(f"\n{time} 시간대:")
    print(bottom_3_routes_per_time[bottom_3_routes_per_time['시간'] == time][['순위', '노선', '시간']])



###########################################################
    import pandas as pd
import folium
import math

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
    #print(num_routes)
    # 하위 3개 순위 구하기: 순위가 (num_routes-1), (num_routes-2), (num_routes-3)인 노선 추출
    bottom_3_routes = time_data[time_data['순위'].isin([num_routes - 2, num_routes - 1, num_routes])]
    bottom_3_routes_per_time = pd.concat([bottom_3_routes_per_time, bottom_3_routes])
    
    # 상위 3개 노선 번호 출력
    top_3_route_numbers = top_3_routes['노선'].unique()
    #print(f"{time} 시간대 상위 3개 노선 번호: {top_3_route_numbers}")
    
    # 하위 3개 노선 번호 출력
    bottom_3_route_numbers = bottom_3_routes['노선'].unique()
    #print(f"{time} 시간대 하위 3개 노선 번호: {bottom_3_route_numbers}")
    #print(top_3_routes.head())

    # top_3_routes, bottom_3_routes에서 노선번호들을 추출
    top_routes = top_3_routes['노선'].values
    bottom_routes = bottom_3_routes['노선'].values
    
    # top_routes에 있는 각 노선번호별로 df_stops에서 해당하는 데이터를 필터링 후, 합치기
    top_routes_stops = pd.concat([df_stops[df_stops['노선번호_x'] == route] for route in top_routes])
    bottom_routes_stops = pd.concat([df_stops[df_stops['노선번호_x'] == route] for route in bottom_routes])

    # 성남시 중심 좌표 (예시)
    center_lat = 37.4292
    center_lon = 127.1375

    # 지도 객체 생성
    m = folium.Map(location=[center_lat, center_lon], zoom_start=12)

    # 색상 목록 (각 노선별로 다른 색상 지정)
    route_colors = {
        'top_1': 'green',
        'top_2': 'blue',
        'top_3': 'purple',
        'bottom_1': 'orange',
        'bottom_2': 'red',
        'bottom_3': 'yellow'
    }


