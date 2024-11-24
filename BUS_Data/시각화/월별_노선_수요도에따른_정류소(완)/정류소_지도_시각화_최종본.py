import pandas as pd
import folium
import math

# 1. 데이터 로드 및 전처리 함수
def load_and_preprocess_data(traffic_file, stops_file):
    # CSV 파일 읽기
    df_traffic = pd.read_csv(traffic_file, encoding='utf-8-sig')
    df_traffic['시간'] = df_traffic['시간'].str.replace('시~', ':00 - ').str.replace('시', ':00')
    df_traffic['시작시간'] = pd.to_datetime(df_traffic['시간'].str.split(' - ').str[0], format='%H:%M')
    
    df_stops = pd.read_csv(stops_file, encoding='utf-8-sig')
    
    return df_traffic, df_stops

# 2. 시간대별 상위/하위 노선 추출 함수
def get_top_bottom_routes(df, month):
    top_3_routes_per_time = pd.DataFrame()
    bottom_3_routes_per_time = pd.DataFrame()
    
    df_month = df[df['월'] == month]
    for time in df_month['시간'].unique():
        time_data = df_month[df_month['시간'] == time]
        
        # 상위 3개 노선
        top_3_routes = time_data[time_data['순위'] <= 3]
        top_3_routes_per_time = pd.concat([top_3_routes_per_time, top_3_routes])

        # 하위 3개 노선
        num_routes = len(time_data)
        bottom_3_routes = time_data[time_data['순위'].isin([num_routes - 2, num_routes - 1, num_routes])]
        bottom_3_routes_per_time = pd.concat([bottom_3_routes_per_time, bottom_3_routes])
    
    return top_3_routes_per_time, bottom_3_routes_per_time

# 3. 지도 생성 및 저장 함수
def create_and_save_map(df_traffic, df_stops, top_routes, bottom_routes, month, time, save_path):
    # 지도 초기화
    m = folium.Map(location=[37.420077, 127.126637], zoom_start=12)
    
    top_routes_stops = df_stops[df_stops['노선번호_x'].isin(top_routes['노선'].values)]
    bottom_routes_stops = df_stops[df_stops['노선번호_x'].isin(bottom_routes['노선'].values)]
    
    combined_routes_stops = pd.concat([top_routes_stops, bottom_routes_stops]).drop_duplicates()
    for _, station in combined_routes_stops.iterrows():
        lat, lon = station['y좌표'], station['x좌표']
        traffic = station['통과노선수']
        marker_size = math.sqrt(traffic) * 3
        marker_color = 'gray'
        folium.Marker(
            location=[lat, lon],
            icon=folium.DivIcon(
                icon_size=(marker_size * 2, marker_size * 2),
                icon_anchor=(marker_size, marker_size),
                html=f'<div style="background-color: {marker_color}; width: {marker_size * 2}px; height: {marker_size * 2}px; border-radius: 50%; display: flex; justify-content: center; align-items: center; color: white; font-size: {marker_size}px;">{traffic}</div>'
            ),
            popup=f"<b>{station['정류소명']}</b><br>통과 노선 수: {traffic}<br>상세위치: {station['상세위치']}"
        ).add_to(m)
    
    top_colors = ['red', 'blue', 'green']
    bottom_colors = ['orange', 'purple', 'cyan']
    for route_df, colors in [(top_routes_stops, top_colors), (bottom_routes_stops, bottom_colors)]:
        unique_routes = route_df['노선번호_x'].unique()
        for i, route in enumerate(unique_routes):
            color = colors[i % len(colors)]
            route_stops = route_df[route_df['노선번호_x'] == route].sort_values('정류소순번')
            coordinates = route_stops[['y좌표', 'x좌표']].values.tolist()
            traffic_data = df_traffic[(df_traffic['노선'] == route) & (df_traffic['시간'] == time) & (df_traffic['월'] == month)]
            if not traffic_data.empty:
                passengers = traffic_data.iloc[0]['이용객수']/2
                interval = traffic_data.iloc[0]['배차간격(평일)(분)']
                traffic = passengers / interval
                weight = (traffic / 10)
            else:
                weight = 3
            folium.PolyLine(
                locations=coordinates,
                color=color,
                weight=weight,
                opacity=0.7,
                tooltip=f"노선: {route}<br>이용객수: {passengers if not traffic_data.empty else '정보 없음'}<br>배차간격: {interval if not traffic_data.empty else '정보 없음'}분"
            ).add_to(m)
    
    time_str = time.replace(':00 - ', '_to_').replace(':00', '')
    m.save(f"{save_path}/{month}월_{time_str}_time_map.html")

# 4. 실행 함수
def process_and_generate_maps(traffic_file, stops_file, save_path):
    df_traffic, df_stops = load_and_preprocess_data(traffic_file, stops_file)
    for month in range(1, 13):
        top_routes, bottom_routes = get_top_bottom_routes(df_traffic, month)
        for time in df_traffic[df_traffic['월'] == month]['시간'].unique():
            create_and_save_map(df_traffic, df_stops, top_routes, bottom_routes, month, time, save_path)

# 5. 특정 월과 시간 지도 시각화
def generate_map_for_month_and_time(month, time):
    """
    특정 월과 시간대에 대한 지도 생성 함수.
    :param month: 생성할 월 (int)
    :param time: 생성할 시간대 (str, e.g., "07:00 - 08:00")
    :return: folium 지도 객체
    """
    # 월과 시간대로 필터링
    df_month = df[df['월'] == month]
    time_data = df_month[df_month['시간'] == time]

    if time_data.empty:
        raise ValueError(f"{month}월의 {time} 시간대에 데이터가 없습니다.")

    # 상위 3개, 하위 3개 노선 추출
    top_3_routes = time_data[time_data['순위'] <= 3]
    num_routes = len(time_data)
    bottom_3_routes = time_data[time_data['순위'].isin([num_routes - 2, num_routes - 1, num_routes])]

    top_routes = top_3_routes['노선'].values
    bottom_routes = bottom_3_routes['노선'].values

    top_routes_stops = df_stops[df_stops['노선번호_x'].isin(top_routes)]
    bottom_routes_stops = df_stops[df_stops['노선번호_x'].isin(bottom_routes)]

    # 지도 초기화
    m = folium.Map(location=[37.420077, 127.126637], zoom_start=12)

    # 통합 데이터프레임 생성
    combined_routes_stops = pd.concat([top_routes_stops, bottom_routes_stops]).drop_duplicates()

    # 정류소마다 마커 생성
    for _, station in combined_routes_stops.iterrows():
        lat, lon = station['y좌표'], station['x좌표']
        traffic = station['통과노선수']
        marker_size = math.sqrt(traffic) * 3  # 마커 크기 계산
        marker_color = 'gray'

        folium.Marker(
            location=[lat, lon],
            icon=folium.DivIcon(
                icon_size=(marker_size * 2, marker_size * 2),
                icon_anchor=(marker_size, marker_size),
                html=f'<div style="background-color: {marker_color}; width: {marker_size * 2}px; height: {marker_size * 2}px; border-radius: 50%; display: flex; justify-content: center; align-items: center; color: white; font-size: {marker_size}px;">{traffic}</div>'
            ),
            popup=f"<b>{station['정류소명']}</b><br>통과 노선 수: {traffic}<br>상세위치: {station['상세위치']}"
        ).add_to(m)

    # 색상 설정
    top_colors = ['red', 'blue', 'green']
    bottom_colors = ['orange', 'purple', 'cyan']

    # 상위 및 하위 노선별 정류소 연결
    for route_df, colors in [(top_routes_stops, top_colors), (bottom_routes_stops, bottom_colors)]:
        unique_routes = route_df['노선번호_x'].unique()
        for i, route in enumerate(unique_routes):
            color = colors[i % len(colors)]
            route_stops = route_df[route_df['노선번호_x'] == route].sort_values('정류소순번')
            coordinates = route_stops[['y좌표', 'x좌표']].values.tolist()

            traffic_data = df_traffic[(df_traffic['노선'] == route) & (df_traffic['시간'] == time) & (df_traffic['월'] == month)]

            if not traffic_data.empty:
                passengers = traffic_data.iloc[0]['이용객수']
                interval = traffic_data.iloc[0]['배차간격(평일)(분)']
                traffic = passengers / interval
                weight = (traffic / 10) + 3
            else:
                passengers, interval, weight = '정보 없음', '정보 없음', 3

            folium.PolyLine(
                locations=coordinates,
                color=color,
                weight=weight,
                opacity=0.7,
                tooltip=f"노선: {route}<br>이용객수: {passengers}<br>배차간격: {interval}분"
            ).add_to(m)

    return m
###############################
# CSV 파일 경로 설정
traffic_file = '23_성남시_일반노선별_시간대_이용객순위_평일_순위수정.csv'  # 예: 교통 데이터 파일
stops_file = '성남시_일반노선_정류소별_노선통과.csv'      # 예: 정류소 데이터 파일

# 데이터 로드
df_traffic, df_stops = load_and_preprocess_data(traffic_file, stops_file)
###################################
month = 1
time = "07:00 - 08:00"

# 지도 생성
m = generate_map_for_month_and_time(month, time)
m
