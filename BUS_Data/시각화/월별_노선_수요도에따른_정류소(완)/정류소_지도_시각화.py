import pandas as pd
import folium
import math

# 데이터 읽기
file_path = '성남시_일반노선_정류소별_노선통과.csv'
df = pd.read_csv(file_path)

# 6번 노선만 필터링
df_6 = df[df['노선번호_x'] == '6']

# 성남시 중심 좌표 (예시)
center_lat = 37.4292
center_lon = 127.1375

# 지도 객체 생성
m = folium.Map(location=[center_lat, center_lon], zoom_start=12)

# 노선 6번의 경로와 정류소 목록
route_coordinates = []

# 각 정류소의 정보를 정리
for _, row in df_6.iterrows():
    station = {
        'lat': row['y좌표'],
        'lon': row['x좌표'],
        'name': row['정류소명'],
        'traffic': row['통과노선수'],  # 통과 노선 수
        'details': row['상세위치']
    }
    
    # 경로 리스트에 좌표 추가
    route_coordinates.append([station['lat'], station['lon']])
    
    # 마커 크기 및 색상 결정 (크기를 좀 더 크게 증가)
    marker_size = math.sqrt(station['traffic']) * 5  # 기존 2배에서 5배로 크기 증가
    marker_color = 'red' if station['traffic'] > 10 else 'blue'  # 트래픽이 크면 빨간색 마커
    
    # DivIcon을 사용하여 원형 마커 내부에 숫자 추가
    folium.Marker(
        location=[station['lat'], station['lon']],
        icon=folium.DivIcon(
            icon_size=(marker_size * 2, marker_size * 2),  # 원형 마커 크기
            icon_anchor=(marker_size, marker_size),  # 아이콘 중심이 위치하도록
            html=f'<div style="background-color: {marker_color}; width: {marker_size * 2}px; height: {marker_size * 2}px; border-radius: 50%; display: flex; justify-content: center; align-items: center; color: white; font-size: {marker_size}px;">{station["traffic"]}</div>'
        ),
        popup=f"<b>{station['name']}</b><br>통과 노선 수: {station['traffic']}<br>상세위치: {station['details']}"
    ).add_to(m)

# 노선 6번의 경로를 선으로 연결
folium.PolyLine(route_coordinates, color='blue', weight=2.5, opacity=1).add_to(m)

# 지도 저장
output_file = '노선_6번_정류소_트래픽_원형마커_숫자_추가.html'
m.save(output_file)

print(f"6번 노선의 정류소와 경로를 포함한 지도 파일이 생성되었습니다: {output_file}")


########################################################################
import pandas as pd
import folium
import math

# CSV 파일 읽기
df = pd.read_csv('23_성남시_일반노선별_시간대_이용객순위_평일_배차포함_100x.csv', parse_dates=False, encoding='utf-8-sig')
df['시간'] = df['시간'].apply(lambda x: x.replace('시~', ':00 - ').replace('시', ':00'))

# 1월부터 12월까지의 데이터를 반복하여 시각화
for month in range(1, 13):
    # 해당 월 데이터만 필터링
    df_month = df[df['월'] == month]
    df_month['시작시간'] = df_month['시간'].str.split(' - ').str[0]
    df_month['시작시간'] = pd.to_datetime(df_month['시작시간'], format='%H:%M')

    # 상위 3개 노선 필터링
    top_3_routes = df_month[df_month['순위'] <= 3]
    top_3_routes = top_3_routes.sort_values(by='시작시간')

    # 시간대별 최하위 3개 노선 필터링
    bottom_rank_per_time = df_month.groupby('시간')['순위'].max()
    bottom_3_routes = pd.DataFrame()
    for time, bottom_rank in bottom_rank_per_time.items():
        # 해당 시간대에서 하위 3개 순위를 추출
        time_data = df_month[(df_month['시간'] == time) & (df_month['순위'] >= bottom_rank - 2)]
        bottom_3_routes = pd.concat([bottom_3_routes, time_data])

    bottom_3_routes = bottom_3_routes.sort_values(by='시작시간')

# 데이터 읽기
file_path = '성남시_일반노선_정류소별_노선통과.csv'
df = pd.read_csv(file_path)

# 노선 리스트 (상위 3개 + 하위 3개 노선)
top_bottom_routes = top_3_routes['노선'].unique().tolist() + bottom_3_routes['노선'].unique().tolist()

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

# 각 노선별 경로와 정류소 목록
for route in top_bottom_routes:
    # 노선 데이터 필터링
    df_route = df[df['노선번호_x'] == str(route)]  # str()으로 변환하여 노선번호에 맞춰 필터링

    # 노선에 따른 색상 구분
    if route in top_3_routes['노선'].unique():
        # top_3_routes에 해당하는 노선 번호를 찾아서 순위 번호를 생성
        route_label = f'top_{list(top_3_routes["노선"].unique()).index(route) + 1}'
    else:
        # bottom_3_routes에 해당하는 노선 번호를 찾아서 순위 번호를 생성
        route_label = f'bottom_{list(bottom_3_routes["노선"].unique()).index(route) + 1}'

    # route_colors 딕셔너리에 있는 색상 할당
    if route_label in route_colors:
        route_color = route_colors[route_label]
    else:
        route_color = 'gray'  # 기본 색상 (기타 노선 처리)

    # 경로 리스트에 좌표 추가
    route_coordinates = []

    # 각 정류소의 정보를 정리
    for _, row in df_route.iterrows():
        station = {
            'lat': row['y좌표'],
            'lon': row['x좌표'],
            'name': row['정류소명'],
            'traffic': row['통과노선수'],  # 통과 노선 수
            'details': row['상세위치']
        }

        # 경로 리스트에 좌표 추가
        route_coordinates.append([station['lat'], station['lon']])

        # 마커 크기 및 색상 결정 (크기를 좀 더 크게 증가)
        marker_size = math.sqrt(station['traffic']) * 5  # 기존 2배에서 5배로 크기 증가
        marker_color = 'red' if station['traffic'] > 10 else route_color  # 트래픽이 크면 빨간색 마커

        # DivIcon을 사용하여 원형 마커 내부에 숫자 추가
        folium.Marker(
            location=[station['lat'], station['lon']],
            icon=folium.DivIcon(
                icon_size=(marker_size * 2, marker_size * 2),  # 원형 마커 크기
                icon_anchor=(marker_size, marker_size),  # 아이콘 중심이 위치하도록
                html=f'<div style="background-color: {marker_color}; width: {marker_size * 2}px; height: {marker_size * 2}px; border-radius: 50%; display: flex; justify-content: center; align-items: center; color: white; font-size: {marker_size}px;">{station["traffic"]}</div>'
            ),
            popup=f"<b>{station['name']}</b><br>통과 노선 수: {station['traffic']}<br>상세위치: {station['details']}"
        ).add_to(m)

    # 노선 경로를 선으로 연결
    folium.PolyLine(route_coordinates, color=route_color, weight=2.5, opacity=1).add_to(m)

# 지도 저장
output_file = '상위_하위_노선_정류소_트래픽_시각화.html'
m.save(output_file)

print(f"상위 3개 및 하위 3개 노선의 정류소와 경로를 포함한 지도 파일이 생성되었습니다: {output_file}")

#########################################################################
#최종본되기 직전
import pandas as pd
import folium
import math

# CSV 파일 읽기
df = pd.read_csv('23_성남시_일반노선별_시간대_이용객순위_평일_순위수정.csv', encoding='utf-8-sig')
df_traffic = df

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

    # 지도 초기화
    m = folium.Map(location=[37.420077, 127.126637], zoom_start=12)  # 성남시 중심 좌표를 기준으로 지도 생성

    # 통합 데이터프레임 생성 (상위와 하위 노선 데이터 병합)
    combined_routes_stops = pd.concat([top_routes_stops, bottom_routes_stops]).drop_duplicates()
    # 정류소마다 마커 생성
    for _, station in combined_routes_stops.iterrows():
        lat, lon = station['y좌표'], station['x좌표']
        traffic = station['통과노선수']
        marker_size = math.sqrt(traffic) * 5  # 마커 크기 계산
        marker_color = 'gray'  # 기본 색상은 회색

        folium.Marker(
            location=[lat, lon],
            icon=folium.DivIcon(
                icon_size=(marker_size * 2, marker_size * 2),  # 원형 마커 크기
                icon_anchor=(marker_size, marker_size),  # 아이콘 중심 위치 조정
                html=f'<div style="background-color: {marker_color}; width: {marker_size * 2}px; height: {marker_size * 2}px; border-radius: 50%; display: flex; justify-content: center; align-items: center; color: white; font-size: {marker_size}px;">{traffic}</div>'
            ),
            popup=f"<b>{station['정류소명']}</b><br>통과 노선 수: {traffic}<br>상세위치: {station['상세위치']}"
        ).add_to(m)
    
    # 노선 색상 설정 (6가지 색상)
    #colors = ['#FF5733', '#FF7F50', '#FFD700', '#1E90FF', '#32CD32', '#8A2BE2']
    # 색상 설정 (고정된 팔레트)
    top_colors = ['red', 'blue', 'green']       # 상위 3개 노선 색상 1, 2, 3
    bottom_colors = ['orange', 'purple', 'cyan']  # 하위 3개 노선 색상 꼴등, 꼴등-1, 꼴등-2

    # 상위 및 하위 노선별 정류소 연결
    for idx, (route_df, colors) in enumerate([(top_routes_stops, top_colors), (bottom_routes_stops, bottom_colors)]):
        unique_routes = route_df['노선번호_x'].unique()  # 노선별로 구분
        for i, route in enumerate(unique_routes):
            # 고정된 색상 선택
            color = colors[i % len(colors)]  # 상위/하위 각각 고유 색상
            
            # 해당 노선의 정류소 데이터 추출
            route_stops = route_df[route_df['노선번호_x'] == route].sort_values('정류소순번')
            
            # 좌표 목록 생성
            coordinates = route_stops[['y좌표', 'x좌표']].values.tolist()

            # PolyLine 추가 (노선 경로)
            folium.PolyLine(
                locations=coordinates,
                color=color,
                weight=5,  # 선 두께
                opacity=0.7,
                tooltip=f"노선: {route}"
            ).add_to(m)

    # 시간대별 지도 저장
    time_str = time.replace(':00 - ', '_to_').replace(':00', '')  # 파일명에 사용할 수 있도록 시간 문자열 변환
    m.save(f"{time_str}_time_map.html")

#####################################################################
#최종본
import pandas as pd
import folium
import math

# CSV 파일 읽기
df = pd.read_csv('23_성남시_일반노선별_시간대_이용객순위_평일_순위수정.csv', encoding='utf-8-sig')
df_traffic = df

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

    # 지도 초기화
    m = folium.Map(location=[37.420077, 127.126637], zoom_start=12)  # 성남시 중심 좌표를 기준으로 지도 생성

    # 통합 데이터프레임 생성 (상위와 하위 노선 데이터 병합)
    combined_routes_stops = pd.concat([top_routes_stops, bottom_routes_stops]).drop_duplicates()
    # 정류소마다 마커 생성
    for _, station in combined_routes_stops.iterrows():
        lat, lon = station['y좌표'], station['x좌표']
        traffic = station['통과노선수']
        marker_size = math.sqrt(traffic) * 3  # 마커 크기 계산
        marker_color = 'gray'  # 기본 색상은 회색

        folium.Marker(
            location=[lat, lon],
            icon=folium.DivIcon(
                icon_size=(marker_size * 2, marker_size * 2),  # 원형 마커 크기
                icon_anchor=(marker_size, marker_size),  # 아이콘 중심 위치 조정
                html=f'<div style="background-color: {marker_color}; width: {marker_size * 2}px; height: {marker_size * 2}px; border-radius: 50%; display: flex; justify-content: center; align-items: center; color: white; font-size: {marker_size}px;">{traffic}</div>'
            ),
            popup=f"<b>{station['정류소명']}</b><br>통과 노선 수: {traffic}<br>상세위치: {station['상세위치']}"
        ).add_to(m)
    
    # 노선 색상 설정 (6가지 색상)
    #colors = ['#FF5733', '#FF7F50', '#FFD700', '#1E90FF', '#32CD32', '#8A2BE2']
    # 색상 설정 (고정된 팔레트)
    top_colors = ['red', 'blue', 'green']       # 상위 3개 노선 색상 1, 2, 3
    bottom_colors = ['orange', 'purple', 'cyan']  # 하위 3개 노선 색상 꼴등, 꼴등-1, 꼴등-2

    # 상위 및 하위 노선별 정류소 연결
    for idx, (route_df, colors) in enumerate([(top_routes_stops, top_colors), (bottom_routes_stops, bottom_colors)]):
        unique_routes = route_df['노선번호_x'].unique()
        for i, route in enumerate(unique_routes):
            # 고정된 색상 선택
            color = colors[i % len(colors)]
            
            # 해당 노선의 정류소 데이터 추출
            route_stops = route_df[route_df['노선번호_x'] == route].sort_values('정류소순번')
            
            # 좌표 목록 생성
            coordinates = route_stops[['y좌표', 'x좌표']].values.tolist()
            
            # 순위수정.csv에서 이용객수와 배차간격 조회
            traffic_data = df_traffic[(df_traffic['노선'] == route) & (df_traffic['시간'] == time) & (df_traffic['월'] == 1)]
            
            if not traffic_data.empty:
                # 이용객수와 배차간격 추출
                passengers = traffic_data.iloc[0]['이용객수']
                interval = traffic_data.iloc[0]['배차간격(평일)(분)']
                
                # 트래픽 계산 및 선 두께 설정
                traffic = passengers / interval
                weight = (traffic / 10) + 3  # 적절히 두께 정규화
                
            # PolyLine 추가 (노선 경로)
            folium.PolyLine(
                locations=coordinates,
                color=color,
                weight=weight,  # 선 두께를 트래픽으로 설정
                opacity=0.7,
                tooltip=f"노선: {route}<br>이용객수: {passengers if not traffic_data.empty else '정보 없음'}<br>배차간격: {interval if not traffic_data.empty else '정보 없음'}분"
            ).add_to(m)
    # 시간대별 지도 저장
    time_str = time.replace(':00 - ', '_to_').replace(':00', '')  # 파일명에 사용할 수 있도록 시간 문자열 변환
    m.save(f"{time_str}_time_map.html")

#############################################################################
#찐 최종본
import pandas as pd
import folium
import math

# CSV 파일 읽기
df = pd.read_csv('23_성남시_일반노선별_시간대_이용객순위_평일_순위수정.csv', encoding='utf-8-sig')
df_traffic = df

# 시간대 문자열 가공
df['시간'] = df['시간'].str.replace('시~', ':00 - ').str.replace('시', ':00')
df['시작시간'] = pd.to_datetime(df['시간'].str.split(' - ').str[0], format='%H:%M')

df_stops = pd.read_csv('성남시_일반노선_정류소별_노선통과.csv', encoding='utf-8-sig')

# 시간대별 상위 3개 노선, 하위 3개 노선 추출
top_3_routes_per_time = pd.DataFrame()
bottom_3_routes_per_time = pd.DataFrame()

# 월별 반복 (1월부터 12월까지)
for month in range(1, 13):
    # 해당 월 데이터 필터링
    df_month = df[df['월'] == month]

    # 각 시간대에 대해 상위 3개, 하위 3개 노선 추출
    for time in df_month['시간'].unique():
        # 해당 시간대 데이터 필터링
        time_data = df_month[df_month['시간'] == time]

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

        # 지도 초기화
        m = folium.Map(location=[37.420077, 127.126637], zoom_start=12)  # 성남시 중심 좌표를 기준으로 지도 생성

        # 통합 데이터프레임 생성 (상위와 하위 노선 데이터 병합)
        combined_routes_stops = pd.concat([top_routes_stops, bottom_routes_stops]).drop_duplicates()
        # 정류소마다 마커 생성
        for _, station in combined_routes_stops.iterrows():
            lat, lon = station['y좌표'], station['x좌표']
            traffic = station['통과노선수']
            marker_size = math.sqrt(traffic) * 3  # 마커 크기 계산
            marker_color = 'gray'  # 기본 색상은 회색

            folium.Marker(
                location=[lat, lon],
                icon=folium.DivIcon(
                    icon_size=(marker_size * 2, marker_size * 2),  # 원형 마커 크기
                    icon_anchor=(marker_size, marker_size),  # 아이콘 중심 위치 조정
                    html=f'<div style="background-color: {marker_color}; width: {marker_size * 2}px; height: {marker_size * 2}px; border-radius: 50%; display: flex; justify-content: center; align-items: center; color: white; font-size: {marker_size}px;">{traffic}</div>'
                ),
                popup=f"<b>{station['정류소명']}</b><br>통과 노선 수: {traffic}<br>상세위치: {station['상세위치']}"
            ).add_to(m)
        
        # 노선 색상 설정 (6가지 색상)
        #colors = ['#FF5733', '#FF7F50', '#FFD700', '#1E90FF', '#32CD32', '#8A2BE2']
        # 색상 설정 (고정된 팔레트)
        top_colors = ['red', 'blue', 'green']       # 상위 3개 노선 색상 1, 2, 3
        bottom_colors = ['orange', 'purple', 'cyan']  # 하위 3개 노선 색상 꼴등, 꼴등-1, 꼴등-2

        # 상위 및 하위 노선별 정류소 연결
        for idx, (route_df, colors) in enumerate([(top_routes_stops, top_colors), (bottom_routes_stops, bottom_colors)]):
            unique_routes = route_df['노선번호_x'].unique()
            for i, route in enumerate(unique_routes):
                # 고정된 색상 선택
                color = colors[i % len(colors)]
                
                # 해당 노선의 정류소 데이터 추출
                route_stops = route_df[route_df['노선번호_x'] == route].sort_values('정류소순번')
                
                # 좌표 목록 생성
                coordinates = route_stops[['y좌표', 'x좌표']].values.tolist()
                
                # 순위수정.csv에서 이용객수와 배차간격 조회
                traffic_data = df_traffic[(df_traffic['노선'] == route) & (df_traffic['시간'] == time) & (df_traffic['월'] == month)]
                
                if not traffic_data.empty:
                    # 이용객수와 배차간격 추출
                    passengers = traffic_data.iloc[0]['이용객수']
                    interval = traffic_data.iloc[0]['배차간격(평일)(분)']
                    
                    # 트래픽 계산 및 선 두께 설정
                    traffic = passengers / interval
                    weight = (traffic / 10) + 3  # 적절히 두께 정규화
                    
                # PolyLine 추가 (노선 경로)
                folium.PolyLine(
                    locations=coordinates,
                    color=color,
                    weight=weight,  # 선 두께를 트래픽으로 설정
                    opacity=0.7,
                    tooltip=f"노선: {route}<br>이용객수: {passengers if not traffic_data.empty else '정보 없음'}<br>배차간격: {interval if not traffic_data.empty else '정보 없음'}분"
                ).add_to(m)
        # 시간대별 지도 저장
        time_str = time.replace(':00 - ', '_to_').replace(':00', '')  # 파일명에 사용할 수 있도록 시간 문자열 변환
        m.save(f"{month}월 {time_str}_time_map.html")


