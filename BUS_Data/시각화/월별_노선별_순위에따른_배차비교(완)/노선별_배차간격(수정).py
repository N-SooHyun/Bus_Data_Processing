##########################################################
#평일 순위에 따른 배차비교
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.dates import DateFormatter, HourLocator
import matplotlib.dates as mdates
from matplotlib.ticker import FuncFormatter

def custom_time_formatter(x, pos):
    hour = int(mdates.num2date(x).strftime('%H'))
    return f"{hour:02}:00 - {hour+1:02}:00" if hour < 23 else "23:00 - 24:00"

# CSV 파일 읽기
df = pd.read_csv('23_성남시_일반노선별_시간대_이용객순위_평일_배차포함_100x.csv', parse_dates=False, encoding='utf-8-sig')
df['시간'] = df['시간'].apply(lambda x: x.replace('시~', ':00 - ').replace('시', ':00'))

# 그래프 설정
sns.set(style="whitegrid")

# 1월부터 12월까지의 데이터를 반복하여 시각화
for month in range(1, 13):
    # 해당 월 데이터만 필터링
    df_month = df[df['월'] == month]
    df_month['시작시간'] = df_month['시간'].str.split(' - ').str[0]
    df_month['시작시간'] = pd.to_datetime(df_month['시작시간'], format='%H:%M')

    # 시간대별 상위 3개 노선 필터링
    top_3_routes = df_month[df_month['순위'] <= 3]
    top_3_routes = top_3_routes.sort_values(by='시작시간')

    # 시간대별 꼴등 순위 찾기 (하위 3개 순위는 len을 이용해 정확히 계산)
    bottom_3_routes = pd.DataFrame()

    for time in df_month['시간'].unique():
        # 해당 시간대 데이터 필터링
        time_data = df_month[df_month['시간'] == time]

        # 해당 시간대에 있는 노선 개수
        num_routes = len(time_data)

        # 하위 3개 순위 구하기: 순위가 (num_routes-2), (num_routes-1), (num_routes)인 노선 추출
        bottom_3_routes_time = time_data[time_data['순위'].isin([num_routes - 2, num_routes - 1, num_routes])]

        # 결과 합치기
        bottom_3_routes = pd.concat([bottom_3_routes, bottom_3_routes_time])

    # 시작 시간 기준으로 정렬
    bottom_3_routes = bottom_3_routes.sort_values(by='시작시간')


    # 그래프 설정
    plt.figure(figsize=(12, 8))

    # 상위 3개 노선 점 찍기
    for route in top_3_routes['노선'].unique():
        route_data = top_3_routes[top_3_routes['노선'] == route]
        x = route_data['시작시간']
        y = route_data['배차간격(평일)(분)']  # 배차 간격으로 변경

        # 상위 3개는 원형 마커로 표시
        plt.scatter(x, y, label=f"Top: {route} (Rank: {route_data['순위'].iloc[0]})", s=100, edgecolors='w', linewidth=2, marker='o')

        # 점 위에 배차 간격 숫자 표시
        for i in range(len(x)):
            plt.text(x.iloc[i], y.iloc[i], f'{y.iloc[i]}', ha='center', va='bottom', fontsize=12, color='black')

    # 하위 3개 노선 점 찍기
    for route in bottom_3_routes['노선'].unique():
        route_data = bottom_3_routes[bottom_3_routes['노선'] == route]
        x = route_data['시작시간']
        y = route_data['배차간격(평일)(분)']  # 배차 간격으로 변경

        # 하위 3개는 사각형 마커로 표시
        plt.scatter(x, y, label=f"Bottom: {route} (Rank: {route_data['순위'].iloc[0]})", s=100, edgecolors='w', linewidth=2, marker='s')

        # 점 위에 배차 간격 숫자 표시
        for i in range(len(x)):
            plt.text(x.iloc[i], y.iloc[i], f'{y.iloc[i]}', ha='center', va='bottom', fontsize=12, color='black')


    # 그래프 제목과 레이블 설정
    plt.title(f"{month} Month - Bus Routes Dispatch Interval", fontsize=16)
    plt.xlabel("Time", fontsize=12)
    plt.ylabel("Dispatch Interval (Minutes)", fontsize=12)
    plt.xticks(rotation=45)

    # x축의 시간만 표시하도록 포맷 설정
    plt.gca().xaxis.set_major_formatter(FuncFormatter(custom_time_formatter))
    plt.gca().xaxis.set_major_locator(HourLocator(interval=1))

    # 범례 설정
    #plt.legend(title="Bus Routes", loc="upper left", fontsize=10)
    plt.legend(title="Bus Routes", loc="upper left", fontsize=10, bbox_to_anchor=(1.05, 1))

    # 그래프 간의 여백을 조정
    plt.tight_layout()

    # 그래프 출력
    plt.show()

###################################################################
#주말 배차 비교 왼쪽 토요일 오른쪽 일요일
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.dates import DateFormatter, HourLocator
import matplotlib.dates as mdates
from matplotlib.ticker import FuncFormatter

def custom_time_formatter(x, pos):
    hour = int(mdates.num2date(x).strftime('%H'))
    return f"{hour:02}:00 - {hour+1:02}:00" if hour < 23 else "23:00 - 24:00"

# CSV 파일 읽기
df = pd.read_csv('23_성남시_일반노선별_시간대_이용객순위_주말_순위수정.csv', parse_dates=False, encoding='utf-8-sig')
df['시간'] = df['시간'].apply(lambda x: x.replace('시~', ':00 - ').replace('시', ':00'))

# 그래프 설정
sns.set(style="whitegrid")

# 1월부터 12월까지의 데이터를 반복하여 시각화
for month in range(1, 13):
    # 해당 월 데이터만 필터링
    df_month = df[df['월'] == month]
    df_month['시작시간'] = df_month['시간'].str.split(' - ').str[0]
    df_month['시작시간'] = pd.to_datetime(df_month['시작시간'], format='%H:%M')

    # 서브 플롯 설정 (2개의 그래프 - 왼쪽: 토요일, 오른쪽: 일요일)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))

    # 토요일 배차 간격
    df_saturday = df_month[['시간', '배차간격(토요일)(분)', '노선', '순위']]
    df_saturday = df_saturday.dropna(subset=['배차간격(토요일)(분)'])
    df_saturday['시작시간'] = pd.to_datetime(df_saturday['시간'].str.split(' - ').str[0], format='%H:%M')

    # 상위 3개 및 하위 3개 노선 구분
    top_3_saturday = df_saturday[df_saturday['순위'] <= 3]
    top_3_saturday = top_3_saturday.sort_values(by='시작시간')

    # 시간대별 하위 3개 순위 찾기 (토요일 데이터)
    bottom_3_saturday = pd.DataFrame()

    for time in df_saturday['시간'].unique():
        # 해당 시간대 데이터 필터링
        time_data = df_saturday[df_saturday['시간'] == time]

        # 해당 시간대에 있는 노선 개수
        num_routes = len(time_data)

        # 하위 3개 순위 구하기: 순위가 (num_routes-2), (num_routes-1), (num_routes)인 노선 추출
        bottom_3_saturday_time = time_data[time_data['순위'].isin([num_routes - 2, num_routes - 1, num_routes])]

        # 결과 합치기
        bottom_3_saturday = pd.concat([bottom_3_saturday, bottom_3_saturday_time])

    # 시작 시간 기준으로 정렬
    bottom_3_saturday = bottom_3_saturday.sort_values(by='시작시간')


    # 토요일 데이터 시각화 (라인 추가)
    for route in top_3_saturday['노선'].unique():
        route_data = top_3_saturday[top_3_saturday['노선'] == route]
        x = route_data['시작시간']
        y = route_data['배차간격(토요일)(분)']
        
        # 라인 추가
        ax1.plot(x, y, label=f"Top: {route} (Rank: {route_data['순위'].iloc[0]})", marker='o', markersize=8, linewidth=2)
        ax1.scatter(x, y, s=100, edgecolors='w', linewidth=2, marker='o')  # 점 추가
        for i in range(len(x)):
            ax1.text(x.iloc[i], y.iloc[i], f'{y.iloc[i]}', ha='center', va='bottom', fontsize=12, color='black')

    for route in bottom_3_saturday['노선'].unique():
        route_data = bottom_3_saturday[bottom_3_saturday['노선'] == route]
        x = route_data['시작시간']
        y = route_data['배차간격(토요일)(분)']
        
        # 라인 추가
        ax1.plot(x, y, label=f"Bottom: {route} (Rank: {route_data['순위'].iloc[0]})", marker='s', markersize=8, linewidth=2, linestyle='--')
        ax1.scatter(x, y, s=100, edgecolors='w', linewidth=2, marker='s')  # 점 추가
        for i in range(len(x)):
            ax1.text(x.iloc[i], y.iloc[i], f'{y.iloc[i]}', ha='center', va='bottom', fontsize=12, color='black')


    ax1.set_title(f"{month} Month - Saturday Bus Routes Dispatch Interval", fontsize=16)
    ax1.set_xlabel("Time", fontsize=12)
    ax1.set_ylabel("Dispatch Interval (Minutes)", fontsize=12)
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45)
    ax1.xaxis.set_major_formatter(FuncFormatter(custom_time_formatter))
    ax1.xaxis.set_major_locator(HourLocator(interval=1))
    ax1.legend(title="Bus Routes", loc="upper left", fontsize=10, bbox_to_anchor=(1.05, 1))

    # 일요일 배차 간격
    df_sunday = df_month[['시간', '배차간격(일요일)(분)', '노선', '순위']]
    df_sunday = df_sunday.dropna(subset=['배차간격(일요일)(분)'])
    df_sunday['시작시간'] = pd.to_datetime(df_sunday['시간'].str.split(' - ').str[0], format='%H:%M')

    # 상위 3개 및 하위 3개 노선 구분
    top_3_sunday = df_sunday[df_sunday['순위'] <= 3]
    top_3_sunday = top_3_sunday.sort_values(by='시작시간')

    # 시간대별 하위 3개 순위 찾기 (일요일 데이터)
    bottom_3_sunday = pd.DataFrame()

    for time in df_sunday['시간'].unique():
        # 해당 시간대 데이터 필터링
        time_data = df_sunday[df_sunday['시간'] == time]

        # 해당 시간대에 있는 노선 개수
        num_routes = len(time_data)

        # 하위 3개 순위 구하기: 순위가 (num_routes-2), (num_routes-1), (num_routes)인 노선 추출
        bottom_3_sunday_time = time_data[time_data['순위'].isin([num_routes - 2, num_routes - 1, num_routes])]

        # 결과 합치기
        bottom_3_sunday = pd.concat([bottom_3_sunday, bottom_3_sunday_time])

    # 시작 시간 기준으로 정렬
    bottom_3_sunday = bottom_3_sunday.sort_values(by='시작시간')


    # 일요일 데이터 시각화 (라인 추가)
    for route in top_3_sunday['노선'].unique():
        route_data = top_3_sunday[top_3_sunday['노선'] == route]
        x = route_data['시작시간']
        y = route_data['배차간격(일요일)(분)']
        
        # 라인 추가
        ax2.plot(x, y, label=f"Top: {route} (Rank: {route_data['순위'].iloc[0]})", marker='o', markersize=8, linewidth=2)
        ax2.scatter(x, y, s=100, edgecolors='w', linewidth=2, marker='o')  # 점 추가
        for i in range(len(x)):
            ax2.text(x.iloc[i], y.iloc[i], f'{y.iloc[i]}', ha='center', va='bottom', fontsize=12, color='black')

    for route in bottom_3_sunday['노선'].unique():
        route_data = bottom_3_sunday[bottom_3_sunday['노선'] == route]
        x = route_data['시작시간']
        y = route_data['배차간격(일요일)(분)']
        
        # 라인 추가
        ax2.plot(x, y, label=f"Bottom: {route} (Rank: {route_data['순위'].iloc[0]})", marker='s', markersize=8, linewidth=2, linestyle='--')
        ax2.scatter(x, y, s=100, edgecolors='w', linewidth=2, marker='s')  # 점 추가
        for i in range(len(x)):
            ax2.text(x.iloc[i], y.iloc[i], f'{y.iloc[i]}', ha='center', va='bottom', fontsize=12, color='black')


    ax2.set_title(f"{month} Month - Sunday Bus Routes Dispatch Interval", fontsize=16)
    ax2.set_xlabel("Time", fontsize=12)
    ax2.set_ylabel("Dispatch Interval (Minutes)", fontsize=12)
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45)
    ax2.xaxis.set_major_formatter(FuncFormatter(custom_time_formatter))
    ax2.xaxis.set_major_locator(HourLocator(interval=1))
    ax2.legend(title="Bus Routes", loc="upper left", fontsize=10, bbox_to_anchor=(1.05, 1))

    # 그래프 간의 여백을 조정
    plt.tight_layout()

    # 그래프 출력
    plt.show()
