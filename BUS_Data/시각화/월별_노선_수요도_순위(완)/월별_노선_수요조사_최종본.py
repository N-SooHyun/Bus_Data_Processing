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
df = pd.read_csv('23_성남시_일반노선별_시간대_이용객순위_평일_순위수정.csv', parse_dates=False, encoding='utf-8-sig')
df['시간'] = df['시간'].apply(lambda x: x.replace('시~', ':00 - ').replace('시', ':00'))

# 그래프 설정
sns.set(style="whitegrid")
reset_font()
# 1월부터 12월까지의 데이터를 반복하여 시각화
# 1월부터 12월까지 데이터를 반복하여 시각화
for month in range(1, 13):
    # 해당 월 데이터만 필터링
    df_month = df[df['월'] == month]
    df_month['시작시간'] = df_month['시간'].str.split(' - ').str[0]
    df_month['시작시간'] = pd.to_datetime(df_month['시작시간'], format='%H:%M')

    # 상위 3개 노선 필터링
    top_3_routes = df_month[df_month['순위'] <= 3].sort_values(by='시작시간')

    # 시간대별 하위 3개 노선 추출
    bottom_3_routes = pd.DataFrame()
    for time in df_month['시간'].unique():
        time_data = df_month[df_month['시간'] == time]  # 해당 시간대 데이터 필터링
        num_routes = len(time_data)  # 해당 시간대의 노선 개수
        # 하위 3개 순위는 (num_routes-2), (num_routes-1), num_routes인 노선
        bottom_3_routes_for_time = time_data[time_data['순위'].isin([num_routes - 2, num_routes - 1, num_routes])]
        bottom_3_routes = pd.concat([bottom_3_routes, bottom_3_routes_for_time])
    bottom_3_routes = bottom_3_routes.sort_values(by='시작시간')

    # 시간대별 순위 데이터를 위한 필터링
    bottom_rank_per_time = df_month.groupby('시간')['순위'].apply(lambda x: len(x))
    rank_ranges = [0, 1, 2]
    rank_lines_bottom = {i: pd.DataFrame() for i in rank_ranges}
    rank_lines_top = {i: pd.DataFrame() for i in range(1, 4)}

    for time, num_routes in bottom_rank_per_time.items():
        bottom_rank = num_routes
        for offset in rank_ranges:
            rank_data_bottom = df_month[(df_month['시간'] == time) & 
                                        (df_month['순위'] == bottom_rank - offset)]
            rank_lines_bottom[offset] = pd.concat([rank_lines_bottom[offset], rank_data_bottom])

        for rank in range(1, 4):
            rank_data_top = df_month[(df_month['시간'] == time) & 
                                      (df_month['순위'] == rank)]
            rank_lines_top[rank] = pd.concat([rank_lines_top[rank], rank_data_top])

    # 그래프 설정
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8), sharey=True)

    # 왼쪽 그래프 (산점도)
    for route in top_3_routes['노선'].unique():
        route_data = top_3_routes[top_3_routes['노선'] == route]
        x = route_data['시작시간']
        y = route_data['이용객수']
        ax1.scatter(x, y, label=f"Top: {route} ({len(route_data)})", s=100, edgecolors='w', linewidth=2, marker='o')
        for i in range(len(x)):
            ax1.text(x.iloc[i], y.iloc[i], f'{y.iloc[i]}', ha='center', va='bottom', fontsize=10)

    for route in bottom_3_routes['노선'].unique():
        route_data = bottom_3_routes[bottom_3_routes['노선'] == route]
        x = route_data['시작시간']
        y = route_data['이용객수']
        ax1.scatter(x, y, label=f"Bottom: {route} ({len(route_data)})", s=100, edgecolors='w', linewidth=2, marker='s')
        for i in range(len(x)):
            ax1.text(x.iloc[i], y.iloc[i], f'{y.iloc[i]}', ha='center', va='bottom', fontsize=10)

    ax1.set_title(f"{month} Month - Scatter Plot", fontsize=16)
    ax1.set_xlabel("Time", fontsize=12)
    ax1.set_ylabel("Users", fontsize=12)
    ax1.xaxis.set_major_formatter(FuncFormatter(custom_time_formatter))
    ax1.xaxis.set_major_locator(HourLocator(interval=1))
    ax1.legend(title="Bus Routes", loc="upper left", fontsize=10)
    ax1.tick_params(axis='x', rotation=45)

    # 오른쪽 그래프 (라인 그래프)
    for rank in range(1, 4):
        line_data_top = rank_lines_top[rank].sort_values(by='시작시간')
        x = line_data_top['시작시간']
        y = line_data_top['이용객수']
        ax2.plot(x, y, label=f"Top {rank}", marker='o', markersize=8, linestyle='-', linewidth=2)
        for i in range(len(x)):
            ax2.text(x.iloc[i], y.iloc[i], f'{y.iloc[i]}', ha='center', va='bottom', fontsize=10)

    for offset in rank_ranges:
        line_data_bottom = rank_lines_bottom[offset].sort_values(by='시작시간')
        x = line_data_bottom['시작시간']
        y = line_data_bottom['이용객수']
        ax2.plot(x, y, label=f"Bottom {offset + 1}", marker='s', markersize=8, linestyle='-', linewidth=2)
        for i in range(len(x)):
            ax2.text(x.iloc[i], y.iloc[i], f'{y.iloc[i]}', ha='center', va='bottom', fontsize=10)

    ax2.set_title(f"{month} Month - Line Plot", fontsize=16)
    ax2.set_xlabel("Time", fontsize=12)
    ax2.xaxis.set_major_formatter(FuncFormatter(custom_time_formatter))
    ax2.xaxis.set_major_locator(HourLocator(interval=1))
    ax2.legend(title="Ranks", loc="upper left", fontsize=10)
    ax2.tick_params(axis='x', rotation=45)

    # 그래프 간 여백 조정 및 출력
    plt.tight_layout()
    plt.show()
