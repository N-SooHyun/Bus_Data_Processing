############################################################
#상위권 3명만

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
df = pd.read_csv('23_성남시_일반노선별_시간대_이용객순위_평일.csv', parse_dates=False, encoding='utf-8-sig')
df['시간'] = df['시간'].apply(lambda x: x.replace('시~', ':00 - ').replace('시', ':00'))

# 그래프 설정
sns.set(style="whitegrid")

# 1월부터 12월까지의 데이터를 반복하여 시각화
for month in range(1, 13):
    # 해당 월 데이터만 필터링
    df_month = df[df['월'] == month]
    df_month['시작시간'] = df_month['시간'].str.split(' - ').str[0]
    df_month['시작시간'] = pd.to_datetime(df_month['시작시간'], format='%H:%M')

    
    # 시간대별로 상위 3개 노선 필터링
    top_3_routes = df_month[df_month['순위'] <= 3]
    top_3_routes = top_3_routes.sort_values(by='시작시간')
   
    
    # 그래프 설정
    plt.figure(figsize=(12, 8))
    
    # 각 노선에 대해 점을 찍고, 그 점들을 표시
    for route in top_3_routes['노선'].unique():
        route_data = top_3_routes[top_3_routes['노선'] == route]
        
        # 시간대별로 이용객 수와 시간대
        x = route_data['시작시간']
        y = route_data['이용객수']
        
        # 점 찍기
        plt.scatter(x, y, label=f"{route} ({len(route_data)})", s=100, edgecolors='w', linewidth=2)
        
        # 점 위에 이용객 수 숫자 표시
        for i in range(len(x)):
            plt.text(x.iloc[i], y.iloc[i], f'{y.iloc[i]}', ha='center', va='bottom', fontsize=12, color='black')

    # 그래프 제목과 레이블 설정
    plt.title(f"{month} month Bus Rank", fontsize=16)
    plt.xlabel("time", fontsize=12)
    plt.ylabel("users", fontsize=12)
    plt.xticks(rotation=45)

    # x축의 시간만 표시하도록 포맷 설정
    #plt.gca().xaxis.set_major_formatter(DateFormatter('%H:%M'))
    plt.gca().xaxis.set_major_formatter(FuncFormatter(custom_time_formatter))
    plt.gca().xaxis.set_major_locator(HourLocator(interval=1))

    # 범례 표시
    plt.legend(title="month number", loc="upper left")

    # 그래프 표시
    plt.tight_layout()
    plt.show()
    


########################################################################
#주말 상위권 3명만
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
df = pd.read_csv('23_성남시_일반노선별_시간대_이용객순위_주말.csv', parse_dates=False, encoding='utf-8-sig')
df['시간'] = df['시간'].apply(lambda x: x.replace('시~', ':00 - ').replace('시', ':00'))

# 그래프 설정
sns.set(style="whitegrid")

# 1월부터 12월까지의 데이터를 반복하여 시각화
for month in range(1, 13):
    # 해당 월 데이터만 필터링
    df_month = df[df['월'] == month]
    df_month['시작시간'] = df_month['시간'].str.split(' - ').str[0]
    df_month['시작시간'] = pd.to_datetime(df_month['시작시간'], format='%H:%M')

    
    # 시간대별로 상위 3개 노선 필터링
    top_3_routes = df_month[df_month['순위'] <= 3]
    top_3_routes = top_3_routes.sort_values(by='시작시간')    
    
    # 그래프 설정
    plt.figure(figsize=(12, 8))
    
    # 각 노선에 대해 점을 찍고, 그 점들을 표시
    for route in top_3_routes['노선'].unique():
        route_data = top_3_routes[top_3_routes['노선'] == route]
        
        # 시간대별로 이용객 수와 시간대
        x = route_data['시작시간']
        y = route_data['이용객수']
        
        # 점 찍기
        plt.scatter(x, y, label=f"{route} ({len(route_data)})", s=100, edgecolors='w', linewidth=2)
        
        # 점 위에 이용객 수 숫자 표시
        for i in range(len(x)):
            plt.text(x.iloc[i], y.iloc[i], f'{y.iloc[i]}', ha='center', va='bottom', fontsize=12, color='black')

    # 그래프 제목과 레이블 설정
    plt.title(f"{month} month Bus Rank", fontsize=16)
    plt.xlabel("time", fontsize=12)
    plt.ylabel("users", fontsize=12)
    plt.xticks(rotation=45)

    # x축의 시간만 표시하도록 포맷 설정
    #plt.gca().xaxis.set_major_formatter(DateFormatter('%H:%M'))
    plt.gca().xaxis.set_major_formatter(FuncFormatter(custom_time_formatter))
    plt.gca().xaxis.set_major_locator(HourLocator(interval=1))

    # 범례 표시
    plt.legend(title="month number", loc="upper left")

    # 그래프 표시
    plt.tight_layout()
    plt.show()
    

###################################################################
#평일 최하위3명
    
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
df = pd.read_csv('23_성남시_일반노선별_시간대_이용객순위_평일.csv', parse_dates=False, encoding='utf-8-sig')
df['시간'] = df['시간'].apply(lambda x: x.replace('시~', ':00 - ').replace('시', ':00'))

# 그래프 설정
sns.set(style="whitegrid")

# 1월부터 12월까지의 데이터를 반복하여 시각화
for month in range(1, 13):
    # 해당 월 데이터만 필터링
    df_month = df[df['월'] == month]
    df_month['시작시간'] = df_month['시간'].str.split(' - ').str[0]
    df_month['시작시간'] = pd.to_datetime(df_month['시작시간'], format='%H:%M')

    
    # 시간대별 꼴등 순위 찾기
    bottom_rank_per_time = df_month.groupby('시간')['순위'].max()
    
    # 시간대별로 하위 3개 노선 필터링
    bottom_3_routes = pd.DataFrame()
    for time, bottom_rank in bottom_rank_per_time.items():
        # 해당 시간대에서 하위 3개 순위를 추출
        time_data = df_month[(df_month['시간'] == time) & (df_month['순위'] >= bottom_rank - 2)]
        bottom_3_routes = pd.concat([bottom_3_routes, time_data])

    bottom_3_routes = bottom_3_routes.sort_values(by='시작시간')

    
    
    # 그래프 설정
    plt.figure(figsize=(12, 8))
    
    # 각 노선에 대해 점을 찍고, 그 점들을 표시
    for route in bottom_3_routes['노선'].unique():
        route_data = bottom_3_routes[bottom_3_routes['노선'] == route]
        
        # 시간대별로 이용객 수와 시간대
        x = route_data['시작시간']
        y = route_data['이용객수']
        
        # 점 찍기
        plt.scatter(x, y, label=f"{route} ({len(route_data)})", s=100, edgecolors='w', linewidth=2)
        
        # 점 위에 이용객 수 숫자 표시
        for i in range(len(x)):
            plt.text(x.iloc[i], y.iloc[i], f'{y.iloc[i]}', ha='center', va='bottom', fontsize=12, color='black')

    # 그래프 제목과 레이블 설정
    plt.title(f"{month} month Bus Rank", fontsize=16)
    plt.xlabel("time", fontsize=12)
    plt.ylabel("users", fontsize=12)
    plt.xticks(rotation=45)

    # x축의 시간만 표시하도록 포맷 설정
    #plt.gca().xaxis.set_major_formatter(DateFormatter('%H:%M'))
    plt.gca().xaxis.set_major_formatter(FuncFormatter(custom_time_formatter))
    plt.gca().xaxis.set_major_locator(HourLocator(interval=1))

    # 범례 표시
    plt.legend(title="month number", loc="upper left")

    # 그래프 표시
    plt.tight_layout()
    plt.show()
    

#####################################################################
#주말 최하위 3명
    
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
df = pd.read_csv('23_성남시_일반노선별_시간대_이용객순위_주말.csv', parse_dates=False, encoding='utf-8-sig')
df['시간'] = df['시간'].apply(lambda x: x.replace('시~', ':00 - ').replace('시', ':00'))

# 그래프 설정
sns.set(style="whitegrid")

# 1월부터 12월까지의 데이터를 반복하여 시각화
for month in range(1, 13):
    # 해당 월 데이터만 필터링
    df_month = df[df['월'] == month]
    df_month['시작시간'] = df_month['시간'].str.split(' - ').str[0]
    df_month['시작시간'] = pd.to_datetime(df_month['시작시간'], format='%H:%M')

    
    # 시간대별 꼴등 순위 찾기
    bottom_rank_per_time = df_month.groupby('시간')['순위'].max()
    
    # 시간대별로 하위 3개 노선 필터링
    bottom_3_routes = pd.DataFrame()
    for time, bottom_rank in bottom_rank_per_time.items():
        # 해당 시간대에서 하위 3개 순위를 추출
        time_data = df_month[(df_month['시간'] == time) & (df_month['순위'] >= bottom_rank - 2)]
        bottom_3_routes = pd.concat([bottom_3_routes, time_data])

    bottom_3_routes = bottom_3_routes.sort_values(by='시작시간')

    
    
    # 그래프 설정
    plt.figure(figsize=(12, 8))
    
    # 각 노선에 대해 점을 찍고, 그 점들을 표시
    for route in bottom_3_routes['노선'].unique():
        route_data = bottom_3_routes[bottom_3_routes['노선'] == route]
        
        # 시간대별로 이용객 수와 시간대
        x = route_data['시작시간']
        y = route_data['이용객수']
        
        # 점 찍기
        plt.scatter(x, y, label=f"{route} ({len(route_data)})", s=100, edgecolors='w', linewidth=2)
        
        # 점 위에 이용객 수 숫자 표시
        for i in range(len(x)):
            plt.text(x.iloc[i], y.iloc[i], f'{y.iloc[i]}', ha='center', va='bottom', fontsize=12, color='black')

    # 그래프 제목과 레이블 설정
    plt.title(f"{month} month Bus Rank", fontsize=16)
    plt.xlabel("time", fontsize=12)
    plt.ylabel("users", fontsize=12)
    plt.xticks(rotation=45)

    # x축의 시간만 표시하도록 포맷 설정
    #plt.gca().xaxis.set_major_formatter(DateFormatter('%H:%M'))
    plt.gca().xaxis.set_major_formatter(FuncFormatter(custom_time_formatter))
    plt.gca().xaxis.set_major_locator(HourLocator(interval=1))

    # 범례 표시
    plt.legend(title="month number", loc="upper left")

    # 그래프 표시
    plt.tight_layout()
    plt.show()
    

######################################################################
#평일 왼쪽 Top3, 오른쪽 Bottom3 한번에
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
df = pd.read_csv('23_성남시_일반노선별_시간대_이용객순위_평일.csv', parse_dates=False, encoding='utf-8-sig')
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

    # 시간대별 최하위 3개 노선 필터링
    bottom_rank_per_time = df_month.groupby('시간')['순위'].max()
    bottom_3_routes = pd.DataFrame()
    for time, bottom_rank in bottom_rank_per_time.items():
        # 해당 시간대에서 하위 3개 순위를 추출
        time_data = df_month[(df_month['시간'] == time) & (df_month['순위'] >= bottom_rank - 2)]
        bottom_3_routes = pd.concat([bottom_3_routes, time_data])

    bottom_3_routes = bottom_3_routes.sort_values(by='시작시간')

    # 서브플롯 설정: 상위 3개와 하위 3개를 나누어 시각화
    fig, ax = plt.subplots(1, 2, figsize=(16, 8))

    # 왼쪽: 상위 3개 노선
    for route in top_3_routes['노선'].unique():
        route_data = top_3_routes[top_3_routes['노선'] == route]
        x = route_data['시작시간']
        y = route_data['이용객수']

        ax[0].scatter(x, y, label=f"{route} ({len(route_data)})", s=100, edgecolors='w', linewidth=2)

        # 점 위에 이용객 수 숫자 표시
        for i in range(len(x)):
            ax[0].text(x.iloc[i], y.iloc[i], f'{y.iloc[i]}', ha='center', va='bottom', fontsize=12, color='black')

    ax[0].set_title(f"{month} Month - Top 3 Bus Routes", fontsize=16)
    ax[0].set_xlabel("Time", fontsize=12)
    ax[0].set_ylabel("Users", fontsize=12)
    ax[0].set_xticklabels(ax[0].get_xticklabels(), rotation=45)
    ax[0].xaxis.set_major_formatter(FuncFormatter(custom_time_formatter))
    ax[0].xaxis.set_major_locator(HourLocator(interval=1))

    # 오른쪽: 하위 3개 노선
    for route in bottom_3_routes['노선'].unique():
        route_data = bottom_3_routes[bottom_3_routes['노선'] == route]
        x = route_data['시작시간']
        y = route_data['이용객수']

        ax[1].scatter(x, y, label=f"{route} ({len(route_data)})", s=100, edgecolors='w', linewidth=2)

        # 점 위에 이용객 수 숫자 표시
        for i in range(len(x)):
            ax[1].text(x.iloc[i], y.iloc[i], f'{y.iloc[i]}', ha='center', va='bottom', fontsize=12, color='black')

    ax[1].set_title(f"{month} Month - Bottom 3 Bus Routes", fontsize=16)
    ax[1].set_xlabel("Time", fontsize=12)
    ax[1].set_ylabel("Users", fontsize=12)
    ax[1].set_xticklabels(ax[1].get_xticklabels(), rotation=45)
    ax[1].xaxis.set_major_formatter(FuncFormatter(custom_time_formatter))
    ax[1].xaxis.set_major_locator(HourLocator(interval=1))

    # 범례 설정
    ax[0].legend(title="Top 3 Routes", loc="upper left", fontsize=10)
    ax[1].legend(title="Bottom 3 Routes", loc="upper left", fontsize=10)

    # 그래프 간의 여백을 조정
    plt.tight_layout()

    # 그래프 출력
    plt.show()

###################################################
#한 개의 그래프로만 Top3, Bottom3표시
    
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

    # 시간대별 하위 3개 노선 추출
    bottom_3_routes = pd.DataFrame()

    # 각 시간대별로 노선 개수를 구하고 하위 3개 노선 추출
    for time in df_month['시간'].unique():
        time_data = df_month[df_month['시간'] == time]  # 해당 시간대 데이터 필터링
        
        # 해당 시간대의 노선 개수 구하기
        num_routes = len(time_data)
        
        # 하위 3개 순위는 (num_routes-2), (num_routes-1), num_routes인 노선
        bottom_3_routes_for_time = time_data[time_data['순위'].isin([num_routes - 2, num_routes - 1, num_routes])]
        
        # 하위 3개 노선 데이터 누적
        bottom_3_routes = pd.concat([bottom_3_routes, bottom_3_routes_for_time])

    bottom_3_routes = bottom_3_routes.sort_values(by='시작시간')

    # 그래프 설정
    plt.figure(figsize=(12, 8))

    # 상위 3개 노선 점 찍기
    for route in top_3_routes['노선'].unique():
        route_data = top_3_routes[top_3_routes['노선'] == route]
        x = route_data['시작시간']
        y = route_data['이용객수']

        # 상위 3개는 원형 마커로 표시
        plt.scatter(x, y, label=f"Top: {route} ({len(route_data)})", s=100, edgecolors='w', linewidth=2, marker='o')

        # 점 위에 이용객 수 숫자 표시
        for i in range(len(x)):
            plt.text(x.iloc[i], y.iloc[i], f'{y.iloc[i]}', ha='center', va='bottom', fontsize=12, color='black')

    # 하위 3개 노선 점 찍기
    for route in bottom_3_routes['노선'].unique():
        route_data = bottom_3_routes[bottom_3_routes['노선'] == route]
        x = route_data['시작시간']
        y = route_data['이용객수']

        # 하위 3개는 사각형 마커로 표시
        plt.scatter(x, y, label=f"Bottom: {route} ({len(route_data)})", s=100, edgecolors='w', linewidth=2, marker='s')

        # 점 위에 이용객 수 숫자 표시
        for i in range(len(x)):
            plt.text(x.iloc[i], y.iloc[i], f'{y.iloc[i]}', ha='center', va='bottom', fontsize=12, color='black')

    # 그래프 제목과 레이블 설정
    plt.title(f"{month} Month - Bus Routes Usage", fontsize=16)
    plt.xlabel("Time", fontsize=12)
    plt.ylabel("Users", fontsize=12)
    plt.xticks(rotation=45)

    # x축의 시간만 표시하도록 포맷 설정
    plt.gca().xaxis.set_major_formatter(FuncFormatter(custom_time_formatter))
    plt.gca().xaxis.set_major_locator(HourLocator(interval=1))

    # 범례 설정
    plt.legend(title="Bus Routes", loc="upper left", fontsize=10)

    # 그래프 간의 여백을 조정
    plt.tight_layout()

    # 그래프 출력
    plt.show()
