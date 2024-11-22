####################################################
#평일 상위권 3명만
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

    # 시간대별로 순위별로 데이터를 나누기
    df_month_sorted = df_month.sort_values(by=['시작시간', '순위'])

    # 그래프 설정
    plt.figure(figsize=(12, 8))

    # 시간대별로 순위별로 이용객 수와 노선 데이터를 선으로 연결
    for rank in range(1, 4):  # 1등, 2등, 3등 순위
        rank_data = df_month_sorted[df_month_sorted['순위'] == rank]
        
        # 각 시간대별로 연결된 선을 그리기
        x = rank_data['시작시간']
        y = rank_data['이용객수']
        
        # 선 그리기
        plt.plot(x, y, label=f"Rank {rank}", marker='o', markersize=8, linestyle='-', linewidth=2)

        # 점 위에 이용객 수 숫자 표시
        for i in range(len(x)):
            plt.text(x.iloc[i], y.iloc[i], f'{y.iloc[i]}', ha='center', va='bottom', fontsize=12, color='black')

    # 그래프 제목과 레이블 설정
    plt.title(f"{month} Month Bus Rank (Ranked by Users)", fontsize=16)
    plt.xlabel("Time", fontsize=12)
    plt.ylabel("Users", fontsize=12)
    plt.xticks(rotation=45)

    # x축의 시간만 표시하도록 포맷 설정
    plt.gca().xaxis.set_major_formatter(FuncFormatter(custom_time_formatter))
    plt.gca().xaxis.set_major_locator(HourLocator(interval=1))

    # 범례 표시
    plt.legend(title="Rank", loc="upper left")

    # 그래프 표시
    plt.tight_layout()
    plt.show()
    


########################################################################
#주말 상위권 3개
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
df = pd.read_csv('23_성남시_일반노선별_시간대_이용객순위_주말_배차포함_100x.csv', parse_dates=False, encoding='utf-8-sig')
df['시간'] = df['시간'].apply(lambda x: x.replace('시~', ':00 - ').replace('시', ':00'))

# 그래프 설정
sns.set(style="whitegrid")

# 1월부터 12월까지의 데이터를 반복하여 시각화
for month in range(1, 13):
    # 해당 월 데이터만 필터링
    df_month = df[df['월'] == month]
    df_month['시작시간'] = df_month['시간'].str.split(' - ').str[0]
    df_month['시작시간'] = pd.to_datetime(df_month['시작시간'], format='%H:%M')

    # 시간대별로 순위별로 데이터를 나누기
    df_month_sorted = df_month.sort_values(by=['시작시간', '순위'])

    # 그래프 설정
    plt.figure(figsize=(12, 8))

    # 시간대별로 순위별로 이용객 수와 노선 데이터를 선으로 연결
    for rank in range(1, 4):  # 1등, 2등, 3등 순위
        rank_data = df_month_sorted[df_month_sorted['순위'] == rank]
        
        # 각 시간대별로 연결된 선을 그리기
        x = rank_data['시작시간']
        y = rank_data['이용객수']
        
        # 선 그리기
        plt.plot(x, y, label=f"Rank {rank}", marker='o', markersize=8, linestyle='-', linewidth=2)

        # 점 위에 이용객 수 숫자 표시
        for i in range(len(x)):
            plt.text(x.iloc[i], y.iloc[i], f'{y.iloc[i]}', ha='center', va='bottom', fontsize=12, color='black')

    # 그래프 제목과 레이블 설정
    plt.title(f"{month} Month Bus Rank (Ranked by Users)", fontsize=16)
    plt.xlabel("Time", fontsize=12)
    plt.ylabel("Users", fontsize=12)
    plt.xticks(rotation=45)

    # x축의 시간만 표시하도록 포맷 설정
    plt.gca().xaxis.set_major_formatter(FuncFormatter(custom_time_formatter))
    plt.gca().xaxis.set_major_locator(HourLocator(interval=1))

    # 범례 표시
    plt.legend(title="Rank", loc="upper left")

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

    # 시간대별 꼴등 순위 찾기
    bottom_rank_per_time = df_month.groupby('시간')['순위'].apply(lambda x: len(x))

    # 순위 범위 (꼴등, 꼴등+1, 꼴등+2) 데이터 필터링
    rank_ranges = [0, 1, 2]  # 꼴등, 꼴등+1, 꼴등+2

    # 라인 그래프 데이터 초기화
    rank_lines = {i: pd.DataFrame() for i in rank_ranges}

    for time, num_routes in bottom_rank_per_time.items():
        # 해당 시간대에서 최하위 순위 계산 (len을 이용)
        bottom_rank = num_routes  # len을 이용하여 최하위 순위는 num_routes 값이 됨
        
        # 해당 시간대에서 꼴등, 꼴등+1, 꼴등+2 추출
        for offset in rank_ranges:
            rank_data = df_month[(df_month['시간'] == time) & 
                                 (df_month['순위'] == bottom_rank - offset)]
            rank_lines[offset] = pd.concat([rank_lines[offset], rank_data])

    # 결과는 rank_lines의 각 데이터프레임에 저장됩니다.

    
    # 그래프 설정
    plt.figure(figsize=(12, 8))

    # 시간대별로 라인 그래프 그리기
    for offset, line_data in rank_lines.items():
        # 각 라인에 대해 시간대별 이용객 수
        line_data = line_data.sort_values(by='시작시간')
        x = line_data['시작시간']
        y = line_data['이용객수']
        
        # 라인 그리기
        plt.plot(x, y, label=f"Rank {offset} (bottom + {offset})", marker='o', markersize=6, linewidth=2)
        
        # 점 위에 이용객 수 숫자 표시
        for i in range(len(x)):
            plt.text(x.iloc[i], y.iloc[i], f'{y.iloc[i]}', ha='center', va='bottom', fontsize=12, color='black')

    # 그래프 제목과 레이블 설정
    plt.title(f"{month} month Bus Rank (Bottom Ranks)", fontsize=16)
    plt.xlabel("Time", fontsize=12)
    plt.ylabel("Users", fontsize=12)
    plt.xticks(rotation=45)

    # x축의 시간만 표시하도록 포맷 설정
    plt.gca().xaxis.set_major_formatter(FuncFormatter(custom_time_formatter))
    plt.gca().xaxis.set_major_locator(HourLocator(interval=1))

    # 범례 표시
    plt.legend(title="Rank Group", loc="upper left")

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
df = pd.read_csv('23_성남시_일반노선별_시간대_이용객순위_주말_배차포함_100x.csv', parse_dates=False, encoding='utf-8-sig')
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
    bottom_rank_per_time = df_month.groupby('시간')['순위'].apply(lambda x: len(x))

    # 순위 범위 (꼴등, 꼴등+1, 꼴등+2) 데이터 필터링
    rank_ranges = [0, 1, 2]  # 꼴등, 꼴등+1, 꼴등+2

    # 라인 그래프 데이터 초기화
    rank_lines = {i: pd.DataFrame() for i in rank_ranges}

    for time, num_routes in bottom_rank_per_time.items():
        # 해당 시간대에서 최하위 순위 계산 (len을 이용)
        bottom_rank = num_routes  # len을 이용하여 최하위 순위는 num_routes 값이 됨
        
        # 해당 시간대에서 꼴등, 꼴등+1, 꼴등+2 추출
        for offset in rank_ranges:
            rank_data = df_month[(df_month['시간'] == time) & 
                                 (df_month['순위'] == bottom_rank - offset)]
            rank_lines[offset] = pd.concat([rank_lines[offset], rank_data])

    # 결과는 rank_lines의 각 데이터프레임에 저장됩니다.

    
    # 그래프 설정
    plt.figure(figsize=(12, 8))

    # 시간대별로 라인 그래프 그리기
    for offset, line_data in rank_lines.items():
        # 각 라인에 대해 시간대별 이용객 수
        line_data = line_data.sort_values(by='시작시간')
        x = line_data['시작시간']
        y = line_data['이용객수']
        
        # 라인 그리기
        plt.plot(x, y, label=f"Rank {offset} (bottom + {offset})", marker='o', markersize=6, linewidth=2)
        
        # 점 위에 이용객 수 숫자 표시
        for i in range(len(x)):
            plt.text(x.iloc[i], y.iloc[i], f'{y.iloc[i]}', ha='center', va='bottom', fontsize=12, color='black')

    # 그래프 제목과 레이블 설정
    plt.title(f"{month} month Bus Rank (Bottom Ranks)", fontsize=16)
    plt.xlabel("Time", fontsize=12)
    plt.ylabel("Users", fontsize=12)
    plt.xticks(rotation=45)

    # x축의 시간만 표시하도록 포맷 설정
    plt.gca().xaxis.set_major_formatter(FuncFormatter(custom_time_formatter))
    plt.gca().xaxis.set_major_locator(HourLocator(interval=1))

    # 범례 표시
    plt.legend(title="Rank Group", loc="upper left")

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

    # 시간대별 하위 순위 개수 찾기 (len을 사용하여 하위 순위를 구함)
    bottom_rank_per_time = df_month.groupby('시간')['순위'].apply(lambda x: len(x))

    # 순위 범위 (꼴등, 꼴등+1, 꼴등+2) 데이터 필터링
    rank_ranges = [0, 1, 2]  # 꼴등, 꼴등+1, 꼴등+2

    # 라인 그래프 데이터 초기화
    rank_lines_bottom = {i: pd.DataFrame() for i in rank_ranges}
    rank_lines_top = {i: pd.DataFrame() for i in range(1, 4)}  # 1등, 2등, 3등

    for time, num_routes in bottom_rank_per_time.items():
        # 해당 시간대에서 하위 3개 순위 (꼴등, 꼴등+1, 꼴등+2) 추출
        bottom_rank = num_routes  # 하위 3개 순위는 len(x)로 얻은 노선 개수로 결정
        
        # 하위 3개 순위 (꼴등, 꼴등+1, 꼴등+2) 추출
        for offset in rank_ranges:
            rank_data_bottom = df_month[(df_month['시간'] == time) & 
                                        (df_month['순위'] == bottom_rank - offset)]
            rank_lines_bottom[offset] = pd.concat([rank_lines_bottom[offset], rank_data_bottom])
        
        # 상위 3등 순위 (1등, 2등, 3등) 추출
        for rank in range(1, 4):  # 상위 3등 (1등, 2등, 3등)
            rank_data_top = df_month[(df_month['시간'] == time) & 
                                      (df_month['순위'] == rank)]
            rank_lines_top[rank] = pd.concat([rank_lines_top[rank], rank_data_top])

    # 결과는 rank_lines_bottom과 rank_lines_top에 저장됩니다.

    
    # 그래프 설정: 상위 3개 순위와 하위 3개 순위 그래프를 나누어 시각화
    fig, ax = plt.subplots(1, 2, figsize=(16, 8))

    # 왼쪽: 상위 3개 순위 (1등, 2등, 3등)
    for rank in range(1, 4):  # 1등, 2등, 3등
        line_data_top = rank_lines_top[rank].sort_values(by='시작시간')
        x = line_data_top['시작시간']
        y = line_data_top['이용객수']
        
        # 라인 그리기
        ax[0].plot(x, y, label=f"Rank {rank}", marker='o', markersize=8, linestyle='-', linewidth=2)

        # 점 위에 이용객 수 숫자 표시
        for i in range(len(x)):
            ax[0].text(x.iloc[i], y.iloc[i], f'{y.iloc[i]}', ha='center', va='bottom', fontsize=12, color='black')

    ax[0].set_title(f"{month} Month - Top 3 Bus Routes", fontsize=16)
    ax[0].set_xlabel("Time", fontsize=12)
    ax[0].set_ylabel("Users", fontsize=12)
    ax[0].set_xticklabels(ax[0].get_xticklabels(), rotation=45)
    ax[0].xaxis.set_major_formatter(FuncFormatter(custom_time_formatter))
    ax[0].xaxis.set_major_locator(HourLocator(interval=1))
    ax[0].legend(title="Top 3 Ranks", loc="upper left", fontsize=10)

    # 오른쪽: 하위 3개 순위 (꼴등, 꼴등+1, 꼴등+2)
    for offset in rank_ranges:
        line_data_bottom = rank_lines_bottom[offset].sort_values(by='시작시간')
        x = line_data_bottom['시작시간']
        y = line_data_bottom['이용객수']
        
        # 라인 그리기
        ax[1].plot(x, y, label=f"Rank {bottom_rank_per_time[line_data_bottom['시간'].iloc[0]] - offset}", 
                   marker='o', markersize=8, linestyle='-', linewidth=2)

        # 점 위에 이용객 수 숫자 표시
        for i in range(len(x)):
            ax[1].text(x.iloc[i], y.iloc[i], f'{y.iloc[i]}', ha='center', va='bottom', fontsize=12, color='black')

    ax[1].set_title(f"{month} Month - Bottom 3 Bus Routes", fontsize=16)
    ax[1].set_xlabel("Time", fontsize=12)
    ax[1].set_ylabel("Users", fontsize=12)
    ax[1].set_xticklabels(ax[1].get_xticklabels(), rotation=45)
    ax[1].xaxis.set_major_formatter(FuncFormatter(custom_time_formatter))
    ax[1].xaxis.set_major_locator(HourLocator(interval=1))
    ax[1].legend(title="Bottom 3 Ranks", loc="upper left", fontsize=10)

    # 그래프 간의 여백을 조정
    plt.tight_layout()

    # 그래프 출력
    plt.show()


###################################################
#한개의 그래프로만 Top3, Bottom3표시
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

# 1월부터 12월까지 데이터를 반복하여 시각화
for month in range(1, 13):
    # 해당 월 데이터만 필터링
    df_month = df[df['월'] == month]
    df_month['시작시간'] = df_month['시간'].str.split(' - ').str[0]
    df_month['시작시간'] = pd.to_datetime(df_month['시작시간'], format='%H:%M')

    # 시간대별 하위 순위 개수 찾기 (len을 사용하여 하위 순위를 구함)
    bottom_rank_per_time = df_month.groupby('시간')['순위'].apply(lambda x: len(x))

    # 순위 범위 (꼴등, 꼴등+1, 꼴등+2) 데이터 필터링
    rank_ranges = [0, 1, 2]  # 꼴등, 꼴등+1, 꼴등+2

    # 라인 그래프 데이터 초기화
    rank_lines_bottom = {i: pd.DataFrame() for i in rank_ranges}
    rank_lines_top = {i: pd.DataFrame() for i in range(1, 4)}  # 1등, 2등, 3등

    for time, num_routes in bottom_rank_per_time.items():
        # 해당 시간대에서 하위 3개 순위 (꼴등, 꼴등+1, 꼴등+2) 추출
        bottom_rank = num_routes  # 하위 3개 순위는 len(x)로 얻은 노선 개수로 결정
        
        for offset in rank_ranges:
            rank_data_bottom = df_month[(df_month['시간'] == time) & 
                                        (df_month['순위'] == bottom_rank - offset)]
            rank_lines_bottom[offset] = pd.concat([rank_lines_bottom[offset], rank_data_bottom])
        
        for rank in range(1, 4):  # 상위 3등 (1등, 2등, 3등)
            rank_data_top = df_month[(df_month['시간'] == time) & 
                                      (df_month['순위'] == rank)]
            rank_lines_top[rank] = pd.concat([rank_lines_top[rank], rank_data_top])

    # 결과는 rank_lines_bottom과 rank_lines_top에 저장됩니다.

    # 그래프 설정: 상위 3개 순위와 하위 3개 순위 한 그래프에 시각화
    fig, ax = plt.subplots(figsize=(12, 8))

    # 상위 3개 순위 (1등, 2등, 3등)
    for rank in range(1, 4):  # 1등, 2등, 3등
        line_data_top = rank_lines_top[rank].sort_values(by='시작시간')
        x = line_data_top['시작시간']
        y = line_data_top['이용객수']
        
        ax.plot(x, y, label=f"Top {rank}", marker='s', markersize=8, linestyle='-', linewidth=2)

        for i in range(len(x)):
            ax.text(x.iloc[i], y.iloc[i], f'{y.iloc[i]}', ha='center', va='bottom', fontsize=10)

    # 하위 3개 순위 (꼴등, 꼴등+1, 꼴등+2)
    for offset in rank_ranges:
        line_data_bottom = rank_lines_bottom[offset].sort_values(by='시작시간')
        x = line_data_bottom['시작시간']
        y = line_data_bottom['이용객수']
        
        ax.plot(x, y, label=f"Bottom {offset + 1}", marker='s', markersize=8, linestyle='-', linewidth=2)

        for i in range(len(x)):
            ax.text(x.iloc[i], y.iloc[i], f'{y.iloc[i]}', ha='center', va='bottom', fontsize=10)

    # 그래프 제목과 레이블
    ax.set_title(f"{month} Month - Top and Bottom 3 Bus Routes", fontsize=16)
    ax.set_xlabel("Time", fontsize=12)
    ax.set_ylabel("Users", fontsize=12)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    ax.xaxis.set_major_formatter(FuncFormatter(custom_time_formatter))
    ax.xaxis.set_major_locator(HourLocator(interval=1))
    ax.legend(title="Ranks", loc="upper left", fontsize=10)

    # 그래프 출력
    plt.tight_layout()
    plt.show()
