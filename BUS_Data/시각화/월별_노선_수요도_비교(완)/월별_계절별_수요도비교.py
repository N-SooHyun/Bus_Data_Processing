import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# CSV 파일 읽기 (예시 파일 경로는 실제 파일 경로로 수정 필요)
weekday_df = pd.read_csv('23_성남시_일반노선별_시간대_이용객_평일.csv', encoding='utf-8')
weekend_df = pd.read_csv('23_성남시_일반노선별_시간대_이용객_주말.csv', encoding='utf-8')

# 데이터의 기본 정보 확인
weekday_df.head(), weekend_df.head()

# 필요한 열만 추출 (예시: 노선, 일시, 시간, 이용객수)
weekday_df = weekday_df[['연도', '월', '노선', '일시', '시간', '이용객수']]
weekend_df = weekend_df[['연도', '월', '노선', '일시', '시간', '이용객수']]

# '월'과 '시간'을 숫자형으로 변환
weekday_df['월'] = weekday_df['월'].astype(int)
weekend_df['월'] = weekend_df['월'].astype(int)
#print(weekday_df.info())

# 예시로 "성남시 노선 100"에 해당하는 데이터를 추출 (노선 번호는 실제 노선에 맞게 수정)
route_id = '100'
weekday_route_df = weekday_df[weekday_df['노선'] == route_id]
weekend_route_df = weekend_df[weekend_df['노선'] == route_id]
weekday_route_df['시간'] = weekday_route_df['시간'].apply(lambda x: x.replace('시~', ':00 - ').replace('시', ':00'))
weekend_route_df['시간'] = weekend_route_df['시간'].apply(lambda x: x.replace('시~', ':00 - ').replace('시', ':00'))
# 데이터 확인
print(weekend_df.head())
print(weekend_df.info())

#########################################################################
#월별 주말 평일

fig, axes = plt.subplots(3, 4, figsize=(18, 12))  # 3행 4열로 월별로 시각화
axes = axes.flatten()

for i, month in enumerate(range(1, 13)):  # 1월부터 12월까지
    # 해당 월에 해당하는 평일 데이터와 주말 데이터 필터링
    month_data_weekday = weekday_route_df[weekday_route_df['월'] == month]
    month_data_weekend = weekend_route_df[weekend_route_df['월'] == month]
    
    # 매치되는 시간대만 필터링
    # '시간' 컬럼을 기준으로 평일과 주말 데이터가 일치하는 시간대만 남기기
    common_times = set(month_data_weekday['시간']).intersection(set(month_data_weekend['시간']))
    
    month_data_weekday = month_data_weekday[month_data_weekday['시간'].isin(common_times)]
    month_data_weekend = month_data_weekend[month_data_weekend['시간'].isin(common_times)]

    # 평일 데이터 그리기 (검정색)
    axes[i].plot(month_data_weekday['시간'], month_data_weekday['이용객수'], marker='o', label=f"Weekday {route_id} Bus", color='black')
    
    # 주말 데이터 그리기 (빨간색)
    axes[i].plot(month_data_weekend['시간'], month_data_weekend['이용객수'], marker='o', label=f"Weekend {route_id} Bus", color='red')

    # 각 점에 이용객 수 표시
    for j, txt in enumerate(month_data_weekday['이용객수']):
        axes[i].text(month_data_weekday['시간'].iloc[j], txt, str(txt), color='black', fontsize=8, verticalalignment='bottom')

    for j, txt in enumerate(month_data_weekend['이용객수']):
        axes[i].text(month_data_weekend['시간'].iloc[j], txt, str(txt), color='red', fontsize=8, verticalalignment='bottom')

    # 그래프 제목 및 레이블 설정
    axes[i].set_title(f"{month}Month - {route_id} Bus")
    axes[i].set_xlabel('Time of Day')
    axes[i].set_ylabel('Passenger Count')
    axes[i].tick_params(axis='x', rotation=45)
    axes[i].grid(True)
    axes[i].legend()

# 레이아웃 조정
plt.tight_layout()
plt.show()


##########################################################################
def get_season(month):
    if month in [3, 4, 5]:
        return 'Spring'  # 봄
    elif month in [6, 7, 8]:
        return 'Summer'  # 여름
    elif month in [9, 10, 11]:
        return 'Fall'    # 가을
    else:
        return 'Winter'  # 겨울

# 계절 컬럼 추가
weekday_route_df['Season'] = weekday_route_df['월'].apply(get_season)

# 계절별로 그래프를 그릴 준비
fig, axes = plt.subplots(2, 2, figsize=(18, 12))  # 2x2 배열로 4개의 그래프 생성
axes = axes.flatten()

# 계절별로 3개의 월에 대한 꺾은선 그래프 그리기
seasons = ['Spring', 'Summer', 'Fall', 'Winter']
labels = {'Spring': 'Spring (3-5 months)', 'Summer': 'Summer (6-8 months)', 'Fall': 'Fall (9-11 months)', 'Winter': 'Winter (12-2 months)'}

for i, season in enumerate(seasons):
    # 계절별로 3개월 데이터만 필터링
    season_data = weekday_route_df[weekday_route_df['Season'] == season]
    
    # 각 계절에 맞는 월 범위 지정
    if season == 'Spring':  # 봄: 3, 4, 5월
        months = [3, 4, 5]
    elif season == 'Summer':  # 여름: 6, 7, 8월
        months = [6, 7, 8]
    elif season == 'Fall':  # 가을: 9, 10, 11월
        months = [9, 10, 11]
    else:  # 겨울: 12, 1, 2월
        months = [12, 1, 2]

    # 각 월에 대한 데이터를 꺾은선 그래프로 그리기
    for month in months:
        month_data = season_data[season_data['월'] == month]
        
        # 각 월에 대해 '시간' 컬럼을 기준으로 공통된 시간대만 남기기
        if i == 0:  # 첫 번째 계절 (Spring)
            common_times = set(month_data['시간'])
        else:
            common_times = common_times.intersection(set(month_data['시간']))
        
        # 공통된 시간대만 남기기
        month_data = month_data[month_data['시간'].isin(common_times)]
        
        # X축은 시간대, Y축은 이용객 수로 꺾은선 그래프를 그리기
        ax = axes[i]
        ax.plot(month_data['시간'], month_data['이용객수'], marker='o', label=f"{month} month")
    
    # 계절별 그래프 설정
    ax.set_title(labels[season])  # 계절 이름
    ax.set_xlabel('Time of Day')
    ax.set_ylabel('Passenger Count')
    ax.tick_params(axis='x', rotation=45)
    ax.grid(True)
    ax.legend()

# 레이아웃 조정
plt.tight_layout()
plt.show()


##########################################################
#계절별 주말만
def get_season(month):
    if month in [3, 4, 5]:
        return 'Spring'  # 봄
    elif month in [6, 7, 8]:
        return 'Summer'  # 여름
    elif month in [9, 10, 11]:
        return 'Fall'    # 가을
    else:
        return 'Winter'  # 겨울

# 계절 컬럼 추가
weekend_route_df['Season'] = weekend_route_df['월'].apply(get_season)

# 계절별로 그래프를 그릴 준비
fig, axes = plt.subplots(2, 2, figsize=(18, 12))  # 2x2 배열로 4개의 그래프 생성
axes = axes.flatten()

# 계절별로 3개의 월에 대한 꺾은선 그래프 그리기
seasons = ['Spring', 'Summer', 'Fall', 'Winter']
labels = {'Spring': 'Spring (3-5 months)', 'Summer': 'Summer (6-8 months)', 'Fall': 'Fall (9-11 months)', 'Winter': 'Winter (12-2 months)'}

for i, season in enumerate(seasons):
    # 계절별로 3개월 데이터만 필터링
    season_data = weekend_route_df[weekend_route_df['Season'] == season]
    
    # 각 계절에 맞는 월 범위 지정
    if season == 'Spring':  # 봄: 3, 4, 5월
        months = [3, 4, 5]
    elif season == 'Summer':  # 여름: 6, 7, 8월
        months = [6, 7, 8]
    elif season == 'Fall':  # 가을: 9, 10, 11월
        months = [9, 10, 11]
    else:  # 겨울: 12, 1, 2월
        months = [12, 1, 2]

    # 각 월에 대한 데이터를 꺾은선 그래프로 그리기
    common_times = None  # 공통 시간대 초기화
    for month in months:
        month_data = season_data[season_data['월'] == month]
        
        # '시간' 컬럼을 기준으로 공통된 시간대만 남기기
        if common_times is None:  # 첫 번째 월에 대해 공통된 시간대를 초기화
            common_times = set(month_data['시간'])
        else:
            common_times = common_times.intersection(set(month_data['시간']))
        
        # 공통된 시간대만 남기기
        month_data = month_data[month_data['시간'].isin(common_times)]
        
        # X축은 시간대, Y축은 이용객 수로 꺾은선 그래프를 그리기
        ax = axes[i]
        ax.plot(month_data['시간'], month_data['이용객수'], marker='o', label=f"{month} month")
    
    # 계절별 그래프 설정
    ax.set_title(labels[season])  # 계절 이름
    ax.set_xlabel('Time of Day')
    ax.set_ylabel('Passenger Count')
    ax.tick_params(axis='x', rotation=45)
    ax.grid(True)
    ax.legend()

# 레이아웃 조정
plt.tight_layout()
plt.show()



