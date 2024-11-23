# 🚌 버스별 수요도 분석

## 📋 목차
1. [프로젝트 개요](#1-project-overview)
   - [Background and Goals](#11-background-and-goals)
   - [Problem Statement](#12-problem-statement)
   - [Data Collection and Analysis Overview](#13-data-collection-and-analysis-overview)
2. [데이터 수집 및 전처리](#2-data-collection-and-preprocessing)
   - [Data Collection Methods and Tools](#21-data-collection-methods-and-tools)
   - [Data Structure and Key Variables](#22-data-structure-and-key-variables)
   - [Data Cleaning and Processing](#23-data-cleaning-and-processing)
3. [데이터 분석 및 결과](#3-data-analysis-and-results)
   - [Time-based Passenger Usage per Route](#31-time-based-passenger-usage-per-route)
   - [Ranking Analysis Across All Routes](#32-ranking-analysis-across-all-routes)
     - [Detailed Rank Visualization](#321-detailed-rank-visualization)
     - [Simplified Rank Visualization](#322-simplified-rank-visualization)
   - [Map Visualization Using Folium](#33-map-visualization-using-folium)
     - [Station Marking by Passing Routes](#331-station-marking-by-passing-routes)
     - [Route Visualization with Line Styling](#332-route-visualization-with-line-styling)
   - [Comparison of Headways by Route Ranking](#34-comparison-of-headways-by-route-ranking)
4. [결론 및 시사점](#4-conclusions-and-implications)
   - [Key Findings](#41-key-findings)
   - [Policy Implications](#42-policy-implications)
   - [Future Work and Improvements](#43-future-work-and-improvements)

---

## 1. 프로젝트 개요
### 1.1 배경 및 목표
버스 노선마다 배차 간격이 다름에 따라 수요도 차이로 인해 이러한 현상이 발생하는지 데이터 분석을 통해 확인하고자 함.

### 1.2 Problem Statement
**"버스승객 수요가 버스 운행 간격에 직접적인 영향을 미치는가?"**

### 1.3 데이터 수집 및 분석 개요
수집 및 처리한 데이터:
- 데이터는 공공데이터 포털 API를 활용해 노선별 시간대별 승객 수 데이터를 수집
- 그 중 성남시만 운행하는 39개 노선의 버스만 Pandas를 활용하여 가공 및 정제
- 시각화 분석도구 Matplotlib, Seaborn, Folium.
---

## 2. 데이터 수집
### 2.1 주요 자료
1. 성남시 경유 노선 목록 데이터
2. 노선별 이용객 시계열 데이터
3. 노선별 정류소 데이터
4. 정류소 간 통과 노선 데이터

### 2.2 수집 도구 및 방식
- 공공데이터 API활용
- 교통카드 빅데이터 시스템

### 2.3 데이터 정리 및 처리
- 간단한 정리는 Excel VBA를 활용하여 정리 및 처리
- Pandas를 활용하여 복잡한 정리 및 처리
---

## 3. 데이터 전처리
### 3.1 성남시 경유 노선 목록 데이터 
- 원본 데이터 : 총 197개의 버스 노선 목록
- 처리 과정:
  1. 성남시를 벗어나는 버스 노선 제거 -> 72개 노선 남음
  2. 고속버스, 마을버스 등 기타버스 제외하고 일반 버스만을 필터링 -> 최종 39개의 노선 확보
- 결과 : 성남시 내 일반 버스 39개의 노선 목록 데이터 완성

### 3.2 노선별 이용객 시계열 데이터
- 원본 데이터 : 23년 기준 버스 이용객 시계열 데이터
- 처리 과정 :
  - 전처리된 39개 버스 노선 목록을 기준으로 관련 없는 데이터 제거
- 결과 : 39개 노선의 이용객 시계열 데이터 생성

### 3.3 노선별 정류소 데이터
- 원본 데이터 : 197개 노선의 경유 정류소 데이터(API활용)
- 처리 과정 :
  1. 1차 API요청으로 197개 노선의 정류소 데이터 수집
  2. 노선 목록 수정(39개 노선으로 축소) 후 다시 API요청
- 결과 : 39개 노선에 대한 정류소 데이터 처리 완료

### 3.4 정류소 간 통과 노선 데이터
- 원본 데이터 : 경기도 기준 정류소 간 통과 노선 데이터
- 처리 과정 :
  1. 성남시 외부 정류소 데이터를 모두 제거
  2. 성남시 내 정류소만 남긴 데이터로 필터링
- 결과 : 성남시 내 정류소 간 통과 노선 데이터 생성

### 3.5 추가 데이터 통합
- 배차 간격 데이터:
  - 노선별 노선 ID를 기준으로 이용객 시계열 데이터(2번)와 병합
- 위치 데이터 통합:
  - 경기도 정류소별 데이터에서 위치 정보를 추출
  - 노선별 정류소 데이터(3번)에 위치 데이터 병합
- 결과 : 정류소와 노선 데이터를 통합한 최종 데이터 생성

### [데이터 전처리 과정](#2./데이터가공/-data-preprocessing)
---

## 4. 데이터 분석 및 시각화
### 4.1 월별 노선별 수요도(평일, 주말)비교 시각화
- 목표:월별로 각 노선의 평일과 주말 이용객 수를 비교
- 시각화 방법:
  - Line Plot을 이용하여 X축은 시간대, Y축은 이용개 수로 설정
  - 노선별로 데이터를 시각화하여 평일과 주말의 수요 차이를 분석
![250노선](https://github.com/user-attachments/assets/ec8e034a-b764-4af6-aa52-5b29d3569a57)

### 4.2 월별 노선 수요도 순위 비교 시각화
- 목표: 각 노선의 월별 이용객 수에 따른 순위를 비교
- 시각화 방법:
  - Plot을 사용하여 X축은 시간대, Y축은 이용객 수를 기준
  - 상위 3개, 하위 3개 노선에 대한 비교 시각화
  - 순위 변동을 시각적으로 파악할 수 있도록 상위/하위 노선의 변화 추이를 나타냄
 ![line_plot_상위하위_1월](https://github.com/user-attachments/assets/11546072-14b9-4c83-b0a3-e931e692e235)

### 4.3 월별 노선별 순위에 따른 배차 간격 비교 시각화
- 목표: 상위 3개 및 하위 3개 노선의 배차 간격 차이를 분석
- 시각화 방법:
  - Plot을 사용하여 X축은 시간대, Y축은 배차 간격(분)을 기준으로 상위/하위 노선의 배차 간격 비교
  - 상위 노선일수록 배차 간격이 비교적 짧고, 하위 노선은 길다는 패턴을 시각적으로 비교
![1월_평일_상하위배차비교](https://github.com/user-attachments/assets/29dce0d6-fc0d-4325-bbc0-ac9987933a8c)

### 4.4 월별 노선 수요도에 따른 정류소 지도 시각화
- 목표: 상위 3개 및 하위 3개 노선이 지나가는 정류소의 트래픽을 비교
- 시각화 방법:
  - Folium 지도 시각화를 사용하여 각 노선이 지나가는 정류소를 Line으로 표시
  - 이용객 수가 많고 배차 간격이 짧은 노선의 라인을 두껍게 처리하여 트래픽을 시각적으로 표현
  - 지도 상에서 각 노선의 위치와 통행량을 한눈에 파악할 수 있도록 시각화
![image](https://github.com/user-attachments/assets/1d527699-2213-45a5-9e72-079e2c30b3cd)

---

## 5. Project Deliverables
### 5.1 Visualization Summary
Key visualizations include:
- Line plots for single-route analysis.
- Scatter and line plots for route rankings.
- Folium-based interactive maps.

---

Feel free to copy and paste this into your `README.md` file. Let me know if you'd like to customize any part further! 😊