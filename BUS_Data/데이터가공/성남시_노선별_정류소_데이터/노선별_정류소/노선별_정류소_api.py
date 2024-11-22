import requests
import csv
import xml.etree.ElementTree as ET

# 버스번호_to_버스ID.csv에서 노선 데이터 읽기
def read_bus_data(file_path):
    bus_data = []
    with open(file_path, mode='r', encoding='utf-8-sig') as file:
        reader = csv.DictReader(file)
        for row in reader:
            bus_data.append(row)
    return bus_data

# API 호출 및 XML 데이터 파싱
def get_bus_route_stations(route_id, service_key):
    url = 'http://apis.data.go.kr/6410000/busrouteservice/getBusRouteStationList'
    params = {'serviceKey': service_key, 'routeId': route_id}

    # API 요청
    response = requests.get(url, params=params)

    if response.status_code == 200:
        return parse_bus_route_stations(response.content)
    else:
        print(f"API 호출 실패: {response.status_code}")
        return []

# XML 데이터를 파싱하여 필요한 정보 추출
def parse_bus_route_stations(xml_content):
    stations = []
    root = ET.fromstring(xml_content)

    # XML에서 필요한 정보 추출
    for station in root.findall('.//busRouteStationList'):
        station_id = station.find('stationId').text if station.find('stationId') is not None else ''
        station_seq = station.find('stationSeq').text if station.find('stationSeq') is not None else ''
        station_name = station.find('stationName').text if station.find('stationName') is not None else ''
        mobile_no = station.find('mobileNo').text.strip() if station.find('mobileNo') is not None else ''
        region_name = station.find('regionName').text if station.find('regionName') is not None else ''
        x = station.find('x').text if station.find('x') is not None else ''
        y = station.find('y').text if station.find('y') is not None else ''
        
        stations.append({
            'stationId': station_id,
            'stationSeq': station_seq,
            'stationName': station_name,
            'mobileNo': mobile_no,
            'regionName': region_name,
            'x': x,
            'y': y
        })
    return stations

# 데이터를 CSV 파일로 저장
def save_to_csv(bus_data, all_stations, output_file):
    fieldnames = [
        '노선번호', '기점', '종점', '노선아이디', '노선유형', '노선유형명', '지역명',
        '정류소아이디', '정류소순번', '정류소명', '정류소번호', '지역명', 'x좌표', 'y좌표'
    ]

    with open(output_file, mode='w', encoding='utf-8-sig', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        # 각 노선별로 정류소 정보를 작성
        for bus in bus_data:
            route_id = bus['노선아이디']
            stations = all_stations.get(route_id, [])
            for station in stations:
                row = {
                    '노선번호': bus['노선번호'],
                    '기점': bus['기점'],
                    '종점': bus['종점'],
                    '노선아이디': bus['노선아이디'],
                    '노선유형': bus['노선유형'],
                    '노선유형명': bus['노선유형명'],
                    '지역명': bus['지역명'],
                    '정류소아이디': station['stationId'],
                    '정류소순번': station['stationSeq'],
                    '정류소명': station['stationName'],
                    '정류소번호': station['mobileNo'],
                    '지역명': station['regionName'],
                    'x좌표': station['x'],
                    'y좌표': station['y']
                }
                writer.writerow(row)

def main():
    # CSV 파일 경로와 서비스 키 설정
    bus_data_file = '버스번호_to_버스ID.csv'
    service_key = 'service_key'  

    # CSV에서 버스 노선 정보 읽기
    bus_data = read_bus_data(bus_data_file)

    # 노선별로 정류소 정보 가져오기
    all_stations = {}
    for bus in bus_data:
        route_id = bus['노선아이디']
        if route_id not in all_stations:
            stations = get_bus_route_stations(route_id, service_key)
            all_stations[route_id] = stations

    # 결과를 CSV 파일로 저장
    output_file = '노선별_정류소.csv'
    save_to_csv(bus_data, all_stations, output_file)

    print(f"CSV 파일 '{output_file}'로 저장 완료")

if __name__ == "__main__":
    main()
