import requests
import csv
import xml.etree.ElementTree as ET

# 서비스 키 설정
service_key = ''  # 여기에 실제 서비스 키를 입력하세요

# 추출된_노선번호_목록.csv 파일을 읽어서 데이터를 처리합니다.
input_file = '추출된_노선번호_목록.csv'
output_file = '버스번호_to_버스ID.csv'

# CSV 파일 읽기
with open(input_file, 'r', encoding='utf-8-sig') as f:
    reader = csv.reader(f)
    header = next(reader)  # 첫 번째 줄은 헤더로 건너뛰기
    routes = [row for row in reader]

# 새로운 CSV 파일 작성
with open(output_file, 'w', newline='', encoding='utf-8-sig') as f:
    writer = csv.writer(f)
    # 새로운 CSV 파일의 헤더
    writer.writerow(['노선번호', '기점', '종점', '노선아이디', '노선번호', '노선유형', '노선유형명', '지역명'])

    # 각 노선번호에 대해 API 호출
    for route in routes:
        route_number = route[0]  # 노선번호
        origin = route[1]  # 기점
        destination = route[2]  # 종점

        # API 요청 파라미터 설정
        url = 'http://apis.data.go.kr/6410000/busrouteservice/getBusRouteList'
        params = {'serviceKey': service_key, 'keyword': route_number}

        # API 요청 및 응답 받기
        response = requests.get(url, params=params)
        if response.status_code == 200:
            # XML 파싱
            root = ET.fromstring(response.content)

            # XML에서 필요한 정보 추출
            bus_route_elements = root.findall('.//busRouteList')

            for bus_route in bus_route_elements:
                # 정확한 노선번호가 일치하는지 확인
                route_name = bus_route.find('routeName').text
                if route_name != route_number:
                    continue  # 노선번호가 정확히 일치하지 않으면 skip

                # 노선 정보 추출
                route_id = bus_route.find('routeId').text
                route_type_cd = bus_route.find('routeTypeCd').text
                route_type_name = bus_route.find('routeTypeName').text
                region_name = bus_route.find('regionName').text

                # CSV에 기록
                writer.writerow([route_number, origin, destination, route_id, route_name, route_type_cd, route_type_name, region_name])
        else:
            print(f"API 호출 오류: 노선번호 {route_number}")

print("버스번호_to_버스ID.csv 파일이 생성되었습니다.")
