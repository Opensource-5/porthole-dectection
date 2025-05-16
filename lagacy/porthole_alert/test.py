import requests
from dotenv import load_dotenv
import os
import re

load_dotenv(override=True)

def get_location_info(coordinates, client_id, client_secret):
    """
    Reverse geocoding API를 호출하여 좌표에 대한 주소 정보를 반환합니다.
    입력 좌표는 (lat, lng) 형식입니다.
    """
    lat, lng = coordinates
    url = "https://maps.apigw.ntruss.com/map-reversegeocode/v2/gc"
    params = {
        "coords": f"{lng},{lat}",  # Naver API는 경도, 위도 순서입니다.
        "sourcecrs": "epsg:4326",
        "orders": "legalcode,admcode,addr,roadaddr",
        "output": "json"
    }
    headers = {
        "x-ncp-apigw-api-key-id": client_id,
        "x-ncp-apigw-api-key": client_secret,
        "User-Agent": "curl/7.64.1",
        "Accept": "application/json"
    }
    try:
        response = requests.get(url, params=params, headers=headers)
        response.raise_for_status()
        data = response.json()
        if data.get('results'):
            return data['results']
        else:
            print("No results found in the reverse geocode response.")
            return None
    except requests.HTTPError as http_err:
        print(f"HTTP error in reverse geocode: {http_err} | Response: {response.text}")
        return None
    except Exception as e:
        print(f"Error in reverse geocode: {e}")
        return None

def format_location_info(location_info):
    """
    reverse geocoding 결과 중 roadaddr 타입만 선택하여
    '시도 시군구 읍면동 도로명' 형식으로 도로명 주소를 반환합니다.
    """
    if not location_info:
        return "주소 정보가 없습니다."
    
    for result in location_info:
        if result.get('name') == 'roadaddr':
            region = result.get('region', {})
            area1 = region.get('area1', {}).get('name', '')
            area2 = region.get('area2', {}).get('name', '')
            area3 = region.get('area3', {}).get('name', '')
            area4 = region.get('area4', {}).get('name', '')
            land = result.get('land', {})
            road_name = land.get('name', '')
            
            # 도로명 주소 형식으로 조합
            address_parts = [part for part in [area1, area2, area3, area4, road_name] if part]
            return " ".join(address_parts)
    
    return "도로명 주소를 찾을 수 없습니다."

def get_building_info(address_query, client_id, client_secret):
    """
    주소 쿼리를 받아서 geocoding API를 호출하여 
    해당 주소와 관련된 건물 정보를 조회합니다.
    건물 이름이 addressElements 중 BUILDING_NAME 타입으로 포함되어 있다면 반환합니다.
    """
    geocode_url = "https://maps.apigw.ntruss.com/map-geocode/v2/geocode"
    params = {
        "query": address_query,
    }
    headers = {
        "x-ncp-apigw-api-key-id": client_id,
        "x-ncp-apigw-api-key": client_secret,
        "Accept": "application/json"
    }
    
    try:
        response = requests.get(geocode_url, params=params, headers=headers)
        response.raise_for_status()
        data = response.json()
        addresses = data.get("addresses", [])
        if not addresses:
            return "No building information"
        
        building_names = []
        for address in addresses:
            for element in address.get("addressElements", []):
                types = element.get("types", [])
                if "BUILDING_NAME" in types:
                    building_name = element.get("longName", "")
                    if building_name and building_name not in building_names:
                        building_names.append(building_name)
        if building_names:
            return "Nearby Building(s): " + ", ".join(building_names)
        else:
            return "No building name found in the geocode response."
    except requests.HTTPError as http_err:
        print(f"HTTP error in geocode API: {http_err} | Response: {response.text}")
        return None
    except Exception as e:
        print(f"Error in geocode API: {e}")
        return None

def get_nearby_building_info_local(address_query, client_id, client_secret):
    """
    네이버 지역 검색 API를 사용하여 입력한 주소(query)와 관련된
    주변 건물/업체 정보를 조회합니다.
    
    공식 문서에 따르면, 지역 검색 API는 아래 URL로 호출하며
    쿼리 파라미터에는 query, display, start, sort 등을 사용합니다.
    """
    url = "https://openapi.naver.com/v1/search/local.json"
    params = {
        "query": address_query,
        "display": 5,     # 한 번에 최대 5건 (최대값: 5)
        "start": 1,
        "sort": "random"  # random: 정확도순 내림차순 정렬
    }
    headers = {
        "X-Naver-Client-Id": client_id,
        "X-Naver-Client-Secret": client_secret,
        "Accept": "application/json"
    }
    try:
        response = requests.get(url, params=params, headers=headers)
        response.raise_for_status()
        data = response.json()
        items = data.get("items", [])
        if not items:
            return "주변 건물 정보가 없습니다."
        
        # 결과 항목에서 HTML 태그 제거 (예: <b>, </b> 등)
        def clean_text(text):
            return re.sub(r'<[^>]*>', '', text)
        
        building_info_list = []
        for item in items:
            title = clean_text(item.get("title", ""))
            road_address = item.get("roadAddress", "")
            address = item.get("address", "")
            category = item.get("category", "")
            # 도로명 주소가 없으면 일반 주소 사용
            display_address = road_address if road_address else address
            building_info_list.append(f"{title} ({category}): {display_address}")
        
        return "\n".join(building_info_list)
    except requests.HTTPError as http_err:
        print(f"HTTP error in local search API: {http_err} | Response: {response.text}")
        return None
    except Exception as e:
        print(f"Error in local search API: {e}")
        return None

if __name__ == "__main__":
    # 환경 변수에서 API 키 로드
    # reverse geocoding 및 geocoding은 NCP API 키 사용, 지역 검색은 네이버 오픈 API 키 사용
    # 만약 두 API의 키가 다르다면 별도로 관리해야 합니다.
    client_id = os.environ["X-NCP-APIGW-API-KEY-ID"].strip()
    client_secret = os.environ["X-NCP-APIGW-API-KEY"].strip()
    
    # 예시 좌표 (숭실대학교: 위도 37.5024, 경도 126.9389)
    coordinates = (37.5024, 126.9389)
    
    # 1. reverse geocoding 결과 출력 (등록된 도로명 주소)
    location_info = get_location_info(coordinates, client_id, client_secret)
    address_query = format_location_info(location_info)
    print("Reverse Geocode Info:")
    print(address_query)
        
    # 2. 기존 geocoding API를 통한 건물 정보 출력 (등록된 주소에 기반)
    building_info = get_building_info(address_query, client_id, client_secret)
    print("\nBuilding Information from Geocode API:")
    print(building_info)
    
    # 3. 지역 검색 API를 통한 주변 건물(업체) 정보 출력
    local_building_info = get_nearby_building_info_local("상도로 369 인근", "Fg2N4KsFo7WaEjX1qjp1", "qq6oUS6SAE")
    print("\nNearby Building Information from Local Search API:")
    print(local_building_info)
