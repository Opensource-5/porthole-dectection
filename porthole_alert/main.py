import datetime
import requests
import time
import json
import logging
import hashlib
from datetime import datetime
from typing import Dict, List, Optional, Any
from pathlib import Path
import threading
import queue

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os
from openai import OpenAI

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("porthole_alert.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("porthole_alert")

# 설정 변수
API_BASE_URL = "http://localhost:8000/api"  # FastAPI 서버 URL
CHECK_INTERVAL = 10  # 초 단위 확인 주기
PROXIMITY_THRESHOLD = 100  # 미터 단위 근접성 임계값

load_dotenv(override=True)

# 알림 큐 - 스트림릿 앱과 알림 시스템 간의 통신용
alert_queue = queue.Queue()

# 최근 알림을 저장하는 전역 변수 - 스트림릿에서 접근
latest_alerts = []
latest_audio_path = None

example_input = {
    "car_id": 1, 
    "distance": 90, 
    "depth": 1232, 
    "porthole_location": (37.5024, 126.9389), 
    "car_location": (37.502, 126.93)
}

def validate_porthole_data(data):
    """
    Validates porthole data received from sensors.
    
    Args:
        data (dict): Dictionary containing porthole information
    
    Returns:
        tuple: (bool, str) - (is_valid, error_message)
    """
    required_fields = ['car_id', 'distance', 'depth', 'porthole_location', 'car_location']
    
    # Check if all required fields are present
    for field in required_fields:
        if field not in data:
            return False, f"Missing required field: {field}"
    
    # Type validation
    if not isinstance(data['car_id'], int):
        return False, "car_id must be an integer"
    if not isinstance(data['distance'], (int, float)):
        return False, "distance must be a number"
    if not isinstance(data['depth'], (int, float)):
        return False, "depth must be a number"
    
    # For porthole_location and car_location, expect a tuple or list with two numbers
    for field in ['porthole_location', 'car_location']:
        value = data[field]
        if not (isinstance(value, (tuple, list)) and len(value) == 2):
            return False, f"{field} must be a tuple or list of two numbers"
        if not all(isinstance(coord, (int, float)) for coord in value):
            return False, f"Both coordinates in {field} must be numbers"
    
    # Value validation
    if data['distance'] < 0:
        return False, "distance cannot be negative"
    if data['depth'] <= 0:
        return False, "depth must be positive"
    
    return True, ""

def preprocess_porthole_data(data):
    """
    Preprocesses and normalizes porthole data.
    
    Args:
        data (dict): Validated porthole data
    
    Returns:
        dict: Preprocessed data
    """
    processed_data = data.copy()
    
    # Round numerical values for consistency
    processed_data['distance'] = round(float(processed_data['distance']), 2)
    processed_data['depth'] = round(float(processed_data['depth']), 2)
    
    # Round each coordinate in porthole_location and car_location
    processed_data['porthole_location'] = (
        round(float(processed_data['porthole_location'][0]), 2),
        round(float(processed_data['porthole_location'][1]), 2)
    )
    processed_data['car_location'] = (
        round(float(processed_data['car_location'][0]), 2),
        round(float(processed_data['car_location'][1]), 2)
    )
    
    # Add derived fields if needed
    processed_data['timestamp'] = datetime.now().isoformat()
    
    return processed_data

def assess_risk(processed_data):
    """
    Assesses risk level based on porthole depth.
    
    Args:
        processed_data (dict): Preprocessed porthole data
    
    Returns:
        dict: Updated data including risk assessment
    """
    depth = processed_data['depth']
    
    # Risk assessment thresholds (adjust these values if needed)
    if depth < 500:
        risk = "Low"
    elif depth < 1500:
        risk = "Medium"
    else:
        risk = "High"
    
    processed_data['risk_level'] = risk
    return processed_data

def process_porthole_input(data):
    """
    Main function to process porthole data from sensors.
    
    Args:
        data (dict): Raw porthole data
    
    Returns:
        dict: Processed data including risk level or None if validation fails
    """
    # Validate input data
    is_valid, error_message = validate_porthole_data(data)
    if not is_valid:
        print(f"Invalid porthole data: {error_message}")
        return None
    
    # Preprocess data
    processed_data = preprocess_porthole_data(data)

    # Enrich data with location information
    # Load environment variables from .env file
    
    # Get API credentials from environment variables
    client_id = os.getenv("X-NCP-APIGW-API-KEY-ID")
    client_secret = os.getenv("X-NCP-APIGW-API-KEY")
    
    if not client_id or not client_secret:
        print("Warning: Naver Maps API credentials not found in environment variables")
        return processed_data
    
    processed_data = enrich_porthole_data_with_location(processed_data, client_id, client_secret)
    
    # Assess risk based on porthole depth
    processed_data = assess_risk(processed_data)
    
    return processed_data

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
    '시도 시군구 읍면동 도로명'의 형식으로 도로명 주소를 반환합니다.
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

def get_nearby_building_info(address_query, client_id, client_secret):
    """
    주소 쿼리를 받아서 geocoding API를 호출하여 
    해당 주소와 관련된 건물 정보를 조회합니다.
    건물 이름이 addressElements 중 BUILDING_NAME 타입으로 포함되어 있다면 반환합니다.
    """
    # geocoding API를 호출하여 건물 정보를 조회합니다.
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
        
        # geocoding 결과에서 addressElements를 순회하며 BUILDING_NAME 타입을 찾습니다.
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


def enrich_porthole_data_with_location(processed_data, client_id, client_secret):
    # 수정: 반환된 리스트에서 "addr" 결과 우선 또는 첫 번째 결과의 region 사용
    lat, lng = processed_data["porthole_location"]
    location_results = get_location_info((lat, lng), client_id, client_secret)
    if location_results:
        # 포트홀 위치 정보 추가
        processed_data["location_info"] = format_location_info(location_results)
    else:  
        print("Warning: No location information found for the porthole location.")
        processed_data["location_info"] = "No location information available."
    return processed_data

# LLM-based alert message generation
def generate_alert_message(processed_data):
    """
    Generates an alert message based on risk level using LangChain with a system prompt.
    
    Args:
        processed_data (dict): Processed porthole data including risk_level.
    
    Returns:
        str: Generated alert message.
    """
    risk_level = processed_data.get('risk_level', 'Unknown')

    # Base instruction based on risk level
    if risk_level == "Low":
        alert_detail = "포트홀의 깊이가 그리 깊지는 않으나 주의가 필요합니다."
    elif risk_level == "Medium":
        alert_detail = "사고를 유발할 수 있는 정도의 포트홀 깊이입니다."
    elif risk_level == "High":
        alert_detail = "위험한 포트홀입니다. 즉각적인 조치가 필요합니다."
    else:
        alert_detail = "상황을 확인할 수 없습니다."

    # Get OpenAI API key from environment variables
    openai_api_key = os.getenv("OPENAI_API_KEY")
    
    if not openai_api_key:
        print("Error: OpenAI API key not found in environment variables")
        return "Unable to generate alert: API key missing"

    # 최신 방식의 프롬프트 템플릿 생성
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant that generates clear, concise, and actionable alert messages based on sensor data."),
        ("human", """
        상황: 차량 인근에 포트홀이 발견되었습니다.
        차량과 포트홀 거리: {distance}m, 포트홀 깊이: {depth}cm.
        위험도: {risk_level}, {alert_detail}
        포트홀 위치: {location}

        위치와 상황을 종합(위험도 포함)하여 운전자에게 경고 알림 메시지를 운전자가 이해하기 쉽게 작성해 주세요.
        """)
    ])
    
    # 최신 LCEL 체이닝 방식 사용
    model = ChatOpenAI(model="gpt-4o-mini", temperature=0.9)
    chain = prompt | model | StrOutputParser()

    # 체인 실행
    location_info = processed_data.get("location_info", {})
    
    alert = chain.invoke({
        "car_id": processed_data["car_id"],
        "distance": processed_data["distance"],
        "depth": processed_data["depth"],
        "risk_level": risk_level,
        "location": location_info,
        "alert_detail": alert_detail
    })
    
    return alert

# 기존 함수 수정: TTS 함수에서 파일 경로를 반환하도록 함
def synthesize_alert_audio(alert_message):
    """음성 합성을 수행하고 생성된 오디오 파일 경로를 반환합니다."""
    
    # OpenAI API 키를 환경 변수에서 명시적으로 가져와 전달
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        print("Error: OpenAI API key not found in environment variables")
        return None
        
    # OpenAI 클라이언트 초기화 - API 키 명시적 전달
    client = OpenAI(api_key=openai_api_key)
    speech_file_path = Path(__file__).parent / "speech.mp3"

    try:
        # 새로운 API 방식으로 음성 합성 요청
        with client.audio.speech.with_streaming_response.create(
            model="gpt-4o-mini-tts",
            voice="coral",
            input=alert_message,
            instructions="Speak in a warning tone."
        ) as response:
            response.stream_to_file(speech_file_path)
            
        return str(speech_file_path)

    except Exception as e:
        print(f"Error synthesizing speech: {e}")
        return None

class PortholeAlert:
    """포트홀 알림 시스템 클래스"""
    
    def __init__(self):
        self.last_alerts = {}  # 마지막으로 전송된 알림을 저장 (차량 ID를 키로 사용)
        self.alert_cooldown = 60  # 동일한 포트홀에 대한 알림 재발송까지의 대기 시간(초)
        self.alert_contents = {}  # 각 차량에 대한 마지막 알림 내용 저장
    
    def check_proximity(self) -> Dict[str, Any]:
        """
        모든 차량과 포트홀 사이의 거리를 확인하여 알림 정보를 반환합니다.
        
        Returns:
            Dict[str, Any]: 알림 정보를 포함한 딕셔너리
        """
        try:
            response = requests.get(f"{API_BASE_URL}/check_proximity")
            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"근접성 확인 API 요청 실패: {response.status_code}")
                return {"alerts": []}
        except Exception as e:
            logger.exception(f"근접성 확인 중 오류 발생: {e}")
            return {"alerts": []}
    
    def filter_discovered_portholes(self, alerts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        '발견됨' 상태의 포트홀만 필터링합니다.
        """
        filtered_alerts = []
        
        for alert in alerts:
            discovered_portholes = [
                p for p in alert.get('alerts', []) 
                if p.get('status') == '발견됨'
            ]
            
            # 발견된 포트홀이 있는 경우만 알림 목록에 추가
            if discovered_portholes:
                filtered_alert = alert.copy()
                filtered_alert['alerts'] = discovered_portholes
                filtered_alerts.append(filtered_alert)
        
        return filtered_alerts
    
    def should_send_alert(self, car_id: int, porthole_id: int) -> bool:
        """특정 차량-포트홀 조합에 대한 알림을 보내야 하는지 확인합니다."""
        alert_key = f"{car_id}_{porthole_id}"
        current_time = time.time()
        
        # 이전에 이 알림이 전송된 적이 있는지 확인
        if alert_key in self.last_alerts:
            last_time = self.last_alerts[alert_key]
            # 쿨다운 시간이 지났는지 확인
            if current_time - last_time < self.alert_cooldown:
                return False  # 아직 쿨다운 중
        
        # 현재 시간을 기록하고 알림 전송
        self.last_alerts[alert_key] = current_time
        return True
    
    def get_alert_hash(self, car_id: int, nearby_portholes: List[Dict[str, Any]]) -> str:
        """알림 내용에 대한 해시값을 생성하여 반환합니다."""
        # 알림 내용의 핵심 정보를 문자열로 변환
        content = f"{car_id}"
        
        # 포트홀 정보 추가 (ID, 거리, 위치)
        for porthole in nearby_portholes:
            content += f"_{porthole['porthole_id']}_{porthole['distance']}_{porthole['location']}"
        
        # 해시 생성 및 반환
        return hashlib.md5(content.encode()).hexdigest()
    
    def is_duplicate_alert(self, car_id: int, nearby_portholes: List[Dict[str, Any]]) -> bool:
        """현재 알림이 이전에 보낸 알림과 동일한 내용인지 확인합니다."""
        # 알림 내용의 해시 생성
        alert_hash = self.get_alert_hash(car_id, nearby_portholes)
        
        # 이전에 동일한 알림을 보냈는지 확인
        if car_id in self.alert_contents and self.alert_contents[car_id] == alert_hash:
            return True
        
        # 새로운 알림 내용 저장
        self.alert_contents[car_id] = alert_hash
        return False
    
    def send_alert(self, car_id: int, nearby_portholes: List[Dict[str, Any]]) -> None:
        """차량에 포트홀 알림을 전송하고 LangChain 및 TTS 처리를 수행합니다."""
        # 중복 알림 확인
        if self.is_duplicate_alert(car_id, nearby_portholes):
            logger.info(f"차량 {car_id}에 대한 중복 알림 감지됨 - 알림 전송 생략")
            return
        
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # 가장 가까운 포트홀 (이미 거리순으로 정렬되어 있음)
        closest_porthole = nearby_portholes[0]
        
        # 알림 로깅
        logger.info(f"차량 {car_id}에 포트홀 알림 전송: "
                   f"포트홀 ID {closest_porthole['porthole_id']}, "
                   f"거리 {closest_porthole['distance']}m, "
                   f"위치 {closest_porthole['location']}")
        
        # 알림 콘솔 출력
        print("\n" + "="*50)
        print(f"⚠️ 포트홀 알림! ⚠️ - {now}")
        print(f"차량 ID: {car_id}")
        print(f"가장 가까운 포트홀: ID {closest_porthole['porthole_id']}")
        print(f"위치: {closest_porthole['location']}")
        print(f"거리: {closest_porthole['distance']}m")
        print(f"깊이: {closest_porthole.get('depth', '정보 없음')} cm")
        
        # 추가 포트홀 정보 표시 (있는 경우)
        if len(nearby_portholes) > 1:
            print(f"\n주변에 {len(nearby_portholes)-1}개의 추가 포트홀이 있습니다.")
            for i, porthole in enumerate(nearby_portholes[1:], 1):
                print(f"  {i}. ID {porthole['porthole_id']} - "
                     f"거리: {porthole['distance']}m, "
                     f"위치: {porthole['location']}")
        
        print("="*50 + "\n")
        
        # 여기서 LangChain 및 TTS 처리
        # 포트홀 데이터를 기존 형식으로 변환
        porthole_data = {
            "car_id": car_id,
            "distance": closest_porthole['distance'],
            "depth": closest_porthole.get('depth', 500),  # 깊이 정보가 없으면 기본값 사용
            "porthole_location": closest_porthole.get('coordinates', (37.5024, 126.9389)),  # 기본 좌표 사용
            "car_location": closest_porthole.get('car_coordinates', (37.502, 126.93)),  # 기본 좌표 사용
            "location_info": closest_porthole.get('location', '주소 정보 없음'),
            "risk_level": "High" if closest_porthole.get('depth', 500) > 1000 else 
                          "Medium" if closest_porthole.get('depth', 500) > 500 else "Low"
        }
        
        # 포트홀 데이터 처리
        processed_data = porthole_data
        
        # LangChain으로 알림 메시지 생성
        alert_message = generate_alert_message(processed_data)
        
        # TTS로 음성 생성
        audio_path = synthesize_alert_audio(alert_message)
        
        # 알림 큐에 추가 (스트림릿 앱에서 처리)
        global latest_alerts, latest_audio_path
        alert_info = {
            "timestamp": now,
            "car_id": car_id,
            "message": alert_message,
            "porthole_info": closest_porthole,
            "audio_path": audio_path
        }
        alert_queue.put(alert_info)
        
        # 최신 알림 저장 (스트림릿이 실시간으로 접근할 수 있도록)
        latest_alerts.append(alert_info)
        if len(latest_alerts) > 5:  # 최대 5개까지만 유지
            latest_alerts.pop(0)
        
        # 최신 오디오 경로 저장
        latest_audio_path = audio_path
    
    def process_alerts(self, proximity_data: Dict[str, Any]) -> None:
        """근접성 데이터를 처리하여 필요한 알림을 전송합니다."""
        alerts = proximity_data.get('alerts', [])
        
        # '발견됨' 상태의 포트홀만 필터링
        discovered_alerts = self.filter_discovered_portholes(alerts)
        
        # 각 차량에 대한 알림 처리
        for alert in discovered_alerts:
            car_id = alert['car_id']
            nearby_portholes = alert['alerts']
            
            if nearby_portholes:
                # 각 포트홀에 대해 알림 전송 여부 확인
                for porthole in nearby_portholes:
                    porthole_id = porthole['porthole_id']
                    
                    if self.should_send_alert(car_id, porthole_id):
                        # 알림 전송
                        self.send_alert(car_id, nearby_portholes)
                        # 한 차량당 한 번만 알림 전송 (가장 가까운 포트홀 기준)
                        break
    
    def run(self, single_run: bool = False) -> None:
        """알림 시스템 실행 메인 루프"""
        logger.info(f"포트홀 알림 시스템 시작 (확인 주기: {CHECK_INTERVAL}초)")
        
        try:
            while True:
                # 테스트 모드인 경우: 서버 연결 없이 예시 데이터 사용
                if self._use_test_data:
                    proximity_data = self._get_test_proximity_data()
                else:
                    # 서버로부터 근접성 데이터 가져오기
                    proximity_data = self.check_proximity()
                
                # 알림 처리
                self.process_alerts(proximity_data)
                
                if single_run:
                    break
                
                # 다음 확인까지 대기
                time.sleep(CHECK_INTERVAL)
                
        except KeyboardInterrupt:
            logger.info("사용자에 의해 포트홀 알림 시스템이 중지되었습니다.")
        except Exception as e:
            logger.exception(f"알림 시스템 실행 중 오류 발생: {e}")
    
    # 테스트용 메서드 추가
    def _get_test_proximity_data(self) -> Dict[str, Any]:
        """테스트용 근접성 데이터 생성"""
        return {
            "alerts": [
                {
                    "car_id": 1,
                    "alerts": [
                        {
                            "porthole_id": 101,
                            "distance": 90,
                            "depth": 1232,
                            "location": "서울특별시 강남구 역삼로 123",
                            "status": "발견됨",
                            "coordinates": (37.5024, 126.9389),
                            "car_coordinates": (37.502, 126.93)
                        }
                    ]
                }
            ]
        }
    
    def set_test_mode(self, use_test_data: bool = True) -> None:
        """테스트 모드 설정 - 테스트 데이터 사용 여부"""
        self._use_test_data = use_test_data

# 알림 시스템 백그라운드 스레드로 실행
def run_alert_system_background(test_mode=True):
    """백그라운드 스레드에서 포트홀 알림 시스템을 실행합니다."""
    alert_system = PortholeAlert()
    alert_system.set_test_mode(test_mode)  # 테스트 모드 설정
    
    # 별도 스레드에서 실행
    thread = threading.Thread(target=alert_system.run)
    thread.daemon = True  # 메인 스레드가 종료되면 함께 종료되도록 설정
    thread.start()
    
    return thread

# 알림 큐에서 알림을 가져오는 함수 (스트림릿 앱에서 사용)
def get_latest_alert():
    """알림 큐에서 가장 최근 알림을 가져옵니다. 없으면 None을 반환합니다."""
    if not alert_queue.empty():
        return alert_queue.get()
    return None

# 테스트용 메인 코드
if __name__ == "__main__":
    # 포트홀 알림 시스템 백그라운드로 실행
    alert_thread = run_alert_system_background(test_mode=True)
    
    # 기존 예시 코드도 그대로 유지
    processed_data = process_porthole_input(example_input)
    if processed_data:
        print("Processed data:")
        print(processed_data)
        
        # LLM-based alert message 생성 테스트
        alert_message = generate_alert_message(processed_data)
        print("\nGenerated Alert Message:")
        print(alert_message)

        # TTS 음성 합성 테스트
        audio_path = synthesize_alert_audio(alert_message)
        print(f"\nAudio synthesis completed: {audio_path}")
    
    # 메인 스레드가 즉시 종료되지 않도록 대기
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("프로그램이 종료되었습니다.")
