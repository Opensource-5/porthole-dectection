import datetime
import requests
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os
from pathlib import Path
from openai import OpenAI


load_dotenv(override=True)

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
    processed_data['timestamp'] = datetime.datetime.now().isoformat()
    
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

# TTS
def synthesize_alert_audio(alert_message):
    
    # OpenAI 클라이언트 초기화
    client = OpenAI()  # OPENAI_API_KEY 환경 변수를 사용
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

    except Exception as e:
        print(f"Error synthesizing speech: {e}")
        return None

# Example usage:
# Assuming 'alert_message' was generated earlier,
# uncomment the following lines to synthesize and save the audio.
#
# alert_message = generate_alert_message(process_porthole_input(example_input))
# if alert_message:
#     synthesize_alert_audio(alert_message)

# Test with example input
if __name__ == "__main__":
    processed_data = process_porthole_input(example_input)
    if processed_data:
        print("Processed data:")
        print(processed_data)
        
        # LLM-based alert message 생성 테스트
        alert_message = generate_alert_message(processed_data)
        print("\nGenerated Alert Message:")
        print(alert_message)

        # TTS 음성 합성 테스트
        synthesize_alert_audio(alert_message)
        print("\nAudio synthesis completed.")
