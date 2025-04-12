"""
포트홀 감지 모듈

이 모듈은 이미지나 센서 데이터로부터 포트홀을 감지하고 
API 서버에 새로운 포트홀 정보를 전송하는 기능을 제공합니다.
"""

import sqlite3
import datetime
import requests
import json
from typing import Dict, Optional, Tuple, Union
import os

# 데이터베이스 경로 (이제 조회용으로만 사용)
DATABASE_PATH = "porthole.db"

# API 서버 URL
API_SERVER_URL = "http://localhost:8000/api/notify_new_porthole"

def get_db_connection() -> sqlite3.Connection:
    """
    SQLite 데이터베이스 연결을 생성하고 반환합니다.
    
    Returns:
        sqlite3.Connection: 데이터베이스 연결 객체
    """
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def detect_porthole_from_image(image_path: str, location: str = "알 수 없음") -> Tuple[bool, Optional[Dict]]:
    """
    이미지로부터 포트홀을 감지하고 API 서버로 전송합니다.
    실제 구현에서는 여기에 포트홀 감지 알고리즘(예: 컴퓨터 비전)이 들어갑니다.
    
    Args:
        image_path (str): 분석할 이미지 파일 경로
        location (str): 위치 정보 (알 수 없는 경우 기본값 사용)
        
    Returns:
        Tuple[bool, Optional[Dict]]: (성공 여부, 감지된 포트홀 정보)
    """
    # 이 예제에서는 간단한 모의 구현을 제공합니다.
    # 실제로는 이미지 처리 라이브러리(OpenCV 등)를 사용하여 포트홀을 감지해야 합니다.
    try:
        print(f"이미지 분석 중: {image_path}")
        
        # 이미지 파일 존재 여부 확인
        if not os.path.exists(image_path):
            print(f"오류: 이미지 파일을 찾을 수 없습니다 - {image_path}")
            return False, None
        
        # 가상의 포트홀 감지 결과 (실제로는 여기서 이미지 분석 수행)
        # 예시: 위도/경도는 실제 값으로 대체필요
        detected = {
            'lat': 37.5665,  # 예시 위도 (실제 구현에서는 이미지 메타데이터 또는 입력에서 가져옴)
            'lng': 126.9780,  # 예시 경도
            'depth': 4.5,     # 감지된 깊이 cm
            'confidence': 0.85  # 감지 신뢰도
        }
        
        # 신뢰도가 임계값 이상인 경우에만 API 서버로 전송
        if detected['confidence'] >= 0.7:
            # API 서버로 포트홀 정보 전송
            send_result = send_porthole_to_server(
                detected['lat'], 
                detected['lng'], 
                detected['depth'], 
                location
            )
            
            if send_result and 'porthole_id' in send_result:
                detected['id'] = send_result['porthole_id']
                print(f"포트홀이 감지되어 API 서버에 전송되었습니다. ID: {send_result['porthole_id']}")
                return True, detected
        
        return False, None
    
    except Exception as e:
        print(f"포트홀 감지 중 오류 발생: {e}")
        return False, None

def detect_porthole_from_sensor(sensor_data: Dict, location: str = "알 수 없음") -> Tuple[bool, Optional[Dict]]:
    """
    센서 데이터로부터 포트홀을 감지하고 API 서버로 전송합니다.
    
    Args:
        sensor_data (Dict): 센서에서 수집한 데이터
        location (str): 위치 정보 (알 수 없는 경우 기본값 사용)
        
    Returns:
        Tuple[bool, Optional[Dict]]: (성공 여부, 감지된 포트홀 정보)
    """
    try:
        # 센서 데이터 유효성 확인 (필수 필드 존재 여부)
        required_fields = ['lat', 'lng', 'acceleration', 'timestamp']
        for field in required_fields:
            if field not in sensor_data:
                print(f"오류: 센서 데이터에 필요한 필드({field})가 없습니다.")
                return False, None
        
        # 가속도 데이터로 포트홀 존재 여부 판단
        # (실제로는 더 복잡한 알고리즘 필요)
        if abs(sensor_data['acceleration']) > 2.0:  # 임계값 이상의 진동이 감지되면
            depth = estimate_depth_from_acceleration(sensor_data['acceleration'])
            
            # API 서버로 포트홀 정보 전송
            send_result = send_porthole_to_server(
                sensor_data['lat'],
                sensor_data['lng'],
                depth,
                location
            )
            
            if send_result and 'porthole_id' in send_result:
                result = {
                    'id': send_result['porthole_id'],
                    'lat': sensor_data['lat'],
                    'lng': sensor_data['lng'],
                    'depth': depth,
                    'confidence': 0.8
                }
                print(f"센서에서 포트홀이 감지되어 API 서버에 전송되었습니다. ID: {send_result['porthole_id']}")
                return True, result
        
        return False, None
    
    except Exception as e:
        print(f"센서 데이터 처리 중 오류 발생: {e}")
        return False, None

def estimate_depth_from_acceleration(acceleration: float) -> float:
    """
    가속도 데이터로부터 대략적인 포트홀 깊이를 추정합니다.
    (이 함수는 단순한 예시이며, 실제로는 더 정교한 모델이 필요합니다)
    
    Args:
        acceleration (float): 측정된 가속도
        
    Returns:
        float: 추정된 깊이(cm)
    """
    # 단순화된 추정 공식 (실제로는 더 복잡한 모델 필요)
    return min(abs(acceleration) * 2.5, 15.0)  # 최대 15cm로 제한

def send_porthole_to_server(lat: float, lng: float, depth: float, location: str) -> Optional[Dict]:
    """
    새로운 포트홀 정보를 API 서버로 전송합니다.
    
    Args:
        lat (float): 위도
        lng (float): 경도
        depth (float): 포트홀 깊이(cm)
        location (str): 위치 설명
        
    Returns:
        Optional[Dict]: API 서버의 응답 또는 None (실패 시)
    """
    try:
        payload = {
            "lat": lat,
            "lng": lng,
            "depth": depth,
            "location": location
        }
        
        response = requests.post(API_SERVER_URL, json=payload)
        
        if response.status_code == 200:
            result = response.json()
            print(f"포트홀 정보가 성공적으로 서버에 전송되었습니다.")
            return result
        else:
            print(f"포트홀 정보 전송 실패. 상태 코드: {response.status_code}")
            return None
    
    except Exception as e:
        print(f"서버 전송 중 오류 발생: {e}")
        return None

# 직접 실행 시 테스트 코드
if __name__ == "__main__":
    # 명령줄에서 테스트를 위한 코드
    import sys
    
    if len(sys.argv) < 2:
        print("사용법: python detect.py [image|sensor] [file_path|sensor_data_json]")
        sys.exit(1)
    
    detection_type = sys.argv[1]
    
    if detection_type == "image" and len(sys.argv) >= 3:
        image_path = sys.argv[2]
        location = sys.argv[3] if len(sys.argv) >= 4 else "알 수 없음"
        success, result = detect_porthole_from_image(image_path, location)
        print(f"감지 결과: {'성공' if success else '실패'}")
        if success:
            print(f"포트홀 정보: {result}")
    
    elif detection_type == "sensor" and len(sys.argv) >= 3:
        try:
            sensor_data = json.loads(sys.argv[2])
            location = sys.argv[3] if len(sys.argv) >= 4 else "알 수 없음"
            success, result = detect_porthole_from_sensor(sensor_data, location)
            print(f"감지 결과: {'성공' if success else '실패'}")
            if success:
                print(f"포트홀 정보: {result}")
        except json.JSONDecodeError:
            print("오류: 올바른 JSON 형식의 센서 데이터가 필요합니다.")
    
    else:
        print("오류: 알 수 없는 감지 유형이거나 필요한 인자가 부족합니다.")


    # python detect.py sensor '{"lat":37.5665,"lng":126.9780,"acceleration":2.5,"timestamp":"2025-04-12T10:30:00"}' "서울시 중구 을지로"