import math
from typing import Dict, List, Tuple
from backend.crud import get_all_portholes, get_all_cars, add_alert, get_car_by_id, get_porthole_by_id
from backend.constants import AlertSettings

# 포트홀 근접 알림을 위한 거리 임계값 (미터 단위)
PROXIMITY_THRESHOLD = AlertSettings.MAX_DISTANCE  # constants.py에서 값을 가져옴

def calculate_distance(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    """
    두 지점(위도/경도) 간의 거리를 미터 단위로 계산합니다.
    Haversine 공식을 사용합니다.
    
    Args:
        lat1: 첫 번째 지점의 위도
        lng1: 첫 번째 지점의 경도
        lat2: 두 번째 지점의 위도
        lng2: 두 번째 지점의 경도
        
    Returns:
        float: 두 지점 간의 거리(미터)
    """
    # 지구 반경 (미터)
    R = 6371 * 1000
    
    # 위도, 경도 차이를 라디안으로 변환
    dLat = math.radians(lat2 - lat1)
    dLng = math.radians(lng2 - lng1)
    
    # Haversine 공식
    a = math.sin(dLat/2) * math.sin(dLat/2) + \
        math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * \
        math.sin(dLng/2) * math.sin(dLng/2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    
    # 거리 계산
    distance = R * c
    
    return distance

def check_car_porthole_proximity() -> List[Dict]:
    """
    모든 차량과 포트홀 간의 거리를 확인하고, 
    임계값(PROXIMITY_THRESHOLD) 이내에 있는 경우 알림을 생성합니다.
    
    Returns:
        List[Dict]: 생성된 알림 정보 목록
    """
    try:
        cars = get_all_cars()
        portholes = get_all_portholes()
        
        created_alerts = []
        
        for car in cars:
            car_lat = car["lat"]
            car_lng = car["lng"]
            car_id = car["id"]
            
            for porthole in portholes:
                try:
                    porthole_details = get_porthole_by_id(porthole["id"])
                    if not porthole_details:
                        print(f"경고: 포트홀 ID {porthole['id']}를 찾을 수 없습니다.")
                        continue
                        
                    # 수리 완료된 포트홀은 알림에서 제외
                    if porthole_details.get("status") == "수리완료":
                        continue
                        
                    porthole_lat = porthole_details["lat"]
                    porthole_lng = porthole_details["lng"]
                    porthole_id = porthole_details["id"]
                    
                    # 차량과 포트홀 간의 거리 계산
                    distance = calculate_distance(car_lat, car_lng, porthole_lat, porthole_lng)
                    
                    # 임계값 이내에 있는 경우 알림 생성
                    if distance <= PROXIMITY_THRESHOLD:
                        try:
                            # ONE_TIME_ALERT가 True인 경우 포트홀에 대해 한 번만 알림 생성
                            alert_id = add_alert(car_id, porthole_id, distance, AlertSettings.ONE_TIME_ALERT)
                            
                            # -1은 one_time_only=True이고 이미 알림이 있었을 경우
                            if alert_id != -1:
                                created_alerts.append({
                                    "alert_id": alert_id,
                                    "car_id": car_id,
                                    "porthole_id": porthole_id,
                                    "distance": distance
                                })
                        except ValueError as ve:
                            print(f"알림 추가 중 유효성 오류: {ve}")
                        except Exception as e:
                            print(f"알림 추가 중 오류 발생: {e}")
                except Exception as e:
                    print(f"포트홀 ID {porthole['id']} 처리 중 오류: {e}")
                    continue
        
        return created_alerts
    except Exception as e:
        print(f"차량-포트홀 근접성 확인 중 오류: {e}")
        return []
