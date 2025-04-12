"""
포트홀 알림 시스템 (alert.py)

이 모듈은 차량과 포트홀 간의 근접성을 주기적으로 확인하고,
차량이 발견된 상태의 포트홀 근처(100m 이내)에 있을 때 알림을 생성합니다.
"""

import requests
import time
import json
import logging
import hashlib
from datetime import datetime
from typing import Dict, List, Optional, Any

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
        
        Args:
            alerts: 포트홀 알림 목록
            
        Returns:
            List[Dict[str, Any]]: 발견됨 상태의 포트홀 알림만 포함된 목록
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
        """
        특정 차량-포트홀 조합에 대한 알림을 보내야 하는지 확인합니다.
        같은 알림이 쿨다운 시간 내에 다시 전송되지 않도록 합니다.
        
        Args:
            car_id: 차량 ID
            porthole_id: 포트홀 ID
            
        Returns:
            bool: 알림을 보내야 하면 True, 아니면 False
        """
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
        """
        알림 내용에 대한 해시값을 생성하여 반환합니다.
        이 해시는 동일한 알림 내용을 식별하는 데 사용됩니다.
        
        Args:
            car_id: 차량 ID
            nearby_portholes: 근처의 포트홀 목록
            
        Returns:
            str: 알림 내용의 해시값
        """
        # 알림 내용의 핵심 정보를 문자열로 변환
        content = f"{car_id}"
        
        # 포트홀 정보 추가 (ID, 거리, 위치)
        for porthole in nearby_portholes:
            content += f"_{porthole['porthole_id']}_{porthole['distance']}_{porthole['location']}"
        
        # 해시 생성 및 반환
        return hashlib.md5(content.encode()).hexdigest()
    
    def is_duplicate_alert(self, car_id: int, nearby_portholes: List[Dict[str, Any]]) -> bool:
        """
        현재 알림이 이전에 보낸 알림과 동일한 내용인지 확인합니다.
        
        Args:
            car_id: 차량 ID
            nearby_portholes: 근처의 포트홀 목록
            
        Returns:
            bool: 중복 알림이면 True, 아니면 False
        """
        # 알림 내용의 해시 생성
        alert_hash = self.get_alert_hash(car_id, nearby_portholes)
        
        # 이전에 동일한 알림을 보냈는지 확인
        if car_id in self.alert_contents and self.alert_contents[car_id] == alert_hash:
            return True
        
        # 새로운 알림 내용 저장
        self.alert_contents[car_id] = alert_hash
        return False
    
    def send_alert(self, car_id: int, nearby_portholes: List[Dict[str, Any]]) -> None:
        """
        차량에 포트홀 알림을 전송합니다.
        실제 시스템에서는 이 메서드를 사용하여 차량으로 알림을 보냅니다.
        
        Args:
            car_id: 차량 ID
            nearby_portholes: 근처의 포트홀 목록
        """
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
        
        # 여기에 실제 차량 알림 시스템 연동 코드 추가 가능
        # 예: 모바일 앱 푸시 알림, 차량 내비게이션 시스템 알림 등
    
    def process_alerts(self, proximity_data: Dict[str, Any]) -> None:
        """
        근접성 데이터를 처리하여 필요한 알림을 전송합니다.
        
        Args:
            proximity_data: 근접성 확인 API 응답 데이터
        """
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
        """
        알림 시스템 실행 메인 루프
        
        Args:
            single_run: True이면 한 번만 실행, False이면 무한 루프 (기본값)
        """
        logger.info(f"포트홀 알림 시스템 시작 (확인 주기: {CHECK_INTERVAL}초)")
        
        try:
            while True:
                # 근접성 데이터 가져오기
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

# 메인 실행 코드
if __name__ == "__main__":
    alert_system = PortholeAlert()
    alert_system.run()