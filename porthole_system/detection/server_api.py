#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
포트홀 서버 통신 모듈 (Porthole Server API Module)

이 모듈은 감지된 포트홀 정보를 API 서버로 전송하는 기능을 제공합니다.

주요 기능:
1. 포트홀 정보 전송
2. 서버 응답 처리
3. 재시도 메커니즘
"""

import time
import requests
from typing import Dict, Optional, Set

# 로컬 모듈 임포트
from config_utils import get_global_config, get_nested_value


class PortholeServerAPI:
    """포트홀 정보 서버 전송을 위한 클래스"""
    
    def __init__(self, config: Optional[Dict] = None):
        """
        PortholeServerAPI 초기화
        
        Args:
            config: 설정 딕셔너리 (None이면 전역 설정 사용)
        """
        self.config = config or get_global_config()
        self.sent_potholes: Set[tuple] = set()  # 중복 전송 방지를 위한 집합
        
        # API 설정 불러오기
        self.server_url = get_nested_value(self.config, 'api.server_url', 
                                         "https://statute-bradford-rebound-paragraph.trycloudflare.com/api/notify_new_porthole")
        self.retry_count = get_nested_value(self.config, 'api.retry_count', 3)
        self.timeout_seconds = get_nested_value(self.config, 'api.timeout_seconds', 10)
        
        # 디버그 설정
        self.print_api_responses = get_nested_value(self.config, 'debug.print_api_responses', True)
    
    def send_pothole_data(self, lat: float, lng: float, depth: float) -> bool:
        """
        새로운 포트홀 정보를 API 서버로 전송합니다.
        
        Args:
            lat: 위도
            lng: 경도
            depth: 포트홀 깊이(mm)
            
        Returns:
            bool: 전송 성공 여부
        """
        try:
            # 이미 전송된 포트홀인지 확인 (위도/경도 기반)
            location_key = (round(lat, 6), round(lng, 6))  # 소수점 6자리로 반올림
            if location_key in self.sent_potholes:
                if self.print_api_responses:
                    print(f"ℹ️  이미 전송된 포트홀 위치입니다: {location_key}")
                return False
                
            payload = {
                "lat": lat,
                "lng": lng,
                "depth": depth,
            }
            
            if self.print_api_responses:
                print(f"📡 서버로 포트홀 정보 전송 중: 위도={lat}, 경도={lng}, 깊이={depth}mm")
            
            for attempt in range(self.retry_count):
                try:
                    response = requests.post(
                        self.server_url, 
                        json=payload, 
                        timeout=self.timeout_seconds
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        if self.print_api_responses:
                            print(f"✅ 포트홀 정보가 성공적으로 서버에 전송되었습니다.")
                            print(f"📄 서버 응답: {result}")
                        
                        # 전송 완료된 위치 기록
                        self.sent_potholes.add(location_key)
                        return True
                    else:
                        if self.print_api_responses:
                            print(f"❌ 포트홀 정보 전송 실패. 상태 코드: {response.status_code}")
                            print(f"응답 내용: {response.text}")
                        
                except requests.RequestException as e:
                    if self.print_api_responses:
                        print(f"⚠️  요청 시도 {attempt+1}/{self.retry_count} 실패: {e}")
                    
                # 마지막 시도가 아니면 잠시 대기 후 재시도
                if attempt < self.retry_count - 1:
                    if self.print_api_responses:
                        print(f"⏳ {attempt+1}초 대기 후 재시도...")
                    time.sleep(attempt + 1)  # 점진적으로 대기 시간 증가
            
            if self.print_api_responses:
                print(f"❌ 모든 재시도 실패. 포트홀 정보 전송을 포기합니다.")
            return False

        except Exception as e:
            if self.print_api_responses:
                print(f"❌ 서버 전송 중 예외 발생: {e}")
            return False
    
    def clear_sent_cache(self) -> None:
        """전송된 포트홀 캐시를 초기화합니다."""
        self.sent_potholes.clear()
        if self.print_api_responses:
            print("🗑️  전송된 포트홀 캐시가 초기화되었습니다.")
    
    def get_sent_count(self) -> int:
        """전송된 포트홀 수를 반환합니다."""
        return len(self.sent_potholes)
    
    def get_sent_locations(self) -> Set[tuple]:
        """전송된 포트홀 위치 목록을 반환합니다."""
        return self.sent_potholes.copy()
        
    def is_location_sent(self, lat: float, lng: float) -> bool:
        """
        특정 위치가 이미 전송되었는지 확인합니다.
        
        Args:
            lat: 위도
            lng: 경도
            
        Returns:
            전송 여부
        """
        location_key = (round(lat, 6), round(lng, 6))
        return location_key in self.sent_potholes


# 하위 호환성을 위한 설정 로드 함수
def load_config(config_path: str = 'config.yaml') -> Dict:
    """
    하위 호환성을 위한 설정 로드 함수
    
    Args:
        config_path: 설정 파일 경로
        
    Returns:
        설정 딕셔너리
    """
    return get_global_config()
