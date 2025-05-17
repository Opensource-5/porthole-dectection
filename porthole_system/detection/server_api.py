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
import yaml
from typing import Dict, Optional, Set, Tuple

# 설정 파일 로드
def load_config(config_path='config.yaml'):
    """
    YAML 설정 파일을 로드합니다.
    
    Args:
        config_path: 설정 파일 경로
        
    Returns:
        설정 딕셔너리
    """
    try:
        with open(config_path, 'r', encoding='utf-8') as file:
            config = yaml.safe_load(file)
        return config
    except Exception as e:
        print(f"설정 파일 로드 중 오류 발생: {e}")
        print("기본 설정을 사용합니다.")
        return {}

# 전역 설정 로드
CONFIG = load_config()

# API 서버 URL 설정 (설정 파일에서 로드, 없으면 기본값 사용)
API_SERVER_URL = CONFIG.get('api', {}).get('server_url', 
    "https://statute-bradford-rebound-paragraph.trycloudflare.com/api/notify_new_porthole")


class PortholeServerAPI:
    """포트홀 정보 서버 전송을 위한 클래스"""
    
    def __init__(self, config=None):
        """
        PortholeServerAPI 초기화
        
        Args:
            config: 설정 딕셔너리 (None이면 전역 CONFIG 사용)
        """
        self.config = config or CONFIG
        self.sent_potholes = set()  # 중복 전송 방지를 위한 집합
        
        # API 설정 불러오기
        self.api_config = self.config.get('api', {})
        self.server_url = self.api_config.get('server_url', API_SERVER_URL)
        self.retry_count = self.api_config.get('retry_count', 3)
        self.timeout_seconds = self.api_config.get('timeout_seconds', 10)
    
    def send_pothole_data(self, lat: float, lng: float, depth: float) -> Optional[Dict]:
        """
        새로운 포트홀 정보를 API 서버로 전송합니다.
        
        Args:
            lat: 위도
            lng: 경도
            depth: 포트홀 깊이(cm)
            
        Returns:
            API 서버의 응답 또는 None (실패 시)
        """
        try:
            # 이미 전송된 포트홀인지 확인 (위도/경도 기반)
            location_key = (lat, lng)
            if location_key in self.sent_potholes:
                print(f"이미 전송된 포트홀 위치입니다: {location_key}")
                return None
                
            payload = {
                "lat": lat,
                "lng": lng,
                "depth": depth,
            }
            
            for attempt in range(self.retry_count):
                try:
                    response = requests.post(self.server_url, json=payload, timeout=self.timeout_seconds)
                    
                    if response.status_code == 200:
                        result = response.json()
                        print(f"포트홀 정보가 성공적으로 서버에 전송되었습니다.")
                        # 전송 완료된 위치 기록
                        
                        return result
                    else:
                        print(f"포트홀 정보 전송 실패. 상태 코드: {response.status_code}")
                        
                except requests.RequestException as e:
                    print(f"요청 시도 {attempt+1}/{self.retry_count} 실패: {e}")
                    
                # 마지막 시도가 아니면 잠시 대기 후 재시도
                if attempt < self.retry_count - 1:
                    time.sleep(1)
                    
            self.sent_potholes.add(location_key)

            return None

        except Exception as e:
            print(f"서버 전송 중 오류 발생: {e}")
            return None
    
    def clear_sent_cache(self):
        """전송된 포트홀 캐시를 초기화합니다."""
        self.sent_potholes.clear()
    
    def get_sent_count(self) -> int:
        """전송된 포트홀 수를 반환합니다."""
        return len(self.sent_potholes)
