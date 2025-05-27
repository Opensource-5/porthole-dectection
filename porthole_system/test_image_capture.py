#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
포트홀 이미지 캡처 기능 테스트 스크립트

이 스크립트는 포트홀 감지 시스템의 이미지 캡처 기능을 테스트합니다.
"""

import requests
import json
import base64
import cv2
import numpy as np
from io import BytesIO
from PIL import Image, ImageDraw, ImageFont

def create_test_image():
    """
    테스트용 포트홀 이미지를 생성합니다.
    """
    # 640x480 크기의 이미지 생성 (도로 색상)
    img = Image.new('RGB', (640, 480), color=(100, 100, 100))
    draw = ImageDraw.Draw(img)
    
    # 포트홀 모양 그리기 (타원형 구멍)
    # 중앙에 큰 포트홀
    draw.ellipse([250, 200, 390, 280], fill=(30, 30, 30), outline=(50, 50, 50), width=3)
    
    # 작은 포트홀들
    draw.ellipse([150, 150, 200, 180], fill=(40, 40, 40), outline=(60, 60, 60), width=2)
    draw.ellipse([450, 320, 480, 350], fill=(35, 35, 35), outline=(55, 55, 55), width=2)
    
    # 도로 선 그리기
    draw.line([(0, 240), (640, 240)], fill=(255, 255, 255), width=4)
    
    # 텍스트 추가
    try:
        # 기본 폰트 사용
        draw.text((10, 10), "Test Porthole Image", fill=(255, 255, 255))
        draw.text((10, 450), f"Generated for testing", fill=(255, 255, 255))
    except:
        pass
    
    return img

def image_to_base64(image):
    """
    PIL 이미지를 base64 문자열로 변환합니다.
    """
    buffer = BytesIO()
    image.save(buffer, format='JPEG', quality=90)
    img_str = base64.b64encode(buffer.getvalue()).decode()
    return img_str

def test_porthole_api():
    """
    포트홀 API에 테스트 이미지를 전송합니다.
    """
    print("🔄 포트홀 API 테스트 시작...")
    
    # 테스트 이미지 생성
    print("📸 테스트 이미지 생성 중...")
    test_image = create_test_image()
    image_base64 = image_to_base64(test_image)
    
    # API 엔드포인트 설정
    api_url = "http://localhost:8000/api/notify_new_porthole"
    
    # 테스트 데이터
    test_data = {
        "lat": 37.5665,  # 서울시청 근처
        "lng": 126.9780,
        "depth": 1500.0,  # 1.5mm
        "image": image_base64
    }
    
    try:
        print("📤 API 요청 전송 중...")
        response = requests.post(api_url, json=test_data)
        
        if response.status_code == 200:
            result = response.json()
            print("✅ API 요청 성공!")
            print(f"   포트홀 ID: {result.get('porthole_id')}")
            print(f"   이미지 경로: {result.get('image_path', '없음')}")
            return result.get('porthole_id')
        else:
            print(f"❌ API 요청 실패: {response.status_code}")
            print(f"   오류 메시지: {response.text}")
            return None
            
    except requests.exceptions.ConnectionError:
        print("❌ 서버 연결 실패! 백엔드 서버가 실행 중인지 확인하세요.")
        print("   실행 명령: python main.py")
        return None
    except Exception as e:
        print(f"❌ 예상치 못한 오류: {e}")
        return None

def test_image_retrieval(porthole_id):
    """
    저장된 이미지를 조회해봅니다.
    """
    if not porthole_id:
        return
        
    print(f"\n🔍 포트홀 {porthole_id} 상세 정보 조회 중...")
    
    try:
        # 포트홀 상세 정보 조회
        api_url = f"http://localhost:8000/api/portholes/{porthole_id}"
        response = requests.get(api_url)
        
        if response.status_code == 200:
            porthole_info = response.json()
            print("✅ 포트홀 정보 조회 성공!")
            print(f"   위치: {porthole_info.get('location', '정보 없음')}")
            print(f"   상태: {porthole_info.get('status', '정보 없음')}")
            print(f"   깊이: {porthole_info.get('depth', '정보 없음')} mm")
            
            image_path = porthole_info.get('image_path')
            if image_path:
                print(f"   이미지 경로: {image_path}")
                # 이미지 URL 생성
                image_url = f"http://localhost:8000{image_path}"
                print(f"   이미지 URL: {image_url}")
                
                # 이미지 접근 테스트
                print("🖼️  이미지 접근 테스트 중...")
                img_response = requests.get(image_url)
                if img_response.status_code == 200:
                    print("✅ 이미지 접근 성공!")
                    print(f"   이미지 크기: {len(img_response.content)} bytes")
                else:
                    print(f"❌ 이미지 접근 실패: {img_response.status_code}")
            else:
                print("   이미지 경로: 없음")
        else:
            print(f"❌ 포트홀 정보 조회 실패: {response.status_code}")
            print(f"   오류 메시지: {response.text}")
            
    except Exception as e:
        print(f"❌ 포트홀 정보 조회 중 오류: {e}")

def main():
    """
    메인 테스트 함수
    """
    print("🚧 포트홀 이미지 캡처 시스템 테스트")
    print("=" * 50)
    
    # 1. 포트홀 API 테스트
    porthole_id = test_porthole_api()
    
    if porthole_id:
        # 2. 이미지 조회 테스트
        test_image_retrieval(porthole_id)
        
        print(f"\n🎉 테스트 완료!")
        print(f"📋 프론트엔드에서 확인해보세요:")
        print(f"   URL: http://localhost:8501")
        print(f"   포트홀 ID {porthole_id}의 상세 정보를 확인하면 이미지가 표시됩니다.")
    else:
        print(f"\n❌ 테스트 실패!")
        print(f"💡 해결 방법:")
        print(f"   1. 백엔드 서버 실행: cd porthole_system && python main.py")
        print(f"   2. 서버 실행 후 다시 테스트: python test_image_capture.py")

if __name__ == "__main__":
    main()
