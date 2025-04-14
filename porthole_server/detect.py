"""
포트홀 감지 모듈

이 모듈은 이미지나 센서 데이터로부터 포트홀을 감지하고 
API 서버에 새로운 포트홀 정보를 전송하는 기능을 제공합니다.
"""

import requests
import json
from typing import Dict, Optional, Tuple, Union, List
import os
import cv2
import torch
import numpy as np

# API 서버 URL
API_SERVER_URL = "http://localhost:8000/api/notify_new_porthole"

# 모델 로드를 위한 전역 변수
yolo_model = None
midas = None
transform = None
device = None

def load_models():
    """
    YOLOv5와 MiDaS 모델을 로드합니다.
    """
    global yolo_model, midas, transform, device
    
    if yolo_model is not None:
        return  # 이미 모델이 로드되었으면 다시 로드하지 않음
    
    try:
        print("포트홀 감지 및 깊이 추정 모델 로드 중...")
        
        # 커스텀 학습된 YOLOv5 모델 로드 (포트홀 탐지용)
        yolo_model = torch.hub.load(
            'ultralytics/yolov5', 'custom',
            path='/content/yolov5/runs/train/gun_yolov5s_results/weights/best.pt'
        )
        yolo_model.eval()

        # MiDaS 모델 로드 (깊이 추정용)
        model_type = "DPT_Hybrid"  # 빠른 처리용 소형 모델
        midas = torch.hub.load("intel-isl/MiDaS", model_type)

        # CPU 또는 GPU 장치 설정
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        midas.to(device)
        midas.eval()

        # MiDaS 입력 이미지 변환 함수 로딩
        midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
        transform = midas_transforms.small_transform
        
        print(f"모델 로드 완료 (장치: {device})")
        return True
        
    except Exception as e:
        print(f"모델 로드 중 오류 발생: {e}")
        return False

def detect_porthole_with_models(image_path: str) -> Tuple[bool, List[Dict]]:
    """
    YOLOv5와 MiDaS 모델을 사용하여 이미지에서 포트홀을 감지하고 깊이를 추정합니다.
    
    Args:
        image_path (str): 분석할 이미지 파일 경로
        
    Returns:
        Tuple[bool, List[Dict]]: (포트홀 감지 여부, 감지된 포트홀 정보 리스트)
    """
    try:
        # 모델 로드 확인
        if not load_models():
            return False, []
            
        # 이미지 파일 읽기
        frame = cv2.imread(image_path)
        if frame is None:
            print(f"오류: 이미지를 열 수 없습니다: {image_path}")
            return False, []

        # ====== MiDaS를 통한 깊이 추정 ======
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # BGR -> RGB 변환
        input_batch = transform(frame_rgb).to(device)        # MiDaS 전처리

        with torch.no_grad():
            depth_prediction = midas(input_batch)             # 깊이 추정 실행
            depth_prediction = torch.nn.functional.interpolate(
                depth_prediction.unsqueeze(1),
                size=frame.shape[:2],  # 원본 이미지 크기로 보간
                mode="bicubic",
                align_corners=False,
            ).squeeze()

        # 깊이 맵 후처리: 정규화 및 컬러맵 적용
        depth_map = depth_prediction.cpu().numpy()
        depth_map_normalized = cv2.normalize(depth_map, None, 0, 1, norm_type=cv2.NORM_MINMAX)
        depth_map_uint8 = (depth_map_normalized * 255).astype(np.uint8)
        depth_colormap = cv2.applyColorMap(depth_map_uint8, cv2.COLORMAP_MAGMA)

        # ====== YOLO를 통한 포트홀 탐지 ======
        results = yolo_model(frame)                    # 이미지에서 객체 탐지
        pothole_boxes = results.xyxy[0].cpu().numpy()    # 바운딩 박스 좌표 및 클래스 정보

        pothole_infos = []       # 포트홀 정보 저장 리스트
        pothole_detected = False # 포트홀 탐지 여부 플래그

        for box in pothole_boxes:
            if len(box) < 6:
                continue  # 정보 부족한 박스 무시

            # 바운딩 박스 좌표 추출 및 정수형 변환
            x1, y1, x2, y2, conf, cls = box
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

            # 해당 박스 영역의 깊이 값 추출 후 중앙값 계산
            pothole_depth_map = depth_map[y1:y2, x1:x2]
            median_depth = float(np.median(pothole_depth_map))

            # 시각화: 바운딩 박스 그리기, 깊이 텍스트 추가, 컬러맵 덧씌우기
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            text = f"Depth: {median_depth:.2f}"
            text_pos = (x1, y1 - 10) if y1 - 10 > 10 else (x1, y1 + 20)
            cv2.putText(frame, text, text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            frame[y1:y2, x1:x2] = cv2.addWeighted(
                frame[y1:y2, x1:x2], 0.6, depth_colormap[y1:y2, x1:x2], 0.4, 0)

            # 포트홀 정보 구성
            pothole_info = {
                "lat": 37.5665,  # 이미지에서 위치 정보를 직접 얻기 어려우므로 예시 값 사용
                "lng": 126.9780,  # 실제 응용에서는 GPS나 위치 정보를 사용해야 함
                "depth": round(median_depth, 2),
                "confidence": float(conf)
            }
            pothole_infos.append(pothole_info)
            pothole_detected = True

        # 결과 이미지 저장 (선택 사항)
        output_dir = os.path.dirname(image_path)
        output_image_path = os.path.join(output_dir, "pothole_result.jpg")
        cv2.imwrite(output_image_path, frame)
        print(f"결과 이미지 저장 완료: {output_image_path}")

        return pothole_detected, pothole_infos

    except Exception as e:
        print(f"포트홀 감지 모델 실행 중 오류 발생: {e}")
        return False, []

def detect_porthole_from_image(image_path: str, location: str = "알 수 없음") -> Tuple[bool, Optional[Dict]]:
    """
    이미지로부터 포트홀을 감지하고 API 서버로 전송합니다.
    
    Args:
        image_path (str): 분석할 이미지 파일 경로
        location (str): 위치 정보 (알 수 없는 경우 기본값 사용)
        
    Returns:
        Tuple[bool, Optional[Dict]]: (성공 여부, 감지된 포트홀 정보)
    """
    try:
        print(f"이미지 분석 중: {image_path}")
        
        # 이미지 파일 존재 여부 확인
        if not os.path.exists(image_path):
            print(f"오류: 이미지 파일을 찾을 수 없습니다 - {image_path}")
            return False, None
        
        # 딥러닝 모델을 사용한 포트홀 감지
        detected, pothole_infos = detect_porthole_with_models(image_path)
        
        if detected and pothole_infos:
            # 여러 포트홀이 감지된 경우, 가장 신뢰도가 높은 하나만 보고
            best_pothole = max(pothole_infos, key=lambda x: x['confidence'])
            
            # API 서버로 포트홀 정보 전송
            send_result = send_porthole_to_server(
                best_pothole['lat'], 
                best_pothole['lng'], 
                best_pothole['depth'], 
                location
            )
            
            if send_result and 'porthole_id' in send_result:
                best_pothole['id'] = send_result['porthole_id']
                print(f"포트홀이 감지되어 API 서버에 전송되었습니다. ID: {send_result['porthole_id']}")
                return True, best_pothole
        
        return False, None
    
    except Exception as e:
        print(f"포트홀 감지 중 오류 발생: {e}")
        return False, None

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
    # 명령줄 인자 처리
    import sys
    
    if len(sys.argv) < 2:
        print("사용법: python detect.py [image_path] [location(optional)]")
        sys.exit(1)
    
    image_path = sys.argv[1]
    location = sys.argv[2] if len(sys.argv) >= 3 else "알 수 없음"
    
    # 이미지에서 포트홀 감지
    success, result = detect_porthole_from_image(image_path, location)
    print(f"감지 결과: {'성공' if success else '실패'}")
    if success:
        print(f"포트홀 정보: {result}")