#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
포트홀 감지 모듈 (Porthole Detection Module)

이 모듈은 실시간 비디오 데이터로부터 포트홀을 감지하고
깊이를 추정하는 기능을 제공합니다.

주요 기능:
1. YOLOv5를 사용한 포트홀 객체 탐지
2. MiDaS를 사용한 깊이 추정
3. 실시간 비디오 스트림 처리
"""

import os
import yaml
import cv2
import numpy as np
import torch
from typing import Dict, List, Optional, Tuple, Union, Any
import pathlib
temp = pathlib.WindowsPath
pathlib.WindowsPath = pathlib.PosixPath

# 서버 API 모듈 임포트
from server_api import PortholeServerAPI

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

# 모델 및 관련 변수 초기화 (타입 힌트 추가)
yolo_model: Optional[Any] = None
midas: Optional[Any] = None
transform: Optional[Any] = None
device: Optional[torch.device] = None


class PortholeDetector:
    """포트홀 감지 및 처리를 위한 클래스 (실시간 웹캠 전용)"""
    
    def __init__(self, config=None, server_api=None):
        """
        PortholeDetector 초기화
        
        Args:
            config: 설정 딕셔너리 (None이면 전역 CONFIG 사용)
            server_api: PortholeServerAPI 인스턴스 (None이면 새로 생성)
        """
        self.config = config or CONFIG
        self.model_path = self.config.get('models', {}).get('yolo', {}).get('path', 
                          'yolov5/runs/train/gun_yolov5s_results/weights/best.pt')
        self.confidence_threshold = self.config.get('models', {}).get('yolo', {}).get('confidence_threshold', 0.5)
        self.img_size = self.config.get('models', {}).get('yolo', {}).get('img_size', 416)
        
        # 서버 API 인스턴스 생성 또는 전달받은 것 사용
        self.server_api = server_api if server_api else PortholeServerAPI(self.config)
        
        # 시각화 설정 불러오기
        self.vis_config = self.config.get('visualization', {})
        self.box_color = tuple(self.vis_config.get('box_color', [0, 255, 0]))
        self.text_color = tuple(self.vis_config.get('text_color', [0, 255, 0]))
        self.text_size = self.vis_config.get('text_size', 0.6)
        self.text_thickness = self.vis_config.get('text_thickness', 2)
        self.box_thickness = self.vis_config.get('box_thickness', 2)
        self.overlay_alpha = self.vis_config.get('overlay_alpha', 0.4)
        
        # 위치 설정 불러오기
        self.location = self.config.get('location', {})
        self.default_lat = self.location.get('latitude', 37.5665)
        self.default_lng = self.location.get('longitude', 126.9780)
        
        # 깊이별 색상 설정 (BGR 형식)
        self.class_colors = [
            [0, 255, 0],    # 얕은 깊이 (초록색)
            [0, 255, 255],  # 중간 깊이 (노란색)
            [0, 0, 255]     # 깊은 깊이 (빨간색)
        ]
        
    def load_models(self) -> bool:
        """
        YOLOv5와 MiDaS 모델을 로드합니다.
        
        Returns:
            bool: 모델 로드 성공 여부
        """
        global yolo_model, midas, transform, device

        if yolo_model is not None:
            return True  # 이미 모델이 로드되었으면 다시 로드하지 않음

        try:
            print("포트홀 감지 및 깊이 추정 모델 로드 중...")

            # CUDA GPU 사용 가능 여부 확인 및 장치 설정
            # Mac(M1/M2)에서 MPS(GPU) 지원 추가
            if torch.backends.mps.is_available():
                device = torch.device("mps")
            else:
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
            # 커스텀 학습된 YOLOv5 모델 로드 (포트홀 탐지용)
            yolo_model = torch.hub.load(
                'ultralytics/yolov5', 'custom', path=self.model_path
            )
            yolo_model.conf = self.confidence_threshold  # 신뢰도 임계값 설정
            yolo_model.img_size = self.img_size  # 이미지 크기 설정
            yolo_model.to(device)
            yolo_model.eval()

            # MiDaS 모델 로드 (깊이 추정용)
            model_type = self.config.get('models', {}).get('midas', {}).get('model_type', "DPT_Hybrid")
            midas = torch.hub.load("intel-isl/MiDaS", model_type)
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
    
    # 이미지 파일 처리 메서드들은 lagacy/porthole_image_methods_legacy.py로 이동되었습니다.
    
    def detect_from_frame(self, frame: np.ndarray) -> Tuple[bool, List[Dict], np.ndarray]:
        """
        프레임에서 포트홀을 감지하고 시각화합니다.
        
        Args:
            frame: 입력 비디오 프레임
            
        Returns:
            (감지 여부, 포트홀 정보 리스트, 시각화된 프레임)
        """
        # 모델 로드 확인
        if not self.load_models():
            return False, [], frame
        
        # 전역 변수 None 체크
        if yolo_model is None or midas is None or transform is None or device is None:
            print("모델이 제대로 로드되지 않았습니다.")
            return False, [], frame

        # MiDaS로 깊이 맵 생성
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        input_batch = transform(frame_rgb).to(device)
        
        with torch.no_grad():
            depth_pred = midas(input_batch)
            depth_map = torch.nn.functional.interpolate(
                depth_pred.unsqueeze(1), size=frame.shape[:2],
                mode="bicubic", align_corners=False
            ).squeeze().cpu().numpy()

        # YOLOv5로 객체 탐지
        results = yolo_model(frame)
        boxes = results.xyxy[0].cpu().numpy()

        infos = []
        detected = False
        
        for box in boxes:
            if len(box) < 6:
                continue
                
            x1, y1, x2, y2, conf, cls = box
            x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
            
            # 깊이 정보 계산
            region = depth_map[y1:y2, x1:x2]
            depth_val = float(np.median(region))    

            if depth_val < 500:
                color = tuple(self.class_colors[0])
            elif depth_val < 1500:
                color = tuple(self.class_colors[1])
            else:
                color = tuple(self.class_colors[2])        
            
            # 시각화: 바운딩 박스와 신뢰도 표시 (설정에서 로드한 값 사용)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, self.box_thickness)
            cv2.putText(frame, f'{conf:.2f}', (x1, y1-5), 
                       cv2.FONT_HERSHEY_SIMPLEX, self.text_size, color, self.text_thickness)
            
            # 깊이 정보 표시
            cv2.putText(
                frame, 
                f'Depth: {depth_val:.2f}', 
                (x1, y1+15), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                self.text_size, 
                color, 
                self.text_thickness
            )
            
            # 설정에서 불러온 위치 정보 사용
            infos.append({
                "lat": self.default_lat,
                "lng": self.default_lng,
                "depth": round(depth_val, 2),
                "confidence": float(conf)
            })
            
            # 설정에서 불러온 신뢰도 임계값 사용
            if conf > self.confidence_threshold:
                detected = True

        return detected, infos, frame
            
    def process_video_stream(self, source=0, display=True):
        """
        비디오 스트림(웹캠 또는 비디오 파일)에서 포트홀을 실시간 감지합니다.
        
        Args:
            source: 비디오 소스 (0=웹캠, 파일 경로=비디오 파일)
            display: 화면에 결과를 표시할지 여부
            
        Returns:
            None
        """
        # 비디오 스트림 초기화
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            print(f"비디오 소스를 열 수 없습니다: {source}")
            return
            
        # 비디오 설정 적용 (웹캠인 경우)
        if isinstance(source, int):
            video_config = self.config.get('video', {})
            width = video_config.get('frame_width', 640)
            height = video_config.get('frame_height', 480)
            fps = video_config.get('fps', 30)
            
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            cap.set(cv2.CAP_PROP_FPS, fps)
            
            print(f"비디오 설정 - 해상도: {width}x{height}, FPS: {fps}")
            
        if not self.load_models():
            print("모델 로드 실패")
            cap.release()
            return
            
        print("실시간 포트홀 감지 시작...")
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                    
                # 프레임에서 포트홀 감지 및 시각화
                detected, pothole_infos, processed_frame = self.detect_from_frame(frame)
                
                # 포트홀이 감지된 경우 서버로 전송
                if detected and pothole_infos:
                    # 가장 신뢰도가 높은 포트홀 정보 선택
                    best_pothole = max(pothole_infos, key=lambda x: x['confidence'])
                    
                    print(f"감지된 포트홀 - 깊이: {best_pothole['depth']} mm, 신뢰도: {best_pothole['confidence']:.2f}")
                    
                    # 서버로 전송
                    self.server_api.send_pothole_data(
                        best_pothole['lat'],
                        best_pothole['lng'],
                        best_pothole['depth']
                    )
                
                # 처리된 프레임 표시
                if display:
                    cv2.imshow('Porthole Detection', processed_frame)
                    
                    # 'q' 키를 누르면 종료
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                        
        finally:
            cap.release()
            if display:
                cv2.destroyAllWindows()
            print("실시간 포트홀 감지 종료")
