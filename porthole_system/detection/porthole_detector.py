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
import cv2
import numpy as np
import torch
import pathlib
import time
import math
from typing import Dict, List, Optional, Tuple, Union, Any, Set

# Windows 경로 호환성을 위한 설정
temp = pathlib.WindowsPath
pathlib.WindowsPath = pathlib.PosixPath

# 로컬 모듈 임포트
from config_utils import get_global_config, get_nested_value
from server_api import PortholeServerAPI

# 모델 및 관련 변수 초기화 (타입 힌트 추가)
yolo_model: Optional[Any] = None
midas: Optional[Any] = None
transform: Optional[Any] = None
device: Optional[torch.device] = None


class PortholeDetector:
    """포트홀 감지 및 처리를 위한 클래스 (실시간 웹캠 전용)"""
    
    def __init__(self, config: Optional[Dict] = None, server_api: Optional[PortholeServerAPI] = None):
        """
        PortholeDetector 초기화
        
        Args:
            config: 설정 딕셔너리 (None이면 전역 설정 사용)
            server_api: PortholeServerAPI 인스턴스 (None이면 새로 생성)
        """
        self.config = config or get_global_config()
        
        # YOLO 모델 설정
        self.model_path = get_nested_value(self.config, 'models.yolo.path', 
                                         'yolov5/runs/train/gun_yolov5s_results/weights/best.pt')
        self.confidence_threshold = get_nested_value(self.config, 'models.yolo.confidence_threshold', 0.5)
        self.img_size = get_nested_value(self.config, 'models.yolo.img_size', 416)
        
        # MiDaS 모델 설정
        self.midas_model_type = get_nested_value(self.config, 'models.midas.model_type', "DPT_Hybrid")
        self.midas_transform_type = get_nested_value(self.config, 'models.midas.transform_type', "small_transform")
        
        # 깊이 분류 임계값
        self.shallow_threshold = get_nested_value(self.config, 'depth_classification.shallow_threshold', 500)
        self.medium_threshold = get_nested_value(self.config, 'depth_classification.medium_threshold', 1500)
        
        # 감지 설정
        self.min_detection_confidence = get_nested_value(self.config, 'detection.min_detection_confidence', 0.3)
        self.send_to_server_confidence = get_nested_value(self.config, 'detection.send_to_server_confidence', 0.5)
        
        # 중복 전송 방지 설정
        self.min_send_interval = get_nested_value(self.config, 'detection.min_send_interval', 5.0)
        self.position_tolerance = get_nested_value(self.config, 'detection.position_tolerance', 0.0001)
        self.max_sent_cache_size = get_nested_value(self.config, 'detection.max_sent_cache_size', 100)
        self.duplicate_detection_distance = get_nested_value(self.config, 'detection.duplicate_detection_distance', 50)
        
        # 중복 방지를 위한 내부 상태
        self.last_send_time = 0
        self.recent_detections: List[Dict] = []  # 최근 감지된 포트홀들
        self.sent_locations: Set[Tuple[float, float]] = set()  # 전송된 위치들
        
        # 서버 API 인스턴스 생성 또는 전달받은 것 사용
        self.server_api = server_api if server_api else PortholeServerAPI(self.config)
        
        # 시각화 설정 불러오기
        vis_config = self.config.get('visualization', {})
        self.class_colors = vis_config.get('class_colors', {
            'shallow': [0, 255, 0],
            'medium': [0, 165, 255], 
            'deep': [0, 0, 255]
        })
        self.text_size = vis_config.get('text_size', 0.6)
        self.text_thickness = vis_config.get('text_thickness', 2)
        self.box_thickness = vis_config.get('box_thickness', 2)
        self.overlay_alpha = vis_config.get('overlay_alpha', 0.4)
        
        # 위치 설정 불러오기
        location_config = self.config.get('location', {})
        self.default_lat = location_config.get('latitude', 37.5665)
        self.default_lng = location_config.get('longitude', 126.9780)
        
        # 디버그 설정
        debug_config = self.config.get('debug', {})
        self.print_detections = debug_config.get('print_detections', True)
        self.print_model_loading = debug_config.get('print_model_loading', True)
        
    def _get_device(self) -> torch.device:
        """
        사용할 디바이스를 결정합니다.
        
        Returns:
            torch.device: 사용할 디바이스
        """
        device_config = self.config.get('device', {})
        
        # CPU 강제 사용 설정이 있으면 CPU 사용
        if device_config.get('force_cpu', False):
            return torch.device("cpu")
        
        # Apple Silicon (M1/M2) MPS 지원 확인
        if device_config.get('use_mps', True) and torch.backends.mps.is_available():
            return torch.device("mps")
        
        # NVIDIA CUDA 지원 확인
        if device_config.get('use_cuda', True) and torch.cuda.is_available():
            return torch.device("cuda")
        
        # 기본적으로 CPU 사용
        return torch.device("cpu")
        
    def load_models(self) -> bool:
        """
        YOLOv5와 MiDaS 모델을 로드합니다.
        
        Returns:
            bool: 모델 로드 성공 여부
        """
        global yolo_model, midas, transform, device

        if yolo_model is not None and midas is not None:
            return True  # 이미 모델이 로드되었으면 다시 로드하지 않음

        try:
            if self.print_model_loading:
                print("포트홀 감지 및 깊이 추정 모델 로드 중...")

            # 디바이스 설정
            device = self._get_device()
            if self.print_model_loading:
                print(f"사용 디바이스: {device}")

            # 커스텀 학습된 YOLOv5 모델 로드 (포트홀 탐지용)
            if not os.path.exists(self.model_path):
                print(f"⚠️  YOLO 모델 파일을 찾을 수 없습니다: {self.model_path}")
                print("기본 YOLOv5s 모델을 사용합니다.")
                yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
            else:
                yolo_model = torch.hub.load('ultralytics/yolov5', 'custom', path=self.model_path)
            
            yolo_model.conf = self.confidence_threshold  # 신뢰도 임계값 설정
            yolo_model.img_size = self.img_size  # 이미지 크기 설정
            yolo_model.to(device)
            yolo_model.eval()

            # MiDaS 모델 로드 (깊이 추정용)
            midas = torch.hub.load("intel-isl/MiDaS", self.midas_model_type)
            midas.to(device)
            midas.eval()
            
            # MiDaS 입력 이미지 변환 함수 로딩
            midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
            
            # transform 타입에 따라 적절한 변환 선택
            try:
                if self.midas_transform_type == "small_transform":
                    transform = midas_transforms.small_transform
                elif self.midas_transform_type == "dpt_transform":
                    transform = midas_transforms.dpt_transform  
                else:
                    # 기본값으로 small_transform 사용
                    transform = midas_transforms.small_transform
                    if self.print_model_loading:
                        print(f"⚠️  알 수 없는 변환 타입: {self.midas_transform_type}, small_transform 사용")
            except AttributeError:
                # 속성에 접근할 수 없는 경우 기본값 사용
                transform = midas_transforms.small_transform
                if self.print_model_loading:
                    print(f"⚠️  변환 함수 로드 실패, 기본 변환 사용")

            if self.print_model_loading:
                print(f"✅ 모델 로드 완료 (장치: {device})")
            return True

        except Exception as e:
            print(f"❌ 모델 로드 중 오류 발생: {e}")
            return False
    
    def _classify_depth(self, depth_value: float) -> Tuple[str, List[int]]:
        """
        깊이값에 따라 분류하고 해당하는 색상을 반환합니다.
        
        Args:
            depth_value: 깊이값
            
        Returns:
            (분류명, BGR 색상)
        """
        if depth_value < self.shallow_threshold:
            return "shallow", self.class_colors['shallow']
        elif depth_value < self.medium_threshold:
            return "medium", self.class_colors['medium']
        else:
            return "deep", self.class_colors['deep']
    
    def _is_duplicate_position(self, lat: float, lng: float) -> bool:
        """
        이미 전송된 위치인지 확인합니다.
        
        Args:
            lat: 위도
            lng: 경도
            
        Returns:
            bool: 중복 위치 여부
        """
        for sent_lat, sent_lng in self.sent_locations:
            if (abs(lat - sent_lat) < self.position_tolerance and 
                abs(lng - sent_lng) < self.position_tolerance):
                return True
        return False
    
    def _add_sent_position(self, lat: float, lng: float) -> None:
        """
        전송된 위치를 캐시에 추가합니다.
        
        Args:
            lat: 위도
            lng: 경도
        """
        # 캐시 크기 제한
        if len(self.sent_locations) >= self.max_sent_cache_size:
            # 가장 오래된 항목 제거 (간단히 첫 번째 항목 제거)
            self.sent_locations.pop()
        
        self.sent_locations.add((lat, lng))
    
    def _is_duplicate_detection(self, bbox: List[int]) -> bool:
        """
        프레임 내에서 중복 감지인지 확인합니다.
        
        Args:
            bbox: 바운딩 박스 [x1, y1, x2, y2]
            
        Returns:
            bool: 중복 감지 여부
        """
        x1, y1, x2, y2 = bbox
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        
        for detection in self.recent_detections:
            det_bbox = detection['bbox']
            det_x1, det_y1, det_x2, det_y2 = det_bbox
            det_center_x = (det_x1 + det_x2) / 2
            det_center_y = (det_y1 + det_y2) / 2
            
            # 거리 계산
            distance = math.sqrt((center_x - det_center_x)**2 + (center_y - det_center_y)**2)
            
            if distance < self.duplicate_detection_distance:
                return True
        
        return False
    
    def _should_send_to_server(self, pothole_infos: List[Dict]) -> Tuple[bool, Optional[Dict]]:
        """
        서버로 전송할지 결정하고 전송할 포트홀을 선택합니다.
        
        Args:
            pothole_infos: 감지된 포트홀 정보 리스트
            
        Returns:
            (전송 여부, 선택된 포트홀 정보)
        """
        current_time = time.time()
        
        # 최소 전송 간격 확인
        if current_time - self.last_send_time < self.min_send_interval:
            return False, None
        
        # 서버 전송 임계값 이상의 포트홀만 필터링
        high_confidence_potholes = [
            p for p in pothole_infos 
            if p['confidence'] >= self.send_to_server_confidence
        ]
        
        if not high_confidence_potholes:
            return False, None
        
        # 중복 위치가 아닌 포트홀만 필터링
        new_potholes = []
        for pothole in high_confidence_potholes:
            if not self._is_duplicate_position(pothole['lat'], pothole['lng']):
                new_potholes.append(pothole)
        
        if not new_potholes:
            return False, None
        
        # 가장 신뢰도가 높은 포트홀 선택
        best_pothole = max(new_potholes, key=lambda x: x['confidence'])
        
        return True, best_pothole
    
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
            print("❌ 모델이 제대로 로드되지 않았습니다.")
            return False, [], frame

        try:
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
                
                # 신뢰도가 최소 임계값보다 낮으면 무시
                if conf < self.min_detection_confidence:
                    continue
                
                # 깊이 정보 계산
                region = depth_map[y1:y2, x1:x2]
                if region.size > 0:
                    depth_val = float(np.median(region))
                else:
                    depth_val = 0.0

                # 깊이 분류 및 색상 결정
                depth_class, color = self._classify_depth(depth_val)
                color = tuple(color)

                # 시각화: 바운딩 박스와 신뢰도 표시
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, self.box_thickness)
                cv2.putText(frame, f'Conf: {conf:.2f}', (x1, y1-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, self.text_size, color, self.text_thickness)
                
                # 깊이 정보 표시
                cv2.putText(frame, f'Depth: {depth_val:.1f} ({depth_class})', (x1, y1+20), 
                           cv2.FONT_HERSHEY_SIMPLEX, self.text_size, color, self.text_thickness)
                
                # 포트홀 정보 저장
                pothole_info = {
                    "lat": self.default_lat,
                    "lng": self.default_lng,
                    "depth": round(depth_val, 2),
                    "confidence": float(conf),
                    "depth_class": depth_class,
                    "bbox": [x1, y1, x2, y2]
                }
                
                # 중복 감지 체크 (같은 프레임 내에서)
                if not self._is_duplicate_detection([x1, y1, x2, y2]):
                    infos.append(pothole_info)
                    
                    # 최근 감지 목록에 추가 (캐시 크기 제한)
                    self.recent_detections.append(pothole_info)
                    if len(self.recent_detections) > 10:  # 최근 10개만 유지
                        self.recent_detections.pop(0)
                
                # 서버 전송 임계값 이상이면 감지됨으로 표시
                if conf >= self.send_to_server_confidence:
                    detected = True
                    
                # 디버그 출력
                if self.print_detections:
                    print(f"감지: 신뢰도={conf:.2f}, 깊이={depth_val:.1f}({depth_class})")

            return detected, infos, frame
            
        except Exception as e:
            print(f"❌ 프레임 처리 중 오류 발생: {e}")
            return False, [], frame
            
    def process_video_stream(self, source: Union[int, str] = 0, display: bool = True) -> None:
        """
        비디오 스트림(웹캠 또는 비디오 파일)에서 포트홀을 실시간 감지합니다.
        
        Args:
            source: 비디오 소스 (0=웹캠, 파일 경로=비디오 파일)
            display: 화면에 결과를 표시할지 여부
        """
        # 비디오 스트림 초기화
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            print(f"❌ 비디오 소스를 열 수 없습니다: {source}")
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
            
            if self.print_model_loading:
                print(f"📹 비디오 설정 - 해상도: {width}x{height}, FPS: {fps}")
            
        # 모델 로드
        if not self.load_models():
            print("❌ 모델 로드 실패")
            cap.release()
            return
            
        print("🎯 실시간 포트홀 감지 시작...")
        print("💡 종료하려면 'q' 키를 누르세요.")
        
        try:
            frame_count = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("❌ 프레임을 읽을 수 없습니다.")
                    break
                    
                frame_count += 1
                
                # 프레임에서 포트홀 감지 및 시각화
                detected, pothole_infos, processed_frame = self.detect_from_frame(frame)
                
                # 포트홀이 감지된 경우 서버로 전송 여부 결정
                if detected and pothole_infos:
                    should_send, selected_pothole = self._should_send_to_server(pothole_infos)
                    
                    if should_send and selected_pothole:
                        print(f"🕳️  새로운 포트홀 감지! 깊이: {selected_pothole['depth']}mm, " +
                              f"신뢰도: {selected_pothole['confidence']:.2f}, " +
                              f"분류: {selected_pothole['depth_class']}")
                        
                        # 서버로 전송 (이미지 포함)
                        success = self.server_api.send_pothole_data(
                            selected_pothole['lat'],
                            selected_pothole['lng'],
                            selected_pothole['depth'],
                            frame  # 원본 프레임을 함께 전송
                        )
                        
                        if success:
                            # 전송 성공 시 위치 캐시에 추가
                            self._add_sent_position(selected_pothole['lat'], selected_pothole['lng'])
                            self.last_send_time = time.time()
                            print(f"✅ 서버 전송 완료")
                        else:
                            print(f"❌ 서버 전송 실패")
                    elif pothole_infos:
                        # 전송하지 않은 이유 출력 (디버깅용)
                        if self.print_detections:
                            print(f"📍 포트홀 감지됨 (전송 안함): 중복 또는 시간 간격 미충족")
                
                # 처리된 프레임 표시
                if display:
                    # 프레임 정보 표시
                    cv2.putText(processed_frame, f'Frame: {frame_count}', (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    
                    cv2.imshow('Porthole Detection System', processed_frame)
                    
                    # 'q' 키를 누르면 종료
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        print("👋 사용자 종료 요청")
                        break
                        
        except KeyboardInterrupt:
            print("\n👋 키보드 인터럽트로 종료")
        except Exception as e:
            print(f"❌ 비디오 처리 중 오류 발생: {e}")
        finally:
            cap.release()
            if display:
                cv2.destroyAllWindows()
            print("✅ 실시간 포트홀 감지 종료")
