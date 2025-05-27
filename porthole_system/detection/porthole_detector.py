#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
포트홀 감지 모듈 (Porthole Detection Module)

이 모듈은 이미지나 비디오 데이터로부터 포트홀을 감지하고
깊이를 추정하는 기능을 제공합니다.

주요 기능:
1. YOLOv5를 사용한 포트홀 객체 탐지
2. MiDaS를 사용한 깊이 추정
3. 이미지 및 비디오 스트림 처리
"""

import os
import yaml
import cv2
import numpy as np
import torch
from glob import glob
from typing import Dict, List, Optional, Tuple, Union
from geopy.geocoders import Nominatim

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

# 모델 및 관련 변수 초기화
yolo_model = None
midas = None
transform = None
device = None


class PortholeDetector:
    """포트홀 감지 및 처리를 위한 클래스"""
    
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
        self.class_colors = self.vis_config.get('class_colors', {})
        # self.box_color = tuple(self.vis_config.get('box_color', [0, 255, 0]))
        # self.text_color = tuple(self.vis_config.get('text_color', [0, 255, 0]))
        self.text_size = self.vis_config.get('text_size', 0.6)
        self.text_thickness = self.vis_config.get('text_thickness', 2)
        self.box_thickness = self.vis_config.get('box_thickness', 2)
        self.overlay_alpha = self.vis_config.get('overlay_alpha', 0.4)
        
        # 위치 설정 불러오기
        self.location = self.config.get('location', {})
        self.default_lat = self.location.get('latitude', 37.5665)
        self.default_lng = self.location.get('longitude', 126.9780)
    
    # 색상 불러올 때 클래스 ID에 따라 동적으로 선택
    def get_class_color(self, class_id):
        color = self.class_colors.get(str(class_id)) or self.class_colors.get(int(class_id))
        return tuple(color) if color else (0, 255, 0)  # 기본값 green

    def coord_into_location(self, lat: float, lng: float):
        """
        위도와 경도로부터 도로명주소를 반환합니다. geopy모듈 사용
        """
        # geolocator 초기화
        geolocator = Nominatim(user_agent="South Korea")

        # 위도, 경도 지정
        location = geolocator.reverse((lat, lng), language='ko')  # 서울시청 좌표

        # 전체 raw 결과 확인
        raw = location.raw

        # 주소 구성요소 추출
        address = raw.get('address', {})

        country = address.get('country', '')
        postcode = address.get('postcode', '') # 도로명주소

        city = address.get('city', '')  #서울
        if city == '서울' :
            city = city +'특별'
        elif city == '부산' or '대구' or '인천' or '광주' or '대전' or '울산' :
            city = city + '광역'

        borough = address.get('borough', '') # 중구
        road = address.get('road', '')        # 세종대로
        # house_number = address.get('house_number', '') # 110

        # 원하는 포맷으로 정리
        output = f"{city}시 {borough} {road}".strip()

        return output

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
    
    def detect_from_image(self, image_path: str) -> Tuple[bool, Optional[Dict]]:
        """
        이미지 파일로부터 포트홀을 감지하고 API 서버로 전송합니다.
        
        Args:
            image_path: 분석할 이미지 파일 경로
            
        Returns:
            (성공 여부, 감지된 포트홀 정보)
        """
        try:
            print(f"이미지 분석 중: {image_path}")

            # 이미지 파일 존재 여부 확인
            if not os.path.exists(image_path):
                print(f"오류: 이미지 파일을 찾을 수 없습니다 - {image_path}")
                return False, None

            # 딥러닝 모델을 사용한 포트홀 감지
            detected, pothole_infos = self._detect_with_models(image_path)

            if detected and pothole_infos:
                # 여러 포트홀이 감지된 경우, 가장 신뢰도가 높은 하나만 보고
                best_pothole = max(pothole_infos, key=lambda x: x['confidence'])
                # API 서버로 포트홀 정보 전송
                send_result = self.server_api.send_pothole_data(
                    best_pothole['lat'],
                    best_pothole['lng'],
                    best_pothole['depth'],
                    best_pothole['location']
                )

                if send_result and 'porthole_id' in send_result:
                    best_pothole['id'] = send_result['porthole_id']
                    print(f"포트홀이 감지되어 API 서버에 전송되었습니다. ID: {send_result['porthole_id']}")
                    return True, best_pothole

            return False, None

        except Exception as e:
            print(f"포트홀 감지 중 오류 발생: {e}")
            return False, None
    
    def _detect_with_models(self, image_path: str) -> Tuple[bool, List[Dict]]:
        """
        YOLOv5와 MiDaS 모델을 사용하여 이미지에서 포트홀을 감지하고 깊이를 추정합니다.
        
        Args:
            image_path: 분석할 이미지 파일 경로
            
        Returns:
            (포트홀 감지 여부, 감지된 포트홀 정보 리스트)
        """
        try:
            # 모델 로드 확인
            if not self.load_models():
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
            
            # 설정에서 깊이 맵 컬러맵 가져오기
            depth_colormap_name = self.vis_config.get('depth_colormap', 'MAGMA').upper()
            colormap_dict = {
                'MAGMA': cv2.COLORMAP_MAGMA,
                'JET': cv2.COLORMAP_JET,
                'VIRIDIS': cv2.COLORMAP_VIRIDIS,
                'PLASMA': cv2.COLORMAP_PLASMA,
                'HOT': cv2.COLORMAP_HOT,
                'BONE': cv2.COLORMAP_BONE
            }
            colormap = colormap_dict.get(depth_colormap_name, cv2.COLORMAP_MAGMA)
            depth_colormap = cv2.applyColorMap(depth_map_uint8, colormap)

            # ====== YOLO를 통한 포트홀 탐지 ======
            results = yolo_model(frame)                    # 이미지에서 객체 탐지
            pothole_boxes = results.xyxy[0].cpu().numpy()    # 바운딩 박스 좌표 및 클래스 정보

            pothole_infos = []       # 포트홀 정보 저장 리스트
            pothole_detected = False # 포트홀 탐지 여부 플래그

            for box in pothole_boxes:
                if len(box) < 6:
                    continue  # 정보 부족한 박스 무시

                # 바운딩 박스 좌표 추출 및 정수형 변환
                x1, y1, x2, y2 = map(int, box[:4])
                conf, cls = box[4], box[5]

                # 해당 박스 영역의 깊이 값 추출 후 중앙값 계산
                pothole_depth_map = depth_map[y1:y2, x1:x2]
                median_depth = float(np.median(pothole_depth_map))

                if median_depth < 500:
                    color = tuple(self.class_colors[0])
                elif median_depth < 1500:
                    color = tuple(self.class_colors[1])
                else:
                    color = tuple(self.class_colors[2])


                # 시각화: 바운딩 박스 그리기, 깊이 텍스트 추가, 컬러맵 덧씌우기
                # cv2.rectangle(frame, (x1, y1), (x2, y2), self.box_color, self.box_thickness)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, self.box_thickness)
                text = f"Depth: {median_depth:.2f}"
                text_pos = (x1, y1 - 10) if y1 - 10 > 10 else (x1, y1 + 20)
                # cv2.putText(frame, text, text_pos, cv2.FONT_HERSHEY_SIMPLEX, self.text_size, self.text_color, self.text_thickness)
                cv2.putText(frame, text, text_pos, cv2.FONT_HERSHEY_SIMPLEX, self.text_size, color, self.text_thickness)
                frame[y1:y2, x1:x2] = cv2.addWeighted(
                    frame[y1:y2, x1:x2], 1 - self.overlay_alpha, depth_colormap[y1:y2, x1:x2], self.overlay_alpha, 0)

                # 포트홀 정보 구성
                pothole_info = {
                    "lat": self.default_lat,  # 설정 파일에서 불러온 기본 위치 정보 사용
                    "lng": self.default_lng,  # 실제 응용에서는 GPS나 위치 정보를 사용해야 함
                    "depth": round(median_depth, 2),
                    "confidence": float(conf),
                    "location" : self.coord_into_location(self.default_lat, self.default_lng)
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
            else :
                color = tuple(self.class_colors[2])        
            
            # 시각화: 바운딩 박스와 신뢰도 표시 (설정에서 로드한 값 사용)
            # cv2.rectangle(frame, (x1, y1), (x2, y2), self.box_color, self.box_thickness)
            # cv2.putText(frame, f'{conf:.2f}', (x1, y1-5), 
            #            cv2.FONT_HERSHEY_SIMPLEX, self.text_size, self.text_color, self.text_thickness)
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
                "confidence": float(conf),
                "location" : self.coord_into_location(self.default_lat, self.default_lng)
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
                        best_pothole['depth'],
                        # best_pothole['location']
                    )
                    print(f"포트홀 정보 전송 완료: {best_pothole['location']}")

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
