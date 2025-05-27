#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
포트홀 이미지 파일 처리 레거시 모듈 (Legacy Image Processing Module)

이 모듈은 정적 이미지 파일로부터 포트홀을 감지하는 기능을 제공합니다.
현재는 사용되지 않으며, 실시간 웹캠 감지만 사용됩니다.

레거시 기능:
1. 이미지 파일에서 포트홀 감지
2. 데이터셋 이미지 경로 수집
3. 정적 이미지 분석
"""

import os
import yaml
import cv2
import numpy as np
import torch
from glob import glob
from typing import Dict, List, Optional, Tuple, Union, Any

def get_image_paths(config=None, data_dir: Optional[str] = None) -> Dict[str, List[str]]:
    """
    데이터 디렉토리에서 이미지 경로를 수집합니다.
    
    Args:
        config: 설정 딕셔너리
        data_dir: 데이터 디렉토리 경로 (설정에서 로드한 값보다 우선)
        
    Returns:
        {'train': [...], 'valid': [...], 'test': [...]} 형태의 딕셔너리
    """
    from server_api import load_config
    
    config = config or load_config()
    data_paths = config.get('data_paths', {})
    
    if data_dir is None:
        data_dir = data_paths.get('base_dir', '.')
    
    # 타입 안전성을 위해 문자열 확인
    data_dir = str(data_dir)
    
    train_path = os.path.join(data_dir, data_paths.get('train_images', 'train/images/*.jpg'))
    valid_path = os.path.join(data_dir, data_paths.get('valid_images', 'valid/images/*.jpg'))
    test_path = os.path.join(data_dir, data_paths.get('test_images', 'test/images/*.jpg'))
    
    train_img_list = glob(train_path)
    valid_img_list = glob(valid_path)
    test_img_list = glob(test_path)
    
    print(f"Train: {len(train_img_list)}")
    print(f"Valid: {len(valid_img_list)}")
    print(f"Test: {len(test_img_list)}")
    
    return {
        'train': train_img_list,
        'valid': valid_img_list,
        'test': test_img_list
    }


# PortholeDetector의 이미지 파일 처리 메서드들 (레거시)
def detect_from_image_legacy(self, image_path: str) -> Tuple[bool, Optional[Dict]]:
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
        detected, pothole_infos = _detect_with_models_legacy(self, image_path)

        if detected and pothole_infos:
            # 여러 포트홀이 감지된 경우, 가장 신뢰도가 높은 하나만 보고
            best_pothole = max(pothole_infos, key=lambda x: x['confidence'])

            # API 서버로 포트홀 정보 전송
            send_result = self.server_api.send_pothole_data(
                best_pothole['lat'],
                best_pothole['lng'],
                best_pothole['depth']
            )

            if send_result and 'porthole_id' in send_result:
                best_pothole['id'] = send_result['porthole_id']
                print(f"포트홀이 감지되어 API 서버에 전송되었습니다. ID: {send_result['porthole_id']}")
                return True, best_pothole

        return False, None

    except Exception as e:
        print(f"포트홀 감지 중 오류 발생: {e}")
        return False, None


def _detect_with_models_legacy(self, image_path: str) -> Tuple[bool, List[Dict]]:
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
        
        # 전역 변수 None 체크
        global yolo_model, midas, transform, device
        if yolo_model is None or midas is None or transform is None or device is None:
            print("모델이 제대로 로드되지 않았습니다.")
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
        # OpenCV normalize 대신 numpy를 사용한 정규화
        depth_min, depth_max = depth_map.min(), depth_map.max()
        if depth_max > depth_min:
            depth_map_normalized = (depth_map - depth_min) / (depth_max - depth_min)
        else:
            depth_map_normalized = np.zeros_like(depth_map)
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
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, self.box_thickness)
            text = f"Depth: {median_depth:.2f}"
            text_pos = (x1, y1 - 10) if y1 - 10 > 10 else (x1, y1 + 20)
            cv2.putText(frame, text, text_pos, cv2.FONT_HERSHEY_SIMPLEX, self.text_size, color, self.text_thickness)
            frame[y1:y2, x1:x2] = cv2.addWeighted(
                frame[y1:y2, x1:x2], 1 - self.overlay_alpha, depth_colormap[y1:y2, x1:x2], self.overlay_alpha, 0)

            # 포트홀 정보 구성
            pothole_info = {
                "lat": self.default_lat,  # 설정 파일에서 불러온 기본 위치 정보 사용
                "lng": self.default_lng,  # 실제 응용에서는 GPS나 위치 정보를 사용해야 함
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
