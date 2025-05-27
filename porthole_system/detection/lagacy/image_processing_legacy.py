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


class LegacyImageProcessor:
    """레거시 이미지 파일 처리 클래스"""
    
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
                    best_pothole['depth']
                )

                if send_result and 'porthole_id' in send_result:
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
        # 이 메서드는 원래 porthole_detector.py에 있던 코드를 그대로 옮긴 것입니다.
        # 실제 구현은 필요시 추가할 수 있습니다.
        pass
