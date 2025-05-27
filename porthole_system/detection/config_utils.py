#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
설정 관리 유틸리티 모듈 (Configuration Management Utility Module)

이 모듈은 시스템 전체의 설정을 중앙에서 관리합니다.

주요 기능:
1. YAML 설정 파일 로드
2. 기본값 제공
3. 설정 검증
"""

import os
import yaml
from typing import Dict, Any, Optional

# 기본 설정값
DEFAULT_CONFIG = {
    'api': {
        'server_url': "https://statute-bradford-rebound-paragraph.trycloudflare.com/api/notify_new_porthole",
        'retry_count': 3,
        'timeout_seconds': 10
    },
    'models': {
        'yolo': {
            'path': "yolov5/runs/train/gun_yolov5s_results/weights/best.pt",
            'confidence_threshold': 0.5,
            'img_size': 416
        },
        'midas': {
            'model_type': "DPT_Hybrid",
            'transform_type': "small_transform"
        }
    },
    'device': {
        'use_mps': True,
        'use_cuda': True,
        'force_cpu': False
    },
    'location': {
        'latitude': 37.5665,
        'longitude': 126.9780,
        'address': "서울특별시 중구 세종대로"
    },
    'depth_classification': {
        'shallow_threshold': 500,
        'medium_threshold': 1500
    },
    'visualization': {
        'class_colors': {
            'shallow': [0, 255, 0],
            'medium': [0, 165, 255],
            'deep': [0, 0, 255]
        },
        'text_size': 0.6,
        'text_thickness': 2,
        'box_thickness': 2,
        'depth_colormap': "MAGMA",
        'overlay_alpha': 0.4
    },
    'video': {
        'webcam_source': 0,
        'frame_width': 640,
        'frame_height': 480,
        'fps': 30
    },
    'detection': {
        'min_detection_confidence': 0.3,
        'send_to_server_confidence': 0.5
    },
    'debug': {
        'print_detections': True,
        'print_model_loading': True,
        'print_api_responses': True
    }
}


def load_config(config_path: str = 'config.yaml') -> Dict[str, Any]:
    """
    YAML 설정 파일을 로드합니다.
    
    Args:
        config_path: 설정 파일 경로
        
    Returns:
        설정 딕셔너리 (기본값과 병합됨)
    """
    try:
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as file:
                file_config = yaml.safe_load(file) or {}
            
            # 기본 설정과 파일 설정을 재귀적으로 병합
            config = merge_configs(DEFAULT_CONFIG, file_config)
            print(f"설정 파일 로드 완료: {config_path}")
            return config
        else:
            print(f"설정 파일을 찾을 수 없습니다: {config_path}")
            print("기본 설정을 사용합니다.")
            return DEFAULT_CONFIG.copy()
            
    except Exception as e:
        print(f"설정 파일 로드 중 오류 발생: {e}")
        print("기본 설정을 사용합니다.")
        return DEFAULT_CONFIG.copy()


def merge_configs(base_config: Dict[str, Any], override_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    두 설정 딕셔너리를 재귀적으로 병합합니다.
    
    Args:
        base_config: 기본 설정
        override_config: 덮어쓸 설정
        
    Returns:
        병합된 설정
    """
    result = base_config.copy()
    
    for key, value in override_config.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_configs(result[key], value)
        else:
            result[key] = value
    
    return result


def get_nested_value(config: Dict[str, Any], key_path: str, default: Any = None) -> Any:
    """
    중첩된 딕셔너리에서 점 표기법으로 값을 가져옵니다.
    
    Args:
        config: 설정 딕셔너리
        key_path: 점으로 구분된 키 경로 (예: "models.yolo.confidence_threshold")
        default: 키가 없을 때 반환할 기본값
        
    Returns:
        해당 키의 값 또는 기본값
    """
    try:
        keys = key_path.split('.')
        value = config
        
        for key in keys:
            value = value[key]
        
        return value
    except (KeyError, TypeError):
        return default


def validate_config(config: Dict[str, Any]) -> bool:
    """
    설정 파일의 유효성을 검사합니다.
    
    Args:
        config: 검사할 설정
        
    Returns:
        유효성 검사 통과 여부
    """
    required_keys = [
        'models.yolo.path',
        'models.yolo.confidence_threshold',
        'api.server_url',
        'location.latitude',
        'location.longitude'
    ]
    
    for key_path in required_keys:
        if get_nested_value(config, key_path) is None:
            print(f"필수 설정값이 누락되었습니다: {key_path}")
            return False
    
    # 신뢰도 임계값 검증
    confidence = get_nested_value(config, 'models.yolo.confidence_threshold', 0)
    if not (0 <= confidence <= 1):
        print(f"신뢰도 임계값이 유효하지 않습니다: {confidence} (0~1 사이여야 함)")
        return False
    
    return True


# 전역 설정 인스턴스
_global_config: Optional[Dict[str, Any]] = None


def get_global_config() -> Dict[str, Any]:
    """
    전역 설정 인스턴스를 반환합니다.
    
    Returns:
        전역 설정
    """
    global _global_config
    if _global_config is None:
        _global_config = load_config()
        if not validate_config(_global_config):
            print("⚠️  설정 검증에 실패했지만 계속 진행합니다.")
    return _global_config


def reload_global_config(config_path: str = 'config.yaml') -> Dict[str, Any]:
    """
    전역 설정을 다시 로드합니다.
    
    Args:
        config_path: 설정 파일 경로
        
    Returns:
        새로 로드된 설정
    """
    global _global_config
    _global_config = load_config(config_path)
    if not validate_config(_global_config):
        print("⚠️  설정 검증에 실패했지만 계속 진행합니다.")
    return _global_config
