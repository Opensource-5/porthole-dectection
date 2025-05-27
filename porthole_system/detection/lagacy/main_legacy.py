#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
포트홀 감지 시스템 메인 스크립트 (Porthole Detection System Main Script)

이 스크립트는 포트홀 감지 시스템의 주요 진입점입니다.
실시간 웹캠으로부터 포트홀을 감지하고 API 서버에 정보를 전송합니다.

주요 기능:
1. 명령행 인수 처리
2. 포트홀 감지기와 서버 API 통합
3. 실시간 웹캠 포트홀 감지
"""

import argparse

# 모듈 임포트
from porthole_detector import PortholeDetector
from server_api import PortholeServerAPI, load_config

def get_image_paths(config=None, data_dir: Optional[str] = None) -> Dict[str, List[str]]:
    """
    데이터 디렉토리에서 이미지 경로를 수집합니다.
    
    Args:
        config: 설정 딕셔너리
        data_dir: 데이터 디렉토리 경로 (설정에서 로드한 값보다 우선)
        
    Returns:
        {'train': [...], 'valid': [...], 'test': [...]} 형태의 딕셔너리
    """
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

def main():
    """메인 함수"""
    # 명령행 인수 처리
    parser = argparse.ArgumentParser(description='포트홀 감지 시스템')
    parser.add_argument('--config', type=str, default='config.yaml', help='설정 파일 경로')
    parser.add_argument('--mode', type=int, help='실행 모드 (1: 이미지 분석, 2: 실시간 웹캠)')
    parser.add_argument('--data-dir', type=str, help='데이터 디렉토리 경로')
    parser.add_argument('--image', type=str, help='분석할 이미지 경로')
    parser.add_argument('--video-source', type=int, default=0, help='비디오 소스 (기본: 0, 웹캠)')
    args = parser.parse_args()
    
    # 설정 파일 로드
    config = load_config(args.config)
    
    # 이미지 경로 수집
    data_dir = args.data_dir or config.get('data_paths', {}).get('base_dir', '.')
    img_paths = get_image_paths(config, data_dir)
    
    # 서버 API 및 포트홀 감지기 초기화
    server_api = PortholeServerAPI(config)
    detector = PortholeDetector(config, server_api)
    
    # 모드 결정 (명령행 인수가 우선)
    mode = args.mode
    if mode is None:
        mode = int(input("모드 선택 (1: 이미지 분석, 2: 실시간 웹캠): "))
    
    if mode == 1:
        # 이미지 분석 모드
        image_path = args.image
        
        if not image_path and img_paths['valid']:
            image_path = img_paths['valid'][0]
        
        if not image_path:
            print("분석할 이미지가 없습니다.")
            return
            
        print(f"이미지 분석: {image_path}")
        success, result = detector.detect_from_image(image_path)
        
        print(f"감지 결과: {'성공' if success else '실패'}")
        if success:
            print(f"포트홀 정보: {result}")
            
    elif mode == 2:
        # 실시간 웹캠 모드
        video_source = args.video_source
        video_config = config.get('video', {})
        
        # 설정에서 웹캠 소스 가져오기
        if video_source == 0 and 'webcam_source' in video_config:
            video_source = video_config.get('webcam_source')
            
        detector.process_video_stream(source=video_source)
        
    else:
        print("잘못된 모드 선택")


if __name__ == "__main__":
    main()
