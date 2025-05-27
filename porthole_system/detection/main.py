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
import sys
import os

# 모듈 임포트
from config_utils import load_config, validate_config, get_nested_value
from porthole_detector import PortholeDetector
from server_api import PortholeServerAPI

def main():
    """메인 함수"""
    # 명령행 인수 처리
    parser = argparse.ArgumentParser(
        description='포트홀 실시간 감지 시스템',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예시:
  python main.py                          # 기본 설정으로 실행
  python main.py --config my_config.yaml  # 사용자 설정 파일 사용
  python main.py --video-source 1         # 두 번째 카메라 사용
  python main.py --no-display            # 화면 출력 없이 실행
        """
    )
    
    parser.add_argument('--config', type=str, default='config.yaml', 
                       help='설정 파일 경로 (기본: config.yaml)')
    parser.add_argument('--video-source', type=int, default=None, 
                       help='비디오 소스 번호 (기본: 설정 파일 값)')
    parser.add_argument('--no-display', action='store_true', 
                       help='화면에 결과를 표시하지 않음')
    parser.add_argument('--debug', action='store_true', 
                       help='디버그 모드 활성화 (모든 출력 활성화)')
    
    args = parser.parse_args()
    
    try:
        # 설정 파일 로드
        print(f"🔧 설정 파일 로드 중: {args.config}")
        config = load_config(args.config)
        
        # 설정 검증
        if not validate_config(config):
            print("⚠️  설정 검증에 실패했지만 계속 진행합니다.")
        
        # 디버그 모드가 활성화된 경우 디버그 설정 오버라이드
        if args.debug:
            if 'debug' not in config:
                config['debug'] = {}
            config['debug'].update({
                'print_detections': True,
                'print_model_loading': True,
                'print_api_responses': True
            })
            print("🐛 디버그 모드가 활성화되었습니다.")
        
        # 비디오 소스 결정
        if args.video_source is not None:
            video_source = args.video_source
        else:
            video_source = get_nested_value(config, 'video.webcam_source', 0)
        
        # 화면 표시 여부
        display = not args.no_display
        
        # 서버 API 및 포트홀 감지기 초기화
        print("🌐 서버 API 초기화 중...")
        server_api = PortholeServerAPI(config)
        
        print("🤖 포트홀 감지기 초기화 중...")
        detector = PortholeDetector(config, server_api)
        
        # 시스템 정보 출력
        print("\n" + "="*60)
        print("🎯 포트홀 실시간 감지 시스템")
        print("="*60)
        print(f"📹 비디오 소스: {video_source}")
        print(f"🌐 API 서버: {get_nested_value(config, 'api.server_url', 'N/A')}")
        print(f"🎨 화면 표시: {'ON' if display else 'OFF'}")
        print(f"🔧 설정 파일: {args.config}")
        
        # 모델 정보 출력
        model_path = get_nested_value(config, 'models.yolo.path', 'N/A')
        confidence = get_nested_value(config, 'models.yolo.confidence_threshold', 'N/A')
        print(f"🧠 YOLO 모델: {os.path.basename(model_path) if model_path != 'N/A' else 'N/A'}")
        print(f"🎯 신뢰도 임계값: {confidence}")
        print("="*60)
        
        if display:
            print("💡 종료하려면 비디오 창에서 'q' 키를 누르세요.")
        else:
            print("💡 종료하려면 Ctrl+C를 누르세요.")
        print()
        
        # 실시간 웹캠 포트홀 감지 시작
        detector.process_video_stream(source=video_source, display=display)
        
    except KeyboardInterrupt:
        print("\n👋 사용자에 의해 중단되었습니다.")
    except FileNotFoundError as e:
        print(f"❌ 파일을 찾을 수 없습니다: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ 예상치 못한 오류가 발생했습니다: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        sys.exit(1)
    finally:
        print("✅ 포트홀 감지 시스템이 종료되었습니다.")


if __name__ == "__main__":
    main()
