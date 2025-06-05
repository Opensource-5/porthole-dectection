#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
포트홀 감지 시스템 메인 스크립트 (Porthole Detection System Main Script)

이 스크립트는 포트홀 감지 시스템의 주요 진입점입니다.
실시간 웹캠, 비디오 파일, 이미지 파일로부터 포트홀을 감지하고 API 서버에 정보를 전송합니다.

주요 기능:
1. 명령행 인수 처리
2. 포트홀 감지기와 서버 API 통합  
3. 실시간 웹캠 포트홀 감지
4. 비디오 파일 처리
5. 이미지 파일 처리 (단일, 일괄, 디렉토리)
"""

import argparse
import sys
import os
import cv2

# 모듈 임포트
from config_utils import load_config, validate_config, get_nested_value
from porthole_detector import PortholeDetector
from server_api import PortholeServerAPI

def main():
    """메인 함수"""
    # 명령행 인수 처리
    parser = argparse.ArgumentParser(
        description='포트홀 감지 시스템 (웹캠, 비디오, 이미지 지원)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예시:
  # 웹캠 모드
  python main.py                              # 기본 설정으로 실행 (웹캠)
  python main.py --config my_config.yaml      # 사용자 설정 파일 사용
  python main.py --video-source 1             # 두 번째 카메라 사용
  
  # 비디오 파일 모드
  python main.py --video-file video.mp4       # 동영상 파일 처리
  python main.py --no-display                 # 화면 출력 없이 실행
  
  # 이미지 모드
  python main.py --mode image --source image.jpg --save           # 단일 이미지 처리
  python main.py --mode batch --source "img1.jpg,img2.jpg" --save --csv # 여러 이미지 일괄 처리 + CSV 저장
  python main.py --mode directory --source /path/to/images --save --csv # 디렉토리 내 모든 이미지 처리 + CSV 저장
        """
    )
    
    # 모드 선택
    parser.add_argument('--mode', type=str, choices=['webcam', 'video', 'image', 'batch', 'directory'], 
                       default='webcam', help='실행 모드 선택')
    
    # 기본 옵션들
    parser.add_argument('--config', type=str, default='config.yaml', 
                       help='설정 파일 경로 (기본: config.yaml)')
    parser.add_argument('--source', type=str, help='입력 소스 (모드에 따라 다름: 웹캠 번호, 파일 경로, 디렉토리 경로)')
    parser.add_argument('--output', type=str, default='results', help='결과 저장 디렉토리 (이미지 모드)')
    parser.add_argument('--save', action='store_true', help='결과 이미지 저장 (이미지 모드)')
    parser.add_argument('--csv', action='store_true', help='감지 결과를 CSV 파일로 저장 (batch, directory 모드)')
    parser.add_argument('--extensions', nargs='+', default=['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'],
                       help='처리할 이미지 확장자 (디렉토리 모드)')
    
    # 레거시 옵션들 (하위 호환성)
    parser.add_argument('--video-source', type=int, default=None, 
                       help='비디오 소스 번호 (기본: 설정 파일 값)')
    parser.add_argument('--video-file', type=str, default=None,
                       help='동영상 파일 경로 (이 옵션 사용 시 파일 모드로 실행)')
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
        
        # 모드별 소스 및 타입 결정
        mode = args.mode
        
        # 레거시 옵션 처리 (하위 호환성)
        if args.video_file:
            mode = 'video'
            source = args.video_file
        elif args.video_source is not None:
            mode = 'webcam'
            source = args.video_source
        else:
            source = args.source
        
        # 모드별 기본값 설정
        if mode == 'webcam' and source is None:
            source = get_nested_value(config, 'video.webcam_source', 0)
        elif mode == 'video' and source is None:
            # 설정 파일에서 비디오 파일 경로 확인
            source = get_nested_value(config, 'video.video_file_path', '')
            if not source:
                print("❌ 비디오 파일 경로를 지정해주세요. (--source 또는 --video-file)")
                sys.exit(1)
        
        # 파일 존재 확인
        if mode in ['video', 'image'] and source:
            if not os.path.exists(source):
                print(f"❌ 파일을 찾을 수 없습니다: {source}")
                sys.exit(1)
        elif mode == 'directory' and source:
            if not os.path.isdir(source):
                print(f"❌ 디렉토리를 찾을 수 없습니다: {source}")
                sys.exit(1)
        
        # 화면 표시 여부
        display = not args.no_display
        
        # 서버 API 및 포트홀 감지기 초기화
        print("🌐 서버 API 초기화 중...")
        server_api = PortholeServerAPI(config)
        
        print("🤖 포트홀 감지기 초기화 중...")
        detector = PortholeDetector(config, server_api)
        
        # 시스템 정보 출력
        print("\n" + "="*60)
        print("🎯 포트홀 감지 시스템")
        print("="*60)
        print(f"🔧 실행 모드: {mode.upper()}")
        
        if mode in ['webcam', 'video']:
            if mode == "video":
                print(f"🎬 비디오 파일: {os.path.basename(source)}")
            else:
                print(f"📹 웹캠 소스: {source}")
            print(f"🎨 화면 표시: {'ON' if display else 'OFF'}")
        elif mode in ['image', 'batch', 'directory']:
            print(f"📁 입력 소스: {source}")
            print(f"💾 결과 저장: {'ON' if args.save else 'OFF'}")
            if args.save:
                print(f"📂 출력 디렉토리: {args.output}")
        
        print(f"🌐 API 서버: {get_nested_value(config, 'api.server_url', 'N/A')}")
        print(f"🔧 설정 파일: {args.config}")
        
        # 모델 정보 출력
        model_path = get_nested_value(config, 'models.yolo.path', 'N/A')
        confidence = get_nested_value(config, 'models.yolo.confidence_threshold', 'N/A')
        print(f"🧠 YOLO 모델: {os.path.basename(model_path) if model_path != 'N/A' else 'N/A'}")
        print(f"🎯 신뢰도 임계값: {confidence}")
        print("="*60)
        
        # 모드별 안내 메시지
        if mode in ['webcam', 'video']:
            if display:
                if mode == "video":
                    print("💡 종료하려면 비디오 창에서 'q' 키를 누르거나 동영상이 끝날 때까지 기다리세요.")
                else:
                    print("💡 종료하려면 비디오 창에서 'q' 키를 누르세요.")
            else:
                print("💡 종료하려면 Ctrl+C를 누르세요.")
        elif mode in ['image', 'batch', 'directory']:
            print("💡 이미지 처리가 시작됩니다...")
        print()
        
        # 모드별 실행
        if mode == 'webcam':
            # 웹캠 모드
            source = source if source is not None else 0
            if isinstance(source, str):
                source = int(source)
            detector.process_video_stream(source=source, display=display)
            
        elif mode == 'video':
            # 비디오 파일 모드
            detector.process_video_stream(source=source, display=display)
            
        elif mode == 'image':
            # 단일 이미지 모드
            if source is None:
                print("❌ 이미지 파일 경로를 지정해주세요. (--source)")
                return
            
            detected, pothole_infos, processed_frame = detector.detect_from_image(
                source, save_result=args.save, output_dir=args.output
            )
            
            # 결과 표시 (GUI 환경에서만)
            if processed_frame is not None and not args.no_display:
                try:
                    cv2.imshow('Porthole Detection Result', processed_frame)
                    print("💡 아무 키나 누르면 창이 닫힙니다.")
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()
                except:
                    print("⚠️  GUI 환경이 아니어서 이미지를 표시할 수 없습니다.")
            
        elif mode == 'batch':
            # 여러 이미지 파일 일괄 처리 모드
            if source is None:
                print("❌ 이미지 파일 경로들을 지정해주세요. (--source)")
                return
            
            # 쉼표로 구분된 파일 경로들을 파싱
            image_paths = [path.strip() for path in source.split(',')]
            results = detector.process_image_batch(
                image_paths, save_results=args.save, output_dir=args.output
            )
            
            # CSV 저장이 요청된 경우
            if args.csv:
                csv_path = os.path.join(args.output, "detection_results.csv")
                os.makedirs(args.output, exist_ok=True)
                detector.save_results_to_csv(results, csv_path)
            
        elif mode == 'directory':
            # 디렉토리 내 모든 이미지 처리 모드
            if source is None:
                print("❌ 디렉토리 경로를 지정해주세요. (--source)")
                return
            
            if args.csv:
                # CSV 저장이 요청된 경우 새로운 메서드 사용
                results = detector.process_directory_with_csv(
                    source, extensions=args.extensions, 
                    save_results=args.save, output_dir=args.output,
                    save_csv=True, csv_filename="detection_results.csv"
                )
            else:
                # 기존 메서드 사용
                results = detector.process_directory(
                    source, extensions=args.extensions, 
                    save_results=args.save, output_dir=args.output
                )
        
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


# uv run python porthole_system/detection/main.py --mode image --source porthole_system/detection/test/images/img-105_jpg.rf.3fe9dff3d1631e79ecb480ff403bcb86.jpg --save --output results