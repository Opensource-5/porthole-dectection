#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
포트홀 감지 시스템 실행 스크립트

이 스크립트는 포트홀 감지 시스템을 실행합니다.
"""

import sys
import os

# 현재 디렉토리를 Python 경로에 추가
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from detection.porthole_detector import PortholeDetector
from detection.server_api import PortholeServerAPI
from detection.config_utils import load_config

def main():
    """
    메인 실행 함수
    """
    print("🚧 포트홀 감지 시스템 시작")
    print("=" * 50)
    
    try:
        # 설정 파일 로드
        print("📋 설정 파일 로딩 중...")
        config = load_config("detection/config.yaml")
        
        # 서버 API 초기화
        print("🌐 서버 API 연결 초기화...")
        server_api = PortholeServerAPI(config)
        
        # 포트홀 감지기 초기화
        print("🔍 포트홀 감지기 초기화...")
        detector = PortholeDetector(config, server_api)
        
        print("✅ 초기화 완료!")
        print("\n🎥 웹캠을 통한 실시간 포트홀 감지를 시작합니다...")
        print("종료하려면 'q'를 누르세요.")
        
        # 실시간 감지 시작
        detector.run_realtime_detection()
        
    except KeyboardInterrupt:
        print("\n⏹️  사용자에 의해 중단되었습니다.")
    except FileNotFoundError as e:
        print(f"❌ 파일을 찾을 수 없습니다: {e}")
        print("💡 현재 디렉토리에서 실행하고 있는지 확인하세요.")
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("🔚 포트홀 감지 시스템 종료")

if __name__ == "__main__":
    main()
