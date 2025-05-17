"""
포트홀 감지 및 관리 시스템

이 시스템은 도로의 포트홀 위치 정보를 관리하고, 차량과의 거리를 계산하여 
알림을 제공하는 기능을 포함합니다.

사용 방법:
1. FastAPI 서버 실행: `uvicorn main:app --reload`
2. Streamlit 앱 실행: `streamlit run frontend/app.py --server.port 8501`
"""

import os
import sys
import uvicorn

# 현재 디렉토리를 시스템 경로에 추가
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

# backend 모듈에서 앱 가져오기
from backend import app

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
