"""
백엔드 앱 초기화 모듈
"""

import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from backend.scheduler import scheduler, monitor_proximity
from backend.db import init_db
from contextlib import asynccontextmanager

# API 라우터 가져오기
from backend.api import porthole, car, alerts

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    앱 실행 및 종료 시 실행될 작업 정의
    """
    # 앱 시작 시 실행
    # 데이터베이스 초기화를 먼저 수행
    try:
        print("데이터베이스 초기화 시작")
        init_db()
        print("데이터베이스 초기화 완료")
    except Exception as e:
        print(f"데이터베이스 초기화 중 오류 발생: {e}")
    
    # 스케줄러 시작
    scheduler.add_job(monitor_proximity, 'interval', seconds=10)
    scheduler.start()
    print("포트홀-차량 모니터링 스케줄러가 시작되었습니다")

    yield  # 앱 실행 중

    # 앱 종료 시 실행
    scheduler.shutdown()
    print("포트홀-차량 모니터링 스케줄러가 종료되었습니다")

# FastAPI 인스턴스 생성
app = FastAPI(
    title="포트홀 감지 API",
    description="도로의 포트홀 위치와 차량 정보를 관리하는 API",
    version="1.0.0",
    lifespan=lifespan
)

# CORS 설정 추가
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 실제 운영 환경에서는 특정 출처만 허용하도록 변경 권장
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 라우터 등록
app.include_router(porthole.router)
app.include_router(car.router)
app.include_router(alerts.router)

# 정적 파일 서빙 설정 - 포트홀 이미지
static_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "static")
if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")
    print(f"정적 파일 서빙 설정 완료: {static_dir}")
else:
    print(f"정적 파일 디렉토리가 존재하지 않습니다: {static_dir}")

# 데이터베이스 초기화
init_db()