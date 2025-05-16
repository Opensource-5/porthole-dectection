"""
백엔드 앱 초기화 모듈
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
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

# 데이터베이스 초기화
init_db()