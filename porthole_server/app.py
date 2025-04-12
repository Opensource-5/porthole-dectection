"""
포트홀 감지 및 관리 시스템 FastAPI 백엔드

이 애플리케이션은 포트홀 위치 정보를 관리하고 차량과의 거리를 계산하는 
REST API를 제공합니다.
"""

from fastapi import FastAPI, Form, HTTPException, Request
from fastapi.responses import RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import sqlite3
import math
import os
import json
from typing import Dict, List, Optional, Union
from datetime import datetime

# ======================================================
# 설정 및 초기화
# ======================================================

# SQLite DB 파일 경로 설정
DATABASE_PATH = "porthole.db"

# FastAPI 앱 생성
app = FastAPI(
    title="포트홀 감지 API",
    description="도로의 포트홀 위치와 차량 정보를 관리하는 API",
    version="1.0.0"
)

# CORS 설정 추가
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 실제 운영 환경에서는 특정 출처만 허용하도록 변경 권장
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 새로 감지된 포트홀 저장 (클라이언트에서 확인할 수 있도록)
# 최근 감지된 포트홀 목록 (최대 10개 유지)
recent_detected_portholes = []
MAX_RECENT_PORTHOLES = 10

# ======================================================
# 데이터베이스 관련 함수
# ======================================================

def get_db_connection() -> sqlite3.Connection:
    """
    SQLite 데이터베이스 연결을 생성하고 반환합니다.
    
    Returns:
        sqlite3.Connection: 데이터베이스 연결 객체
    """
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db() -> None:
    """
    데이터베이스가 존재하지 않으면 초기화합니다.
    필요한 테이블(포트홀, 차량)을 생성하고 예시 데이터를 추가합니다.
    """
    db_exists = os.path.exists(DATABASE_PATH)
    
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    
    # 포트홀 테이블 생성
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS porthole (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            lat REAL NOT NULL,            -- 위도
            lng REAL NOT NULL,            -- 경도
            depth REAL,                   -- 깊이(cm)
            location TEXT,                -- 위치 설명
            date TEXT,                    -- 발견 날짜
            status TEXT                   -- 상태 (발견됨, 수리중, 수리완료 등)
        )
    ''')
    
    # 차량 테이블 생성
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS car (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            lat REAL NOT NULL,            -- 위도
            lng REAL NOT NULL             -- 경도
        )
    ''')
    
    # 기존 데이터베이스가 없었을 경우에만 샘플 데이터 추가
    if not db_exists:
        # 샘플 포트홀 데이터 추가
        porthole_samples = [
            (37.5665, 126.9780, 5.2, '서울시 중구 을지로', '2023-04-15', '발견됨'),
            (37.5113, 127.0980, 3.8, '서울시 송파구 올림픽로', '2023-04-16', '수리중'),
            (37.4989, 127.0280, 2.5, '서울시 강남구 테헤란로', '2023-04-10', '수리완료')
        ]
        cursor.executemany('''
            INSERT INTO porthole (lat, lng, depth, location, date, status)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', porthole_samples)
        
        # 샘플 차량 데이터 추가
        car_samples = [
            (37.5668, 126.9785),  # 을지로 근처
            (37.5115, 127.0990),  # 올림픽로 근처
            (37.4992, 127.0275)   # 테헤란로 근처
        ]
        cursor.executemany('''
            INSERT INTO car (lat, lng)
            VALUES (?, ?)
        ''', car_samples)
        
        print("예시 데이터 3개씩 추가 완료")
    
    conn.commit()
    conn.close()
    print("데이터베이스 초기화 완료")

def get_all_portholes() -> List[Dict]:
    """
    모든 포트홀 목록을 조회합니다.
    
    Returns:
        List[Dict]: 포트홀 정보 목록
    """
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT id, location, depth, date, status FROM porthole")
        results = cursor.fetchall()
        conn.close()
        return [dict(row) for row in results]
    except Exception as e:
        print(f"get_all_portholes 오류: {e}")
        return []

def get_porthole_by_id(porthole_id: int) -> Optional[Dict]:
    """
    ID로 특정 포트홀 정보를 조회합니다.
    
    Args:
        porthole_id: 포트홀 ID
        
    Returns:
        Optional[Dict]: 포트홀 정보 또는 None (찾지 못한 경우)
    """
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM porthole WHERE id = ?", (porthole_id,))
        result = cursor.fetchone()
        conn.close()
        return dict(result) if result else None
    except Exception as e:
        print(f"get_porthole_by_id 오류: {e}")
        return None

def get_car_by_id(car_id: int) -> Optional[Dict]:
    """
    ID로 특정 차량 정보를 조회합니다.
    
    Args:
        car_id: 차량 ID
        
    Returns:
        Optional[Dict]: 차량 정보 또는 None (찾지 못한 경우)
    """
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM car WHERE id = ?", (car_id,))
        result = cursor.fetchone()
        conn.close()
        return dict(result) if result else None
    except Exception as e:
        print(f"get_car_by_id 오류: {e}")
        return None

def get_all_cars() -> List[Dict]:
    """
    모든 차량 정보를 조회합니다.
    
    Returns:
        List[Dict]: 차량 정보 목록
    """
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM car")
        results = cursor.fetchall()
        conn.close()
        return [dict(row) for row in results]
    except Exception as e:
        print(f"get_all_cars 오류: {e}")
        return []

def update_porthole_status(porthole_id: int, new_status: str) -> bool:
    """
    포트홀의 상태를 업데이트합니다.
    
    Args:
        porthole_id: 포트홀 ID
        new_status: 새로운 상태 값
        
    Returns:
        bool: 업데이트 성공 여부
    """
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("UPDATE porthole SET status = ? WHERE id = ?", (new_status, porthole_id))
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        print(f"update_porthole_status 오류: {e}")
        return False

# ======================================================
# 유틸리티 함수
# ======================================================

def calculate_distance(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    """
    두 지점 사이의 거리를 계산합니다 (하버사인 공식 사용).
    
    Args:
        lat1: 첫 번째 지점의 위도
        lng1: 첫 번째 지점의 경도
        lat2: 두 번째 지점의 위도
        lng2: 두 번째 지점의 경도
        
    Returns:
        float: 미터 단위 거리
    """
    # 지구 반지름 (미터 단위)
    R = 6371 * 1000
    
    # 각도를 라디안으로 변환
    dLat = math.radians(lat2 - lat1)
    dLng = math.radians(lng2 - lng1)

    # 하버사인 공식
    a = math.sin(dLat / 2) ** 2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dLng / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    # 거리 계산 (미터 단위)
    return R * c

# 포트홀 근접 알림을 위한 거리 임계값 (미터 단위)
PROXIMITY_THRESHOLD = 100  # 100미터 이내

# ======================================================
# API 엔드포인트
# ======================================================

@app.get("/", response_description="API 루트 페이지")
def root():
    """API 루트 페이지"""
    return {
        "message": "포트홀 감지 API",
        "endpoints": [
            "/api/portholes - 모든 포트홀 목록 조회",
            "/update_status - 포트홀 상태 업데이트(POST)",
            "/api/portholes/{porthole_id} - 특정 포트홀 상세 정보 조회",
            "/api/nearby_portholes/{car_id} - 차량 근처의 포트홀 알림",
            "/api/check_proximity - 포트홀 근처의 차량 검사 및 알림",
            "/api/notify_new_porthole - 새로운 포트홀 감지 알림 수신(POST)",
            "/api/new_portholes - 최근 감지된 포트홀 목록 조회"
        ]
    }

@app.get("/api/portholes", response_description="포트홀 목록")
def api_portholes():
    """
    모든 포트홀 목록을 조회하는 API 엔드포인트
    
    Returns:
        List[Dict]: 포트홀 정보 목록
    """
    return get_all_portholes()

@app.get("/api/portholes/{porthole_id}", response_description="포트홀 상세 정보")
def api_porthole_details(porthole_id: int):
    """
    특정 ID의 포트홀 상세 정보를 조회하는 API 엔드포인트
    
    Args:
        porthole_id: 포트홀 ID
        
    Returns:
        Dict: 포트홀 상세 정보
        
    Raises:
        HTTPException: 포트홀 데이터가 없는 경우
    """
    porthole = get_porthole_by_id(porthole_id)
    if not porthole:
        raise HTTPException(status_code=404, detail=f"포트홀 ID {porthole_id}를 찾을 수 없습니다.")
    return porthole

@app.get("/api/nearby_portholes/{car_id}", response_description="차량 근처의 포트홀 알림")
def nearby_portholes(car_id: int):
    """
    특정 차량 주변의 포트홀을 알려주는 API 엔드포인트
    
    Args:
        car_id: 차량 ID
        
    Returns:
        Dict: 근처 포트홀 목록과 거리 정보
        
    Raises:
        HTTPException: 차량 데이터가 없는 경우
    """
    car = get_car_by_id(car_id)
    if not car:
        raise HTTPException(status_code=404, detail=f"차량 ID {car_id}를 찾을 수 없습니다.")
    
    # 모든 포트홀 조회
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM porthole")
    portholes = [dict(row) for row in cursor.fetchall()]
    conn.close()
    
    # 차량과 각 포트홀 사이의 거리 계산
    nearby_results = []
    for porthole in portholes:
        distance = calculate_distance(car['lat'], car['lng'], porthole['lat'], porthole['lng'])
        
        # 임계값 이하의 포트홀만 포함
        if distance <= PROXIMITY_THRESHOLD:
            nearby_results.append({
                "porthole_id": porthole["id"],
                "location": porthole["location"],
                "distance": round(distance, 2),  # 소수점 2자리까지 반올림
                "depth": porthole["depth"],
                "status": porthole["status"]
            })
    
    # 거리순으로 정렬
    nearby_results.sort(key=lambda x: x["distance"])
    
    return {
        "car_id": car_id,
        "car_location": {"lat": car["lat"], "lng": car["lng"]},
        "nearby_portholes": nearby_results,
        "proximity_threshold": PROXIMITY_THRESHOLD
    }

@app.get("/api/check_proximity", response_description="포트홀 근처의 차량 검사 및 알림")
def check_proximity():
    """
    모든 차량과 포트홀 간의 거리를 계산하고 일정 거리 이하인 경우 알림 정보를 제공하는 API
    
    Returns:
        Dict: 각 차량에 대한 근처 포트홀 알림 정보
    """
    # 모든 차량과 포트홀 정보 조회
    cars = get_all_cars()
    
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM porthole")
    portholes = [dict(row) for row in cursor.fetchall()]
    conn.close()
    
    # 각 차량에 대한 알림 정보를 저장할 목록
    alerts = []
    
    # 모든 차량에 대해 검사
    for car in cars:
        car_alerts = []
        
        # 각 포트홀에 대한 거리 계산
        for porthole in portholes:
            distance = calculate_distance(car['lat'], car['lng'], porthole['lat'], porthole['lng'])
            
            # 임계값 이하인 경우 알림 추가
            if distance <= PROXIMITY_THRESHOLD:
                car_alerts.append({
                    "porthole_id": porthole["id"],
                    "location": porthole["location"],
                    "distance": round(distance, 2),
                    "depth": porthole["depth"],
                    "status": porthole["status"]
                })
        
        # 거리순으로 정렬
        car_alerts.sort(key=lambda x: x["distance"])
        
        # 차량에 대한 알림이 있는 경우만 추가
        if car_alerts:
            alerts.append({
                "car_id": car["id"],
                "car_location": {"lat": car["lat"], "lng": car["lng"]},
                "alerts": car_alerts
            })
    
    return {
        "timestamp": "알림 생성 시각",
        "proximity_threshold": PROXIMITY_THRESHOLD,
        "alerts": alerts
    }

@app.post("/update_status", response_description="포트홀 상태 업데이트")
async def update_status(porthole_id: int = Form(...), new_status: str = Form(...)):
    """
    포트홀의 상태를 업데이트하는 API 엔드포인트
    
    Args:
        porthole_id: 포트홀 ID (폼 데이터)
        new_status: 새로운 상태 값 (폼 데이터)
        
    Returns:
        RedirectResponse: API 포트홀 목록으로 리다이렉트
        
    Raises:
        HTTPException: 업데이트 실패 시
    """
    print(f"상태 업데이트 요청: ID={porthole_id}, 새 상태={new_status}")
    
    # 포트홀 존재 여부 확인
    porthole = get_porthole_by_id(porthole_id)
    if not porthole:
        raise HTTPException(status_code=404, detail=f"포트홀 ID {porthole_id}를 찾을 수 없습니다.")
    
    # 상태 업데이트 수행
    success = update_porthole_status(porthole_id, new_status)
    if not success:
        raise HTTPException(status_code=500, detail="포트홀 상태 업데이트 중 오류가 발생했습니다.")
    
    return RedirectResponse(url="/api/portholes", status_code=303)

@app.post("/api/notify_new_porthole", response_description="새로운 포트홀 감지 알림 수신")
async def notify_new_porthole(request: Request):
    """
    외부 모듈(detect.py)에서 새로운 포트홀 감지 시 호출되는 웹훅 엔드포인트
    
    Args:
        request: FastAPI Request 객체
        
    Returns:
        Dict: 처리 결과
    """
    try:
        # 요청 데이터 파싱
        data = await request.json()
        
        # 필수 필드 확인
        required_fields = ['lat', 'lng', 'depth', 'location']
        for field in required_fields:
            if field not in data:
                raise HTTPException(status_code=400, detail=f"필수 필드 '{field}'가 없습니다")
        
        # 현재 날짜 설정
        current_date = datetime.now().strftime("%Y-%m-%d")
        
        # 데이터베이스에 포트홀 정보 저장
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO porthole (lat, lng, depth, location, date, status)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (data['lat'], data['lng'], data['depth'], data['location'], current_date, '발견됨'))
        
        # 저장된 포트홀의 ID 가져오기
        porthole_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        # 포트홀 정보 가져오기
        porthole = get_porthole_by_id(porthole_id)
        if not porthole:
            raise HTTPException(status_code=500, detail=f"포트홀 정보 저장 후 조회 실패")
        
        # 알림 데이터 생성
        notification = {
            "porthole_id": porthole['id'],
            "lat": porthole['lat'],
            "lng": porthole['lng'],
            "depth": porthole['depth'],
            "location": porthole['location'],
            "date": porthole['date'],
            "status": porthole['status'],
            "detected_at": datetime.now().isoformat()
        }
        
        # 최근 감지된 포트홀 목록에 추가
        global recent_detected_portholes
        recent_detected_portholes.append(notification)
        
        # 최대 개수 유지
        if len(recent_detected_portholes) > MAX_RECENT_PORTHOLES:
            recent_detected_portholes = recent_detected_portholes[-MAX_RECENT_PORTHOLES:]
        
        print(f"새로운 포트홀이 감지되어 DB에 저장되었습니다. ID: {porthole['id']}, 위치: {porthole['location']}")
        
        return {
            "success": True,
            "message": "새로운 포트홀이 성공적으로 저장되었습니다",
            "porthole_id": porthole['id']
        }
    
    except HTTPException as he:
        raise he
    except Exception as e:
        print(f"포트홀 저장 중 오류 발생: {e}")
        raise HTTPException(status_code=500, detail=f"내부 서버 오류: {str(e)}")

@app.get("/api/new_portholes", response_description="최근 감지된 포트홀 목록")
def get_new_portholes():
    """
    최근에 감지된 포트홀 목록을 조회하는 API 엔드포인트
    Streamlit 앱에서는 이 엔드포인트를 주기적으로 폴링하여 새로운 포트홀을 확인할 수 있습니다.
    
    Returns:
        Dict: 최근 감지된 포트홀 목록
    """
    global recent_detected_portholes
    
    # 최근에 감지된 포트홀 목록이 있는 경우 반환
    return {
        "count": len(recent_detected_portholes),
        "portholes": recent_detected_portholes
    }

# 앱 시작 시 DB 초기화
init_db()

# 직접 실행 시 서버 시작
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
