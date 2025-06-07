import sqlite3
import os
from contextlib import contextmanager

DATABASE_PATH = "porthole.db"

def get_db_connection():
    """
    SQLite 데이터베이스 연결을 생성하고 반환합니다.
    
    Returns:
        sqlite3.Connection: 데이터베이스 연결 객체
    """
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    return conn

@contextmanager
def db_transaction():
    """
    트랜잭션을 관리하는 컨텍스트 매니저입니다.
    트랜잭션 내에서 예외가 발생하면 자동으로 롤백하고, 성공하면 커밋합니다.
    
    Yields:
        sqlite3.Connection: 데이터베이스 연결 객체
        sqlite3.Cursor: 데이터베이스 커서 객체
    
    Example:
        with db_transaction() as (conn, cursor):
            cursor.execute("INSERT INTO ...")
            # 예외가 발생하지 않으면 자동으로 커밋
    """
    conn = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        yield conn, cursor
        conn.commit()
    except Exception as e:
        if conn:
            conn.rollback()
        raise
    finally:
        if conn:
            conn.close()

def init_db():
    """
    데이터베이스가 존재하지 않으면 초기화합니다.
    필요한 테이블(포트홀, 차량)을 생성하고 예시 데이터를 추가합니다.
    """
    db_exists = os.path.exists(DATABASE_PATH)
    
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # 포트홀 테이블 생성
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS porthole (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            lat REAL NOT NULL,            -- 위도
            lng REAL NOT NULL,            -- 경도
            depth REAL,                   -- 깊이
            location TEXT,                -- 위치 설명
            date TEXT,                    -- 발견 날짜
            status TEXT,                  -- 상태 (발견됨, 수리중, 수리완료 등)
            image_path TEXT               -- 이미지 파일 경로
        )
    ''')
    
    # 기존 포트홀 테이블에 image_path 컬럼이 없는 경우 추가
    try:
        cursor.execute("ALTER TABLE porthole ADD COLUMN image_path TEXT")
        print("포트홀 테이블에 image_path 컬럼 추가됨")
    except sqlite3.OperationalError:
        # 컬럼이 이미 존재하는 경우 무시
        pass
    
    # 차량 테이블 생성
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS car (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            lat REAL NOT NULL,            -- 위도
            lng REAL NOT NULL             -- 경도
        )
    ''')
    
    # 포트홀 알림 테이블 생성
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS alert (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            car_id INTEGER,               -- 차량 ID
            porthole_id INTEGER,          -- 포트홀 ID  
            distance REAL,                -- 거리(m)
            created_at TEXT,              -- 생성 시간
            acknowledged BOOLEAN,         -- 확인 여부
            FOREIGN KEY (car_id) REFERENCES car (id),
            FOREIGN KEY (porthole_id) REFERENCES porthole (id)
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
