from backend.db import get_db_connection
from typing import List, Dict, Optional, Union
from datetime import datetime

# ======================================================
# 포트홀 관련 CRUD 함수
# ======================================================

from backend.db import get_db_connection, db_transaction
from backend.logger import get_logger
from typing import List, Dict, Optional, Union
from datetime import datetime

# 로거 가져오기
logger = get_logger()

def get_all_portholes() -> List[Dict]:
    """
    모든 포트홀 목록을 조회합니다.
    
    Returns:
        List[Dict]: 포트홀 정보 목록
    """
    try:
        with db_transaction() as (conn, cursor):
            cursor.execute("SELECT id, location, depth, date, status, lat, lng FROM porthole")
            results = cursor.fetchall()
            return [dict(row) for row in results]
    except Exception as e:
        logger.error(f"포트홀 목록 조회 중 오류 발생: {e}")
        
        # 데이터베이스 테이블이 없는 경우 (no such table 오류) 데이터베이스를 초기화
        if "no such table" in str(e):
            logger.info("데이터베이스 테이블이 없습니다. 데이터베이스를 초기화합니다.")
            from backend.db import init_db
            init_db()
            # 초기화 후 다시 시도
            try:
                with db_transaction() as (conn, cursor):
                    cursor.execute("SELECT id, location, depth, date, status, lat, lng FROM porthole")
                    results = cursor.fetchall()
                    return [dict(row) for row in results]
            except Exception as e2:
                logger.error(f"데이터베이스 초기화 후에도 조회 중 오류 발생: {e2}")
        
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
        logger.error(f"get_porthole_by_id 오류: {e}")
        
        # 데이터베이스 테이블이 없는 경우 데이터베이스를 초기화
        if "no such table" in str(e):
            logger.info("데이터베이스 테이블이 없습니다. 데이터베이스를 초기화합니다.")
            from backend.db import init_db
            init_db()
            # 초기화 후 다시 시도
            try:
                conn = get_db_connection()
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM porthole WHERE id = ?", (porthole_id,))
                result = cursor.fetchone()
                conn.close()
                return dict(result) if result else None
            except Exception as e2:
                logger.error(f"데이터베이스 초기화 후에도 조회 중 오류 발생: {e2}")
        
        return None

def add_porthole(porthole_data: Dict) -> int:
    """
    새로운 포트홀 정보를 데이터베이스에 추가합니다.
    
    Args:
        porthole_data: 포트홀 정보 (위도, 경도, 깊이, 위치, 상태)
        
    Returns:
        int: 추가된 포트홀의 ID
    """
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # 현재 날짜를 기본값으로 사용
        current_date = datetime.now().strftime("%Y-%m-%d")
        
        cursor.execute(
            "INSERT INTO porthole (lat, lng, depth, location, date, status) VALUES (?, ?, ?, ?, ?, ?)",
            (
                porthole_data["lat"], 
                porthole_data["lng"], 
                porthole_data.get("depth"), 
                porthole_data.get("location"), 
                current_date,
                porthole_data.get("status", "발견됨")
            )
        )
        porthole_id = cursor.lastrowid
        conn.commit()
        conn.close()
        return porthole_id
    except Exception as e:
        print(f"add_porthole 오류: {e}")
        raise

def delete_porthole(porthole_id: int) -> bool:
    """
    특정 ID의 포트홀 정보를 데이터베이스에서 삭제합니다.
    
    Args:
        porthole_id: 삭제할 포트홀 ID
        
    Returns:
        bool: 삭제 성공 여부
    """
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("DELETE FROM porthole WHERE id = ?", (porthole_id,))
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        print(f"delete_porthole 오류: {e}")
        return False

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
# 차량 관련 CRUD 함수
# ======================================================

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

def add_car(car_data: Dict) -> int:
    """
    새로운 차량 정보를 데이터베이스에 추가합니다.
    
    Args:
        car_data: 차량 정보 (위도, 경도)
        
    Returns:
        int: 추가된 차량의 ID
    """
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO car (lat, lng) VALUES (?, ?)",
            (car_data["lat"], car_data["lng"])
        )
        car_id = cursor.lastrowid
        conn.commit()
        conn.close()
        return car_id
    except Exception as e:
        print(f"add_car 오류: {e}")
        raise

def delete_car(car_id: int) -> bool:
    """
    특정 ID의 차량 정보를 데이터베이스에서 삭제합니다.
    
    Args:
        car_id: 삭제할 차량 ID
        
    Returns:
        bool: 삭제 성공 여부
    """
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("DELETE FROM car WHERE id = ?", (car_id,))
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        print(f"delete_car 오류: {e}")
        return False

def update_car_location(car_id: int, lat: float, lng: float) -> bool:
    """
    차량의 위치 정보를 업데이트합니다.
    
    Args:
        car_id: 차량 ID
        lat: 새로운 위도 값
        lng: 새로운 경도 값
        
    Returns:
        bool: 업데이트 성공 여부
    """
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("UPDATE car SET lat = ?, lng = ? WHERE id = ?", (lat, lng, car_id))
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        print(f"update_car_location 오류: {e}")
        return False

# ======================================================
# 알림 관련 CRUD 함수
# ======================================================

def get_car_alerts(car_id: int, include_acknowledged: bool = False) -> List[Dict]:
    """
    특정 차량에 대한 포트홀 근접 알림을 조회합니다.
    
    Args:
        car_id: 차량 ID
        include_acknowledged: 확인된 알림도 포함할지 여부
        
    Returns:
        List[Dict]: 알림 정보 목록
    """
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        query = """
            SELECT a.*, p.depth, p.location, p.status
            FROM alert a
            JOIN porthole p ON a.porthole_id = p.id
            WHERE a.car_id = ?
        """
        
        if not include_acknowledged:
            query += " AND a.acknowledged = 0"
            
        cursor.execute(query, (car_id,))
        results = cursor.fetchall()
        conn.close()
        
        alerts = []
        for row in results:
            alert_dict = dict(row)
            # 위험도 계산 (거리와, 포트홀 깊이 기반)
            depth = alert_dict.get('depth') or 0
            distance = alert_dict.get('distance') or 0
            
            if depth > 2000:
                risk_level = "High"
            elif depth > 1000:
                risk_level = "Medium"
            else:
                risk_level = "Low"
                
            alert_dict["risk_level"] = risk_level
            alerts.append(alert_dict)
            
        return alerts
    except Exception as e:
        print(f"get_car_alerts 오류: {e}")
        return []

def add_alert(car_id: int, porthole_id: int, distance: float, one_time_only: bool = False) -> int:
    """
    새로운 알림을 추가합니다.
    
    Args:
        car_id: 차량 ID
        porthole_id: 포트홀 ID
        distance: 차량과 포트홀 간 거리(미터)
        one_time_only: True인 경우, 해당 포트홀에 대해 이전에 알림을 받은 적이 있으면
                       (확인 여부와 관계없이) 새 알림을 생성하지 않습니다.
        
    Returns:
        int: 추가된 알림의 ID, one_time_only=True이고 이미 알림이 있었다면 -1
        
    Raises:
        ValueError: 차량이나 포트홀 ID가 유효하지 않을 경우
        Exception: 데이터베이스 오류 발생시
    """
    conn = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # 차량과 포트홀이 존재하는지 검증
        cursor.execute("SELECT id FROM car WHERE id = ?", (car_id,))
        if not cursor.fetchone():
            raise ValueError(f"차량 ID {car_id}를 찾을 수 없습니다.")
            
        cursor.execute("SELECT id FROM porthole WHERE id = ?", (porthole_id,))
        if not cursor.fetchone():
            raise ValueError(f"포트홀 ID {porthole_id}를 찾을 수 없습니다.")
        
        if one_time_only:
            # 같은 차량-포트홀 조합으로 알림을 받은 적이 있는지 확인 (확인 여부와 관계없이)
            cursor.execute("""
                SELECT id FROM alert 
                WHERE car_id = ? AND porthole_id = ?
            """, (car_id, porthole_id))
            
            if cursor.fetchone():
                # 이미 알림이 생성된 적이 있으므로 새 알림을 생성하지 않음
                return -1
        
        # 중복 알림 확인 (같은 차량-포트홀 조합으로 미확인 알림이 이미 있는지)
        cursor.execute("""
            SELECT id FROM alert 
            WHERE car_id = ? AND porthole_id = ? AND acknowledged = 0
        """, (car_id, porthole_id))
        
        existing = cursor.fetchone()
        if existing:
            # 이미 존재하는 경우 거리만 업데이트
            cursor.execute("""
                UPDATE alert SET distance = ?, created_at = ? WHERE id = ?
            """, (distance, datetime.now().isoformat(), existing["id"]))
            alert_id = existing["id"]
        else:
            # 새로운 알림 추가
            cursor.execute("""
                INSERT INTO alert (car_id, porthole_id, distance, created_at, acknowledged)
                VALUES (?, ?, ?, ?, ?)
            """, (car_id, porthole_id, distance, datetime.now().isoformat(), False))
            alert_id = cursor.lastrowid
            
        conn.commit()
        return alert_id
    except Exception as e:
        print(f"add_alert 오류: {e}")
        if conn:
            conn.rollback()  # 트랜잭션 롤백
        raise
    finally:
        if conn:
            conn.close()  # 연결 항상 닫기

def acknowledge_alerts(car_id: int, alert_ids: List[int]) -> bool:
    """
    여러 알림을 확인 처리합니다.
    
    Args:
        car_id: 차량 ID
        alert_ids: 확인할 알림 ID 목록
        
    Returns:
        bool: 처리 성공 여부
    """
    if not alert_ids:
        return True
        
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # 해당 차량의 알림 ID만 처리 (보안상 중요)
        placeholders = ",".join(["?"] * len(alert_ids))
        query = f"""
            UPDATE alert SET acknowledged = 1
            WHERE car_id = ? AND id IN ({placeholders})
        """
        
        params = [car_id] + alert_ids
        cursor.execute(query, params)
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        print(f"acknowledge_alerts 오류: {e}")
        return False
