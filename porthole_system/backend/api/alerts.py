from fastapi import APIRouter, HTTPException
from backend.models import AlertAcknowledgeModel
from backend.crud import get_car_alerts, get_car_by_id, acknowledge_alerts
from typing import Dict, List, Optional

router = APIRouter(prefix="/api/alerts", tags=["Alert"])

@router.get("/car/{car_id}")
def get_alerts_for_car(car_id: int, include_acknowledged: bool = False):
    """
    특정 차량에 대한 포트홀 근접 알림을 조회합니다.
    
    Parameters:
    - car_id: 차량 ID
    - include_acknowledged: 확인된 알림도 포함할지 여부 (기본값: False)
    
    Returns:
    - alerts: 알림 목록
    - count: 알림 수
    """
    # 차량 존재 여부 확인
    car = get_car_by_id(car_id)
    if not car:
        raise HTTPException(status_code=404, detail="차량을 찾을 수 없습니다")
    
    # 알림 조회
    alerts = get_car_alerts(car_id, include_acknowledged)
    
    return {
        "car_id": car_id,
        "alerts": alerts,
        "count": len(alerts)
    }

@router.post("/car/{car_id}/acknowledge")
def acknowledge_car_alerts(car_id: int, data: AlertAcknowledgeModel):
    """
    특정 차량의 알림을 확인 처리합니다.
    
    Parameters:
    - car_id: 차량 ID
    - data: 확인할 알림 ID 목록
    
    Returns:
    - success: 처리 성공 여부
    - count: 처리된 알림 수
    """
    # 차량 존재 여부 확인
    car = get_car_by_id(car_id)
    if not car:
        raise HTTPException(status_code=404, detail="차량을 찾을 수 없습니다")
    
    # 알림 확인 처리
    success = acknowledge_alerts(car_id, data.alert_ids)
    
    if success:
        return {
            "success": True,
            "message": f"{len(data.alert_ids)}개의 알림이 확인 처리되었습니다.",
            "count": len(data.alert_ids)
        }
    else:
        raise HTTPException(status_code=500, detail="알림 확인 처리 중 오류가 발생했습니다.")
