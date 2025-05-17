from fastapi import APIRouter, HTTPException, Body
from backend.models import PortholeModel
from backend.crud import get_all_portholes, get_porthole_by_id, add_porthole, delete_porthole, update_porthole_status
from typing import Dict, List

router = APIRouter(tags=["Porthole"])
portholes_router = APIRouter(prefix="/api/portholes")

@router.post("/api/notify_new_porthole")
def notify_new_porthole(
    lat: float = Body(..., description="포트홀의 위도 좌표"),
    lng: float = Body(..., description="포트홀의 경도 좌표"), 
    depth: float = Body(..., description="포트홀의 깊이(mm)")
):
    """
    감지 시스템에서 전송한 새로운 포트홀 정보를 받아 데이터베이스에 저장합니다.
    """
    porthole_data = {
        "lat": lat,
        "lng": lng,
        "depth": depth,
        "status": "발견됨"
    }
    
    try:
        porthole_id = add_porthole(porthole_data)
        return {
            "id": porthole_id,
            "message": "포트홀이 성공적으로 등록되었습니다."
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"포트홀 등록 중 오류가 발생했습니다: {str(e)}")

@portholes_router.get("", response_model=List[Dict])
def list_portholes():
    """
    모든 포트홀 정보를 조회합니다.
    """
    return get_all_portholes()

@portholes_router.get("/{porthole_id}", response_model=Dict)
def get_porthole_detail(porthole_id: int):
    """
    특정 포트홀의 상세 정보를 조회합니다.
    """
    porthole = get_porthole_by_id(porthole_id)
    if not porthole:
        raise HTTPException(status_code=404, detail="포트홀을 찾을 수 없습니다")
    return porthole

@portholes_router.post("/add")
def create_porthole(porthole: PortholeModel):
    """
    새로운 포트홀을 추가합니다.
    """
    porthole_id = add_porthole(porthole.dict())
    return {
        "id": porthole_id,
        "message": "포트홀이 성공적으로 추가되었습니다."
    }

@portholes_router.delete("/{porthole_id}")
def remove_porthole(porthole_id: int):
    """
    특정 포트홀을 삭제합니다.
    """
    if not get_porthole_by_id(porthole_id):
        raise HTTPException(status_code=404, detail="포트홀을 찾을 수 없습니다")
    
    if delete_porthole(porthole_id):
        return {"message": "포트홀이 성공적으로 삭제되었습니다."}
    else:
        raise HTTPException(status_code=500, detail="포트홀 삭제 중 오류가 발생했습니다.")

@portholes_router.put("/{porthole_id}/status")
def update_status(porthole_id: int, status: str):
    """
    포트홀의 상태를 업데이트합니다.
    """
    if not get_porthole_by_id(porthole_id):
        raise HTTPException(status_code=404, detail="포트홀을 찾을 수 없습니다")
        
    if status not in ["발견됨", "수리중", "수리완료"]:
        raise HTTPException(status_code=400, detail="유효하지 않은 상태입니다. '발견됨', '수리중', '수리완료' 중 하나여야 합니다.")
        
    if update_porthole_status(porthole_id, status):
        return {"message": f"포트홀 상태가 '{status}'(으)로 업데이트되었습니다."}
    else:
        raise HTTPException(status_code=500, detail="포트홀 상태 업데이트 중 오류가 발생했습니다.")

# portholes_router를 메인 router에 포함
router.include_router(portholes_router)
