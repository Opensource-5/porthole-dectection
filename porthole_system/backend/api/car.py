from fastapi import APIRouter, HTTPException
from backend.models import CarModel
from backend.crud import get_all_cars, get_car_by_id, add_car, delete_car, update_car_location
from typing import Dict, List

router = APIRouter(prefix="/api/cars", tags=["Car"])

@router.get("", response_model=List[Dict])
def list_cars():
    """
    모든 차량 정보를 조회합니다.
    """
    return get_all_cars()

@router.get("/{car_id}", response_model=Dict)
def get_car_detail(car_id: int):
    """
    특정 차량의 상세 정보를 조회합니다.
    """
    car = get_car_by_id(car_id)
    if not car:
        raise HTTPException(status_code=404, detail="차량을 찾을 수 없습니다")
    return car

@router.post("/add")
def create_car(car: CarModel):
    """
    새로운 차량을 추가합니다.
    """
    car_id = add_car(car.dict())
    return {
        "id": car_id,
        "message": "차량이 성공적으로 추가되었습니다."
    }

@router.delete("/{car_id}")
def remove_car(car_id: int):
    """
    특정 차량을 삭제합니다.
    """
    if not get_car_by_id(car_id):
        raise HTTPException(status_code=404, detail="차량을 찾을 수 없습니다")
    
    if delete_car(car_id):
        return {"message": "차량이 성공적으로 삭제되었습니다."}
    else:
        raise HTTPException(status_code=500, detail="차량 삭제 중 오류가 발생했습니다.")

@router.put("/{car_id}/location")
def update_location(car_id: int, car: CarModel):
    """
    차량의 위치 정보를 업데이트합니다.
    """
    if not get_car_by_id(car_id):
        raise HTTPException(status_code=404, detail="차량을 찾을 수 없습니다")
        
    if update_car_location(car_id, car.lat, car.lng):
        return {"message": "차량 위치가 업데이트되었습니다."}
    else:
        raise HTTPException(status_code=500, detail="차량 위치 업데이트 중 오류가 발생했습니다.")
