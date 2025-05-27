from fastapi import APIRouter, HTTPException, Body
import base64
import os
from datetime import datetime
from backend.models import PortholeModel
from backend.crud import get_all_portholes, get_porthole_by_id, add_porthole, delete_porthole, update_porthole_status, update_porthole_image_path
from typing import Dict, List, Optional
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderUnavailable

router = APIRouter(tags=["Porthole"])
portholes_router = APIRouter(prefix="/api/portholes")

def coord_into_location(lat: float, lng: float):
    """
    위도와 경도로부터 도로명주소를 반환합니다. geopy모듈 사용
    """
    try:
        # geolocator 초기화
        geolocator = Nominatim(user_agent="South Korea")

        # 위도, 경도 지정
        location = geolocator.reverse((lat, lng), language='ko')  # 서울시청 좌표

        # 전체 raw 결과 확인
        raw = location.raw

        # 주소 구성요소 추출
        address = raw.get('address', {})

        city = address.get('city', '')  #서울
        if city == '서울':
            city += '특별'
        elif city in ['부산', '대구', '인천', '광주', '대전', '울산']:
            city += '광역'

        borough = address.get('borough', '') # 중구
        road = address.get('road', '')        # 세종대로
        # house_number = address.get('house_number', '') # 110

        # 원하는 포맷으로 정리
        output = f"{city}시 {borough} {road}".strip()

        return output

    except (GeocoderTimedOut, GeocoderUnavailable) as geo_err:
        return "주소 조회 실패 (지오코더 오류)"
    except Exception as e:
        return f"주소 조회 실패: {str(e)}"


def save_porthole_image(image_base64: str, porthole_id: int, image_format: str = "jpg") -> Optional[str]:
    """
    포트홀 이미지를 서버에 저장합니다.
    
    Args:
        image_base64: base64로 인코딩된 이미지 데이터
        porthole_id: 포트홀 ID
        image_format: 이미지 포맷 (jpg, png)
        
    Returns:
        저장된 이미지 파일 경로 또는 None
    """
    try:
        # 이미지 저장 디렉토리 생성
        images_dir = "static/porthole_images"
        os.makedirs(images_dir, exist_ok=True)
        
        # 이미지 파일명 생성
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"porthole_{porthole_id}_{timestamp}.{image_format}"
        filepath = os.path.join(images_dir, filename)
        
        # base64 디코딩 및 파일 저장
        image_data = base64.b64decode(image_base64)
        with open(filepath, 'wb') as f:
            f.write(image_data)
        
        # 상대 경로 반환 (웹에서 접근 가능한 경로)
        return f"/{filepath}"
        
    except Exception as e:
        print(f"이미지 저장 중 오류 발생: {e}")
        return None

@router.post("/api/notify_new_porthole")
def notify_new_porthole(
    lat: float = Body(..., description="포트홀의 위도 좌표"),
    lng: float = Body(..., description="포트홀의 경도 좌표"), 
    depth: float = Body(..., description="포트홀의 깊이(mm)"),
    image: Optional[str] = Body(None, description="base64로 인코딩된 포트홀 이미지")
):
    """
    감지 시스템에서 전송한 새로운 포트홀 정보를 받아 데이터베이스에 저장합니다.
    """
    # 이미지가 제공된 경우 먼저 저장
    image_path = None
    if image:
        # 임시 ID로 이미지 저장 (나중에 실제 ID로 업데이트)
        temp_id = int(datetime.now().timestamp())
        image_path = save_porthole_image(image, temp_id)
    
    porthole_data = {
        "lat": lat,
        "lng": lng,
        "depth": depth,
        "status": "발견됨",
        "location": coord_into_location(lat, lng),
        "image_path": image_path
    }
    
    try:
        porthole_id = add_porthole(porthole_data)
        
        # 실제 ID로 이미지 파일명 업데이트
        if image_path and porthole_id:
            # 새로운 파일명 생성
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"porthole_{porthole_id}_{timestamp}.jpg"
            new_filepath = f"static/porthole_images/{filename}"
            
            # 파일 이름 변경
            try:
                import shutil
                old_filepath = image_path[1:]  # '/' 제거
                shutil.move(old_filepath, new_filepath)
                image_path = f"/{new_filepath}"
                
                # 데이터베이스에서 이미지 경로 업데이트
                update_porthole_image_path(porthole_id, image_path)
            except Exception as e:
                print(f"이미지 파일명 업데이트 중 오류: {e}")
        
        return {
            "id": porthole_id,
            "image_path": image_path,
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
