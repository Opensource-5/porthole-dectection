from pydantic import BaseModel, Field
from typing import Optional, List

class CarModel(BaseModel):
    lat: float = Field(..., description="차량의 위도 좌표")
    lng: float = Field(..., description="차량의 경도 좌표")

class PortholeModel(BaseModel):
    lat: float = Field(..., description="포트홀의 위도 좌표")
    lng: float = Field(..., description="포트홀의 경도 좌표")
    depth: Optional[float] = Field(None, description="포트홀의 깊이(cm)")
    location: Optional[str] = Field(None, description="포트홀의 위치 설명")
    status: str = Field("발견됨", description="포트홀의 상태 (발견됨, 수리중, 수리완료 등)")
    image_path: Optional[str] = Field(None, description="포트홀 이미지 파일 경로")

class AlertModel(BaseModel):
    car_id: int = Field(..., description="차량 ID")
    porthole_id: int = Field(..., description="포트홀 ID")
    distance: float = Field(..., description="차량과 포트홀 간의 거리(미터)")
    created_at: str = Field(..., description="알림 생성 시간")
    acknowledged: bool = Field(False, description="알림 확인 여부")

class AlertAcknowledgeModel(BaseModel):
    alert_ids: List[int] = Field(..., description="확인할 알림 ID 목록")
