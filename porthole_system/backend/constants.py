"""
애플리케이션에서 사용하는 상수 정의 모듈
"""

# 포트홀 상태 정의
class PortholeStatus:
    FOUND = "발견됨"
    REPAIRING = "수리중"
    REPAIRED = "수리완료"
    
    @classmethod
    def all_statuses(cls):
        """모든 유효한 포트홀 상태 값 반환"""
        return [cls.FOUND, cls.REPAIRING, cls.REPAIRED]
    
    @classmethod
    def is_valid_status(cls, status):
        """주어진 상태가 유효한지 확인"""
        return status in cls.all_statuses()

# 알림 관련 상수
class AlertSettings:
    # 최대 거리 임계값 (미터)
    MAX_DISTANCE = 100
    
    # 포트홀 알림 설정
    ONE_TIME_ALERT = True  # 각 포트홀에 대해 한 번만 알림을 생성할지 여부
    
    # 위험도 레벨 임계값
    class RiskLevel:
        HIGH = "High"
        MEDIUM = "Medium"
        LOW = "Low"
        
        @classmethod
        def get_risk_level(cls, depth, distance):
            """
            포트홀 깊이와 거리에 따른 위험도 레벨 계산
            
            Args:
                depth (float): 포트홀 깊이(cm)
                distance (float): 차량과 포트홀 간 거리(m)
                
            Returns:
                str: 위험도 레벨
            """
            depth = depth or 0
            distance = distance or 0
            
            if depth > 5 or distance < 10:
                return cls.HIGH
            elif depth > 3 or distance < 30:
                return cls.MEDIUM
            else:
                return cls.LOW
