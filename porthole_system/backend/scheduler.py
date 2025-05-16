from apscheduler.schedulers.background import BackgroundScheduler
from backend.logic import check_car_porthole_proximity
from backend.constants import AlertSettings

# 스케줄러 인스턴스 생성
scheduler = BackgroundScheduler()

def monitor_proximity():
    """
    차량과 포트홀 간의 거리를 모니터링하고 필요한 경우 알림을 생성하는 스케줄링 작업
    """
    print("[Scheduler] 차량-포트홀 거리 모니터링 중...")
    print(f"[Scheduler] 알림 모드: {'한 번만 알림' if AlertSettings.ONE_TIME_ALERT else '지속적 알림'}")
    
    alerts = check_car_porthole_proximity()
    
    if alerts:
        print(f"[Scheduler] {len(alerts)}개의 새로운 알림이 생성되었습니다.")
        for alert in alerts:
            print(f"[Scheduler] 차량 ID {alert['car_id']}가 포트홀 ID {alert['porthole_id']}에서 {alert['distance']:.1f}m 거리에 있습니다.")
    else:
        print("[Scheduler] 새로운 알림이 없습니다.")
