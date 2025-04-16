"""
포트홀 알림 대시보드

이 애플리케이션은 지정된 차량 ID에 대한 포트홀 근접 알림을 실시간으로 
조회하고 표시합니다. 사용자는 알림을 확인하여 처리할 수 있습니다.
"""

import streamlit as st
import requests
import time
import pandas as pd
from datetime import datetime

# FastAPI 서버 URL
API_BASE_URL = "http://localhost:8000"

# 페이지 설정
st.set_page_config(
    page_title="포트홀 알림 대시보드",
    page_icon="🚧",
    layout="wide"
)

# 세션 상태 초기화
if "car_id" not in st.session_state:
    st.session_state.car_id = None
if "last_check_time" not in st.session_state:
    st.session_state.last_check_time = None
if "alert_count" not in st.session_state:
    st.session_state.alert_count = 0

def get_car_alerts(car_id):
    """
    특정 차량의 알림을 조회하는 함수
    
    Args:
        car_id: 차량 ID
        
    Returns:
        dict: 알림 정보 또는 None(오류 발생 시)
    """
    try:
        response = requests.get(f"{API_BASE_URL}/api/car_alerts/{car_id}")
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"알림 조회 오류: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        st.error(f"서버 연결 오류: {str(e)}")
        return None

def acknowledge_alert(car_id, alert_ids):
    """
    알림을 확인 처리하는 함수
    
    Args:
        car_id: 차량 ID
        alert_ids: 확인할 알림 ID 목록
        
    Returns:
        bool: 처리 성공 여부
    """
    try:
        response = requests.post(
            f"{API_BASE_URL}/api/car_alerts/{car_id}/acknowledge", 
            json={"alert_ids": alert_ids}
        )
        if response.status_code == 200:
            return True
        else:
            st.error(f"알림 확인 처리 오류: {response.status_code} - {response.text}")
            return False
    except Exception as e:
        st.error(f"서버 연결 오류: {str(e)}")
        return False

# 헤더 및 설명
st.title("🚧 포트홀 알림 대시보드")
st.markdown("""
이 대시보드는 특정 차량에 대한 포트홀 근접 알림을 실시간으로 조회하고 표시합니다.
차량 ID를 입력하면 해당 차량에 대한 알림을 지속적으로 확인합니다.
""")

# 사이드바에 차량 ID 입력 폼
with st.sidebar:
    st.header("차량 정보")
    
    with st.form("car_form"):
        input_car_id = st.number_input("차량 ID", min_value=1, step=1, value=1)
        submit_button = st.form_submit_button("저장")
        
        if submit_button:
            st.session_state.car_id = input_car_id
            st.session_state.last_check_time = None
            st.success(f"차량 ID {input_car_id}에 대한 알림을 조회합니다.")

    if st.session_state.car_id:
        st.info(f"현재 조회 중인 차량 ID: {st.session_state.car_id}")
        
        # 조회 간격 설정
        st.subheader("설정")
        refresh_interval = st.slider("알림 조회 간격(초)", min_value=5, max_value=60, value=10)

# 메인 화면 - 알림 표시
if st.session_state.car_id:
    # 데이터 컨테이너를 루프 밖에서 선언
    alert_container = st.empty()
    
    # 자동 새로고침 영역
    auto_refresh = st.empty()
    
    # 메인 루프
    while True:
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # 새로운 컨테이너로 갱신
        with alert_container.container():
            st.subheader("포트홀 근접 알림")
            
            # 알림 데이터 조회
            alert_data = get_car_alerts(st.session_state.car_id)
            
            if alert_data and "alerts" in alert_data:
                alerts = alert_data["alerts"]
                alert_count = len(alerts)
                
                # 새 알림 확인
                if st.session_state.last_check_time and alert_count > st.session_state.alert_count:
                    st.warning(f"⚠️ {alert_count - st.session_state.alert_count}개의 새로운 알림이 있습니다!")
                
                st.session_state.alert_count = alert_count
                
                if alert_count > 0:
                    st.info(f"총 {alert_count}개의 알림이 있습니다.")
                    
                    # 알림 목록을 DataFrame으로 변환
                    alert_list = []
                    for alert in alerts:
                        alert_list.append({
                            "포트홀 ID": alert["porthole_id"],
                            "위치": alert.get("location", "알 수 없음"),
                            "거리(m)": alert.get("distance", 0),
                            "깊이(cm)": alert.get("depth", "알 수 없음"),
                            "상태": alert.get("status", "알 수 없음"),
                            "시간": alert.get("timestamp", "알 수 없음"),
                            "알림 ID": alert["porthole_id"]  # 확인 처리를 위해 porthole_id를 사용
                        })
                    
                    if alert_list:
                        df = pd.DataFrame(alert_list)
                        
                        # 각 알림에 대한 확인 버튼 추가
                        for idx, row in df.iterrows():
                            col1, col2 = st.columns([4, 1])
                            
                            with col1:
                                alert_info = (
                                    f"**포트홀 ID**: {row['포트홀 ID']} | "
                                    f"**위치**: {row['위치']} | "
                                    f"**거리**: {row['거리(m)']}m | "
                                    f"**깊이**: {row['깊이(cm)']}cm | "
                                    f"**상태**: {row['상태']} | "
                                    f"**시간**: {row['시간']}"
                                )
                                st.markdown(alert_info)
                            
                            with col2:
                                # 고유한 키를 위해 알림 ID, idx, 시간, 상태까지 조합
                                button_key = f"ack_{row['알림 ID']}_{idx}_{row['시간']}_{row['상태']}"
                                if st.button("확인", key=button_key):
                                    # 디버깅 메시지 추가
                                    st.write(f"확인 요청 전송 중... 알림 ID: {row['알림 ID']}")
                                    
                                    # 확인 처리 요청 보내기
                                    success = acknowledge_alert(st.session_state.car_id, [row['알림 ID']])
                                    
                                    if success:
                                        st.success(f"포트홀 ID {row['포트홀 ID']} 알림을 확인했습니다.")
                                        # 세션 상태에 확인된 알림 표시
                                        if "acknowledged_alerts" not in st.session_state:
                                            st.session_state.acknowledged_alerts = []
                                        st.session_state.acknowledged_alerts.append(row['알림 ID'])
                                        # 잠시 대기 후 페이지 새로고침
                                        time.sleep(1)
                                        st.rerun()
                                    else:
                                        st.error(f"알림 확인 처리에 실패했습니다. 다시 시도해 주세요.")
                            
                            st.divider()
                else:
                    st.success("현재 알림이 없습니다.")
            else:
                st.warning("알림 데이터를 가져오지 못했습니다.")
            
            st.session_state.last_check_time = current_time
        
        # 자동 새로고침 카운터 표시 (매번 새로운 컨테이너로 갱신)
        with auto_refresh.container():
            st.caption(f"마지막 업데이트: {current_time} (다음 업데이트까지 {refresh_interval}초)")
            
            # 진행 바 표시
            progress_bar = st.progress(0)
            for i in range(refresh_interval):
                progress_bar.progress((i + 1) / refresh_interval)
                time.sleep(1)
        
        # st.rerun() 호출은 제거 - 불필요한 페이지 리로드를 방지하고 루프가 자연스럽게 반복되도록 함
        # 대신 이전 데이터를 명시적으로 제거하기 위해 empty().container() 사용
else:
    st.info("사이드바에서 차량 ID를 입력하고 저장 버튼을 클릭하세요.")