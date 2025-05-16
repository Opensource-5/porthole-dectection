import streamlit as st
from pathlib import Path
import os
import time
import threading

# Import functions and example input from main.py
from main import (
    example_input,
    process_porthole_input,
    generate_alert_message,
    synthesize_alert_audio,
    run_alert_system_background,
    get_latest_alert,
    latest_alerts,
    latest_audio_path
)

def main():
    st.title("포트홀 알림 시스템")
    st.write("센서 데이터를 처리하고 운전자에게 경고 메시지 및 음성 알림을 제공합니다.")

    # 사이드바에 기본 정보 표시
    st.sidebar.header("시스템 정보")
    st.sidebar.write("이 시스템은 포트홀을 감지하고 운전자에게 경고합니다.")
    
    st.sidebar.header("예제 센서 데이터")
    st.sidebar.json(example_input)
    
    # 탭 생성: 실시간 알림과 데모 생성
    tab1, tab2 = st.tabs(["실시간 알림", "데모 알림 생성"])
    
    # 실시간 알림 탭
    with tab1:
        st.header("🔴 실시간 포트홀 알림")
        st.write("서버로부터 수신된 포트홀 알림이 실시간으로 표시됩니다.")

        # 차량 번호 입력 및 상태 관리
        car_id = st.number_input("조회할 차량 번호를 입력하세요", min_value=1, value=st.session_state.get('selected_car_id', 1), key="car_id_input")
        if 'selected_car_id' not in st.session_state or st.session_state.selected_car_id != car_id:
            st.session_state.selected_car_id = car_id
            st.session_state.acknowledged_alerts = set()

        # 알림 시스템 시작/중지 버튼
        alert_system_status = st.session_state.get('alert_system_status', False)
        alert_thread = st.session_state.get('alert_thread', None)

        col1, col2 = st.columns(2)
        with col1:
            if st.button("알림 시스템 시작" if not alert_system_status else "알림 시스템 재시작"):
                st.session_state.alert_system_status = True
                if alert_thread:
                    st.warning("기존 알림 시스템을 재시작합니다.")
                st.session_state.alert_thread = run_alert_system_background(test_mode=True)
                st.success("알림 시스템이 시작되었습니다!")

        with col2:
            if st.button("테스트 알림 생성"):
                # 테스트 알림 생성
                processed_data = process_porthole_input(example_input)
                if processed_data:
                    alert_message = generate_alert_message(processed_data)
                    audio_path = synthesize_alert_audio(alert_message)
                    st.session_state.test_alert = {
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "message": alert_message,
                        "audio_path": audio_path,
                        "car_id": example_input["car_id"],
                        "porthole_info": {
                            "distance": example_input["distance"],
                            "depth": example_input["depth"],
                            "location": "테스트 위치",
                        }
                    }
                    st.success("테스트 알림이 생성되었습니다!")

        # 실시간 알림 표시 영역
        alert_container = st.container()
        with alert_container:
            # 차량별 알림 필터링
            filtered_alerts = [a for a in latest_alerts if a['car_id'] == car_id]
            st.subheader(f"차량 {car_id}에 대한 실시간 알림")
            if filtered_alerts:
                for i, alert in enumerate(filtered_alerts[::-1]):
                    alert_key = f"{alert['timestamp']}_{alert['car_id']}"
                    acknowledged = alert_key in st.session_state.get('acknowledged_alerts', set())
                    with st.expander(f"알림 {i+1} - {alert['timestamp']}"):
                        st.write(f"**알림 메시지**: {alert['message']}")
                        porthole = alert['porthole_info']
                        st.write(f"**포트홀 정보**: 거리 {porthole.get('distance', '?')}m, 깊이 {porthole.get('depth', '?')}cm, 위치: {porthole.get('location', '?')}")
                        if alert.get('audio_path') and Path(alert['audio_path']).exists():
                            with open(alert['audio_path'], "rb") as audio_file:
                                st.audio(audio_file.read(), format="audio/mp3")
                        if not acknowledged:
                            if st.button(f"확인 (Ack) - 알림 {i+1}", key=f"ack_{alert_key}"):
                                st.session_state.acknowledged_alerts.add(alert_key)
                                st.success("알림을 확인했습니다.")
                                st.experimental_rerun()
                        else:
                            st.info("이 알림은 확인(Ack) 처리됨.")
            else:
                st.write("해당 차량에 대한 수신된 알림이 없습니다.")

            # 무한 재실행 방지를 위해 자동 새로고침 부분 수정
            # 타이밍 로직을 추가하여 5초마다 새로고침하도록 변경
            if 'last_refresh' not in st.session_state:
                st.session_state.last_refresh = time.time()
            
            current_time = time.time()
            if current_time - st.session_state.last_refresh > 5:  # 5초마다 새로고침
                st.session_state.last_refresh = current_time
                st.experimental_rerun()
    
    # 데모 알림 생성 탭
    with tab2:
        st.header("🧪 포트홀 알림 데모")
        st.write("센서 데이터를 입력하고 알림을 생성할 수 있습니다.")
        
        # 사용자 입력
        with st.form("alert_form"):
            car_id = st.number_input("차량 ID", min_value=1, value=1)
            distance = st.slider("포트홀까지의 거리 (미터)", min_value=10, max_value=200, value=90, key="distance_slider")
            depth = st.slider("포트홀 깊이 (밀리미터)", min_value=100, max_value=2000, value=1232, key="depth_slider")
            location = st.text_input("위치", "서울특별시 강남구 테헤란로")
            
            # 좌표 입력
            st.write("포트홀 좌표")
            col1, col2 = st.columns(2)
            with col1:
                lat = st.number_input("위도", value=37.5024, format="%.4f")
            with col2:
                lng = st.number_input("경도", value=126.9389, format="%.4f")
            
            submit_button = st.form_submit_button("알림 생성")
            
            if submit_button:
                # 사용자 입력으로 데이터 생성
                user_input = {
                    "car_id": car_id,
                    "distance": distance,
                    "depth": depth,
                    "porthole_location": (lat, lng),
                    "car_location": (lat-0.0004, lng-0.009),  # 약간 떨어진 위치
                    "location_info": location
                }
                
                # 데이터 처리 및 알림 생성
                processed_data = process_porthole_input(user_input)
                if processed_data:
                    st.success("데이터가 성공적으로 처리되었습니다.")
                    
                    # 처리된 데이터 표시
                    with st.expander("처리된 데이터"):
                        st.json(processed_data)
                    
                    # 알림 메시지 생성
                    alert_message = generate_alert_message(processed_data)
                    st.subheader("생성된 알림 메시지")
                    st.write(alert_message)
                    
                    # 음성 합성
                    st.subheader("알림 음성")
                    with st.spinner("음성을 합성하는 중..."):
                        audio_path = synthesize_alert_audio(alert_message)
                        if audio_path and Path(audio_path).exists():
                            with open(audio_path, "rb") as audio_file:
                                st.audio(audio_file.read(), format="audio/mp3")
                        else:
                            st.error("음성 합성에 실패했습니다.")
                else:
                    st.error("데이터 처리에 실패했습니다.")

if __name__ == "__main__":
    main()