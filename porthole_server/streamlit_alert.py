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
import os
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from openai import OpenAI
from pathlib import Path

# FastAPI 서버 URL
API_BASE_URL = "http://localhost:8000"

# 환경 변수 로드
load_dotenv(override=True)

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

def generate_alert_message(alert):
    """
    LangChain을 이용해 알림 메시지 생성
    """
    risk_level = alert.get('risk_level', 'Unknown')
    if risk_level == "Low":
        alert_detail = "포트홀의 깊이가 그리 깊지는 않으나 주의가 필요합니다."
    elif risk_level == "Medium":
        alert_detail = "사고를 유발할 수 있는 정도의 포트홀 깊이입니다."
    elif risk_level == "High":
        alert_detail = "위험한 포트홀입니다. 즉각적인 조치가 필요합니다."
    else:
        alert_detail = "상황을 확인할 수 없습니다."
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant that generates clear, concise, and actionable alert messages based on sensor data."),
        ("human", """
        상황: 차량 인근에 포트홀이 발견되었습니다.
        차량과 포트홀 거리: {distance}m, 포트홀 깊이: {depth}cm.
        위험도: {risk_level}, {alert_detail}
        포트홀 위치: {location}
        위치와 상황을 종합(위험도 포함)하여 운전자에게 경고 알림 메시지를 운전자가 이해하기 쉽게 작성해 주세요.
        """)
    ])
    model = ChatOpenAI(model="gpt-4o-mini", temperature=0.9)
    chain = prompt | model | StrOutputParser()
    location_info = alert.get("location", "알 수 없음")
    return chain.invoke({
        "car_id": alert.get("car_id", 0),
        "distance": alert.get("distance", 0),
        "depth": alert.get("depth", 0),
        "risk_level": risk_level,
        "location": location_info,
        "alert_detail": alert_detail
    })

def synthesize_alert_audio(alert_message, filename_hint="speech"):
    """
    OpenAI TTS로 알림 메시지 음성 파일 생성, 파일 경로 반환
    """
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        st.warning("OpenAI API 키가 환경 변수에 없습니다. 음성 합성 불가.")
        return None
    client = OpenAI(api_key=openai_api_key)
    speech_file_path = Path(__file__).parent / f"{filename_hint}.mp3"
    try:
        with client.audio.speech.with_streaming_response.create(
            model="gpt-4o-mini-tts",
            voice="coral",
            input=alert_message,
            instructions="Speak in a warning tone."
        ) as response:
            response.stream_to_file(speech_file_path)
        return str(speech_file_path)
    except Exception as e:
        st.warning(f"TTS 합성 오류: {e}")
        return None

def get_car_alerts(car_id):
    """
    특정 차량의 알림을 조회하고, 각 알림에 대해 LangChain 메시지 및 TTS 생성
    Returns: list of dicts with alert info, message, audio_path
    """
    try:
        response = requests.get(f"{API_BASE_URL}/api/car_alerts/{car_id}")
        if response.status_code == 200:
            alert_data = response.json()
            alerts = alert_data.get("alerts", [])
            processed_alerts = []
            for alert in alerts:
                # 위험도 산정
                depth = alert.get("depth", 0)
                if depth < 500:
                    risk = "Low"
                elif depth < 1500:
                    risk = "Medium"
                else:
                    risk = "High"
                alert["risk_level"] = risk
                # LangChain 메시지 생성
                alert_message = generate_alert_message(alert)
                # TTS 음성 파일 생성 (포트홀 ID로 파일명 구분)
                audio_path = synthesize_alert_audio(alert_message, filename_hint=f"speech_{alert.get('porthole_id', 'x')}")
                processed_alerts.append({
                    **alert,
                    "alert_message": alert_message,
                    "audio_path": audio_path
                })
            return processed_alerts
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
    alert_container = st.empty()
    auto_refresh = st.empty()
    while True:
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with alert_container.container():
            st.subheader("포트홀 근접 알림")
            alert_list = get_car_alerts(st.session_state.car_id)
            if alert_list is not None:
                alert_count = len(alert_list)
                if st.session_state.last_check_time and alert_count > st.session_state.alert_count:
                    st.warning(f"⚠️ {alert_count - st.session_state.alert_count}개의 새로운 알림이 있습니다!")
                st.session_state.alert_count = alert_count
                if alert_count > 0:
                    st.info(f"총 {alert_count}개의 알림이 있습니다.")
                    for alert in alert_list:
                        col1, col2 = st.columns([4, 1])
                        with col1:
                            alert_info = (
                                f"**포트홀 ID**: {alert['porthole_id']} | "
                                f"**위치**: {alert.get('location', '알 수 없음')} | "
                                f"**거리**: {alert.get('distance', 0)}m | "
                                f"**깊이**: {alert.get('depth', '알 수 없음')}cm | "
                                f"**상태**: {alert.get('status', '알 수 없음')} | "
                                f"**시간**: {alert.get('timestamp', '알 수 없음')}"
                            )
                            st.markdown(alert_info)
                            st.markdown(f"**경고 메시지:** {alert['alert_message']}")
                            if alert.get("audio_path"):
                                st.audio(alert["audio_path"], format="audio/mp3")
                        with col2:
                            button_key = f"ack_{alert['porthole_id']}_{alert.get('timestamp', '')}_{alert.get('status', '')}"
                            if st.button("확인", key=button_key):
                                st.write(f"확인 요청 전송 중... 알림 ID: {alert['porthole_id']}")
                                success = acknowledge_alert(st.session_state.car_id, [alert['porthole_id']])
                                if success:
                                    st.success(f"포트홀 ID {alert['porthole_id']} 알림을 확인했습니다.")
                                    if "acknowledged_alerts" not in st.session_state:
                                        st.session_state.acknowledged_alerts = []
                                    st.session_state.acknowledged_alerts.append(alert['porthole_id'])
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
        with auto_refresh.container():
            st.caption(f"마지막 업데이트: {current_time} (다음 업데이트까지 {refresh_interval}초)")
            progress_bar = st.progress(0)
            for i in range(refresh_interval):
                progress_bar.progress((i + 1) / refresh_interval)
                time.sleep(1)
else:
    st.info("사이드바에서 차량 ID를 입력하고 저장 버튼을 클릭하세요.")