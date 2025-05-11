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
import uuid
import tempfile

# FastAPI 서버 URL
API_BASE_URL = "http://localhost:8000"

print("[DEBUG] 애플리케이션 시작, API 주소:", API_BASE_URL)

# 환경 변수 로드
load_dotenv(override=True)
print("[DEBUG] 환경 변수 로드 완료")

# 페이지 설정
st.set_page_config(
    page_title="포트홀 알림 대시보드",
    page_icon="🚧",
    layout="wide"
)
print("[DEBUG] Streamlit 페이지 설정 완료")

# 세션 상태 초기화
if "car_id" not in st.session_state:
    st.session_state.car_id = None
    print("[DEBUG] 세션 상태 car_id 초기화: None")
if "last_check_time" not in st.session_state:
    st.session_state.last_check_time = None
    print("[DEBUG] 세션 상태 last_check_time 초기화: None")
if "alert_count" not in st.session_state:
    st.session_state.alert_count = 0
    print("[DEBUG] 세션 상태 alert_count 초기화: 0")

# 포트홀 ID별 메시지/TTS 캐시
alert_message_cache = {}
alert_audio_cache = {}
print("[DEBUG] 메시지/TTS 캐시 초기화 완료")

def generate_alert_message(alert):
    """
    LangChain을 이용해 알림 메시지 생성
    """
    print(f"[DEBUG] generate_alert_message 시작: {alert}")
    risk_level = alert.get('risk_level', 'Unknown')
    if risk_level == "Low":
        alert_detail = "포트홀의 깊이가 그리 깊지는 않으나 주의가 필요합니다."
    elif risk_level == "Medium":
        alert_detail = "사고를 유발할 수 있는 정도의 포트홀 깊이입니다."
    elif risk_level == "High":
        alert_detail = "위험한 포트홀입니다. 즉각적인 조치가 필요합니다."
    else:
        alert_detail = "상황을 확인할 수 없습니다."
    print(f"[DEBUG] 위험도 {risk_level}에 따른 상세 설명: {alert_detail}")
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
    print("[DEBUG] ChatPromptTemplate 생성 완료")
    
    try:
        print(f"[DEBUG] ChatOpenAI 모델 로드: gpt-4o-mini")
        model = ChatOpenAI(model="gpt-4o-mini", temperature=0.9)
        print("[DEBUG] 모델 로드 완료, 체인 구성 시작")
        chain = prompt | model | StrOutputParser()
        print("[DEBUG] 체인 구성 완료")
        
        location_info = alert.get("location", "알 수 없음")
        input_data = {
            "car_id": alert.get("car_id", 0),
            "distance": alert.get("distance", 0),
            "depth": alert.get("depth", 0),
            "risk_level": risk_level,
            "location": location_info,
            "alert_detail": alert_detail
        }
        print(f"[DEBUG] LangChain 입력 데이터: {input_data}")
        
        message = chain.invoke(input_data)
        print(f"[DEBUG] 생성된 알림 메시지: {message[:50]}...")
        return message
    except Exception as e:
        print(f"[ERROR] 메시지 생성 중 오류 발생: {e}")
        return f"포트홀 경고: 위치 {location_info}에 포트홀이 발견되었습니다. 주의하세요."

def synthesize_alert_audio(alert_message, filename_hint="speech"):
    """
    OpenAI TTS로 알림 메시지 음성 파일 생성, 파일 경로 반환 (임시파일 사용)
    """
    print(f"[DEBUG] synthesize_alert_audio 시작: {filename_hint}")
    print(f"[DEBUG] 메시지 길이: {len(alert_message)} 문자")
    
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        print("[ERROR] OpenAI API 키가 환경 변수에 없습니다.")
        st.warning("OpenAI API 키가 환경 변수에 없습니다. 음성 합성 불가.")
        return None
    
    print("[DEBUG] OpenAI 클라이언트 초기화")
    client = OpenAI(api_key=openai_api_key)
    
    print(f"[DEBUG] 임시 파일 생성: suffix=.mp3")
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmpfile:
        speech_file_path = tmpfile.name
        print(f"[DEBUG] 임시 파일 경로: {speech_file_path}")
        
    try:
        print(f"[DEBUG] TTS API 요청 시작: 모델=gpt-4o-mini-tts, 음성=coral")
        with client.audio.speech.with_streaming_response.create(
            model="gpt-4o-mini-tts",
            voice="coral",
            input=alert_message,
            instructions="Speak in a warning tone."
        ) as response:
            print(f"[DEBUG] TTS 응답 수신, 파일에 저장 중...")
            response.stream_to_file(speech_file_path)
            
        print(f"[DEBUG] TTS 파일 생성 완료: {speech_file_path}")
        
        # 파일 크기 확인
        file_size = os.path.getsize(speech_file_path)
        print(f"[DEBUG] 생성된 파일 크기: {file_size} bytes")
        
        return speech_file_path
    except Exception as e:
        print(f"[ERROR] TTS 합성 오류: {e}")
        st.warning(f"TTS 합성 오류: {e}")
        return None

def get_car_alerts(car_id):
    """
    특정 차량의 알림을 조회하고, 각 알림에 대해 LangChain 메시지 및 TTS 생성
    같은 포트홀 ID에 대해서는 한 번만 메시지/TTS를 생성 (캐시 활용)
    Returns: list of dicts with alert info, message, audio_path
    """
    print(f"[DEBUG] get_car_alerts 시작: car_id={car_id}")
    
    try:
        api_url = f"{API_BASE_URL}/api/car_alerts/{car_id}"
        print(f"[DEBUG] API 요청 URL: {api_url}")
        
        response = requests.get(api_url)
        print(f"[DEBUG] API 응답 상태 코드: {response.status_code}")
        
        if response.status_code == 200:
            alert_data = response.json()
            print(f"[DEBUG] 응답 데이터: {alert_data.keys()}")
            
            alerts = alert_data.get("alerts", [])
            print(f"[DEBUG] 알림 수: {len(alerts)}")
            
            processed_alerts = []
            processed_porthole_ids = set()  # 이미 처리한 포트홀 ID 집합
            
            for i, alert in enumerate(alerts):
                print(f"[DEBUG] 알림 {i+1}/{len(alerts)} 처리 중")
                porthole_id = alert.get("porthole_id")
                print(f"[DEBUG] 포트홀 ID: {porthole_id}")
                
                if porthole_id in processed_porthole_ids:
                    print(f"[DEBUG] 포트홀 ID {porthole_id}는 이미 처리됨. 건너뜀")
                    continue
                    
                processed_porthole_ids.add(porthole_id)
                print(f"[DEBUG] 처리할 포트홀 ID: {porthole_id}")
                
                # 위험도 산정
                depth = alert.get("depth", 0)
                if depth < 500:
                    risk = "Low"
                elif depth < 1500:
                    risk = "Medium"
                else:
                    risk = "High"
                alert["risk_level"] = risk
                print(f"[DEBUG] 깊이 {depth}에 따른 위험도: {risk}")
                
                # 메시지 캐시 확인
                if porthole_id in alert_message_cache:
                    print(f"[DEBUG] 포트홀 ID {porthole_id}의 메시지가 캐시에 존재함")
                    alert_message = alert_message_cache[porthole_id]
                else:
                    print(f"[DEBUG] 포트홀 ID {porthole_id}의 메시지를 새로 생성")
                    alert_message = generate_alert_message(alert)
                    alert_message_cache[porthole_id] = alert_message
                    print(f"[DEBUG] 메시지 캐시에 저장 완료")
                
                # TTS 캐시 확인
                if porthole_id in alert_audio_cache:
                    print(f"[DEBUG] 포트홀 ID {porthole_id}의 오디오가 캐시에 존재함")
                    audio_path = alert_audio_cache[porthole_id]
                    if not os.path.exists(audio_path):
                        print(f"[DEBUG] 캐시된 오디오 파일 {audio_path}가 존재하지 않음. 다시 생성")
                        audio_path = synthesize_alert_audio(alert_message, filename_hint=f"speech_{porthole_id}")
                        alert_audio_cache[porthole_id] = audio_path
                else:
                    print(f"[DEBUG] 포트홀 ID {porthole_id}의 오디오를 새로 생성")
                    audio_path = synthesize_alert_audio(alert_message, filename_hint=f"speech_{porthole_id}")
                    alert_audio_cache[porthole_id] = audio_path
                    print(f"[DEBUG] 오디오 캐시에 저장 완료: {audio_path}")
                    
                processed_alert = {
                    **alert,
                    "alert_message": alert_message,
                    "audio_path": audio_path
                }
                processed_alerts.append(processed_alert)
                print(f"[DEBUG] 알림 {i+1} 처리 완료")
            
            print(f"[DEBUG] 전체 {len(processed_alerts)}개의 알림 처리 완료")
            return processed_alerts
        else:
            print(f"[ERROR] 알림 조회 API 오류: {response.status_code} - {response.text}")
            st.error(f"알림 조회 오류: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        print(f"[ERROR] 서버 연결 오류: {str(e)}")
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
    print(f"[DEBUG] acknowledge_alert 시작: car_id={car_id}, alert_ids={alert_ids}")
    
    try:
        api_url = f"{API_BASE_URL}/api/car_alerts/{car_id}/acknowledge"
        request_data = {"alert_ids": alert_ids}
        print(f"[DEBUG] API 요청 URL: {api_url}")
        print(f"[DEBUG] 요청 데이터: {request_data}")
        
        response = requests.post(api_url, json=request_data)
        print(f"[DEBUG] API 응답 상태 코드: {response.status_code}")
        
        if response.status_code == 200:
            print("[DEBUG] 알림 확인 처리 성공")
            return True
        else:
            print(f"[ERROR] 알림 확인 처리 API 오류: {response.status_code} - {response.text}")
            st.error(f"알림 확인 처리 오류: {response.status_code} - {response.text}")
            return False
    except Exception as e:
        print(f"[ERROR] 알림 확인 처리 요청 중 오류: {str(e)}")
        st.error(f"서버 연결 오류: {str(e)}")
        return False

# 헤더 및 설명
print("[DEBUG] 웹 인터페이스 렌더링 시작")
st.title("🚧 포트홀 알림 대시보드")
st.markdown("""
이 대시보드는 특정 차량에 대한 포트홀 근접 알림을 실시간으로 조회하고 표시합니다.
차량 ID를 입력하면 해당 차량에 대한 알림을 지속적으로 확인합니다.
""")
print("[DEBUG] 헤더 및 설명 렌더링 완료")

# 사이드바에 차량 ID 입력 폼
print("[DEBUG] 사이드바 렌더링 시작")
with st.sidebar:
    print("[DEBUG] 사이드바: 차량 정보 섹션")
    st.header("차량 정보")
    
    with st.form("car_form"):
        print("[DEBUG] 차량 ID 입력 폼")
        input_car_id = st.number_input("차량 ID", min_value=1, step=1, value=1)
        submit_button = st.form_submit_button("저장")
        
        if submit_button:
            print(f"[DEBUG] 차량 ID 저장 버튼 클릭: {input_car_id}")
            st.session_state.car_id = input_car_id
            st.session_state.last_check_time = None
            print(f"[DEBUG] 세션 상태 업데이트: car_id={st.session_state.car_id}, last_check_time=None")
            st.success(f"차량 ID {input_car_id}에 대한 알림을 조회합니다.")

    if st.session_state.car_id:
        print(f"[DEBUG] 사이드바: 현재 조회 중인 차량 ID {st.session_state.car_id}")
        st.info(f"현재 조회 중인 차량 ID: {st.session_state.car_id}")
        
        # 조회 간격 설정
        print("[DEBUG] 사이드바: 조회 간격 설정")
        st.subheader("설정")
        refresh_interval = st.slider("알림 조회 간격(초)", min_value=5, max_value=60, value=10)
        print(f"[DEBUG] 알림 조회 간격 설정: {refresh_interval}초")

print("[DEBUG] 사이드바 렌더링 완료")

# 메인 화면 - 알림 표시
print("[DEBUG] 메인 화면 렌더링 시작")
if st.session_state.car_id:
    print(f"[DEBUG] 차량 ID {st.session_state.car_id}에 대한 알림 표시")
    alert_container = st.empty()
    auto_refresh = st.empty()
    print("[DEBUG] 무한 루프 시작: 알림 자동 갱신")
    while True:
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[DEBUG] 현재 시간: {current_time}")
        
        with alert_container.container():
            print(f"[DEBUG] 알림 컨테이너 업데이트")
            st.subheader("포트홀 근접 알림")
            
            print(f"[DEBUG] 차량 ID {st.session_state.car_id}의 알림 목록 조회")
            alert_list = get_car_alerts(st.session_state.car_id)
            
            if alert_list is not None:
                alert_count = len(alert_list)
                print(f"[DEBUG] 알림 수: {alert_count}")
                
                if st.session_state.last_check_time and alert_count > st.session_state.alert_count:
                    print(f"[DEBUG] 새 알림 발견: {alert_count - st.session_state.alert_count}개")
                    st.warning(f"⚠️ {alert_count - st.session_state.alert_count}개의 새로운 알림이 있습니다!")
                
                st.session_state.alert_count = alert_count
                print(f"[DEBUG] 세션 상태 업데이트: alert_count={alert_count}")
                
                if alert_count > 0:
                    st.info(f"총 {alert_count}개의 알림이 있습니다.")
                    print(f"[DEBUG] {alert_count}개 알림 표시 시작")
                    
                    for i, alert in enumerate(alert_list):
                        print(f"[DEBUG] 알림 {i+1}/{alert_count} 표시")
                        col1, col2 = st.columns([4, 1])
                        
                        with col1:
                            print(f"[DEBUG] 알림 {i+1} 정보 표시")
                            alert_info = (
                                f"**포트홀 ID**: {alert['porthole_id']} | "
                                f"**위치**: {alert.get('location', '알 수 없음')} | "
                                f"**거리**: {alert.get('distance', 0)}m | "
                                f"**깊이**: {alert.get('depth', '알 수 없음')}cm | "
                                f"**상태**: {alert.get('status', '알 수 없음')} | "
                                f"**시간**: {alert.get('timestamp', '알 수 없음')}"
                            )
                            st.markdown(alert_info)
                            print(f"[DEBUG] 알림 {i+1} 메시지 표시")
                            st.markdown(f"**경고 메시지:** {alert['alert_message']}")
                            
                            if alert.get("audio_path") and os.path.exists(alert["audio_path"]) and os.path.getsize(alert["audio_path"]) > 0:
                                print(f"[DEBUG] 알림 {i+1} 오디오 파일 로드: {alert['audio_path']}")
                                file_size = os.path.getsize(alert["audio_path"])
                                print(f"[DEBUG] 오디오 파일 크기: {file_size} bytes")
                                try:
                                    with open(alert["audio_path"], "rb") as audio_file:
                                        audio_data = audio_file.read()
                                        print(f"[DEBUG] 오디오 데이터 로드됨: {len(audio_data)} bytes")
                                        st.audio(audio_data, format="audio/mp3")
                                except Exception as e:
                                    print(f"[ERROR] 오디오 파일 로드 오류: {e}")
                                    st.warning(f"오디오 파일 로드 중 오류 발생: {e}")
                            else:
                                audio_path = alert.get("audio_path", "None")
                                print(f"[DEBUG] 오디오 파일 문제: path={audio_path}, exists={os.path.exists(audio_path) if audio_path else False}")
                                st.warning("TTS 음성 파일이 생성되지 않았거나 접근할 수 없습니다.")
                        
                        with col2:
                            print(f"[DEBUG] 알림 {i+1} '확인' 버튼 표시")
                            button_key = f"ack_{alert['porthole_id']}_{alert.get('timestamp', '')}_{alert.get('status', '')}_{i}_{uuid.uuid4()}"
                            print(f"[DEBUG] 버튼 키: {button_key}")
                            
                            if st.button("확인", key=button_key):
                                print(f"[DEBUG] 알림 {i+1} '확인' 버튼 클릭")
                                st.write(f"확인 요청 전송 중... 알림 ID: {alert['porthole_id']}")
                                
                                print(f"[DEBUG] 알림 확인 처리 요청: car_id={st.session_state.car_id}, alert_id={alert['porthole_id']}")
                                success = acknowledge_alert(st.session_state.car_id, [alert['porthole_id']])
                                
                                if success:
                                    print(f"[DEBUG] 알림 확인 처리 성공: {alert['porthole_id']}")
                                    st.success(f"포트홀 ID {alert['porthole_id']} 알림을 확인했습니다.")
                                    
                                    if "acknowledged_alerts" not in st.session_state:
                                        print("[DEBUG] acknowledged_alerts 세션 초기화")
                                        st.session_state.acknowledged_alerts = []
                                        
                                    st.session_state.acknowledged_alerts.append(alert['porthole_id'])
                                    print(f"[DEBUG] 확인된 알림 목록에 추가: {alert['porthole_id']}")
                                    print(f"[DEBUG] 확인된 알림 목록: {st.session_state.acknowledged_alerts}")
                                    
                                    time.sleep(1)
                                    print("[DEBUG] 페이지 재로드 (rerun)")
                                    st.rerun()
                                else:
                                    print(f"[ERROR] 알림 확인 처리 실패: {alert['porthole_id']}")
                                    st.error(f"알림 확인 처리에 실패했습니다. 다시 시도해 주세요.")
                        
                        print(f"[DEBUG] 알림 {i+1} 구분선 추가")
                        st.divider()
                        
                    print("[DEBUG] 모든 알림 표시 완료")
                else:
                    print("[DEBUG] 알림 없음 표시")
                    st.success("현재 알림이 없습니다.")
            else:
                print("[ERROR] 알림 데이터 조회 실패")
                st.warning("알림 데이터를 가져오지 못했습니다.")
                
            st.session_state.last_check_time = current_time
            print(f"[DEBUG] 마지막 조회 시간 업데이트: {current_time}")
            
        with auto_refresh.container():
            print(f"[DEBUG] 자동 갱신 표시: 마지막 업데이트={current_time}, 간격={refresh_interval}초")
            st.caption(f"마지막 업데이트: {current_time} (다음 업데이트까지 {refresh_interval}초)")
            progress_bar = st.progress(0)
            
            for i in range(refresh_interval):
                progress = (i + 1) / refresh_interval
                print(f"[DEBUG] 다음 갱신까지 진행률: {progress:.2f} ({i+1}/{refresh_interval}초)")
                progress_bar.progress(progress)
                time.sleep(1)
                
            print("[DEBUG] 대기 시간 완료, 다음 갱신 시작")
else:
    print("[DEBUG] 차량 ID가 없음, 안내 메시지 표시")
    st.info("사이드바에서 차량 ID를 입력하고 저장 버튼을 클릭하세요.")