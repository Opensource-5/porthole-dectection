import streamlit as st
import requests
from datetime import datetime
import time
import tempfile
import os
import json
from dotenv import load_dotenv
from openai import OpenAI

# 환경 변수 로드 (OpenAI API 키 등)
load_dotenv(override=True)

# 알림 메시지/TTS 캐시
if "alert_message_cache" not in st.session_state:
    st.session_state.alert_message_cache = {}
if "alert_audio_cache" not in st.session_state:
    st.session_state.alert_audio_cache = {}

@st.cache_data(ttl=30)  # 30초 캐싱
def fetch_car_alerts(api_url: str, car_id: int, include_acknowledged: bool = False):
    """
    특정 차량의 포트홀 알림을 가져오는 함수
    """
    try:
        response = requests.get(f"{api_url}/alerts/car/{car_id}", 
                               params={"include_acknowledged": str(include_acknowledged).lower()})
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"알림 데이터를 가져오는데 실패했습니다. 상태코드: {response.status_code}")
            return {"car_id": car_id, "alerts": [], "count": 0}
    except Exception as e:
        st.error(f"알림 데이터 요청 중 오류: {str(e)}")
        return {"car_id": car_id, "alerts": [], "count": 0}
        
def generate_alert_message(alert):
    """
    알림 메시지 생성 함수 (OpenAI API 사용)
    """
    # 캐시에서 메시지 확인
    alert_id = alert.get("id")
    if alert_id in st.session_state.alert_message_cache:
        return st.session_state.alert_message_cache[alert_id]
    
    # 위험도 정보 구성
    risk_level = alert.get('risk_level', 'Unknown')
    if risk_level == "Low":
        alert_detail = "포트홀의 깊이가 그리 깊지는 않으나 주의가 필요합니다."
    elif risk_level == "Medium":
        alert_detail = "사고를 유발할 수 있는 정도의 포트홀 깊이입니다."
    elif risk_level == "High":
        alert_detail = "위험한 포트홀입니다. 즉각적인 조치가 필요합니다."
    else:
        alert_detail = "상황을 확인할 수 없습니다."
    
    # OpenAI API를 사용하여 메시지 생성
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        message = f"포트홀 경고: 위치 {alert.get('location', '알 수 없음')}에 포트홀이 발견되었습니다. 주의하세요."
    else:
        try:
            client = OpenAI(api_key=openai_api_key)
            
            prompt = f"""
            상황: 차량 인근에 포트홀이 발견되었습니다.
            차량과 포트홀 거리: {alert.get('distance', 0)}m, 포트홀 깊이: {alert.get('depth', 0)}mm.
            위험도: {risk_level}, {alert_detail}
            포트홀 위치: {alert.get('location', '알 수 없음')}
            
            위치와 상황을 종합(위험도 포함)하여 운전자에게 경고 알림 메시지를 운전자가 이해하기 쉽게 작성해 주세요.
            """
            
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that generates clear, concise, and actionable alert messages based on sensor data."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=150
            )
            
            message = response.choices[0].message.content.strip()
        except Exception as e:
            st.warning(f"메시지 생성 중 오류 발생: {e}")
            message = f"포트홀 경고: 위치 {alert.get('location', '알 수 없음')}에 포트홀이 발견되었습니다. 주의하세요."
    
    # 캐시에 저장
    st.session_state.alert_message_cache[alert_id] = message
    return message

import os
import tempfile
from openai import OpenAI

import os
import tempfile
from openai import OpenAI

def synthesize_alert_audio(alert_message, filename_hint="speech"):
    """
    OpenAI TTS로 알림 메시지 음성 파일 생성 (Echo voice + Custom instructions 적용)
    """
    # OpenAI API 키 확인
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        print("OpenAI API 키가 환경 변수에 없습니다. 음성 합성 불가.")
        return None

    # 음성 스타일 설명
    instructions = """
    The following message is a warning to alert the driver about the risk of a pothole (road damage) ahead.
    - Please deliver the message clearly and articulately, with a calm yet alert tone.
    - Emphasize key information such as location, risk level, and the need for action so the driver can quickly understand the situation.
    - Encourage safe driving without causing unnecessary anxiety.
    - Speak at a natural pace, neither too fast nor too slow.
    - At the end of the message, add: "For your safety, please reduce your speed."
    """

    # 임시 파일 생성
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmpfile:
        speech_file_path = tmpfile.name

    try:
        # OpenAI 클라이언트 생성
        client = OpenAI(api_key=openai_api_key)

        # 음성 생성 요청
        response = client.audio.speech.create(
            model="gpt-4o-mini-tts",
            voice="echo",
            instructions=instructions,
            input=alert_message,
            response_format="mp3"
        )

        # 결과 저장
        with open(speech_file_path, "wb") as out_file:
            out_file.write(response.content)

        return speech_file_path
    except Exception as e:
        print(f"TTS 합성 오류: {e}")
        return None



def acknowledge_alerts(api_url: str, car_id: int, alert_ids: list):
    """
    알림 확인 처리 함수
    """
    if not alert_ids:
        return True
        
    try:
        response = requests.post(
            f"{api_url}/alerts/car/{car_id}/acknowledge",
            json={"alert_ids": alert_ids}
        )
        
        if response.status_code == 200:
            return True
        else:
            st.error(f"알림 확인 처리 실패: {response.text}")
            return False
    except Exception as e:
        st.error(f"알림 확인 요청 중 오류: {str(e)}")
        return False

def render_alert_tab(api_url: str):
    """알림 탭 UI를 렌더링하는 함수"""
    st.header("🚨 포트홀 알림 대시보드")
    
    # 차량 ID 입력 폼
    st.subheader("차량 선택")
    car_id = st.number_input("차량 ID", min_value=1, step=1, value=1)
    include_acknowledged = st.checkbox("확인된 알림도 표시", value=False)
    
    if st.button("알림 조회", key="fetch_alerts"):
        st.session_state.selected_car_id = car_id
        st.session_state.include_acknowledged = include_acknowledged
        st.cache_data.clear()  # 캐시 갱신
    
    # 자동 새로고침 설정
    auto_refresh = st.checkbox("자동 새로고침", value=True)
    refresh_interval = st.slider("새로고침 간격(초)", min_value=5, max_value=60, value=10, key="alert_tab_refresh_interval")
    
    # 선택된 차량에 대한 알림 표시
    if 'selected_car_id' in st.session_state:
        # 타이틀 표시
        st.subheader(f"차량 ID: {st.session_state.selected_car_id}의 알림")
        
        # 알림 조회
        alert_data = fetch_car_alerts(
            api_url, 
            st.session_state.selected_car_id, 
            st.session_state.get('include_acknowledged', include_acknowledged)
        )
        
        alerts = alert_data.get("alerts", [])
        alert_count = len(alerts)
        
        if alert_count > 0:
            st.success(f"{alert_count}개의 알림이 있습니다.")
            
            # 알림 리스트 표시
            for i, alert in enumerate(alerts):
                alert_id = alert.get("id")
                porthole_id = alert.get("porthole_id")
                distance = alert.get("distance", 0)
                depth = alert.get("depth", 0)
                location = alert.get("location", "알 수 없음")
                risk_level = alert.get("risk_level", "Unknown")
                acknowledged = alert.get("acknowledged", False)
                created_at = alert.get("created_at", "")
                
                # 위험도에 따른 색상 결정
                if risk_level == "High":
                    card_color = "#FFCCCC"  # 연한 빨강
                    text_color = "#990000"  # 진한 빨강
                    emoji = "🚨"
                elif risk_level == "Medium":
                    card_color = "#FFEEBB"  # 연한 노랑
                    text_color = "#996600"  # 갈색
                    emoji = "⚠️"
                else:
                    card_color = "#CCFFCC"  # 연한 녹색
                    text_color = "#006600"  # 진한 녹색
                    emoji = "ℹ️"
                
                # 알림 메시지 생성
                alert_message = generate_alert_message(alert)
                
                # 알림 카드 표시
                with st.container():
                    st.markdown(f"""
                    <div style="
                        background-color: {card_color};
                        padding: 10px;
                        border-radius: 10px;
                        margin-bottom: 10px;
                        color: black;
                    ">
                        <h3 style="color: {text_color};">{emoji} {risk_level} 위험도 알림</h3>
                        <p><b>메시지:</b> {alert_message}</p>
                        <p><b>위치:</b> {location}</p>
                        <p><b>포트홀 ID:</b> {porthole_id} | <b>거리:</b> {distance:.1f}m | <b>깊이:</b> {depth}cm</p>
                        <p><b>생성시간:</b> {created_at} | <b>상태:</b> {"확인됨" if acknowledged else "미확인"}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # 알림 확인 버튼 (미확인 알림만)
                    if not acknowledged:
                        if st.button(f"알림 확인", key=f"confirm_alert_{alert_id}"):
                            if acknowledge_alerts(api_url, st.session_state.selected_car_id, [alert_id]):
                                st.success("알림이 확인 처리되었습니다.")
                                st.cache_data.clear()  # 캐시 갱신
                                st.rerun()
                                
                    # 음성 재생 버튼
                    if 'OPENAI_API_KEY' in os.environ:
                        # 이미 캐시에 있는지 확인
                        audio_path = st.session_state.alert_audio_cache.get(alert_id)
                        
                        # 없으면 새로 생성
                        if not audio_path:
                            audio_path = synthesize_alert_audio(alert_message, f"alert_{alert_id}")
                            if audio_path:
                                st.session_state.alert_audio_cache[alert_id] = audio_path
                        
                        if audio_path and os.path.exists(audio_path):
                            with open(audio_path, "rb") as f:
                                st.audio(f.read(), format="audio/mpeg", start_time=0)
                    
                # 구분선
                st.markdown("---")
            
            # 모든 미확인 알림 한번에 확인하기
            unconfirmed_alerts = [a["id"] for a in alerts if not a.get("acknowledged", False)]
            if unconfirmed_alerts:
                if st.button("모든 알림 확인", key="confirm_all_alerts"):
                    if acknowledge_alerts(api_url, st.session_state.selected_car_id, unconfirmed_alerts):
                        st.success(f"{len(unconfirmed_alerts)}개의 알림이 모두 확인 처리되었습니다.")
                        st.cache_data.clear()  # 캐시 갱신
                        st.rerun()
        else:
            st.info("현재 알림이 없습니다.")
        
        # 자동 새로고침
        if auto_refresh:
            time_placeholder = st.empty()
            time_placeholder.text(f"{refresh_interval}초 후 자동 새로고침...")
            time.sleep(refresh_interval)
            st.cache_data.clear()  # 캐시 갱신
            st.rerun()
    else:
        st.info("차량 ID를 입력하고 알림 조회 버튼을 클릭하세요.")
