import streamlit as st
from ui.porthole_ui import render_porthole_tab
from ui.car_ui import render_car_tab
from ui.map_ui import render_map_tab
from ui.alert_ui import render_alert_tab

# 페이지 설정
st.set_page_config(
    page_title="포트홀 감지 대시보드",
    page_icon="🚧",
    layout="wide"
)

# API URL 설정
API_BASE_URL = "http://localhost:8000/api"

# 전역 상태 초기화
if 'last_check_time' not in st.session_state:
    st.session_state.last_check_time = None
if 'auto_refresh' not in st.session_state:
    st.session_state.auto_refresh = True
if 'alert_message_cache' not in st.session_state:
    st.session_state['alert_message_cache'] = {}


# 헤더
st.title("🚧 포트홀 감지 대시보드")
st.markdown("""
이 대시보드는 도로의 포트홀 정보를 관리하고 차량과의 근접도를 모니터링합니다.
""")

# 사이드바 설정
with st.sidebar:
    st.header("설정")
    
    # 자동 새로고침 설정
    auto_refresh_enabled = st.checkbox(
        "자동 새로고침 활성화", 
        value=st.session_state.auto_refresh
    )
    st.session_state.auto_refresh = auto_refresh_enabled
    
    refresh_interval = st.slider("새로고침 간격(초)", 
                                min_value=5, 
                                max_value=60, 
                                value=10,
                                key="sidebar_refresh_interval")
    
    # 수동 새로고침 버튼
    if st.button("데이터 새로고침"):
        st.cache_data.clear()
        st.rerun()
        
# 탭 구성
tab1, tab2, tab3, tab4 = st.tabs(["포트홀 목록", "차량 목록", "지도 보기", "알림"])

# 각 탭 렌더링
with tab1:
    render_porthole_tab(API_BASE_URL)

with tab2:
    render_car_tab(API_BASE_URL)

with tab3:
    render_map_tab(API_BASE_URL)

with tab4:
    render_alert_tab(API_BASE_URL)
