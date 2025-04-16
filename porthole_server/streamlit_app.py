import streamlit as st
import requests
import pandas as pd
import folium
from streamlit_folium import st_folium  # folium_static 대신 st_folium을 사용
import time
import threading

# Streamlit 앱 제목
st.title("Pothole Detection Dashboard")

# API URL 설정
API_BASE_URL = "http://localhost:8000/api"

# 세션 상태 초기화 (알림 관련)
if 'last_check_time' not in st.session_state:
    st.session_state.last_check_time = time.time()
if 'new_portholes' not in st.session_state:
    st.session_state.new_portholes = []
if 'show_notification' not in st.session_state:
    st.session_state.show_notification = False
if 'notification_message' not in st.session_state:
    st.session_state.notification_message = ""
if 'notification_details' not in st.session_state:
    st.session_state.notification_details = {}
if 'auto_refresh' not in st.session_state:
    st.session_state.auto_refresh = True

# 포트홀 데이터 가져오기
@st.cache_data(ttl=60)  # 1분 캐싱
def fetch_portholes():
    response = requests.get(f"{API_BASE_URL}/portholes")
    if response.status_code == 200:
        return response.json()
    else:
        st.error("포트홀 데이터를 가져오는데 실패했습니다.")
        return []

# 최근 감지된 포트홀 데이터 가져오기 (새 기능)
def fetch_new_portholes():
    try:
        response = requests.get(f"{API_BASE_URL}/new_portholes")
        if response.status_code == 200:
            data = response.json()
            return data.get('portholes', [])
        else:
            return []
    except Exception as e:
        print(f"새 포트홀 데이터를 가져오는 중 오류 발생: {str(e)}")
        return []

# 새로운 포트홀 알림 확인 함수
def check_new_portholes():
    current_time = time.time()
    # 마지막 확인 이후의 새로운 포트홀만 필터링
    new_portholes = fetch_new_portholes()
    if new_portholes:
        # 마지막 확인 시간 이후에 감지된 포트홀만 필터링
        recent_portholes = []
        for porthole in new_portholes:
            detected_time = porthole.get('detected_at', '')
            if detected_time:
                try:
                    # ISO 형식 문자열을 시간으로 변환하여 비교 (간략화된 버전)
                    if 'T' in detected_time and detected_time > st.session_state.last_check_time:
                        recent_portholes.append(porthole)
                except:
                    pass  # 시간 파싱 오류 무시
        
        # 새로운 포트홀이 있으면 알림 표시
        if recent_portholes:
            st.session_state.new_portholes = recent_portholes
            st.session_state.show_notification = True
            
            # 최신 포트홀을 알림 메시지로 표시
            latest = recent_portholes[-1]
            st.session_state.notification_message = f"새로운 포트홀이 감지되었습니다! 위치: {latest['location']}"
            st.session_state.notification_details = latest
            
            # 자동 새로고침이 활성화된 경우 페이지 리로드
            if st.session_state.auto_refresh:
                st.rerun()
    
    # 마지막 확인 시간 업데이트
    st.session_state.last_check_time = current_time

# 차량 데이터 가져오기 함수 수정
@st.cache_data(ttl=60)  # 1분 캐싱
def fetch_cars():
    try:
        response = requests.get(f"{API_BASE_URL}/cars")
        if response.status_code == 200:
            data = response.json()
            cars = []
            for car in data:
                cars.append({
                    'id': car['id'],
                    'lat': car['lat'],
                    'lng': car['lng'],
                    'nearby_portholes': None  # 근접 포트홀 수는 별도 API에서 필요시 조회
                })
            return cars
        else:
            st.error("차량 데이터를 가져오는데 실패했습니다.")
            return []
    except Exception as e:
        st.error(f"차량 데이터 가져오기 오류: {str(e)}")
        return []

# 포트홀 상세 정보 가져오기
@st.cache_data(ttl=60)  # 1분 캐싱
def fetch_porthole_details(porthole_id):
    response = requests.get(f"{API_BASE_URL}/portholes/{porthole_id}")
    if response.status_code == 200:
        return response.json()
    else:
        st.error(f"포트홀 ID {porthole_id}의 상세 정보를 가져오는데 실패했습니다.")
        return None

# 포트홀 상세 정보를 표시하는 함수
def display_porthole_details(selected_porthole_details, tab_name="list"):
    if selected_porthole_details:
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**ID:** {selected_porthole_details['id']}")
            st.write(f"**위치:** {selected_porthole_details['location']}")
            st.write(f"**발견 날짜:** {selected_porthole_details['date']}")
            st.write(f"**상태:** {selected_porthole_details['status']}")
            st.write(f"**깊이:** {selected_porthole_details.get('depth', '정보 없음')} cm")
            st.write(f"**좌표:** {selected_porthole_details['lat']}, {selected_porthole_details['lng']}")
            
            # 상태 업데이트 폼 - 탭 이름을 포함한 고유 키 생성
            with st.form(key=f"update_status_form_{tab_name}_{selected_porthole_details['id']}"):
                status_options = ["발견됨", "수리중", "수리완료"]
                new_status = st.selectbox("새 상태", options=status_options, 
                                         index=status_options.index(selected_porthole_details['status']) 
                                         if selected_porthole_details['status'] in status_options else 0)
                submit_button = st.form_submit_button("상태 업데이트")

                if submit_button:
                    try:
                        st.info("상태 업데이트 요청 중...")
                        response = requests.post(
                            f"{API_BASE_URL.replace('/api', '')}/update_status",
                            data={"porthole_id": selected_porthole_details['id'], "new_status": new_status}
                        )
                        st.write(f"응답 상태 코드: {response.status_code}")
                        
                        # 303 See Other는 리다이렉션 응답으로, 요청이 성공적으로 처리되었음을 의미합니다
                        if response.status_code == 303 or response.status_code == 200:
                            st.success(f"상태가 '{new_status}'(으)로 성공적으로 업데이트되었습니다!")
                            st.cache_data.clear()
                            st.rerun()  # 페이지 새로고침
                        else:
                            st.error(f"상태 업데이트에 실패했습니다. (상태 코드: {response.status_code})")
                            if response.text:
                                st.error(f"오류 메시지: {response.text}")
                    except Exception as e:
                        st.error(f"요청 중 오류가 발생했습니다: {str(e)}")
        
        # col2에 포트홀 이미지와 깊이 정보 시각화
        with col2:
            # 포트홀 이미지 표시 (이미지가 있는 경우)
            if 'image_url' in selected_porthole_details and selected_porthole_details['image_url']:
                st.image(selected_porthole_details['image_url'], caption="포트홀 이미지", use_column_width=True)
            else:
                st.info("이미지가 없습니다")
            
            # 포트홀 깊이 시각화
            st.subheader("포트홀 깊이")
            depth = selected_porthole_details.get('depth')
            if depth is not None and isinstance(depth, (int, float)):
                # 깊이에 따른 위험도 색상 결정
                if depth < 3:
                    risk = "낮음"
                    color = "초록색"
                elif depth < 7:
                    risk = "중간"
                    color = "주황색"
                else:
                    risk = "높음"
                    color = "빨간색"
                
                # 깊이 게이지 표시
                st.progress(min(depth/10, 1.0), text=f"{depth} cm")
                st.write(f"**위험도:** {risk} ({color})")
            else:
                st.warning("깊이 정보가 없습니다")
    else:
        st.info("선택된 포트홀 정보가 없습니다.")

# 새로운 포트홀 알림 표시
def show_notification_if_needed():
    if st.session_state.show_notification:
        # 알림 컨테이너 생성
        with st.container():
            # 알림 메시지
            st.warning(st.session_state.notification_message)
            
            # 세부 정보가 있으면 표시
            if st.session_state.notification_details:
                with st.expander("상세 정보 보기"):
                    porthole = st.session_state.notification_details
                    st.write(f"**포트홀 ID:** {porthole.get('porthole_id', '정보 없음')}")
                    st.write(f"**위치:** {porthole.get('location', '정보 없음')}")
                    st.write(f"**깊이:** {porthole.get('depth', '정보 없음')} cm")
                    st.write(f"**좌표:** {porthole.get('lat', '정보 없음')}, {porthole.get('lng', '정보 없음')}")
                    st.write(f"**상태:** {porthole.get('status', '정보 없음')}")
                    st.write(f"**감지 시간:** {porthole.get('detected_at', '정보 없음')}")
            
            # 알림 제거 버튼
            if st.button("알림 닫기"):
                st.session_state.show_notification = False
                st.rerun()
        
        # 구분선 추가
        st.markdown("---")

# 사이드바 - 필터링
st.sidebar.header("필터 옵션")
status_filter = st.sidebar.multiselect(
    "상태 필터링",
    ["발견됨", "수리중", "수리완료"],
    default=[],
)

# 자동 새로고침 옵션
st.sidebar.header("알림 설정")
auto_refresh_enabled = st.sidebar.checkbox(
    "새 포트홀 감지 시 자동 새로고침", 
    value=st.session_state.auto_refresh
)
st.session_state.auto_refresh = auto_refresh_enabled

# 새 포트홀 수동 확인 버튼
if st.sidebar.button("새 포트홀 확인"):
    check_new_portholes()

# 사이드바에 데이터 새로고침 버튼 추가
if st.sidebar.button("데이터 새로고침"):
    st.cache_data.clear()
    st.rerun()

# 알림 표시(있는 경우)
show_notification_if_needed()

# 메인 콘텐츠
st.markdown("## 📊 포트홀 현황")

# 탭 생성 - 차량 정보 탭 추가
tab1, tab2, tab3 = st.tabs(["포트홀 목록", "차량 목록", "지도 보기"])

# 포트홀 데이터 가져오기
portholes = fetch_portholes()
filtered_portholes = [p for p in portholes if not status_filter or p['status'] in status_filter]

# 차량 데이터 가져오기
cars = fetch_cars()

# 탭 1: 포트홀 목록 보기
with tab1:
    # --- 포트홀 추가 폼 ---
    with st.expander("➕ 포트홀 추가"):
        with st.form("add_porthole_form"):
            lat = st.number_input("위도", format="%.6f")
            lng = st.number_input("경도", format="%.6f")
            depth = st.number_input("깊이(cm)", min_value=0.0, format="%.2f")
            location = st.text_input("위치 설명")
            status = st.selectbox("상태", ["발견됨", "수리중", "수리완료"], index=0)
            submitted = st.form_submit_button("포트홀 추가")
            if submitted:
                try:
                    resp = requests.post(
                        f"{API_BASE_URL}/portholes/add",
                        json={
                            "lat": lat,
                            "lng": lng,
                            "depth": depth,
                            "location": location,
                            "status": status
                        }
                    )
                    if resp.status_code == 200:
                        st.success("포트홀 추가 성공!")
                        st.cache_data.clear()
                        st.rerun()
                    else:
                        st.error(f"포트홀 추가 실패: {resp.text}")
                except Exception as e:
                    st.error(f"포트홀 추가 오류: {str(e)}")

    # --- 포트홀 삭제 폼 ---
    with st.expander("🗑️ 포트홀 삭제"):
        if filtered_portholes:
            delete_id = st.selectbox(
                "삭제할 포트홀 선택",
                options=[p['id'] for p in filtered_portholes],
                format_func=lambda x: f"ID: {x}"
            )
            if st.button("포트홀 삭제"):
                try:
                    resp = requests.delete(f"{API_BASE_URL}/portholes/{delete_id}")
                    if resp.status_code == 200:
                        st.success("포트홀 삭제 성공!")
                        st.cache_data.clear()
                        st.rerun()
                    else:
                        st.error(f"포트홀 삭제 실패: {resp.text}")
                except Exception as e:
                    st.error(f"포트홀 삭제 오류: {str(e)}")
        else:
            st.info("삭제할 포트홀이 없습니다.")

    if filtered_portholes:
        # 데이터프레임으로 변환하여 표시
        df = pd.DataFrame(filtered_portholes)
        st.dataframe(df, use_container_width=True)
        
        # 구분선 추가
        st.markdown("---")
        
        # 선택한 포트홀 상세 정보 (더 명확한 시각적 구분 추가)
        st.markdown("## 🔍 포트홀 상세 정보")
        selected_pothole_id = st.selectbox(
            "자세히 볼 포트홀 선택",
            options=[p['id'] for p in filtered_portholes],
            format_func=lambda x: f"ID: {x}",
            key="pothole_select_list_view"  # 고유 키 추가
        )
        
        # 상세 정보 가져오기
        selected_pothole_details = fetch_porthole_details(selected_pothole_id)
        display_porthole_details(selected_pothole_details, "list")  # "list" 탭 식별자 추가
    else:
        st.info("표시할 포트홀이 없습니다.")

# 탭 2: 차량 목록 보기
with tab2:
    # --- 차량 추가 폼 ---
    with st.expander("➕ 차량 추가"):
        with st.form("add_car_form"):
            car_lat = st.number_input("차량 위도", format="%.6f", key="car_lat")
            car_lng = st.number_input("차량 경도", format="%.6f", key="car_lng")
            car_submitted = st.form_submit_button("차량 추가")
            if car_submitted:
                try:
                    resp = requests.post(
                        f"{API_BASE_URL}/cars/add",
                        json={"lat": car_lat, "lng": car_lng}
                    )
                    if resp.status_code == 200:
                        st.success("차량 추가 성공!")
                        st.cache_data.clear()
                        st.rerun()
                    else:
                        st.error(f"차량 추가 실패: {resp.text}")
                except Exception as e:
                    st.error(f"차량 추가 오류: {str(e)}")

    # --- 차량 삭제 폼 ---
    with st.expander("🗑️ 차량 삭제"):
        if cars:
            delete_car_id = st.selectbox(
                "삭제할 차량 선택",
                options=[car['id'] for car in cars],
                format_func=lambda x: f"ID: {x}",
                key="delete_car_select"
            )
            if st.button("차량 삭제"):
                try:
                    resp = requests.delete(f"{API_BASE_URL}/cars/{delete_car_id}")
                    if resp.status_code == 200:
                        st.success("차량 삭제 성공!")
                        st.cache_data.clear()
                        st.rerun()
                    else:
                        st.error(f"차량 삭제 실패: {resp.text}")
                except Exception as e:
                    st.error(f"차량 삭제 오류: {str(e)}")
        else:
            st.info("삭제할 차량이 없습니다.")

    if cars:
        # 차량 데이터를 데이터프레임으로 변환하여 표시
        df_cars = pd.DataFrame(cars)
        # 컬럼 이름을 한글로 변경
        df_cars.columns = ['ID', '위도', '경도', '주변 포트홀 수']
        st.dataframe(df_cars, use_container_width=True)
        
        # 구분선 추가
        st.markdown("---")
        
        # 선택한 차량 상세 정보
        st.markdown("## 🚗 차량 상세 정보")
        if cars:
            selected_car_id = st.selectbox(
                "자세히 볼 차량 선택",
                options=[car['id'] for car in cars],
                format_func=lambda x: f"ID: {x}",
                key="car_select_list_view"
            )
            
            # 선택한 차량 정보 표시
            selected_car = next((car for car in cars if car['id'] == selected_car_id), None)
            if selected_car:
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"**ID:** {selected_car['id']}")
                    st.write(f"**위도:** {selected_car['lat']}")
                    st.write(f"**경도:** {selected_car['lng']}")
                    st.write(f"**주변 포트홀 수:** {selected_car['nearby_portholes']}")
                
                # 미니 지도로 차량 위치 표시
                with col2:
                    mini_map = folium.Map(location=[selected_car['lat'], selected_car['lng']], zoom_start=15)
                    folium.Marker(
                        location=[selected_car['lat'], selected_car['lng']],
                        tooltip=f"차량 {selected_car['id']}",
                        icon=folium.Icon(color='blue', icon='car', prefix='fa')
                    ).add_to(mini_map)
                    st_folium(mini_map, width=300, height=200)
    else:
        st.info("표시할 차량이 없습니다.")

# 탭 3: 지도 보기 (기존 tab2를 tab3로 변경)
with tab3:
    # 포트홀 데이터와 차량 데이터를 모두 가져오기
    portholes_with_coords = []
    for p in filtered_portholes:
        details = fetch_porthole_details(p['id'])
        if details and 'lat' in details and 'lng' in details:
            portholes_with_coords.append(details)
    
    # 모든 데이터가 없는 경우에만 메시지 표시
    if not portholes_with_coords and not cars:
        st.info("표시할 포트홀 또는 차량 데이터가 없습니다.")
    else:
        # 지도에 표시할 좌표 수집
        all_coords = []
        if portholes_with_coords:
            all_coords.extend([(p['lat'], p['lng']) for p in portholes_with_coords])
        if cars:
            all_coords.extend([(car['lat'], car['lng']) for car in cars])
        
        # 최소 하나의 좌표가 있으면 지도 생성
        if all_coords:
            # 모든 좌표의 평균 계산
            avg_lat = sum(coord[0] for coord in all_coords) / len(all_coords)
            avg_lng = sum(coord[1] for coord in all_coords) / len(all_coords)
            
            # 지도 생성
            m = folium.Map(location=[avg_lat, avg_lng], zoom_start=13)
            
            # 포트홀 마커 추가 (항상 처리)
            if portholes_with_coords:
                for p in portholes_with_coords:
                    # 상태에 따른 색상 지정
                    if p['status'] == '수리완료':
                        color = 'green'
                    elif p['status'] == '수리중':
                        color = 'orange'
                    else:
                        color = 'red'
                        
                    folium.Marker(
                        location=[p['lat'], p['lng']],
                        popup=f"포트홀 ID: {p['id']}<br>위치: {p['location']}<br>상태: {p['status']}",
                        tooltip=f"포트홀 {p['id']}",
                        icon=folium.Icon(color=color)
                    ).add_to(m)
            
            # 차량 마커 추가 (항상 처리)
            if cars:
                for car in cars:
                    folium.Marker(
                        location=[car['lat'], car['lng']],
                        popup=f"차량 ID: {car['id']}<br>주변 포트홀 수: {car['nearby_portholes']}",
                        tooltip=f"차량 {car['id']}",
                        icon=folium.Icon(color='blue', icon='car', prefix='fa')
                    ).add_to(m)
            
            # 지도 표시
            st.subheader("📍 포트홀 및 차량 위치")
            st.write("지도에서 포트홀(빨강/주황/녹색)과 차량(파란색)의 위치를 확인하세요.")
            map_data = st_folium(m, width=700, height=500)
            
            # 구분선 추가
            st.markdown("---")
            
            # 지도 클릭 이벤트 처리 - 포트홀과 차량 모두 처리
            st.markdown("## 🔍 상세 정보")
            
            # 클릭된 마커 처리
            selected_id = None
            selected_type = None  # "porthole" 또는 "car"
            
            # 안전하게 map_data 체크
            if map_data is not None and 'last_clicked' in map_data and map_data['last_clicked'] is not None:
                try:
                    clicked_lat = map_data['last_clicked'].get('lat')
                    clicked_lng = map_data['last_clicked'].get('lng')
                    
                    # lat와 lng가 있는 경우에만 처리
                    if clicked_lat is not None and clicked_lng is not None:
                        # 클릭한 위치와 가장 가까운 포트홀 또는 차량 찾기
                        min_distance = float('inf')
                        
                        # 포트홀 확인
                        for p in portholes_with_coords:
                            distance = ((p['lat'] - clicked_lat) ** 2 + (p['lng'] - clicked_lng) ** 2) ** 0.5
                            if distance < min_distance:
                                min_distance = distance
                                selected_id = p['id']
                                selected_type = "porthole"
                        
                        # 차량 확인
                        for car in cars:
                            distance = ((car['lat'] - clicked_lat) ** 2 + (car['lng'] - clicked_lng) ** 2) ** 0.5
                            if distance < min_distance:
                                min_distance = distance
                                selected_id = car['id']
                                selected_type = "car"
                        
                except Exception as e:
                    st.warning(f"마커 클릭 이벤트 처리 중 오류 발생: {str(e)}")
            
            # 데이터 유형에 따라 다른 탭 표시
            map_tabs = st.tabs(["포트홀 정보", "차량 정보"])
            
            with map_tabs[0]:
                # 포트홀 상세 정보 표시
                available_porthole_ids = [p['id'] for p in portholes_with_coords]
                
                # selected_id가 유효하고 포트홀 타입인 경우에만 index 설정
                porthole_index = 0  # 기본값
                if selected_id is not None and selected_type == "porthole" and selected_id in available_porthole_ids:
                    porthole_index = available_porthole_ids.index(selected_id)
                    
                if available_porthole_ids:
                    selected_porthole_id = st.selectbox(
                        "자세히 볼 포트홀 선택",
                        options=available_porthole_ids,
                        format_func=lambda x: f"ID: {x}",
                        index=porthole_index,
                        key="porthole_select_map_view"
                    )
                    
                    # 상세 정보 가져오기
                    selected_porthole_details = fetch_porthole_details(selected_porthole_id)
                    display_porthole_details(selected_porthole_details, "map")
                else:
                    st.info("선택 가능한 포트홀이 없습니다.")
            
            with map_tabs[1]:
                # 차량 상세 정보 표시
                if cars:
                    # selected_id가 유효하고 차량 타입인 경우에만 index 설정
                    car_index = 0  # 기본값
                    car_ids = [car['id'] for car in cars]
                    if selected_id is not None and selected_type == "car" and selected_id in car_ids:
                        car_index = car_ids.index(selected_id)
                        
                    selected_car_id = st.selectbox(
                        "자세히 볼 차량 선택",
                        options=car_ids,
                        format_func=lambda x: f"ID: {x}",
                        index=car_index,
                        key="car_select_map_view"
                    )
                    
                    # 선택한 차량 정보 표시
                    selected_car = next((car for car in cars if car['id'] == selected_car_id), None)
                    if selected_car:
                        st.write(f"**ID:** {selected_car['id']}")
                        st.write(f"**위도:** {selected_car['lat']}")
                        st.write(f"**경도:** {selected_car['lng']}")
                        st.write(f"**주변 포트홀 수:** {selected_car['nearby_portholes']}")
                        
                        # 주변 포트홀 정보 요청
                        try:
                            proximity_response = requests.get(f"{API_BASE_URL}/car_alerts/{selected_car_id}")
                            if proximity_response.status_code == 200:
                                proximity_data = proximity_response.json()
                                nearby_portholes = proximity_data.get('alerts', [])
                                
                                if nearby_portholes:
                                    st.subheader("주변 포트홀 목록")
                                    # 데이터 구조 변환 - alerts 배열의 각 요소에서 필요한 필드만 추출
                                    formatted_portholes = []
                                    for alert in nearby_portholes:
                                        formatted_portholes.append({
                                            "포트홀 ID": alert.get("porthole_id", "N/A"),
                                            "위치": alert.get("location", "N/A"),
                                            "깊이": alert.get("depth", "N/A"),
                                            "거리": f"{alert.get('distance', 'N/A')}m",
                                            "상태": alert.get("status", "N/A")
                                        })
                                    nearby_df = pd.DataFrame(formatted_portholes)
                                    st.dataframe(nearby_df, use_container_width=True)
                                else:
                                    st.info(f"차량 {selected_car_id} 주변에 알림이 발생한 포트홀이 없습니다.")
                        except Exception as e:
                            st.error(f"주변 포트홀 정보를 가져오는 도중 오류 발생: {str(e)}")
                else:
                    st.info("표시할 차량이 없습니다.")
        else:
            st.warning("지도에 표시할 좌표 정보가 없습니다.")

# 백그라운드에서 새로운 포트홀 확인
# Streamlit은 백그라운드 작업을 직접 지원하지 않으므로 주기적인 통신이 필요할 경우 
# 웹앱이 표시되는 동안 주기적으로 실행
check_new_portholes()