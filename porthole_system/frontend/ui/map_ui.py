import streamlit as st
import folium
from streamlit_folium import st_folium
import requests
import pandas as pd

# 포트홀 데이터를 가져오는 함수
@st.cache_data(ttl=60)
def fetch_portholes_for_map(api_url: str):
    """포트홀 정보를 지도에 표시하기 위해 가져옵니다."""
    try:
        response = requests.get(f"{api_url}/portholes")
        if response.status_code == 200:
            portholes = response.json()
            result = []
            
            # 각 포트홀의 상세 정보 가져오기
            for p in portholes:
                detail_response = requests.get(f"{api_url}/portholes/{p['id']}")
                if detail_response.status_code == 200:
                    detail = detail_response.json()
                    if 'lat' in detail and 'lng' in detail:
                        result.append(detail)
            return result
        else:
            st.error(f"포트홀 데이터를 가져오는데 실패했습니다. 상태코드: {response.status_code}")
            return []
    except Exception as e:
        st.error(f"포트홀 데이터 요청 중 오류: {str(e)}")
        return []
        
# 차량 데이터를 가져오는 함수
@st.cache_data(ttl=60)
def fetch_cars_for_map(api_url: str):
    """차량 정보를 지도에 표시하기 위해 가져옵니다."""
    try:
        response = requests.get(f"{api_url}/cars")
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"차량 데이터를 가져오는데 실패했습니다. 상태코드: {response.status_code}")
            return []
    except Exception as e:
        st.error(f"차량 데이터 요청 중 오류: {str(e)}")
        return []

def render_map_tab(api_url: str):
    """지도 탭 UI를 렌더링하는 함수"""
    st.header("🗺️ 포트홀 및 차량 위치 지도")
    
    # 데이터 로딩
    portholes = fetch_portholes_for_map(api_url)
    cars = fetch_cars_for_map(api_url)
    
    # 표시할 데이터가 있는지 확인
    if not portholes and not cars:
        st.info("표시할 포트홀 또는 차량 데이터가 없습니다.")
        return
    
    # 지도 중심 좌표 결정 (첫 번째 포트홀 또는 차량 위치, 또는 서울 중심부)
    center_lat = 37.5665
    center_lng = 126.9780
    
    if portholes:
        center_lat = portholes[0].get('lat', center_lat)
        center_lng = portholes[0].get('lng', center_lng)
    elif cars:
        center_lat = cars[0].get('lat', center_lat)
        center_lng = cars[0].get('lng', center_lng)
    
    # folium 지도 생성
    m = folium.Map(location=[center_lat, center_lng], zoom_start=15)
    
    # 포트홀 마커 추가
    for porthole in portholes:
        lat = porthole.get('lat')
        lng = porthole.get('lng')
        
        if not lat or not lng:
            continue
            
        # 상태에 따라 아이콘 색상 결정
        status = porthole.get('status', '발견됨')
        if status == '발견됨':
            color = 'red'
        elif status == '수리중':
            color = 'orange' 
        elif status == '수리완료':
            color = 'green'
        else:
            color = 'gray'
        
        # 팝업 내용 생성
        popup_html = f"""
        <b>포트홀 ID:</b> {porthole.get('id')}<br>
        <b>위치:</b> {porthole.get('location', '정보 없음')}<br>
        <b>깊이:</b> {porthole.get('depth', '정보 없음')} cm<br>
        <b>상태:</b> {status}<br>
        <b>발견 날짜:</b> {porthole.get('date', '정보 없음')}<br>
        """
        
        folium.Marker(
            location=[lat, lng],
            popup=folium.Popup(popup_html, max_width=300),
            tooltip=f"포트홀 #{porthole.get('id')} - {status}",
            icon=folium.Icon(color=color, icon='warning-sign', prefix='glyphicon')
        ).add_to(m)
    
    # 차량 마커 추가
    for car in cars:
        lat = car.get('lat')
        lng = car.get('lng')
        
        if not lat or not lng:
            continue
        
        # 팝업 내용 생성
        popup_html = f"""
        <b>차량 ID:</b> {car.get('id')}<br>
        <b>위도:</b> {lat}<br>
        <b>경도:</b> {lng}<br>
        """
        
        folium.Marker(
            location=[lat, lng],
            popup=folium.Popup(popup_html, max_width=300),
            tooltip=f"차량 #{car.get('id')}",
            icon=folium.Icon(color='blue', icon='info-sign', prefix='glyphicon')
        ).add_to(m)
    
    # 열 나누기
    col1, col2 = st.columns([3, 1])
    
    # 열 1: 지도 표시
    with col1:
        # 지도 컨테이너 스타일링
        st.write("##### 포트홀 및 차량 위치")
        map_data = st_folium(m, width=700, height=500)
    
    # 열 2: 범례 및 정보
    with col2:
        st.write("##### 지도 범례")
        st.markdown("""
        - <span style='color:red'>&#9679;</span> 발견된 포트홀
        - <span style='color:orange'>&#9679;</span> 수리중인 포트홀
        - <span style='color:green'>&#9679;</span> 수리완료 포트홀
        - <span style='color:blue'>&#9679;</span> 차량 위치
        """, unsafe_allow_html=True)
        
        st.write("##### 요약 정보")
        st.write(f"총 포트홀 수: {len(portholes)}")
        st.write(f"총 차량 수: {len(cars)}")
        
        status_counts = {
            '발견됨': len([p for p in portholes if p.get('status') == '발견됨']),
            '수리중': len([p for p in portholes if p.get('status') == '수리중']),
            '수리완료': len([p for p in portholes if p.get('status') == '수리완료'])
        }
        
        st.write("포트홀 상태 분포:")
        for status, count in status_counts.items():
            st.write(f"- {status}: {count}개")
    
    # 지도 클릭 이벤트 처리
    if map_data and map_data.get("last_clicked"):
        clicked_lat = map_data["last_clicked"]["lat"]
        clicked_lng = map_data["last_clicked"]["lng"]
        
        st.write("##### 선택한 위치")
        st.write(f"위도: {clicked_lat:.6f}, 경도: {clicked_lng:.6f}")
        
        # 클릭한 위치에 새 포트홀이나 차량 추가하는 옵션
        st.write("이 위치에 새로운 항목을 추가하시겠습니까?")
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("포트홀 추가", key="add_porthole_from_map"):
                st.session_state.map_clicked_lat = clicked_lat
                st.session_state.map_clicked_lng = clicked_lng
                st.session_state.active_tab = "포트홀 목록"
                st.rerun()
                
        with col2:
            if st.button("차량 추가", key="add_car_from_map"):
                st.session_state.map_clicked_lat = clicked_lat
                st.session_state.map_clicked_lng = clicked_lng
                st.session_state.active_tab = "차량 목록"
                st.rerun()
