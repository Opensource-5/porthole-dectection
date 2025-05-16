import streamlit as st
import requests
import pandas as pd
from datetime import datetime

@st.cache_data(ttl=60)
def fetch_cars(api_url: str):
    """차량 목록을 가져오는 함수"""
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
        
@st.cache_data(ttl=60)
def fetch_car_details(api_url: str, car_id: int):
    """차량 상세 정보를 가져오는 함수"""
    try:
        response = requests.get(f"{api_url}/cars/{car_id}")
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"차량 상세 정보를 가져오는데 실패했습니다. 상태코드: {response.status_code}")
            return None
    except Exception as e:
        st.error(f"차량 상세 정보 요청 중 오류: {str(e)}")
        return None

def render_car_tab(api_url: str):
    """차량 탭 UI를 렌더링하는 함수"""
    st.header("🚗 차량 관리")
    
    # 차량 추가 폼
    with st.expander("➕ 차량 추가"):
        with st.form("add_car_form"):
            lat = st.number_input("위도", format="%.6f", value=37.5665)
            lng = st.number_input("경도", format="%.6f", value=126.9780)
            
            submitted = st.form_submit_button("차량 추가")
            if submitted:
                try:
                    response = requests.post(
                        f"{api_url}/cars/add", 
                        json={
                            "lat": lat,
                            "lng": lng
                        }
                    )
                    if response.status_code == 200:
                        st.success("차량이 추가되었습니다.")
                        # 캐시 갱신
                        st.cache_data.clear()
                    else:
                        st.error(f"차량 추가 실패: {response.text}")
                except Exception as e:
                    st.error(f"요청 중 오류 발생: {str(e)}")
    
    # 차량 목록 조회
    cars = fetch_cars(api_url)
    
    if cars:
        # 데이터프레임으로 변환하고 컬럼 이름 변경
        df = pd.DataFrame(cars)
        df.columns = [col if col != 'id' else 'ID' for col in df.columns]
        df.columns = [col if col != 'lat' else '위도' for col in df.columns]
        df.columns = [col if col != 'lng' else '경도' for col in df.columns]
        
        st.dataframe(df, use_container_width=True)
        
        # 차량 삭제 폼
        with st.expander("🗑️ 차량 삭제"):
            delete_id = st.selectbox(
                "삭제할 차량 선택",
                options=[car['id'] for car in cars],
                format_func=lambda x: f"차량 ID: {x}"
            )
            if st.button("차량 삭제"):
                try:
                    response = requests.delete(f"{api_url}/cars/{delete_id}")
                    if response.status_code == 200:
                        st.success("차량이 삭제되었습니다.")
                        # 캐시 갱신
                        st.cache_data.clear()
                        st.rerun()
                    else:
                        st.error(f"삭제 실패: {response.text}")
                except Exception as e:
                    st.error(f"요청 중 오류 발생: {str(e)}")
        
        # 차량 위치 업데이트
        with st.expander("📍 차량 위치 업데이트"):
            update_car_id = st.selectbox(
                "위치를 업데이트할 차량 선택",
                options=[car['id'] for car in cars],
                format_func=lambda x: f"차량 ID: {x}",
                key="update_car_id"
            )
            
            selected_car = next((car for car in cars if car['id'] == update_car_id), None)
            if selected_car:
                with st.form("update_car_location_form"):
                    new_lat = st.number_input(
                        "새 위도", 
                        format="%.6f", 
                        value=float(selected_car.get('lat', 37.5665))
                    )
                    new_lng = st.number_input(
                        "새 경도", 
                        format="%.6f", 
                        value=float(selected_car.get('lng', 126.9780))
                    )
                    
                    loc_submitted = st.form_submit_button("위치 업데이트")
                    if loc_submitted:
                        try:
                            response = requests.put(
                                f"{api_url}/cars/{update_car_id}/location", 
                                json={
                                    "lat": new_lat,
                                    "lng": new_lng
                                }
                            )
                            if response.status_code == 200:
                                st.success("차량 위치가 업데이트되었습니다.")
                                # 캐시 갱신
                                st.cache_data.clear()
                                st.rerun()
                            else:
                                st.error(f"위치 업데이트 실패: {response.text}")
                        except Exception as e:
                            st.error(f"요청 중 오류 발생: {str(e)}")
    else:
        st.info("표시할 차량이 없습니다. 새로운 차량을 추가해 보세요.")
