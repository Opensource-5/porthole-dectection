import streamlit as st
import requests
import pandas as pd
from datetime import datetime
import time

@st.cache_data(ttl=60)
def fetch_portholes(api_url: str):
    """포트홀 목록을 가져오는 함수"""
    try:
        response = requests.get(f"{api_url}/portholes")
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"포트홀 데이터를 가져오는데 실패했습니다. 상태코드: {response.status_code}")
            return []
    except Exception as e:
        st.error(f"포트홀 데이터 요청 중 오류: {str(e)}")
        return []
        
@st.cache_data(ttl=60)
def fetch_porthole_details(api_url: str, porthole_id: int):
    """포트홀 상세 정보를 가져오는 함수"""
    try:
        response = requests.get(f"{api_url}/portholes/{porthole_id}")
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"포트홀 상세 정보를 가져오는데 실패했습니다. 상태코드: {response.status_code}")
            return None
    except Exception as e:
        st.error(f"포트홀 상세 정보 요청 중 오류: {str(e)}")
        return None

def render_porthole_tab(api_url: str):
    """포트홀 탭 UI를 렌더링하는 함수"""
    st.header("📊 포트홀 관리")
    
    # 포트홀 추가 폼
    with st.expander("➕ 포트홀 추가"):
        with st.form("add_porthole_form"):
            lat = st.number_input("위도", format="%.6f", value=37.5665)
            lng = st.number_input("경도", format="%.6f", value=126.9780)
            depth = st.number_input("깊이(mm)", min_value=0.0, format="%.2f", value=3.0)
            location = st.text_input("위치 설명", value="서울시 중구")
            status = st.selectbox("상태", ["발견됨", "수리중", "수리완료"], index=0)
            
            submitted = st.form_submit_button("포트홀 추가")
            if submitted:
                try:
                    response = requests.post(
                        f"{api_url}/portholes/add", 
                        json={
                            "lat": lat,
                            "lng": lng,
                            "depth": depth,
                            "location": location,
                            "status": status
                        }
                    )
                    if response.status_code == 200:
                        st.success("포트홀이 추가되었습니다.")
                        # 캐시 갱신
                        st.cache_data.clear()
                    else:
                        st.error(f"포트홀 추가 실패: {response.text}")
                except Exception as e:
                    st.error(f"요청 중 오류 발생: {str(e)}")
    
    # 포트홀 목록 조회
    portholes = fetch_portholes(api_url)
    
    if portholes:
        # 필터링 기능 추가
        status_filter = st.multiselect(
            "상태 필터링",
            ["발견됨", "수리중", "수리완료"],
            default=[]
        )
        
        filtered_portholes = portholes
        if status_filter:
            filtered_portholes = [p for p in portholes if p.get('status', '') in status_filter]
        
        # 데이터프레임 표시
        if filtered_portholes:
            # 이미지 여부를 표시하기 위해 데이터 전처리
            for porthole in filtered_portholes:
                porthole['has_image'] = '✅' if porthole.get('image_path') else '❌'
            
            df = pd.DataFrame(filtered_portholes)
            
            # 표시할 컬럼 선택 및 순서 조정
            display_columns = ['id', 'location', 'status', 'depth', 'has_image', 'date']
            available_columns = [col for col in display_columns if col in df.columns]
            if available_columns:
                df_display = df[available_columns].copy()
                df_display.columns = ['ID', '위치', '상태', '깊이(mm)', '이미지', '발견일']
                st.dataframe(df_display, use_container_width=True)
            
            # 포트홀 삭제 폼
            with st.expander("🗑️ 포트홀 삭제"):
                delete_id = st.selectbox(
                    "삭제할 포트홀 선택",
                    options=[p['id'] for p in filtered_portholes],
                    format_func=lambda x: f"ID: {x}"
                )
                if st.button("포트홀 삭제"):
                    try:
                        response = requests.delete(f"{api_url}/portholes/{delete_id}")
                        if response.status_code == 200:
                            st.success("포트홀이 삭제되었습니다.")
                            # 캐시 갱신
                            st.cache_data.clear()
                            st.rerun()
                        else:
                            st.error(f"삭제 실패: {response.text}")
                    except Exception as e:
                        st.error(f"요청 중 오류 발생: {str(e)}")
            
            # 포트홀 상세 정보 보기
            st.subheader("🔍 포트홀 상세 정보")
            selected_porthole_id = st.selectbox(
                "상세 정보를 볼 포트홀 선택",
                options=[p['id'] for p in filtered_portholes],
                format_func=lambda x: f"ID: {x}"
            )
            
            if selected_porthole_id:
                porthole_detail = fetch_porthole_details(api_url, selected_porthole_id)
                if porthole_detail:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write(f"**ID:** {porthole_detail['id']}")
                        st.write(f"**위치:** {porthole_detail.get('location', '정보 없음')}")
                        st.write(f"**발견 날짜:** {porthole_detail.get('date', '정보 없음')}")
                        st.write(f"**상태:** {porthole_detail.get('status', '정보 없음')}")
                        st.write(f"**깊이:** {porthole_detail.get('depth', '정보 없음')} mm")
                        st.write(f"**좌표:** {porthole_detail.get('lat', '정보 없음')}, {porthole_detail.get('lng', '정보 없음')}")
                        
                        # 상태 업데이트 폼
                        with st.form(key=f"update_status_form_{selected_porthole_id}"):
                            new_status = st.selectbox(
                                "상태 업데이트",
                                ["발견됨", "수리중", "수리완료"],
                                index=["발견됨", "수리중", "수리완료"].index(porthole_detail.get('status', '발견됨'))
                            )
                            status_submitted = st.form_submit_button("상태 업데이트")
                            
                            if status_submitted and new_status != porthole_detail.get('status'):
                                try:
                                    response = requests.put(
                                        f"{api_url}/portholes/{selected_porthole_id}/status",
                                        params={"status": new_status}
                                    )
                                    if response.status_code == 200:
                                        st.success("포트홀 상태가 업데이트되었습니다.")
                                        # 캐시 갱신
                                        st.cache_data.clear()
                                        st.rerun()
                                    else:
                                        st.error(f"상태 업데이트 실패: {response.text}")
                                except Exception as e:
                                    st.error(f"요청 중 오류 발생: {str(e)}")
                    
                    with col2:
                        # 포트홀 이미지 표시
                        if porthole_detail.get('image_path'):
                            st.subheader("📸 포트홀 이미지")
                            try:
                                # 이미지 URL 생성 (백엔드 서버 주소에 static 경로 추가)
                                base_url = api_url.replace('/api', '')  # '/api' 제거
                                image_url = f"{base_url}{porthole_detail['image_path']}"
                                
                                # 이미지 표시
                                st.image(image_url, caption=f"포트홀 ID: {porthole_detail['id']}", use_column_width=True)
                                
                                # 이미지 다운로드 링크
                                st.markdown(f"[원본 이미지 다운로드]({image_url})")
                                
                            except Exception as e:
                                st.error(f"이미지 로딩 중 오류: {str(e)}")
                                st.write("이미지를 표시할 수 없습니다.")
                        else:
                            st.info("이 포트홀에는 이미지가 없습니다.")
                        
                        # 깊이에 따른 위험도 표시
                        depth = porthole_detail.get('depth', 0)
                        if depth is not None:
                            st.subheader("포트홀 깊이 시각화")
                            if depth > 2000:
                                color = "red"
                                risk = "높음"
                            elif depth > 1000:
                                color = "orange"
                                risk = "중간"
                            else:
                                color = "green"
                                risk = "낮음"
                                
                            st.markdown(f"**위험도:** <span style='color:{color};font-weight:bold;'>{risk}</span>", unsafe_allow_html=True)
                            st.progress(min(depth / 10, 1.0))  # 최대 10mm를 기준으로
                            st.write(f"{depth} mm")
                        
                        # 지도에 포트홀 위치 표시 추가 가능
        else:
            st.info("조건에 맞는 포트홀이 없습니다.")
    else:
        st.info("표시할 포트홀이 없습니다. 새로운 포트홀을 추가해 보세요.")
