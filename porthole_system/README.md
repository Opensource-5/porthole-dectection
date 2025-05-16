# 포트홀 감지 시스템 (Porthole Detection System)

이 프로젝트는 도로의 포트홀 정보를 관리하고, 차량과 포트홀 간의 거리를 계산하여 실시간 알림을 제공하는 시스템입니다.

## 주요 기능

- 포트홀 등록, 조회, 상태 업데이트, 삭제
- 차량 등록, 조회, 위치 업데이트, 삭제
- 차량과 포트홀 간의 근접 거리 모니터링
- 포트홀 알림 생성 및 관리
- 지도 기반 시각화

## 시스템 구조

```
porthole_system/
├── backend/           # 백엔드 모듈
│   ├── api/           # API 엔드포인트
│   │   ├── alerts.py  # 알림 API
│   │   ├── car.py     # 차량 API
│   │   └── porthole.py# 포트홀 API
│   ├── crud.py        # 데이터베이스 CRUD 작업
│   ├── db.py          # 데이터베이스 연결 및 초기화
│   ├── logic.py       # 비즈니스 로직
│   ├── models.py      # 데이터 모델
│   └── scheduler.py   # 백그라운드 작업 스케줄러
├── frontend/          # 프론트엔드 모듈
│   ├── app.py         # Streamlit 앱 진입점
│   └── ui/            # UI 컴포넌트
│       ├── alert_ui.py    # 알림 UI
│       ├── car_ui.py      # 차량 UI
│       ├── map_ui.py      # 지도 UI
│       └── porthole_ui.py # 포트홀 UI
├── main.py            # 애플리케이션 진입점
└── requirements.txt   # 필요한 패키지 목록
```

## 설치 방법

1. 저장소 클론

```bash
git clone https://github.com/yourusername/porthole-system.git
cd porthole-system
```

2. 가상환경 생성 및 활성화

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

3. 필요한 패키지 설치

```bash
pip install -r requirements.txt
```

## 실행 방법

1. 백엔드 서버 실행

```bash
uvicorn main:app --reload
```

2. 프론트엔드 앱 실행 (새 터미널에서)

```bash
streamlit run frontend/app.py
```

3. 웹 브라우저에서 접속
   - 백엔드 API 문서: http://localhost:8000/docs
   - 프론트엔드 대시보드: http://localhost:8501

## 환경 변수 설정

`.env` 파일을 생성하고 다음 변수를 설정할 수 있습니다:

```
OPENAI_API_KEY=your_api_key_here  # 알림 메시지 생성 및 TTS를 위한 OpenAI API 키
```

## 개발 환경

- Python 3.8+
- FastAPI
- Streamlit
- SQLite
- OpenAI API (선택적)

## 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다.
