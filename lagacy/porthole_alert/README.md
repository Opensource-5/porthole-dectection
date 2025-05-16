# 포트홀 경고 시스템 (Porthole Alert System)

## 기본 기능 정의

1. **포트홀 데이터 입력 처리**
   - 센서로부터 수집된 포트홀 정보 처리
   - 데이터 포맷 검증 및 전처리

2. **깊이 분석 및 위험도 평가**
   - 포트홀 깊이에 따른 위험도 분류
   - 차량과의 거리, 속도를 고려한 위험 수준 계산

3. **LLM 기반 경고 메시지 생성**
   - 위험도에 따른 맞춤형 경고 메시지 생성
   - 상황 맥락을 고려한 자연스러운 경고문 작성

4. **음성 변환 및 출력**
   - 텍스트 경고 메시지를 음성으로 변환
   - 운전자에게 명확한 경고 전달


## 개발 접근 방식

### 프로토타입 개발
- **초기 단계**: `main.py`와 최소한의 모듈로 기본 기능 구현
- **확장 단계**: 사용자 피드백 및 테스트 결과에 따라 모듈 확장

### 단계적 구현
1. 데이터 모델 및 입력 처리 구현
2. 위험도 평가 알고리즘 개발
3. LLM 연동 및 메시지 생성 시스템 구축
4. TTS 시스템 통합
5. 차량 간 통신 시스템 구현

## 필요 라이브러리/API

### LLM 연동
- OpenAI API (GPT 모델)
- Hugging Face Transformers

### 음성 변환 (TTS)
- Google Cloud Text-to-Speech
- AWS Polly
- Mozilla TTS (오픈소스 대안)

### 차량 간 통신
- MQTT 프로토콜
- WebSocket 기반 실시간 통신
- 클라우드 기반 메시지 브로커

### 기타 필요 라이브러리
- FastAPI/Flask (API 서버)
- PyTorch/TensorFlow (추가 ML 모델링)
- Pandas (데이터 처리)

## 시스템 프롬프트 설정

LLM 기반 경고 메시지 생성을 위해 시스템 프롬프트를 설정합니다. LangChain을 사용하여 시스템 프롬프트를 정의하고 OpenAI 모델에 적용합니다.

```python
from langchain.chat_models import ChatOpenAI
from langchain.schema import SystemMessage
from langchain.prompts.chat import ChatPromptTemplate
from langchain.chains import LLMChain

# 시스템 메시지 정의
system_message = SystemMessage(content="당신은 도로 안전 전문가입니다. 포트홀 상황에 대한 적절한 안전 조치 사항을 제안해 주세요.")

# OpenAI 모델 생성 시 시스템 메시지 적용
llm = ChatOpenAI(temperature=0.7, model="gpt-3.5-turbo")

# 시스템 메시지와 사용자 메시지를 포함한 체인 구성
message_template = (
    "상황: 차량 {car_id}가 감지되었습니다.\n"
    "수색 거리: {distance}m, 깊이: {depth}m.\n"
    "위험도: {risk_level}\n\n"
    "{alert_message}\n\n"
    "상황을 종합하여 추가 조치 사항과 권고 메시지를 자연스럽게 작성해 주세요."
)

prompt = ChatPromptTemplate.from_messages([
    ("system", system_message.content),
    ("user", message_template)
])

chain = LLMChain(llm=llm, prompt=prompt)
