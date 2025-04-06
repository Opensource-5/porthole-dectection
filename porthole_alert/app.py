import streamlit as st
from pathlib import Path
import os

# Import functions and example input from main.py
from main import (
    example_input,
    process_porthole_input,
    generate_alert_message,
    synthesize_alert_audio,
)

def main():
    st.title("포트홀 알림 시스템")
    st.write("센서 데이터를 처리하고 운전자에게 경고 메시지 및 음성 알림을 제공합니다.")

    st.sidebar.header("예제 센서 데이터")
    st.sidebar.json(example_input)

    if st.button("알림 생성"):
        # Process and enrich sensor data
        processed_data = process_porthole_input(example_input)
        if not processed_data:
            st.error("센서 데이터 처리에 실패하였습니다.")
            return

        st.subheader("처리된 데이터")
        st.json(processed_data)

        # Generate alert message with LLM
        alert_message = generate_alert_message(processed_data)
        st.subheader("경고 메시지")
        st.write(alert_message)

        # Synthesize audio alert
        synthesize_alert_audio(alert_message)
        audio_file_path = Path(__file__).parent / "speech.mp3"
        if audio_file_path.exists():
            st.subheader("알림 음성 재생")
            with open(audio_file_path, "rb") as audio_file:
                st.audio(audio_file.read(), format="audio/mp3")
        else:
            st.error("음성 합성 파일을 찾을 수 없습니다.")

if __name__ == "__main__":
    main()