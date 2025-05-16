"""
로깅 시스템 설정 모듈
"""
import logging
import sys
from logging.handlers import RotatingFileHandler
import os

# 로그 디렉토리 생성
log_dir = "logs"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# 로거 생성
logger = logging.getLogger("porthole_system")
logger.setLevel(logging.DEBUG)

# 파일 핸들러 설정 (로그 파일에 기록)
file_handler = RotatingFileHandler(
    os.path.join(log_dir, "porthole.log"),
    maxBytes=10*1024*1024,  # 10MB
    backupCount=5
)
file_handler.setLevel(logging.DEBUG)

# 콘솔 핸들러 설정 (터미널에 출력)
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)

# 포맷터 설정
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

# 핸들러 추가
logger.addHandler(file_handler)
logger.addHandler(console_handler)

def get_logger():
    """
    애플리케이션의 로거를 반환합니다.
    
    Returns:
        logging.Logger: 로거 인스턴스
    """
    return logger
