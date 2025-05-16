-- 기존 데이터베이스 목록 보기
SHOW DATABASES;

-- MapDB 데이터베이스 없으면 생성
CREATE DATABASE IF NOT EXISTS MapDB;

-- 사용
USE MapDB;

-- 기존 테이블 제거 (선택적으로)
DROP TABLE IF EXISTS maptb;
DROP TABLE IF EXISTS markertb;
DROP TABLE IF EXISTS porthole;

-- 새로운 테이블 생성
CREATE TABLE IF NOT EXISTS porthole (
  id INT NOT NULL PRIMARY KEY,
  lat DOUBLE NOT NULL,
  lng DOUBLE NOT NULL,
  location VARCHAR(255),
  date DATE,
  status VARCHAR(20) DEFAULT 'INCOMPLETED'
);

-- 예시 데이터 삽입
INSERT INTO porthole (id, lat, lng, location, date, status)
VALUES 
  (1, 37.496111, 126.953889, '서울시 서대문구 일번고속도로', '2025-04-03', 'INCOMPLETED'),
  (2, 37.495888, 126.953555, '서울시 노원구 이번고속도로', '2025-04-08', 'INCOMPLETED'),
  (3, 37.496333, 126.954222, '서울시 중구 삼번고속도로', '2025-04-05', 'INCOMPLETED');

-- 전체 확인
SELECT * FROM porthole;
