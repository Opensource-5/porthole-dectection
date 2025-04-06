from flask import Flask, render_template, request, redirect, jsonify
import pymysql
import math

import pymysql.cursors

app = Flask(__name__)

# MySQL 연결 설정
db = pymysql.connect(
    host='127.0.0.1',
    port=3306,
    user='root',
    password='root', #SQL비밀번호
    db='MapDB',
    charset='utf8mb4',
    cursorclass=pymysql.cursors.DictCursor
)


## 차량과 포트홀 사이 거리 계산 파트

# 하버사인 거리 계산 함수
def calculate_distance(lat1, lng1, lat2, lng2):
    R = 6371 * 1000  # 지구 반지름 (미터 단위)
    dLat = math.radians(lat2 - lat1)
    dLng = math.radians(lng2 - lng1)

    a = math.sin(dLat / 2) ** 2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dLng / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    return R * c  # 거리 (미터 단위)

def get_porthole_distance(porthole_id, car_id):
    porthole = get_porthole_by_id(porthole_id)
    car = get_car_by_id(car_id)
    distance = calculate_distance(porthole['lat'], porthole['lng'], car['lat'], car['lng'])
    return distance

def get_car_by_id(car_id):
    try:
        with db.cursor(pymysql.cursors.DictCursor) as cursor:
            sql = "select * FROM car WHERE id = %s"
            cursor.execute(sql, (car_id,))
            result = cursor.fetchone()
            return result
    except Exception as e:
        print("get_car_by_id 오류:", e)
        return None




## 여기서부터 웹페이지 데이터 가져오는 파트 


# ✅ 전체 포트홀 목록 가져오기
def get_all_portholes():
    try:
        with db.cursor() as cursor:
            sql = "SELECT id, location, date, status FROM porthole"
            cursor.execute(sql)
            return cursor.fetchall()
    except Exception as e:
        print("get_all_portholes 오류:", e)
        return []

# ✅ 특정 포트홀 상세 정보 가져오기
def get_porthole_by_id(porthole_id):
    try:
        with db.cursor(pymysql.cursors.DictCursor) as cursor:  # <-- 여기 수정!
            sql = "SELECT * FROM porthole WHERE id = %s"
            cursor.execute(sql, (porthole_id,))
            result = cursor.fetchone()
            return result
    except Exception as e:
        print("get_porthole_by_id 오류:", e)
        return None


# ✅ 포트홀 상태 업데이트
def update_porthole_status(porthole_id, new_status):
    try:
        with db.cursor() as cursor:
            sql = "UPDATE porthole SET status = %s WHERE id = %s"
            cursor.execute(sql, (new_status, porthole_id))
        db.commit()
    except Exception as e:
        print("update_porthole_status 오류:", e)

# ✅ API: JSON 응답
@app.route("/api/portholes")
def api_portholes():
    return jsonify(get_all_portholes())

# ✅ 메인 페이지
@app.route('/')
def index():
    return render_template("index.html", portholes=get_all_portholes())



## 포트홀과 차량 간 거리 
@app.route("/api/distance/<int:porthole_id>/<int:car_id>")
def distance(porthole_id, car_id):
    porthole = get_porthole_by_id(porthole_id)
    car = get_car_by_id(car_id)

    if not porthole or not car:
        return jsonify({"error": "잘못된 포트홀 ID 또는 차량 ID입니다."}), 404

    distance_value = calculate_distance(
        porthole["lat"], porthole["lng"],
        car["lat"], car["lng"]
    )
    distance_str = f"{round(distance_value, 2)}m"

    return jsonify({
        "car_lat": car["lat"],
        "car_lng": car["lng"],
        "porthole_lat": porthole["lat"],
        "porthole_lng": porthole["lng"],
        "distance": distance_str
    })


# ✅ 상세 페이지
@app.route("/detail/<int:porthole_id>")
def detail(porthole_id):
    porthole = get_porthole_by_id(porthole_id)

    if not porthole:
        return "해당 포트홀을 찾을 수 없습니다.", 404

    # 포트홀 정보를 변수로 분해해서 템플릿에 전달
    return render_template(
        "detail.html",
        id=porthole["id"],
        lat=porthole["lat"],
        lng=porthole["lng"],
        location=porthole["location"],
        status=porthole["status"],
        date=porthole["date"],

        markers=[porthole]  # 지도 마커용으로 넘김
    )

# ✅ 상태 업데이트 처리
@app.route("/update_status", methods=["POST"])
def update_status():
    porthole_id = int(request.form.get("porthole_id"))  
    new_status = request.form.get("new_status")
    
    print("업데이트 시도: ID =", porthole_id, "→", new_status)  
    
    update_porthole_status(porthole_id, new_status)
    return redirect(f"/detail/{porthole_id}")

# ✅ 서버 실행
if __name__ == "__main__":
    app.run(debug=True)
