from flask import Flask, jsonify, render_template, request
import pymysql

app = Flask(__name__)

# MySQL 연결 설정
connection = pymysql.connect(host='localhost',
                             user='root',
                             password='root',
                             database='MapDB',
                             cursorclass=pymysql.cursors.DictCursor)

# 구현할 때 다음과 같은 구조로 map과 marker table을 만들었습니다. 
# cf) data_map은 marker가 어느 map에 해당하는지, 즉 map의 id와 일치합니다.    

# CREATE TABLE if not exists maptb 
# ( id   	INT NOT NULL PRIMARY KEY,
# lat   	FLOAT8 NOT NULL,
# lng 	FLOAT8 NOT NULL
# );

# CREATE TABLE if not exists markertb 
# ( id   	INT NOT NULL,
# lat   	FLOAT8 NOT NULL,
# lng 	FLOAT8 NOT NULL,
# data_map INT NOT NULL
# );

@app.route('/detail/<int:id>') # redirect할 때 detail 뒤에 id를 적어서 넣으면 id에 해당하는 지도로 가도록
def detail(id):
    try:
        with connection.cursor() as cursor: 
            sql = "SELECT lat, lng FROM maptb WHERE id = %s" # maptb에서 lat과 lng 정보를 fetch해서 가져옵니다.
            cursor.execute(sql, (id,))
            data = cursor.fetchone()
            if not data:
                return "ID not found", 404
            
            sql = "SELECT lat, lng FROM markertb WHERE data_map = %s" #위와 같이 markertb에서 data_map이 일치하는 것을 fetch해서 전부 가져옵니다.
            cursor.execute(sql, (id,))
            markers = cursor.fetchall()  # 여러 개의 마커 리스트

            return render_template('detail.html', id=id, lat=data['lat'], lng=data['lng'], markers = markers) #위에서 fetch해온 정보로 detail.html로 연결합니다.
    except Exception as e:
        return jsonify({'error': str(e)})

# @app.route('/api/update_maptb', methods=['POST']) // 업데이트 필요시 남겨둠
# def update_maptb():
#     try:
#         data = request.get_json()
#         maptb_id = data['id']
#         new_lat = data['lat']
#         new_lng = data['lng']

#         with connection.cursor() as cursor:
#             sql = "UPDATE maptb SET lat = %s, lng = %s WHERE id = %s"
#             cursor.execute(sql, (new_lat, new_lng, maptb_id))
#             connection.commit()

#         return jsonify({'success': True})
#     except Exception as e:
#         return jsonify({'error': str(e)})

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)