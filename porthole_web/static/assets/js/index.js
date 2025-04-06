let maps = {};
let markers = {}; 

function updateFooterInfo() { // 맵과 마커 정보 이용하려고 만든 함수입니다. 이전에 짰던 건데 지금은 mysql로 가져와서 안봐도 될 것 같습니다.
    let footerMaps = document.getElementById("footer-maps-info");
    let footerMarkers = document.getElementById("footer-markers-info");

    if (!footerMaps || !footerMarkers) return; // footer가 없으면 에러 방지

    let mapsInfo = "지도 정보:<br>";
    for (let mapId in maps) {
        let center = maps[mapId].getCenter();
        mapsInfo += `${mapId}: lat = ${center.lat()}, lng = ${center.lng()}<br>`;
    }

    let markersInfo = "마커 정보:<br>";
    for (let mapId in markers) {
        markers[mapId].forEach((marker, index) => {
            let pos = marker.getPosition();
            markersInfo += `${mapId} - 마커 ${index + 1}: lat = ${pos.lat()}, lng = ${pos.lng()}<br>`;
        });
    }

    footerMaps.innerHTML = mapsInfo;
    footerMarkers.innerHTML = markersInfo;
}

function initMap() { //map과 marker를 표시하는 함수입니다.
    let mapElements = document.querySelectorAll('.map'); //class가 맵인 요소를 가져옵니다.

    mapElements.forEach(function(mapElement) { // 해당 요소에서 id, lat, lng를 추출합니다.
        let lat = parseFloat(mapElement.getAttribute('lat'));
        let lng = parseFloat(mapElement.getAttribute('lng'));
        let mapId = mapElement.getAttribute('id');

        let position = { lat: lat, lng: lng };

        if (mapId.startsWith('map')) { //해당 id가 맵이면 중앙에 해당 position과 zoom정보로 맵 표시합니다.
            maps[mapId] = new google.maps.Map(mapElement, {
                zoom: 16,
                center: position
            });

            markers[mapId] = []; // 아래에 marker를 찍기 위해 mapId별 markers 배열 초기화

            
            let mainMarker = new google.maps.Marker({ // 해당 맵에 기본 마커를 추가합니다. (빨간색)
                position: position,
                map: maps[mapId],
                icon: "http://maps.google.com/mapfiles/ms/icons/red-dot.png"
            });

            markers[mapId].push(mainMarker); 
        } 
    });

    let markerElements = document.querySelectorAll('.marker'); // class가 marker인 요소를 가져옵니다.

    markerElements.forEach(function(markerElement) { 
        let lat = parseFloat(markerElement.getAttribute('lat'));
        let lng = parseFloat(markerElement.getAttribute('lng'));
        let targetMapId = markerElement.getAttribute('data-map');

        let position = { lat: lat, lng: lng };

        if (maps[targetMapId]) { //위에 maps 배열 안에 id가 data-map의 id와 일치하면 마커로 표시합니다.
            let marker = new google.maps.Marker({
                position: position,
                map: maps[targetMapId],
                icon: "http://maps.google.com/mapfiles/ms/icons/blue-dot.png"
            });

            markers[targetMapId].push(marker);
        }
    });

    updateFooterInfo();
}
