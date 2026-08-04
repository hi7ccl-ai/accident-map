import geopandas as gpd
import numpy as np
import pandas as pd
import folium
from folium.plugins import HeatMap
import plotly.express as px
from sklearn.neighbors import BallTree
import streamlit as st
from streamlit_folium import st_folium

# -----------------------------------
# 0. 스트림릿 페이지 및 데이터 로드
# -----------------------------------
st.set_page_config(page_title="대전 교통사고 분석지도", layout="wide")
st.title("🚨 대전청 교통사고 분석지도")


# 사고 데이터 불러오기 (Parquet 파일 로드 & 캐싱)
@st.cache_data
def load_data():
    df = pd.read_parquet("정제완료(21~25).parquet")

    # [명칭 변경] 가해/피해차량 차종 명칭 치환 처리
    for col in ["wrngdo_vhcle_asort_dc", "dmge_vhcle_asort_dc"]:
        if col in df.columns:
            df[col] = (
                df[col]
                .astype(str)
                .str.replace("개인형이동수단(PM)", "PM", regex=False)
            )
            df[col] = (
                df[col]
                .astype(str)
                .str.replace("사륜오토바이(ATV)", "ATV", regex=False)
            )

    # 시간대 필터링용 정수형 변환
    if "occrrnc_time_dc" in df.columns:
        df["time_num"] = (
            df["occrrnc_time_dc"]
            .astype(str)
            .str.replace("시", "", regex=False)
        )
        df["time_num"] = (
            pd.to_numeric(df["time_num"], errors="coerce").fillna(0).astype(int)
        )

    return df


df = load_data()


# -----------------------------------
# 0-1. 사고다발지점 연산 함수
# 사용자 지정 반경 및 표시 개수 적용
# 반경 내 사고 행 번호도 함께 저장
# -----------------------------------
def get_top_hotspots(target_df, radius_m, top_n):
    """선택 반경 내 사고 건수가 많은 지점을 상위 개수만큼 추출"""

    if target_df.empty:
        return pd.DataFrame()

    earth_radius_m = 6371000

    # 이후 iloc으로 반경 내 사고를 추출하기 위해
    # 인덱스를 0부터 다시 정리
    df_temp = target_df.reset_index(drop=True).copy()

    coords_rad = np.radians(
        df_temp[["latitude", "longitude"]].to_numpy()
    )

    tree = BallTree(
        coords_rad,
        metric="haversine",
    )

    radius_rad = radius_m / earth_radius_m

    # 각 지점을 중심으로 선택 반경 안에 들어오는
    # 사고 데이터의 위치 인덱스를 반환
    nearby_indices_array = tree.query_radius(
        coords_rad,
        r=radius_rad,
        return_distance=False,
    )

    # 반경 내 사고 건수
    df_temp["nearby_count"] = [
        len(indices)
        for indices in nearby_indices_array
    ]

    # 반경 내 사고의 위치 인덱스
    # 팝업의 가해차량·법규위반 집계에 사용
    df_temp["nearby_indices"] = list(
        nearby_indices_array
    )

    sorted_df = (
        df_temp
        .sort_values(
            by="nearby_count",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    selected_rows = []

    for _, row in sorted_df.iterrows():
        is_far_enough = True

        # 이미 선정된 사고다발지점과 겹치는 후보 제외
        for selected_row in selected_rows:
            lat1 = np.radians(row["latitude"])
            lon1 = np.radians(row["longitude"])
            lat2 = np.radians(selected_row["latitude"])
            lon2 = np.radians(selected_row["longitude"])

            dlat = lat2 - lat1
            dlon = lon2 - lon1

            a = (
                np.sin(dlat / 2) ** 2
                + np.cos(lat1)
                * np.cos(lat2)
                * np.sin(dlon / 2) ** 2
            )

            distance_m = (
                2
                * earth_radius_m
                * np.arcsin(np.sqrt(a))
            )

            if distance_m < radius_m:
                is_far_enough = False
                break

        if is_far_enough:
            selected_rows.append(row)

        if len(selected_rows) >= top_n:
            break

    return pd.DataFrame(selected_rows)


# -----------------------------------
# 1. SHP 파일 및 데이터 읽기 (캐싱 처리)
# -----------------------------------
@st.cache_data
def load_gis_data():
    file_path = "pss/PSS_Daejeon.shp"
    gdf = gpd.read_file(file_path)
    gdf = gdf[gdf.geometry.notnull() & gdf.is_valid]

    gdf["PS_SHORT"] = (
        gdf["PSNAME"]
        .astype(str)
        .str.replace("대전", "")
        .str.replace("경찰서", "")
        .str.strip()
    )

    ps_boundary = gdf.dissolve(by="PS_SHORT").reset_index()

    gdf_center = gdf.to_crs(epsg=5186)
    center_geom = gdf_center.unary_union.centroid
    center_gdf = gpd.GeoSeries([center_geom], crs=5186).to_crs(epsg=4326)
    center_coords = [center_gdf.y.iloc[0], center_gdf.x.iloc[0]]

    gdf = gdf.to_crs(epsg=4326)
    ps_boundary = ps_boundary.to_crs(epsg=4326)

    return gdf, ps_boundary, center_coords


gdf, ps_boundary, center = load_gis_data()

# -----------------------------------
# 2. 스트림릿 사이드바 필터 구현
# -----------------------------------
st.sidebar.header("🔍 분석 필터 설정")

# -----------------------------------
# [순서 1] 관할 경찰서 선택
# 전체 → 중부 → 동부 → 서부 → 대덕 → 둔산 → 유성
# -----------------------------------
station_order = [
    "중부",
    "동부",
    "서부",
    "대덕",
    "둔산",
    "유성",
]

station_values = set(
    df["관할"]
    .dropna()
    .astype(str)
    .unique()
)

station_options = ["전체"] + [
    station
    for station in station_order
    if station in station_values
]

selected_ps = st.sidebar.selectbox(
    "관할 경찰서 선택",
    station_options,
)

# -----------------------------------
# [순서 2] 발생 연도 선택
# -----------------------------------
min_year = (
    int(df["acdnt_year"].min())
    if "acdnt_year" in df.columns
    else 2021
)

max_year = (
    int(df["acdnt_year"].max())
    if "acdnt_year" in df.columns
    else 2025
)

start_year, end_year = st.sidebar.slider(
    "발생 연도 범위 선택",
    min_value=min_year,
    max_value=max_year,
    value=(min_year, max_year),
    step=1,
    format="%d년",
)

# -----------------------------------
# [순서 3] 사고분류 선택
# 사망사고 → 중상사고 → 경상사고 → 부상신고사고
# -----------------------------------
accident_type_order = [
    "사망사고",
    "중상사고",
    "경상사고",
    "부상신고사고",
]

if "acdnt_gae_dc" in df.columns:
    accident_type_values = set(
        df["acdnt_gae_dc"]
        .dropna()
        .astype(str)
        .unique()
    )

    type_options = [
        accident_type
        for accident_type in accident_type_order
        if accident_type in accident_type_values
    ]

else:
    type_options = []

selected_types = st.sidebar.multiselect(
    "사고분류 (복수 선택)",
    type_options,
    placeholder="전체 (미선택 시)",
)

# -----------------------------------
# [순서 4] 사고종별 선택
# 차대차 → 차대사람 → 차량단독
# -----------------------------------
accident_hdc_order = [
    "차대차",
    "차대사람",
    "차량단독",
]

if "acdnt_hdc" in df.columns:
    accident_hdc_values = set(
        df["acdnt_hdc"]
        .dropna()
        .astype(str)
        .unique()
    )

    hdc_options = ["전체"] + [
        accident_hdc
        for accident_hdc in accident_hdc_order
        if accident_hdc in accident_hdc_values
    ]

else:
    hdc_options = ["전체"]

selected_hdc = st.sidebar.selectbox(
    "사고종별",
    hdc_options,
)

# -----------------------------------
# [순서 5] 시간대 선택
# 시작·종료시간을 한 행 2열로 배치
# 시작시간이 종료시간보다 크면 자정을 넘는 시간대로 처리
# -----------------------------------
time_options = list(range(24))
time_col1, time_col2 = st.sidebar.columns(2)

with time_col1:
    start_time = st.selectbox(
        "시작시간",
        options=time_options,
        index=0,
        format_func=lambda x: f"{x:02d}시",
        key="start_time",
    )

with time_col2:
    end_time = st.selectbox(
        "종료시간",
        options=time_options,
        index=23,
        format_func=lambda x: f"{x:02d}시",
        key="end_time",
    )

if start_time <= end_time:
    st.sidebar.caption(
        f"선택 시간대: "
        f"{start_time:02d}시 ~ {end_time:02d}시"
    )

else:
    st.sidebar.caption(
        f"선택 시간대: "
        f"{start_time:02d}시 ~ 익일 {end_time:02d}시"
    )

# -----------------------------------
# [순서 6] 가해차량 차종 선택
# 승용 → 승합 → 화물 → 이륜 → 원동기 →
# ATV → 자전거 → PM
# 보행자는 선택항목에서 제외
# -----------------------------------
wrngdo_vehicle_order = [
    "승용",
    "승합",
    "화물",
    "이륜",
    "원동기",
    "ATV",
    "자전거",
    "PM",
]

if "wrngdo_vhcle_asort_dc" in df.columns:
    wrngdo_vehicle_values = set(
        df["wrngdo_vhcle_asort_dc"]
        .dropna()
        .astype(str)
        .unique()
    )

    wrngdo_options = [
        vehicle
        for vehicle in wrngdo_vehicle_order
        if vehicle in wrngdo_vehicle_values
    ]

else:
    wrngdo_options = wrngdo_vehicle_order

selected_wrngdo = st.sidebar.multiselect(
    "가해차량 차종 (복수 선택)",
    wrngdo_options,
    placeholder="전체 (미선택 시)",
)

# -----------------------------------
# [순서 7] 피해차량 차종 선택
# 보행자 → 승용 → 승합 → 화물 → 이륜 →
# 원동기 → ATV → 자전거 → PM
# -----------------------------------
damage_vehicle_order = [
    "보행자",
    "승용",
    "승합",
    "화물",
    "이륜",
    "원동기",
    "ATV",
    "자전거",
    "PM",
]

if "dmge_vhcle_asort_dc" in df.columns:
    damage_vehicle_values = set(
        df["dmge_vhcle_asort_dc"]
        .dropna()
        .astype(str)
        .unique()
    )

    dmge_options = [
        vehicle
        for vehicle in damage_vehicle_order
        if vehicle in damage_vehicle_values
    ]

else:
    dmge_options = damage_vehicle_order

selected_dmge = st.sidebar.multiselect(
    "피해차량 차종 (복수 선택)",
    dmge_options,
    placeholder="전체 (미선택 시)",
)

# -----------------------------------
# [순서 8] 사망자 유형 선택
# 보행자 → 승용 → 승합 → 화물 → 이륜 →
# 원동기 → ATV → 자전거 → PM → 기타불명
# -----------------------------------
fatal_type_order = [
    "보행자",
    "승용",
    "승합",
    "화물",
    "이륜",
    "원동기",
    "ATV",
    "자전거",
    "PM",
    "기타불명",
]

if "fatal_type" in df.columns:
    fatal_type_values = set(
        df["fatal_type"]
        .dropna()
        .astype(str)
        .unique()
    )

    fatal_type_options = [
        fatal_type
        for fatal_type in fatal_type_order
        if fatal_type in fatal_type_values
    ]

else:
    fatal_type_options = []

selected_fatal_type = st.sidebar.multiselect(
    "사망자 유형 (복수 선택)",
    fatal_type_options,
    placeholder="전체 (미선택 시)",
)

# -----------------------------------
# [순서 9] 사망자 연령대 선택
# 기존 정렬 방식 유지
# -----------------------------------
fatal_age_options = (
    sorted(
        df["fatal_age_group"]
        .dropna()
        .unique()
        .tolist()
    )
    if "fatal_age_group" in df.columns
    else []
)

selected_fatal_age = st.sidebar.multiselect(
    "사망자 연령대 (복수 선택)",
    fatal_age_options,
    placeholder="전체 (미선택 시)",
)

# -----------------------------------
# [순서 10] 날씨 선택
# 맑음 → 흐림 → 안개 → 비 → 눈 → 기타
# -----------------------------------
weather_order = [
    "맑음",
    "흐림",
    "안개",
    "비",
    "눈",
    "기타",
]

if "wether_sttus_dc" in df.columns:
    weather_values = set(
        df["wether_sttus_dc"]
        .dropna()
        .astype(str)
        .unique()
    )

    wether_options = [
        weather
        for weather in weather_order
        if weather in weather_values
    ]

else:
    wether_options = []

selected_wether = st.sidebar.multiselect(
    "날씨 (복수 선택)",
    wether_options,
    placeholder="전체 (미선택 시)",
)

# -----------------------------------
# [순서 11] 법규위반유형 선택
# '기타'는 마지막에 배치
# -----------------------------------
raw_violt_options = (
    df["lrg_violt_1_dc"]
    .dropna()
    .unique()
    .tolist()
    if "lrg_violt_1_dc" in df.columns
    else []
)

violt_options = sorted(
    raw_violt_options,
    key=lambda x: (
        str(x) == "기타",
        str(x),
    ),
)

selected_violt = st.sidebar.multiselect(
    "법규위반유형 (복수 선택)",
    violt_options,
    placeholder="전체 (미선택 시)",
)

# -----------------------------------
# 3. 데이터 동적 필터링 처리
# -----------------------------------
filtered_df = df.copy()

# [필터 1] 관할 경찰서
if selected_ps != "전체":
    filtered_gdf = gdf[gdf["PS_SHORT"] == selected_ps]

    filtered_boundary = ps_boundary[
        ps_boundary["PS_SHORT"] == selected_ps
    ]

    filtered_df = filtered_df[
        filtered_df["관할"] == selected_ps
    ]

    if not filtered_gdf.empty:
        filtered_center_geom = (
            filtered_gdf
            .to_crs(epsg=5186)
            .unary_union
            .centroid
        )

        filtered_center_gdf = gpd.GeoSeries(
            [filtered_center_geom],
            crs=5186,
        ).to_crs(epsg=4326)

        map_center = [
            filtered_center_gdf.y.iloc[0],
            filtered_center_gdf.x.iloc[0],
        ]

        zoom_level = 12

    else:
        map_center = center
        zoom_level = 11

else:
    filtered_gdf = gdf
    filtered_boundary = ps_boundary
    map_center = center
    zoom_level = 11

# [필터 2] 발생 연도
if "acdnt_year" in filtered_df.columns:
    filtered_df = filtered_df[
        (filtered_df["acdnt_year"] >= start_year)
        & (filtered_df["acdnt_year"] <= end_year)
    ]

# [필터 3] 사고분류
if selected_types and "acdnt_gae_dc" in filtered_df.columns:
    filtered_df = filtered_df[
        filtered_df["acdnt_gae_dc"].isin(selected_types)
    ]

# [필터 4] 사고종별
if selected_hdc != "전체" and "acdnt_hdc" in filtered_df.columns:
    filtered_df = filtered_df[
        filtered_df["acdnt_hdc"] == selected_hdc
    ]

# -----------------------------------
# [필터 5] 시간대
# 일반 시간대: 시작시간 이상 AND 종료시간 이하
# 자정 통과: 시작시간 이상 OR 종료시간 이하
# -----------------------------------
if "time_num" in filtered_df.columns:
    if start_time <= end_time:
        filtered_df = filtered_df[
            (filtered_df["time_num"] >= start_time)
            & (filtered_df["time_num"] <= end_time)
        ]

    else:
        filtered_df = filtered_df[
            (filtered_df["time_num"] >= start_time)
            | (filtered_df["time_num"] <= end_time)
        ]

# [필터 6] 가해차량 차종
if (
    selected_wrngdo
    and "wrngdo_vhcle_asort_dc" in filtered_df.columns
):
    filtered_df = filtered_df[
        filtered_df["wrngdo_vhcle_asort_dc"].isin(
            selected_wrngdo
        )
    ]

# [필터 7] 피해차량 차종
if (
    selected_dmge
    and "dmge_vhcle_asort_dc" in filtered_df.columns
):
    filtered_df = filtered_df[
        filtered_df["dmge_vhcle_asort_dc"].isin(
            selected_dmge
        )
    ]

# [필터 8] 사망자 유형
if selected_fatal_type and "fatal_type" in filtered_df.columns:
    filtered_df = filtered_df[
        filtered_df["fatal_type"].isin(
            selected_fatal_type
        )
    ]

# [필터 9] 사망자 연령대
if (
    selected_fatal_age
    and "fatal_age_group" in filtered_df.columns
):
    filtered_df = filtered_df[
        filtered_df["fatal_age_group"].isin(
            selected_fatal_age
        )
    ]

# [필터 10] 날씨
if (
    selected_wether
    and "wether_sttus_dc" in filtered_df.columns
):
    filtered_df = filtered_df[
        filtered_df["wether_sttus_dc"].isin(
            selected_wether
        )
    ]

# [필터 11] 법규위반유형
if (
    selected_violt
    and "lrg_violt_1_dc" in filtered_df.columns
):
    filtered_df = filtered_df[
        filtered_df["lrg_violt_1_dc"].isin(
            selected_violt
        )
    ]

# 히트맵 데이터 가공
filtered_df["latitude"] = pd.to_numeric(
    filtered_df["latitude"], errors="coerce"
)
filtered_df["longitude"] = pd.to_numeric(
    filtered_df["longitude"], errors="coerce"
)
heat_data = filtered_df[["latitude", "longitude"]].dropna().values.tolist()


# -----------------------------------
# 지도 위 사고다발지점 표시 설정
# 입력값은 적용 버튼을 눌렀을 때만 지도에 반영
# 기본값: 반경 100m / 상위 5개
# -----------------------------------
if "hotspot_radius" not in st.session_state:
    st.session_state.hotspot_radius = 100

if "hotspot_top_n" not in st.session_state:
    st.session_state.hotspot_top_n = 5

st.markdown("입력한 반경(m)에 따라, 지정한 수만큼 사고가 많은 지역이 현출됩니다.)")

with st.form("hotspot_settings_form"):
    hotspot_col1, hotspot_col2, hotspot_col3 = st.columns(
        [1, 1, 0.7]
    )

    with hotspot_col1:
        radius_input = st.number_input(
            "분석 반경 (m)",
            min_value=50,
            max_value=300,
            value=int(st.session_state.hotspot_radius),
            step=10,
            format="%d",
        )

    with hotspot_col2:
        top_n_input = st.number_input(
            "사고다발지역 수",
            min_value=1,
            max_value=10,
            value=int(st.session_state.hotspot_top_n),
            step=1,
            format="%d",
        )

    with hotspot_col3:
        # 입력창과 버튼의 세로 위치 맞춤
        st.write("")
        st.write("")

        hotspot_apply = st.form_submit_button(
            "설정 적용",
            use_container_width=True,
        )

# 적용 버튼을 눌렀을 때만 실제 설정값 변경
if hotspot_apply:
    st.session_state.hotspot_radius = int(radius_input)
    st.session_state.hotspot_top_n = int(top_n_input)

hotspot_radius = st.session_state.hotspot_radius
hotspot_top_n = st.session_state.hotspot_top_n

st.caption(
    f"현재 적용값: 반경 {hotspot_radius}m · "
    f"사고다발지점 상위 {hotspot_top_n}개"
)

# -----------------------------------
# 4. Folium 지도 시각화 생성
# 레이어 생성과 지도 추가 순서를 분리
# -----------------------------------

# 배경지도는 LayerControl 범례에서 숨김
m2 = folium.Map(
    location=map_center,
    zoom_start=zoom_level,
    tiles=None,
)

folium.TileLayer(
    tiles="OpenStreetMap",
    name="OpenStreetMap",
    control=False,
).add_to(m2)

# -----------------------------------
# 지도 레이어 생성
# 아래에서는 레이어에 객체만 담고,
# 실제 지도 추가는 마지막에 원하는 범례 순서대로 처리
# -----------------------------------
hotspot_layer = folium.FeatureGroup(
    name=f"사고다발지점 {hotspot_radius}m TOP {hotspot_top_n}",
    show=True,
)

fatal_group = folium.FeatureGroup(
    name="사망사고",
    show=True,
)

heatmap_layer = folium.FeatureGroup(
    name="사고 히트맵",
    show=True,
)

police_boundary_layer = folium.FeatureGroup(
    name="경찰서 경계",
    show=True,
)

district_layer = folium.FeatureGroup(
    name="지구대 관할",
    show=True,
)

# -----------------------------------
# 4-1. 경찰서 경계 레이어 구성
# -----------------------------------
folium.GeoJson(
    filtered_boundary,
    style_function=lambda feature: {
        "fill": False,
        "color": "#404040",
        "weight": 2.5,
    },
    tooltip=folium.GeoJsonTooltip(
        fields=["PSNAME"],
        aliases=["경찰서:"],
        sticky=True,
    ),
).add_to(police_boundary_layer)

# -----------------------------------
# 4-2. 지구대 관할 레이어 구성
# -----------------------------------
folium.GeoJson(
    filtered_gdf,
    style_function=lambda feature: {
        "fillColor": "#BDBDBD",
        "color": "#7A7A7A",
        "weight": 1,
        "fillOpacity": 0.4,
    },
    tooltip=folium.GeoJsonTooltip(
        fields=["PSNAME", "DEPT_NM"],
        aliases=["경찰서:", "지구대:"],
        sticky=True,
    ),
).add_to(district_layer)

# -----------------------------------
# 4-3. 사고 히트맵 레이어 구성
# -----------------------------------
if heat_data:
    HeatMap(
        data=heat_data,
        radius=15,
        blur=18,
        min_opacity=0.25,
    ).add_to(heatmap_layer)

# -----------------------------------
# 사고다발지점 표시
# 사용자 지정 반경 + 사용자 지정 표시 개수
# 반경 내 가해차량 및 법규위반 상위 3개 표시
# -----------------------------------
top_hotspot_df = get_top_hotspots(
    filtered_df,
    hotspot_radius,
    hotspot_top_n,
)

hotspot_layer = folium.FeatureGroup(
    name=f"사고다발지점 {hotspot_radius}m TOP {hotspot_top_n}",
    show=True,
)

if not top_hotspot_df.empty:

    # get_top_hotspots() 함수에서 사용한 데이터와
    # 동일한 행 순서를 유지하기 위해 인덱스 초기화
    hotspot_source_df = filtered_df.reset_index(drop=True)

    for rank, (_, row) in enumerate(
        top_hotspot_df.iterrows(),
        start=1,
    ):
        lat = row["latitude"]
        lon = row["longitude"]
        count = int(row["nearby_count"])

        location_name = row.get(
            "legaldong_name",
            "",
        )

        if pd.isna(location_name):
            location_name = ""

        # -----------------------------------
        # 해당 사고다발지점 반경 내 사고 추출
        # -----------------------------------
        nearby_indices = row["nearby_indices"]

        nearby_accidents = hotspot_source_df.iloc[
            nearby_indices
        ]

                # -----------------------------------
        # 가해차량 상위 3개 집계
        # -----------------------------------
        vehicle_series = (
            nearby_accidents["wrngdo_vhcle_asort_dc"]
            .dropna()
            .astype(str)
            .str.strip()
        )

        vehicle_series = vehicle_series[
            (vehicle_series != "")
            & (vehicle_series.str.lower() != "nan")
        ]

        top3_vehicles = vehicle_series.value_counts().head(3)

        if not top3_vehicles.empty:
            vehicle_html = "<br>".join(
                f"{order}. {vehicle_type} "
                f"<span style='color:#000000; font-weight:normal;'>"
                f"{int(vehicle_count)}건 "
                f"({vehicle_count / count * 100:.1f}%)"
                f"</span>"
                for order, (vehicle_type, vehicle_count)
                in enumerate(top3_vehicles.items(), start=1)
            )
        else:
            vehicle_html = (
                "<span style='color:#777777;'>"
                "집계 가능한 데이터 없음"
                "</span>"
            )

        # -----------------------------------
        # 피해차량 상위 3개 집계
        # -----------------------------------
        damage_vehicle_series = (
            nearby_accidents["dmge_vhcle_asort_dc"]
            .dropna()
            .astype(str)
            .str.strip()
        )

        damage_vehicle_series = damage_vehicle_series[
            (damage_vehicle_series != "")
            & (damage_vehicle_series.str.lower() != "nan")
        ]

        top3_damage_vehicles = (
            damage_vehicle_series
            .value_counts()
            .head(3)
        )

        if not top3_damage_vehicles.empty:
            damage_vehicle_html = "<br>".join(
                f"{order}. {vehicle_type} "
                f"<span style='color:#000000; font-weight:normal;'>"
                f"{int(vehicle_count)}건 "
                f"({vehicle_count / count * 100:.1f}%)"
                f"</span>"
                for order, (vehicle_type, vehicle_count)
                in enumerate(top3_damage_vehicles.items(), start=1)
            )
        else:
            damage_vehicle_html = (
                "<span style='color:#777777;'>"
                "집계 가능한 데이터 없음"
                "</span>"
            )

        # -----------------------------------
        # 법규위반 상위 3개 집계
        # -----------------------------------
        violation_series = (
            nearby_accidents["lrg_violt_1_dc"]
            .dropna()
            .astype(str)
            .str.strip()
        )

        violation_series = violation_series[
            (violation_series != "")
            & (violation_series.str.lower() != "nan")
        ]

        top3_violations = violation_series.value_counts().head(3)

        if not top3_violations.empty:
            violation_html = "<br>".join(
                f"{order}. {violation_type} "
                f"<span style='color:#000000; font-weight:normal;'>"
                f"{int(violation_count)}건 "
                f"({violation_count / count * 100:.1f}%)"
                f"</span>"
                for order, (violation_type, violation_count)
                in enumerate(top3_violations.items(), start=1)
            )
        else:
            violation_html = (
                "<span style='color:#777777;'>"
                "집계 가능한 데이터 없음"
                "</span>"
            )

        # -----------------------------------
        # 피해차량 상위 3개 집계
        # dmge_vhcle_asort_dc 사용
        # 항목별 건수와 전체 사고 대비 비율 표시
        # -----------------------------------
        if "dmge_vhcle_asort_dc" in nearby_accidents.columns:
            damage_vehicle_series = (
                nearby_accidents["dmge_vhcle_asort_dc"]
                .dropna()
                .astype(str)
                .str.strip()
            )

            damage_vehicle_series = damage_vehicle_series[
                (damage_vehicle_series != "")
                & (damage_vehicle_series.str.lower() != "nan")
            ]

            top3_damage_vehicles = (
                damage_vehicle_series
                .value_counts()
                .head(3)
            )

        else:
            top3_damage_vehicles = pd.Series(dtype="int64")

        if not top3_damage_vehicles.empty:
            damage_vehicle_html = "<br>".join(
                [
                    (
                        f"{order}. {vehicle_type} "
                        f"<span style='color:#000000;'>"
                        f"{int(vehicle_count)}건 "
                        f"({vehicle_count / count * 100:.1f}%)"
                        f"</span>"
                    )
                    for order, (
                        vehicle_type,
                        vehicle_count,
                    ) in enumerate(
                        top3_damage_vehicles.items(),
                        start=1,
                    )
                ]
            )

        else:
            damage_vehicle_html = (
                "<span style='color:#777777;'>"
                "집계 가능한 데이터 없음"
                "</span>"
            )

        # -----------------------------------
        # 법규위반 상위 3개 집계
        # 항목별 건수와 전체 사고 대비 비율 표시
        # -----------------------------------
        if "lrg_violt_1_dc" in nearby_accidents.columns:
            violation_series = (
                nearby_accidents["lrg_violt_1_dc"]
                .dropna()
                .astype(str)
                .str.strip()
            )

            violation_series = violation_series[
                (violation_series != "")
                & (violation_series.str.lower() != "nan")
            ]

            top3_violations = (
                violation_series
                .value_counts()
                .head(3)
            )

        else:
            top3_violations = pd.Series(dtype="int64")

        if not top3_violations.empty:
            violation_html = "<br>".join(
                [
                    (
                        f"{order}. {violation_type} "
                        f"<span style='color:#000000;'>"
                        f"{int(violation_count)}건 "
                        f"({violation_count / count * 100:.1f}%)"
                        f"</span>"
                    )
                    for order, (
                        violation_type,
                        violation_count,
                    ) in enumerate(
                        top3_violations.items(),
                        start=1,
                    )
                ]
            )

        else:
            violation_html = (
                "<span style='color:#777777;'>"
                "집계 가능한 데이터 없음"
                "</span>"
            )
        
        # -----------------------------------
        # 사용자가 선택한 분석 반경 원 표시
        # -----------------------------------
        folium.Circle(
            location=[lat, lon],
            radius=hotspot_radius,
            color="#1565C0",
            weight=2,
            fill=True,
            fill_color="#42A5F5",
            fill_opacity=0.25,
            tooltip=folium.Tooltip(
                f"<b>{rank}위 사고다발지점</b><br>"
                f"반경 {hotspot_radius}m 내 사고 {count}건",
                sticky=True,
            ),
        ).add_to(hotspot_layer)

                # -----------------------------------
        # 사고다발지점 상세 팝업
        # -----------------------------------
        popup_html = f"""
        <div style="
            width:270px;
            font-size:13px;
            line-height:1.7;
            font-family:Arial, sans-serif;
            color:#000000;
        ">
            <div style="
                color:#1565C0;
                font-size:15px;
                font-weight:bold;
                margin-bottom:6px;
            ">
                사고다발지점 TOP {rank}
            </div>

            <b>위치</b> : {location_name}<br>
            <b>분석반경</b> : {hotspot_radius}m<br>
            <b>사고건수</b> : {count}건

            <div style="
                margin-top:8px;
                padding-top:7px;
                border-top:1px solid #DDDDDD;
                color:#000000;
            ">
                <b>가해차량</b><br>
                {vehicle_html}
            </div>

            <div style="
                margin-top:8px;
                padding-top:7px;
                border-top:1px solid #DDDDDD;
                color:#000000;
            ">
                <b>피해차량</b><br>
                {damage_vehicle_html}
            </div>

            <div style="
                margin-top:8px;
                padding-top:7px;
                border-top:1px solid #DDDDDD;
                color:#000000;
            ">
                <b>법규위반</b><br>
                {violation_html}
            </div>
        </div>
        """

        # -----------------------------------
        # 파란색 원형 순위 배지
        # -----------------------------------
        badge_html = f"""
        <div style="
            width:38px;
            height:38px;
            border-radius:50%;
            background-color:#1565C0;
            border:3px solid #FFFFFF;
            box-shadow:0 2px 7px rgba(0,0,0,0.45);
            color:#FFFFFF;
            font-size:15px;
            font-weight:bold;
            font-family:Arial, sans-serif;
            display:flex;
            align-items:center;
            justify-content:center;
            box-sizing:border-box;
            cursor:pointer;
        ">
            {rank}
        </div>
        """

        folium.Marker(
            location=[lat, lon],
            popup=folium.Popup(
                popup_html,
                max_width=320,
            ),
            tooltip=folium.Tooltip(
                f"사고다발지점 {rank}위 · {count}건",
                sticky=True,
            ),
            icon=folium.DivIcon(
                html=badge_html,
                icon_size=(38, 38),
                icon_anchor=(19, 19),
            ),
        ).add_to(hotspot_layer)

# -----------------------------------
# 4-5. 사망사고 마커 레이어 구성
# SVG 형태의 붉은색 핀 아이콘
# -----------------------------------
if "is_fatal" in filtered_df.columns:
    fatal_df = filtered_df[
        filtered_df["is_fatal"] == 1
    ]

    for _, row in fatal_df.iterrows():
        popup_html = f"""
        <div style="
            width:220px;
            font-size:13px;
            line-height:1.7;
        ">
            <div style="
                color:#B71C1C;
                font-size:15px;
                font-weight:bold;
                margin-bottom:5px;
            ">
                사망사고
            </div>

            <b>관할</b> : {row['관할']}<br>
            <b>발생연도</b> : {row['acdnt_year']}년<br>
            <b>발생일</b> : {row['acdnt_month']} {row['acdnt_day']}<br>
            <b>발생시간</b> : {row['occrrnc_time_dc']}<br>
            <b>사고종별</b> : {row['acdnt_hdc']}<br>
            <b>사망자</b> :
            <span style="
                color:#B71C1C;
                font-weight:bold;
            ">
                {row['fatal_type']} {row['dprs_cnt']}명
            </span>
        </div>
        """

        pin_svg = """
        <div style="
            width:24px;
            height:32px;
            filter:drop-shadow(1px 2px 2px rgba(0,0,0,0.35));
            cursor:pointer;
        ">
            <svg
                xmlns="http://www.w3.org/2000/svg"
                width="24"
                height="32"
                viewBox="0 0 24 32"
            >
                <path
                    d="
                        M12 1
                        C5.9 1 1 5.9 1 12
                        C1 20 12 31 12 31
                        C12 31 23 20 23 12
                        C23 5.9 18.1 1 12 1
                        Z
                    "
                    fill="#D32F2F"
                    stroke="#FFFFFF"
                    stroke-width="1.5"
                />

                <circle
                    cx="12"
                    cy="12"
                    r="5"
                    fill="#FFFFFF"
                />

                <circle
                    cx="12"
                    cy="12"
                    r="2.2"
                    fill="#D32F2F"
                />
            </svg>
        </div>
        """

        folium.Marker(
            location=[
                row["latitude"],
                row["longitude"],
            ],
            popup=folium.Popup(
                popup_html,
                max_width=300,
            ),
            tooltip="사망사고 상세정보",
            icon=folium.DivIcon(
                html=pin_svg,
                icon_size=(24, 32),
                icon_anchor=(12, 32),
            ),
        ).add_to(fatal_group)

# -----------------------------------
# 4-6. 레이어를 원하는 범례 순서대로 지도에 추가
# 1. 사고다발지점
# 2. 사망사고
# 3. 사고 히트맵
# 4. 경찰서 경계
# 5. 지구대 관할
# -----------------------------------
hotspot_layer.add_to(m2)
fatal_group.add_to(m2)
heatmap_layer.add_to(m2)
police_boundary_layer.add_to(m2)
district_layer.add_to(m2)

# 레이어 선택창 추가
folium.LayerControl(
    collapsed=False,
).add_to(m2)

# -----------------------------------
# 5. 메인 레이아웃: 지도 배치
# -----------------------------------
st_folium(m2, use_container_width=True, height=650, returned_objects=[])

# ============================================================
# 6. 하단 레이아웃: 분석 통계 배치
# ============================================================

st.markdown("---")
st.subheader("📊 필터링된 사고 통계", anchor=False)


# ============================================================
# 6-1. 그래프 공통 디자인 설정
# ============================================================

# 일반 그래프용 색상
COLOR_PRIMARY = "#2563EB"
COLOR_SECONDARY = "#14B8A6"
COLOR_PURPLE = "#7C3AED"
COLOR_ORANGE = "#F97316"
COLOR_FATAL = "#DC2626"
COLOR_FATAL_LIGHT = "#F87171"
COLOR_TEXT = "#1F2937"
COLOR_GRID = "#E5E7EB"
COLOR_MUTED = "#64748B"

# 사고분류별 고정 색상
ACCIDENT_CLASS_COLORS = {
    "사망사고": "#DC2626",
    "중상사고": "#F97316",
    "경상사고": "#EAB308",
    "부상신고사고": "#3B82F6",
}

# 사고종별 고정 색상
ACCIDENT_TYPE_COLORS = {
    "차대차": "#2563EB",
    "차대사람": "#F97316",
    "차량단독": "#7C3AED",
}

# 차종 그래프 팔레트
VEHICLE_COLORS = [
    "#2563EB",
    "#14B8A6",
    "#7C3AED",
    "#F97316",
    "#0891B2",
    "#65A30D",
    "#D97706",
    "#DB2777",
    "#64748B",
]

# 사망사고 그래프 팔레트
FATAL_COLORS = [
    "#991B1B",
    "#B91C1C",
    "#DC2626",
    "#EF4444",
    "#F87171",
    "#FCA5A5",
    "#FDBA74",
    "#FB923C",
    "#C2410C",
]


def apply_common_chart_style(
    fig,
    height=320,
    show_legend=False,
    horizontal=False,
    top_margin=20,
):
    """
    Plotly 그래프 공통 스타일 적용
    """

    fig.update_layout(
        height=height,
        margin=dict(
            l=25,
            r=35,
            t=top_margin,
            b=30,
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(
            family='"Malgun Gothic", "Apple SD Gothic Neo", sans-serif',
            size=14,
            color=COLOR_TEXT,
        ),
        showlegend=show_legend,
        hoverlabel=dict(
            bgcolor="white",
            bordercolor="#CBD5E1",
            font=dict(
                size=14,
                color=COLOR_TEXT,
                family='"Malgun Gothic", sans-serif',
            ),
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=13),
            title=None,
        ),
    )

    fig.update_xaxes(
        title_text="",
        showline=True,
        linewidth=1,
        linecolor="#CBD5E1",
        tickfont=dict(
            size=13,
            color=COLOR_TEXT,
        ),
        title_font=dict(size=14),
        automargin=True,
        zeroline=False,
    )

    fig.update_yaxes(
        title_text="",
        tickfont=dict(
            size=13,
            color=COLOR_TEXT,
        ),
        title_font=dict(size=14),
        automargin=True,
        zeroline=False,
    )

    if horizontal:
        fig.update_xaxes(
            showgrid=True,
            gridcolor=COLOR_GRID,
            gridwidth=1,
        )
        fig.update_yaxes(showgrid=False)

    else:
        fig.update_xaxes(showgrid=False)
        fig.update_yaxes(
            showgrid=True,
            gridcolor=COLOR_GRID,
            gridwidth=1,
        )

    return fig


def render_plotly_chart(fig):
    """
    Streamlit Plotly 출력 공통 설정
    """
    st.plotly_chart(
        fig,
        use_container_width=True,
        config={
            "displayModeBar": False,
            "responsive": True,
        },
    )


# ============================================================
# 6-2. 기본 정보 카드
# ============================================================

info_col1, info_col2 = st.columns(2)

with info_col1:
    st.metric(
        label="선택된 관할",
        value=selected_ps,
    )

with info_col2:
    st.metric(
        label="분석된 사고 건수",
        value=f"{len(heat_data):,}건",
    )

st.markdown("##### 📈 세부 필터 항목별 사고 현황")


# ============================================================
# 6-3. 필터링 결과가 있는 경우
# ============================================================

if not filtered_df.empty:

    # ========================================================
    # 첫 번째 줄: 연도별 / 사고분류 / 사고종별
    # ========================================================

    row1_col1, row1_col2, row1_col3 = st.columns(3)

    # --------------------------------------------------------
    # 발생 연도별 추이
    # --------------------------------------------------------
    with row1_col1:
        st.markdown("**📅 발생 연도별 추이**")

        year_counts = (
            filtered_df["acdnt_year"]
            .value_counts()
            .sort_index()
            .rename_axis("연도")
            .reset_index(name="사고건수")
        )

        fig_line = px.line(
            year_counts,
            x="연도",
            y="사고건수",
            markers=True,
        )

        fig_line.update_traces(
            line=dict(
                color=COLOR_PRIMARY,
                width=3,
            ),
            marker=dict(
                size=8,
                color="white",
                line=dict(
                    color=COLOR_PRIMARY,
                    width=2.5,
                ),
            ),
            hovertemplate=(
                "<b>%{x}년</b><br>"
                "사고 건수: %{y:,}건"
                "<extra></extra>"
            ),
        )

        fig_line.update_xaxes(
            dtick=1,
        )

        fig_line = apply_common_chart_style(
            fig_line,
            height=300,
        )

        render_plotly_chart(fig_line)

    # --------------------------------------------------------
    # 사고분류 도넛차트
    # --------------------------------------------------------
    with row1_col2:
        st.markdown("**📂 사고 분류 비율**")

        accident_class_order = [
            "사망사고",
            "중상사고",
            "경상사고",
            "부상신고사고",
        ]

        type_counts = (
            filtered_df["acdnt_gae_dc"]
            .value_counts()
            .reindex(accident_class_order)
            .dropna()
            .rename_axis("사고분류")
            .reset_index(name="사고건수")
        )

        fig_type = px.pie(
            type_counts,
            values="사고건수",
            names="사고분류",
            hole=0.58,
            color="사고분류",
            color_discrete_map=ACCIDENT_CLASS_COLORS,
        )

        fig_type.update_traces(
            textposition="outside",
            textinfo="percent",
            textfont=dict(size=13),
            marker=dict(
                line=dict(
                    color="white",
                    width=2,
                )
            ),
            hovertemplate=(
                "<b>%{label}</b><br>"
                "사고 건수: %{value:,}건<br>"
                "비율: %{percent}"
                "<extra></extra>"
            ),
        )

        fig_type.update_layout(
            height=300,
            margin=dict(
                l=20,
                r=20,
                t=10,
                b=10,
            ),
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(
                family='"Malgun Gothic", sans-serif',
                size=13,
                color=COLOR_TEXT,
            ),
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="top",
                y=-0.02,
                xanchor="center",
                x=0.5,
                font=dict(size=12),
                title=None,
            ),
            uniformtext=dict(
                mode="hide",
                minsize=12,
            ),
        )

        fig_type.add_annotation(
            x=0.5,
            y=0.5,
            text=(
                f"<b>{type_counts['사고건수'].sum():,}</b>"
                "<br><span style='font-size:12px'>전체 사고</span>"
            ),
            showarrow=False,
            font=dict(
                size=18,
                color=COLOR_TEXT,
            ),
        )

        render_plotly_chart(fig_type)

    # --------------------------------------------------------
    # 사고종별 도넛차트
    # --------------------------------------------------------
    with row1_col3:
        st.markdown("**🚑 사고 종별 비율**")

        accident_type_order = [
            "차대차",
            "차대사람",
            "차량단독",
        ]

        hdc_counts = (
            filtered_df["acdnt_hdc"]
            .value_counts()
            .reindex(accident_type_order)
            .dropna()
            .rename_axis("사고종별")
            .reset_index(name="사고건수")
        )

        fig_hdc = px.pie(
            hdc_counts,
            values="사고건수",
            names="사고종별",
            hole=0.58,
            color="사고종별",
            color_discrete_map=ACCIDENT_TYPE_COLORS,
        )

        fig_hdc.update_traces(
            textposition="outside",
            textinfo="percent",
            textfont=dict(size=13),
            marker=dict(
                line=dict(
                    color="white",
                    width=2,
                )
            ),
            hovertemplate=(
                "<b>%{label}</b><br>"
                "사고 건수: %{value:,}건<br>"
                "비율: %{percent}"
                "<extra></extra>"
            ),
        )

        fig_hdc.update_layout(
            height=300,
            margin=dict(
                l=20,
                r=20,
                t=10,
                b=10,
            ),
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(
                family='"Malgun Gothic", sans-serif',
                size=13,
                color=COLOR_TEXT,
            ),
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="top",
                y=-0.02,
                xanchor="center",
                x=0.5,
                font=dict(size=12),
                title=None,
            ),
            uniformtext=dict(
                mode="hide",
                minsize=12,
            ),
        )

        fig_hdc.add_annotation(
            x=0.5,
            y=0.5,
            text=(
                f"<b>{hdc_counts['사고건수'].sum():,}</b>"
                "<br><span style='font-size:12px'>전체 사고</span>"
            ),
            showarrow=False,
            font=dict(
                size=18,
                color=COLOR_TEXT,
            ),
        )

        render_plotly_chart(fig_hdc)

    # ========================================================
    # 두 번째 줄: 차량 분석
    # ========================================================

    st.markdown("---")
    st.markdown("##### 🚗 차량 분석")

    row2_col1, row2_col2 = st.columns(2)

    # --------------------------------------------------------
# 가해차량 차종
# --------------------------------------------------------
    with row2_col1:
        st.markdown("**🚗 가해차량 차종**")

        wrngdo_counts = (
            filtered_df["wrngdo_vhcle_asort_dc"]
            .dropna()
            .value_counts()
            .rename_axis("차종")
            .reset_index(name="사고건수")
            .sort_values("사고건수", ascending=False)
        )

        fig_wrngdo = px.bar(
            wrngdo_counts,
            x="사고건수",
            y="차종",
            orientation="h",
            color="차종",
            color_discrete_sequence=VEHICLE_COLORS,
            text="사고건수",
        )

        fig_wrngdo.update_traces(
            texttemplate="%{text:,}",
            textposition="outside",
            textfont=dict(size=13),
            cliponaxis=False,
            marker_line_width=0,
            hovertemplate=(
                "<b>%{y}</b><br>"
                "사고 건수: %{x:,}건"
                "<extra></extra>"
            ),
        )

        fig_wrngdo = apply_common_chart_style(
            fig_wrngdo,
            height=max(330, len(wrngdo_counts) * 36),
            horizontal=True,
        )

        # 건수가 가장 많은 차종을 위쪽에 표시
        fig_wrngdo.update_yaxes(
            autorange="reversed",
            categoryorder="array",
            categoryarray=wrngdo_counts["차종"].tolist(),
        )

        render_plotly_chart(fig_wrngdo)

    # --------------------------------------------------------
# 피해차량 차종
# --------------------------------------------------------
    with row2_col2:
        st.markdown("**🚙 피해차량 차종**")

        # 실제 결측값 제거
        dmge_series = (
            filtered_df["dmge_vhcle_asort_dc"]
            .dropna()
        )

        # 문자열로 저장된 nan도 제거
        dmge_series = dmge_series[
            dmge_series.astype(str)
            .str.strip()
            .str.lower()
            .ne("nan")
        ]

        dmge_counts = (
            dmge_series
            .value_counts()
            .rename_axis("차종")
            .reset_index(name="사고건수")
            .sort_values("사고건수", ascending=False)
        )

        fig_dmge = px.bar(
            dmge_counts,
            x="사고건수",
            y="차종",
            orientation="h",
            color="차종",
            color_discrete_sequence=VEHICLE_COLORS,
            text="사고건수",
        )

        fig_dmge.update_traces(
            texttemplate="%{text:,}",
            textposition="outside",
            textfont=dict(size=13),
            cliponaxis=False,
            marker_line_width=0,
            hovertemplate=(
                "<b>%{y}</b><br>"
                "사고 건수: %{x:,}건"
                "<extra></extra>"
            ),
        )

        fig_dmge = apply_common_chart_style(
            fig_dmge,
            height=max(330, len(dmge_counts) * 36),
            horizontal=True,
        )

        # 건수가 가장 많은 차종을 위쪽에 표시
        fig_dmge.update_yaxes(
            autorange="reversed",
            categoryorder="array",
            categoryarray=dmge_counts["차종"].tolist(),
        )

        render_plotly_chart(fig_dmge)

    # ========================================================
    # 세 번째 줄: 사망사고 분석
    # ========================================================

    st.markdown("---")
    st.markdown("##### 🚨 사망사고 분석")

    row3_col1, row3_col2 = st.columns(2)


# --------------------------------------------------------
# 사망자 유형
# --------------------------------------------------------
# --------------------------------------------------------
# 사망자 유형
# --------------------------------------------------------
    with row3_col1:
        st.markdown("**🚨 사망자 유형**")

        fatal_type_counts = (
            filtered_df["fatal_type"]
            .dropna()
            .value_counts()
            .rename_axis("사망자 유형")
            .reset_index(name="사망사고 건수")
            .sort_values("사망사고 건수", ascending=False)
        )

        if not fatal_type_counts.empty:
            fig_fatal_type = px.bar(
                fatal_type_counts,
                x="사망사고 건수",
                y="사망자 유형",
                orientation="h",
                color="사망자 유형",
                color_discrete_sequence=FATAL_COLORS,
                text="사망사고 건수",
            )

            fig_fatal_type.update_traces(
                texttemplate="%{text:,}",
                textposition="outside",
                textfont=dict(size=13),
                cliponaxis=False,
                marker_line_width=0,
                hovertemplate=(
                    "<b>%{y}</b><br>"
                    "사망사고 건수: %{x:,}건"
                    "<extra></extra>"
                ),
            )

            fig_fatal_type = apply_common_chart_style(
                fig_fatal_type,
                height=max(330, len(fatal_type_counts) * 38),
                horizontal=True,
            )

            # 건수가 가장 많은 유형부터 위쪽에 표시
            fig_fatal_type.update_yaxes(
                autorange="reversed",
                categoryorder="array",
                categoryarray=fatal_type_counts[
                    "사망자 유형"
                ].tolist(),
            )

            render_plotly_chart(fig_fatal_type)

        else:
            st.info(
                "선택된 조건에 해당하는 "
                "사망자 유형 데이터가 없습니다."
            )


    # --------------------------------------------------------
    # 사망자 연령대
    # --------------------------------------------------------
    with row3_col2:
        st.markdown("**👤 사망자 연령대**")

        fatal_age_order = [
            "20세 이하",
            "21-30세",
            "31-40세",
            "41-50세",
            "51-60세",
            "61-70세",
            "71세 이상",
            "미상",
        ]

        fatal_age_counts = (
            filtered_df["fatal_age_group"]
            .dropna()
            .value_counts()
            .reindex(fatal_age_order)
            .dropna()
            .rename_axis("사망자 연령대")
            .reset_index(name="사망사고 건수")
        )

        # 가로 막대에서 위쪽부터 낮은 연령대로 보이도록 역순 배치
        fatal_age_counts = fatal_age_counts.iloc[::-1]

        if not fatal_age_counts.empty:

            fig_fatal_age = px.bar(
                fatal_age_counts,
                x="사망사고 건수",
                y="사망자 연령대",
                orientation="h",
                color="사망자 연령대",
                color_discrete_sequence=FATAL_COLORS,
                text="사망사고 건수",
            )

            fig_fatal_age.update_traces(
                texttemplate="%{text:,}",
                textposition="outside",
                textfont=dict(size=13),
                cliponaxis=False,
                marker_line_width=0,
                hovertemplate=(
                    "<b>%{y}</b><br>"
                    "사망사고 건수: %{x:,}건"
                    "<extra></extra>"
                ),
            )

            fig_fatal_age = apply_common_chart_style(
                fig_fatal_age,
                height=max(330, len(fatal_age_counts) * 38),
                horizontal=True,
            )

            render_plotly_chart(fig_fatal_age)

        else:
            st.info(
                "선택된 조건에 해당하는 "
                "사망자 연령대 데이터가 없습니다."
            )

    # ========================================================
    # 네 번째 줄: 상황별 사고 분석
    # ========================================================

    st.markdown("---")
    st.markdown("##### ⏱️ 상황별 사고 분석")

    row4_col1, row4_col2 = st.columns(2)

    # --------------------------------------------------------
    # 시간대별
    # --------------------------------------------------------
    with row4_col1:
        st.markdown("**⏰ 시간대별**")

        time_counts = (
            filtered_df["time_num"]
            .value_counts()
            .rename_axis("시간")
            .reset_index(name="사고건수")
        )

        # 자정을 넘는 시간대는 시작시간부터 순서대로 정렬
        if start_time > end_time:
            time_order = (
                list(range(start_time, 24))
                + list(range(0, end_time + 1))
            )

            time_counts["시간순서"] = pd.Categorical(
                time_counts["시간"],
                categories=time_order,
                ordered=True,
            )

            time_counts = (
                time_counts
                .sort_values("시간순서")
                .drop(columns=["시간순서"])
            )

        # 일반 시간대는 숫자 순서대로 정렬
        else:
            time_counts = time_counts.sort_values("시간")

        # 그래프 표시용 라벨
        time_counts["시간표시"] = (
            time_counts["시간"]
            .astype(int)
            .map(lambda x: f"{x:02d}시")
        )

        fig_time = px.bar(
            time_counts,
            x="시간표시",
            y="사고건수",
            text="사고건수",
            color="사고건수",
            color_continuous_scale=[
                [0.0, "#DBEAFE"],
                [0.5, "#60A5FA"],
                [1.0, "#1D4ED8"],
            ],
        )

        fig_time.update_traces(
            texttemplate="%{text:,}",
            textposition="outside",
            textfont=dict(size=12),
            cliponaxis=False,
            marker_line_width=0,
            hovertemplate=(
                "<b>%{x}</b><br>"
                "사고 건수: %{y:,}건"
                "<extra></extra>"
            ),
        )

        fig_time.update_xaxes(
            categoryorder="array",
            categoryarray=time_counts["시간표시"].tolist(),
            tickangle=-45 if len(time_counts) > 12 else 0,
        )

        fig_time.update_coloraxes(showscale=False)

        fig_time = apply_common_chart_style(
            fig_time,
            height=350,
        )

        render_plotly_chart(fig_time)

    # --------------------------------------------------------
    # 법규위반별
    # --------------------------------------------------------
    with row4_col2:
        st.markdown("**⚖️ 법규위반별**")

        violt_counts = (
            filtered_df["lrg_violt_1_dc"]
            .value_counts()
            .rename_axis("법규위반유형")
            .reset_index(name="사고건수")
        )

        # '기타'를 마지막에 배치
        violt_counts["is_etc"] = (
            violt_counts["법규위반유형"]
            .astype(str)
            .eq("기타")
        )

        violt_counts = (
            violt_counts
            .sort_values(
                by=["is_etc", "사고건수"],
                ascending=[False, True],
            )
            .drop(columns=["is_etc"])
        )

        fig_violt = px.bar(
            violt_counts,
            x="사고건수",
            y="법규위반유형",
            orientation="h",
            text="사고건수",
            color="사고건수",
            color_continuous_scale=[
                [0.0, "#FEF3C7"],
                [0.5, "#FB923C"],
                [1.0, "#C2410C"],
            ],
        )

        fig_violt.update_traces(
            texttemplate="%{text:,}",
            textposition="outside",
            textfont=dict(size=13),
            cliponaxis=False,
            marker_line_width=0,
            hovertemplate=(
                "<b>%{y}</b><br>"
                "사고 건수: %{x:,}건"
                "<extra></extra>"
            ),
        )

        fig_violt.update_coloraxes(showscale=False)

        fig_violt = apply_common_chart_style(
            fig_violt,
            height=max(350, len(violt_counts) * 38),
            horizontal=True,
        )

        render_plotly_chart(fig_violt)


# ============================================================
# 6-4. 필터링 결과가 없는 경우
# ============================================================

else:
    st.info(
        "선택된 필터 조건에 부합하는 데이터가 존재하지 않아 "
        "그래프를 표시할 수 없습니다."
    )