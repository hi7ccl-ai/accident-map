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
st.title("🚨 대전청 교통사고 분석지도(히트맵)")


# 사고 데이터 불러오기 (Parquet 파일 로드 & 캐싱)
@st.cache_data
def load_data():
    df = pd.read_parquet("data_cleaned.parquet")

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
## -----------------------------------
# 0-1. 핫스팟 (반경 100m TOP 10) 연산 함수
# -----------------------------------
def get_top10_hotspots_100m(target_df):
    """필터링된 데이터에서 반경 100m 이내 사고 건수가 가장 많은 TOP 10 지점 추출"""
    if len(target_df) == 0:
        return pd.DataFrame()

    EARTH_RADIUS_METERS = 6371000
    coords_rad = np.radians(target_df[['latitude', 'longitude']].to_numpy())

    tree = BallTree(coords_rad, metric='haversine')
    radius_rad = 100.0 / EARTH_RADIUS_METERS

    counts = tree.query_radius(coords_rad, r=radius_rad, count_only=True)

    df_temp = target_df.copy()
    df_temp['nearby_100m_count'] = counts

    top10_list = []
    sorted_df = df_temp.sort_values(
        by='nearby_100m_count', ascending=False
    ).reset_index(drop=True)

    for idx, row in sorted_df.iterrows():
        if len(top10_list) == 0:
            top10_list.append(row)
        else:
            is_far_enough = True
            for top_row in top10_list:
                lat_diff = abs(row['latitude'] - top_row['latitude'])
                lon_diff = abs(row['longitude'] - top_row['longitude'])
                if lat_diff < 0.0009 and lon_diff < 0.0011:  # 약 100m 이내 중복 제외
                    is_far_enough = False
                    break
            if is_far_enough:
                top10_list.append(row)

        if len(top10_list) == 10:  # 상위 10개 추출
            break

    return pd.DataFrame(top10_list)


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

# [순서 1] 관할 경찰서 선택 (단일 선택)
station_options = ["전체"] + sorted(df["관할"].dropna().unique().tolist())
selected_ps = st.sidebar.selectbox("관할 경찰서 선택", station_options)

# [순서 2] 발생 연도 선택 (슬라이더 바)
min_year = int(df["acdnt_year"].min()) if "acdnt_year" in df.columns else 2021
max_year = int(df["acdnt_year"].max()) if "acdnt_year" in df.columns else 2025
start_year, end_year = st.sidebar.slider(
    "발생 연도 범위 선택",
    min_value=min_year,
    max_value=max_year,
    value=(min_year, max_year),
    step=1,
    format="%d년",
)

# [순서 3] 사고분류 선택 (복수 선택)
type_options = (
    sorted(df["acdnt_gae_dc"].dropna().unique().tolist())
    if "acdnt_gae_dc" in df.columns
    else []
)
selected_types = st.sidebar.multiselect(
    "사고분류 (복수 선택)", type_options, placeholder="전체 (미선택 시)"
)

# [순서 4] 사고종별 선택 (단일 선택)
hdc_options = (
    ["전체"] + sorted(df["acdnt_hdc"].dropna().unique().tolist())
    if "acdnt_hdc" in df.columns
    else ["전체"]
)
selected_hdc = st.sidebar.selectbox("사고종별", hdc_options)

# [순서 5] 시간대 선택 범위 슬라이더 바
start_time, end_time = st.sidebar.slider(
    "시간대 범위 선택 (시)",
    min_value=0,
    max_value=23,
    value=(0, 23),
    format="%d시",
)

# [순서 6] 가해차량 차종 선택 (복수 선택)
vhcle_options = [
    "승용",
    "승합",
    "화물",
    "이륜",
    "원동기",
    "자전거",
    "PM",
    "보행자",
    "ATV",
]
selected_wrngdo = st.sidebar.multiselect(
    "가해차량 차종 (복수 선택)",
    vhcle_options,
    placeholder="전체 (미선택 시)",
)

# [순서 7] 피해차량 차종 선택 (복수 선택)
selected_dmge = st.sidebar.multiselect(
    "피해차량 차종 (복수 선택)",
    vhcle_options,
    placeholder="전체 (미선택 시)",
)

# [순서 8] 피해자 연령대 선택 (복수 선택)
age_options = (
    sorted(df["acdnt_age_2_dc"].dropna().unique().tolist())
    if "acdnt_age_2_dc" in df.columns
    else []
)
selected_age = st.sidebar.multiselect(
    "피해자 연령대 (복수 선택)", age_options, placeholder="전체 (미선택 시)"
)

# [순서 9] 날씨 선택 (복수 선택)
wether_options = (
    sorted(df["wether_sttus_dc"].dropna().unique().tolist())
    if "wether_sttus_dc" in df.columns
    else []
)
selected_wether = st.sidebar.multiselect(
    "날씨 (복수 선택)", wether_options, placeholder="전체 (미선택 시)"
)

# [순서 10] 법규위반유형 선택 (복수 선택)
raw_violt_options = (
    df["lrg_violt_1_dc"].dropna().unique().tolist()
    if "lrg_violt_1_dc" in df.columns
    else []
)
violt_options = sorted(raw_violt_options, key=lambda x: (x == "기타", x))
selected_violt = st.sidebar.multiselect(
    "법규위반유형 (복수 선택)",
    violt_options,
    placeholder="전체 (미선택 시)",
)

# -----------------------------------
# 3. 데이터 동적 필터링 처리
# -----------------------------------
filtered_df = df.copy()

# [필터 1] 관할 경찰서 조건 적용
if selected_ps != "전체":
    filtered_gdf = gdf[gdf["PS_SHORT"] == selected_ps]
    filtered_boundary = ps_boundary[ps_boundary["PS_SHORT"] == selected_ps]
    filtered_df = filtered_df[filtered_df["관할"] == selected_ps]

    if not filtered_gdf.empty:
        filtered_center_geom = filtered_gdf.to_crs(
            epsg=5186
        ).unary_union.centroid
        filtered_center_gdf = gpd.GeoSeries(
            [filtered_center_geom], crs=5186
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

# [필터 2] 연도 범위 조건 적용
if "acdnt_year" in filtered_df.columns:
    filtered_df = filtered_df[
        (filtered_df["acdnt_year"] >= start_year)
        & (filtered_df["acdnt_year"] <= end_year)
    ]

# [필터 3] 사고분류 조건 적용
if selected_types and "acdnt_gae_dc" in filtered_df.columns:
    filtered_df = filtered_df[filtered_df["acdnt_gae_dc"].isin(selected_types)]

# [필터 4] 사고종별 조건 적용
if selected_hdc != "전체" and "acdnt_hdc" in filtered_df.columns:
    filtered_df = filtered_df[filtered_df["acdnt_hdc"] == selected_hdc]

# [필터 5] 시간대 범위 조건 적용
if "time_num" in filtered_df.columns:
    filtered_df = filtered_df[
        (filtered_df["time_num"] >= start_time)
        & (filtered_df["time_num"] <= end_time)
    ]

# [필터 6] 가해차량 차종 조건 적용
if selected_wrngdo and "wrngdo_vhcle_asort_dc" in filtered_df.columns:
    filtered_df = filtered_df[
        filtered_df["wrngdo_vhcle_asort_dc"].isin(selected_wrngdo)
    ]

# [필터 7] 피해차량 차종 조건 적용
if selected_dmge and "dmge_vhcle_asort_dc" in filtered_df.columns:
    filtered_df = filtered_df[
        filtered_df["dmge_vhcle_asort_dc"].isin(selected_dmge)
    ]

# [필터 8] 피해자 연령대 조건 적용
if selected_age and "acdnt_age_2_dc" in filtered_df.columns:
    filtered_df = filtered_df[filtered_df["acdnt_age_2_dc"].isin(selected_age)]

# [필터 9] 날씨 조건 적용
if selected_wether and "wether_sttus_dc" in filtered_df.columns:
    filtered_df = filtered_df[
        filtered_df["wether_sttus_dc"].isin(selected_wether)
    ]

# [필터 10] 법규위반유형 조건 적용
if selected_violt and "lrg_violt_1_dc" in filtered_df.columns:
    filtered_df = filtered_df[
        filtered_df["lrg_violt_1_dc"].isin(selected_violt)
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
# 4. Folium 지도 시각화 생성
# -----------------------------------
m2 = folium.Map(location=map_center, zoom_start=zoom_level, tiles="OpenStreetMap")

# 지구대 관할 표시
folium.GeoJson(
    filtered_gdf,
    name="지구대 관할",
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
).add_to(m2)

# 경찰서 경계 표시
folium.GeoJson(
    filtered_boundary,
    name="경찰서 경계",
    style_function=lambda feature: {
        "fill": False,
        "color": "#404040",
        "weight": 2.5,
    },
    tooltip=folium.GeoJsonTooltip(
        fields=["PSNAME"], aliases=["경찰서:"], sticky=True
    ),
).add_to(m2)

# 히트맵 추가
if heat_data:
    HeatMap(
        data=heat_data, name="사고 히트맵", radius=15, blur=18, min_opacity=0.25
    ).add_to(m2)

# -----------------------------------
# 반경 100m 사고다발 TOP 10 표시
# 파란색 반경 원 + 원형 순위 배지
# -----------------------------------
top10_df = get_top10_hotspots_100m(filtered_df)

if not top10_df.empty:
    hotspot_layer = folium.FeatureGroup(
        name="사고다발지점 TOP 10",
        show=True,
    )

    for rank, (_, row) in enumerate(top10_df.iterrows(), start=1):
        lat = row["latitude"]
        lon = row["longitude"]
        count = int(row["nearby_100m_count"])

        # 데이터가 없거나 NaN인 경우 빈 문자열로 처리
        location_name = row.get("legaldong_name", "")
        main_cause = row.get("acdnt_mdclrg_violt_1_dc", "")

        if pd.isna(location_name):
            location_name = ""

        if pd.isna(main_cause):
            main_cause = ""

        # 사고다발지점 반경 100m 표시
        folium.Circle(
            location=[lat, lon],
            radius=100,
            color="#1565C0",
            weight=2,
            fill=True,
            fill_color="#42A5F5",
            fill_opacity=0.25,
            tooltip=folium.Tooltip(
                f"<b>{rank}위 사고다발지점</b><br>"
                f"반경 100m 내 사고 {count}건",
                sticky=True,
            ),
        ).add_to(hotspot_layer)

        # 사고다발지점 클릭 시 표시할 팝업
        popup_html = f"""
        <div style="
            width: 220px;
            font-size: 13px;
            line-height: 1.7;
            font-family: Arial, sans-serif;
        ">
            <div style="
                color: #1565C0;
                font-size: 15px;
                font-weight: bold;
                margin-bottom: 6px;
            ">
                사고다발지점 TOP {rank}
            </div>

            <b>위치</b> : {location_name}<br>
            <b>반경</b> : 100m<br>
            <b>사고건수</b> :
            <span style="
                color: #1565C0;
                font-weight: bold;
            ">
                {count}건
            </span><br>
            <b>주요 원인</b> : {main_cause}
        </div>
        """

        # 원형 순위 배지
        badge_html = f"""
        <div style="
            width: 38px;
            height: 38px;
            border-radius: 50%;
            background-color: #1565C0;
            border: 3px solid #FFFFFF;
            box-shadow: 0 2px 7px rgba(0, 0, 0, 0.45);
            color: #FFFFFF;
            font-size: 15px;
            font-weight: bold;
            font-family: Arial, sans-serif;
            display: flex;
            align-items: center;
            justify-content: center;
            box-sizing: border-box;
            cursor: pointer;
        ">
            {rank}
        </div>
        """

        folium.Marker(
            location=[lat, lon],
            popup=folium.Popup(
                popup_html,
                max_width=280,
            ),
            tooltip=folium.Tooltip(
                f"사고다발지점 {rank}위 · {count}건",
                sticky=True,
            ),
            icon=folium.DivIcon(
                html=badge_html,
                icon_size=(38, 38),

                # 원의 정중앙을 실제 좌표에 맞춤
                icon_anchor=(19, 19),
            ),
        ).add_to(hotspot_layer)

    hotspot_layer.add_to(m2)

# -----------------------------------
# 사망사고 마커 표시
# 붉은색 핀 형태 아이콘
# -----------------------------------
if "is_fatal" in filtered_df.columns:
    fatal_df = filtered_df.loc[
        filtered_df["is_fatal"] == 1,
        [
            "latitude",
            "longitude",
            "관할",
            "acdnt_year",
            "acdnt_month",
            "acdnt_day",
            "occrrnc_time_dc",
            "acdnt_hdc",
            "fatal_type",
            "dprs_cnt",
        ],
    ]

    fatal_group = folium.FeatureGroup(
        name="사망사고",
        show=True,
    )

    for row in fatal_df.itertuples(index=False):
        popup_html = f"""
        <div style="width:220px; font-size:13px; line-height:1.7;">
            <div style="
                color:#B71C1C;
                font-size:15px;
                font-weight:bold;
                margin-bottom:5px;
            ">
                사망사고
            </div>
            <b>관할</b> : {row.관할}<br>
            <b>발생연도</b> : {row.acdnt_year}년<br>
            <b>발생일</b> : {row.acdnt_month} {row.acdnt_day}<br>
            <b>발생시간</b> : {row.occrrnc_time_dc}<br>
            <b>사고종별</b> : {row.acdnt_hdc}<br>
            <b>사망자</b> :
            <span style="color:#B71C1C; font-weight:bold;">
                {row.fatal_type} {row.dprs_cnt}명
            </span>
        </div>
        """

        folium.Marker(
            location=[row.latitude, row.longitude],
            popup=folium.Popup(popup_html, max_width=300),
            tooltip="사망사고 상세정보",
            icon=folium.Icon(
                color="red",
                icon="map-marker",
                prefix="fa",
            ),
        ).add_to(fatal_group)

    fatal_group.add_to(m2)

# 레이어 선택창 추가
folium.LayerControl(collapsed=False).add_to(m2)

# -----------------------------------
# 5. 메인 레이아웃: 지도 배치
# -----------------------------------
st_folium(m2, use_container_width=True, height=550, returned_objects=[])

# -----------------------------------
# 6. 하단 레이아웃: 분석 통계 배치
# -----------------------------------
st.markdown("---")
st.subheader("📊 필터링된 사고 통계", anchor=False)

# 1) 기본 정보 가로 2칸 배치
info_col1, info_col2 = st.columns(2)
with info_col1:
    st.metric(label="선택된 관할", value=selected_ps)
with info_col2:
    st.metric(label="분석된 사고 건수", value=f"{len(heat_data):,} 건")

st.markdown("##### 📈 세부 필터 항목별 사고 현황")

if not filtered_df.empty:
    # 첫 번째 줄 격자 (가로 3열: 연도별 / 사고 분류 / 사고 종별)
    row1_col1, row1_col2, row1_col3 = st.columns(3)

    with row1_col1:
        st.write("**📅 발생 연도별 추이**")
        year_counts = (
            filtered_df["acdnt_year"].value_counts().sort_index().reset_index()
        )
        year_counts.columns = ["연도", "사고건수"]

        fig_line = px.line(year_counts, x="연도", y="사고건수", markers=True)
        fig_line.update_yaxes(autorange=True, matches=None, title_text="")
        fig_line.update_xaxes(dtick=1)
        fig_line.update_traces(hovertemplate="%{x}년: %{y}건<extra></extra>")
        fig_line.update_layout(margin=dict(l=20, r=20, t=15, b=15), height=260)
        st.plotly_chart(fig_line, use_container_width=True)

    with row1_col2:
        st.write("**📂 사고 분류 비율**")
        type_counts = filtered_df["acdnt_gae_dc"].value_counts().reset_index()

        fig_type = px.pie(
            type_counts, values="count", names="acdnt_gae_dc", hole=0.3
        )
        fig_type.update_traces(
            textposition="inside",
            textinfo="label+percent",
            insidetextfont=dict(size=11),
            hovertemplate="%{label}: %{value}건 (%{percent})<extra></extra>",
        )
        fig_type.update_layout(
            margin=dict(l=10, r=10, t=10, b=10),
            height=260,
            showlegend=False,
            uniformtext=dict(mode="hide", minsize=11),
        )
        st.plotly_chart(fig_type, use_container_width=True)

    with row1_col3:
        st.write("**🚑 사고 종별 비율**")
        hdc_counts = filtered_df["acdnt_hdc"].value_counts().reset_index()

        fig_hdc = px.pie(
            hdc_counts, values="count", names="acdnt_hdc", hole=0.3
        )
        fig_hdc.update_traces(
            textposition="inside",
            textinfo="label+percent",
            insidetextfont=dict(size=11),
            hovertemplate="%{label}: %{value}건 (%{percent})<extra></extra>",
        )
        fig_hdc.update_layout(
            margin=dict(l=10, r=10, t=10, b=10),
            height=260,
            showlegend=False,
            uniformtext=dict(mode="hide", minsize=11),
        )
        st.plotly_chart(fig_hdc, use_container_width=True)

    # 두 번째 줄 격자 (가로 3열 확장: 가해 / 피해 / 피해자 연령대)
    st.write("---")
    st.markdown("##### 🏢 차량 및 피해 대상 분석")
    row2_col1, row2_col2, row2_col3 = st.columns(3)

    with row2_col1:
        st.write("**🚗 가해차량 차종**")
        wrngdo_counts = (
            filtered_df["wrngdo_vhcle_asort_dc"].value_counts().reset_index()
        )
        wrngdo_counts.columns = ["차종", "사고건수"]

        fig_wrngdo = px.bar(wrngdo_counts, x="차종", y="사고건수")
        fig_wrngdo.update_yaxes(title_text="")
        fig_wrngdo.update_traces(hovertemplate="%{x}: %{y}건<extra></extra>")
        fig_wrngdo.update_layout(
            margin=dict(l=20, r=20, t=15, b=15), height=260
        )
        st.plotly_chart(fig_wrngdo, use_container_width=True)

    with row2_col2:
        st.write("**🚶 피해차량 차종**")
        dmge_counts = (
            filtered_df["dmge_vhcle_asort_dc"].value_counts().reset_index()
        )
        dmge_counts.columns = ["차종", "사고건수"]

        fig_dmge = px.bar(dmge_counts, x="차종", y="사고건수")
        fig_dmge.update_yaxes(title_text="")
        fig_dmge.update_traces(hovertemplate="%{x}: %{y}건<extra></extra>")
        fig_dmge.update_layout(margin=dict(l=20, r=20, t=15, b=15), height=260)
        st.plotly_chart(fig_dmge, use_container_width=True)

    with row2_col3:
        st.write("**👵 피해자 연령대**")
        age_counts = (
            filtered_df["acdnt_age_2_dc"].value_counts().reset_index()
        )
        age_counts.columns = ["연령대", "사고건수"]

        fig_age = px.bar(age_counts, x="연령대", y="사고건수")
        fig_age.update_yaxes(title_text="")
        fig_age.update_traces(hovertemplate="%{x}: %{y}건<extra></extra>")
        fig_age.update_layout(margin=dict(l=20, r=20, t=15, b=15), height=260)
        st.plotly_chart(fig_age, use_container_width=True)

    # 세 번째 줄 격자 (가로 2열: 시간대별 vs 법규위반유형)
    st.write("---")
    st.markdown("##### ⏱️ 상황별 사고 분석")
    row3_col1, row3_col2 = st.columns(2)

    with row3_col1:
        st.write("**⏰ 시간대별**")
        time_counts = (
            filtered_df["time_num"].value_counts().sort_index().reset_index()
        )
        time_counts.columns = ["시간", "사고건수"]

        fig_time = px.bar(time_counts, x="시간", y="사고건수")
        fig_time.update_yaxes(title_text="")
        fig_time.update_xaxes(dtick=2, ticksuffix="시")
        fig_time.update_traces(hovertemplate="%{x}: %{y}건<extra></extra>")
        fig_time.update_layout(margin=dict(l=20, r=20, t=15, b=15), height=260)
        st.plotly_chart(fig_time, use_container_width=True)

    with row3_col2:
        st.write("**⚖️ 법규위반별**")
        violt_counts = (
            filtered_df["lrg_violt_1_dc"].value_counts().reset_index()
        )
        violt_counts.columns = ["법규위반유형", "사고건수"]

        # '기타' 항목 정렬용 보조 컬럼 생성 (Categorical 타입 MultiIndex 오류 방지)
        violt_counts["is_etc"] = (violt_counts["법규위반유형"].astype(str) == "기타")
        violt_counts = violt_counts.sort_values(
            by=["is_etc", "사고건수"], 
            ascending=[True, False]
        ).drop(columns=["is_etc"])

        fig_violt = px.bar(violt_counts, x="법규위반유형", y="사고건수")
        fig_violt.update_yaxes(title_text="")
        fig_violt.update_traces(hovertemplate="%{x}: %{y}건<extra></extra>")
        fig_violt.update_layout(
            margin=dict(l=20, r=20, t=15, b=15), height=260
        )
        st.plotly_chart(fig_violt, use_container_width=True)

        
else:
    st.info("선택된 필터 조건에 부합하는 데이터가 존재하지 않아 그래프를 표시할 수 없습니다.")