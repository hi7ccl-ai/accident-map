import geopandas as gpd
import hashlib
import json
import urllib.error
import urllib.parse
import urllib.request
import numpy as np
import pandas as pd
import folium
from folium.plugins import HeatMap
import plotly.express as px
from sklearn.neighbors import BallTree
import streamlit as st
from streamlit_folium import st_folium

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

# -----------------------------------
# 0. 스트림릿 페이지 및 데이터 로드
# -----------------------------------
st.set_page_config(page_title="대전 교통사고 분석지도", layout="wide")
st.title("🚨 대전청 교통사고 분석지도")

# ============================================================
# 공통 UI 디자인
# - 사이드바 배경·그룹 카드·버튼
# - KPI 아래 현재 검색조건 박스
# ============================================================
st.markdown(
    """
    <style>
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #F8FAFC 0%, #F1F5F9 100%);
        border-right: 1px solid #E2E8F0;
    }

    [data-testid="stSidebar"] [data-testid="stSidebarContent"] {
        padding-top: 1.2rem;
    }

    [data-testid="stSidebar"] h2 {
        color: #1D4ED8;
        letter-spacing: -0.02em;
    }

    [data-testid="stSidebar"] details {
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        background-color: #FFFFFF;
        padding: 2px 8px;
        margin: 8px 0;
        box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
    }

    [data-testid="stSidebar"] details summary {
        font-weight: 700;
        color: #334155;
    }

    [data-testid="stSidebar"] .stButton > button {
        border-radius: 10px;
        border: 1px solid #CBD5E1;
        background-color: #FFFFFF;
        color: #334155;
        font-weight: 700;
    }

    [data-testid="stSidebar"] .stButton > button:hover {
        border-color: #2563EB;
        color: #1D4ED8;
        background-color: #EFF6FF;
    }

    .search-condition-box {
        display: flex;
        align-items: center;
        flex-wrap: wrap;
        gap: 7px;
        margin-top: 6px;
        margin-bottom: 18px;
        padding: 11px 15px;
        border: 1px solid #DBEAFE;
        border-left: 4px solid #2563EB;
        border-radius: 10px;
        background-color: #F8FAFC;
        color: #334155;
        font-size: 13px;
        line-height: 1.6;
    }

    .search-condition-title {
        color: #1D4ED8;
        font-weight: 700;
        margin-right: 5px;
    }

    .search-condition-item {
        white-space: nowrap;
    }

    .search-condition-divider {
        color: #CBD5E1;
        margin: 0 2px;
    }

    /* 사이드바 접기 그룹 카드 */
    [data-testid="stSidebar"] [data-testid="stExpander"] {
        border: 1px solid #D7E0EA !important;
        border-radius: 12px !important;
        background-color: #FFFFFF !important;
        margin: 9px 0 !important;
        overflow: hidden !important;
        box-shadow: 0 2px 5px rgba(15, 23, 42, 0.05) !important;
    }

    [data-testid="stSidebar"] [data-testid="stExpander"] details {
        border: 0 !important;
        background: transparent !important;
    }

    [data-testid="stSidebar"] [data-testid="stExpander"] summary {
        padding: 0.35rem 0.25rem !important;
        font-weight: 700 !important;
        color: #334155 !important;
    }

    [data-testid="stSidebar"] [data-testid="stExpander"] summary:hover {
        color: #1D4ED8 !important;
        background-color: #F8FAFC !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


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
# 0-1. 좌표를 상세주소로 변환
# Kakao Local API 사용
# - .streamlit/secrets.toml의 KAKAO_REST_API_KEY 자동 사용
# - 도로명주소와 지번주소를 함께 반환
# - 키가 없거나 호출에 실패하면 기존 법정동명으로 대체
# -----------------------------------
@st.cache_data(ttl=86400, show_spinner=False)
def get_address_info(latitude, longitude, fallback=""):
    """WGS84 위도·경도를 도로명주소와 지번주소로 변환"""

    default_address = str(fallback).strip() if fallback else "주소 확인 불가"

    try:
        kakao_key = st.secrets.get("KAKAO_REST_API_KEY", "")
    except Exception:
        kakao_key = ""

    if not kakao_key:
        return {
            "road_address": "확인 불가",
            "jibun_address": default_address,
            "display_address": default_address,
        }

    try:
        query = urllib.parse.urlencode(
            {
                "x": float(longitude),
                "y": float(latitude),
                "input_coord": "WGS84",
            }
        )

        request = urllib.request.Request(
            (
                "https://dapi.kakao.com/v2/local/geo/"
                f"coord2address.json?{query}"
            ),
            headers={
                "Authorization": f"KakaoAK {kakao_key}",
            },
        )

        with urllib.request.urlopen(request, timeout=5) as response:
            result = json.loads(response.read().decode("utf-8"))

        documents = result.get("documents", [])
        if not documents:
            return {
                "road_address": "확인 불가",
                "jibun_address": default_address,
                "display_address": default_address,
            }

        document = documents[0]

        address_data = document.get("address") or {}
        jibun_address = (
            address_data.get("address_name", "")
            or default_address
        )

        road_address_data = document.get("road_address") or {}
        road_address = (
            road_address_data.get("address_name", "")
            or "확인 불가"
        )

        display_address = (
            road_address
            if road_address != "확인 불가"
            else jibun_address
        )

        return {
            "road_address": road_address,
            "jibun_address": jibun_address,
            "display_address": display_address,
        }

    except (
        urllib.error.URLError,
        urllib.error.HTTPError,
        TimeoutError,
        ValueError,
        json.JSONDecodeError,
    ):
        return {
            "road_address": "확인 불가",
            "jibun_address": default_address,
            "display_address": default_address,
        }


def get_jibun_address(latitude, longitude, fallback=""):
    """기존 AI 분석 코드와의 호환성을 위해 지번주소만 반환"""

    return get_address_info(
        latitude=latitude,
        longitude=longitude,
        fallback=fallback,
    )["jibun_address"]


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



# ============================================================
# 0-2. 생성형 AI 분석 보고서용 함수
# - 현재 필터링 결과를 JSON 구조로 집계
# - 원본 사고 행·접수번호·좌표는 외부 API로 전송하지 않음
# ============================================================

def _clean_series(dataframe, column_name):
    """결측값과 문자열 nan을 제거한 문자열 Series 반환"""
    if column_name not in dataframe.columns:
        return pd.Series(dtype="object")

    series = (
        dataframe[column_name]
        .dropna()
        .astype(str)
        .str.strip()
    )

    return series[
        (series != "")
        & (series.str.lower() != "nan")
        & (series.str.lower() != "none")
    ]


def _distribution_dict(dataframe, column_name, top_n=None):
    """항목별 건수와 유효값 기준 비율을 JSON용 dict로 변환"""
    series = _clean_series(dataframe, column_name)

    if series.empty:
        return {}

    counts = series.value_counts()
    if top_n is not None:
        counts = counts.head(top_n)

    valid_total = int(series.shape[0])

    return {
        str(label): {
            "count": int(count),
            "ratio_percent": round(count / valid_total * 100, 1),
        }
        for label, count in counts.items()
    }


def _cross_table_dict(
    dataframe,
    row_column,
    column_column,
    max_rows=8,
    max_columns=8,
):
    """상위 항목 중심의 교차표를 JSON용 중첩 dict로 변환"""
    if (
        row_column not in dataframe.columns
        or column_column not in dataframe.columns
    ):
        return {}

    temp = dataframe[[row_column, column_column]].copy()
    temp[row_column] = temp[row_column].astype(str).str.strip()
    temp[column_column] = temp[column_column].astype(str).str.strip()

    temp = temp[
        temp[row_column].notna()
        & temp[column_column].notna()
        & temp[row_column].str.lower().ne("nan")
        & temp[column_column].str.lower().ne("nan")
        & temp[row_column].ne("")
        & temp[column_column].ne("")
    ]

    if temp.empty:
        return {}

    top_rows = (
        temp[row_column]
        .value_counts()
        .head(max_rows)
        .index
    )
    top_columns = (
        temp[column_column]
        .value_counts()
        .head(max_columns)
        .index
    )

    temp = temp[
        temp[row_column].isin(top_rows)
        & temp[column_column].isin(top_columns)
    ]

    table = pd.crosstab(
        temp[row_column],
        temp[column_column],
    )

    result = {}
    for row_label, row_values in table.iterrows():
        result[str(row_label)] = {
            str(column_label): int(value)
            for column_label, value in row_values.items()
            if int(value) > 0
        }

    return result


def _year_distribution(dataframe):
    """연도별 사고 건수 반환"""
    if "acdnt_year" not in dataframe.columns:
        return {}

    counts = (
        dataframe["acdnt_year"]
        .dropna()
        .astype(int)
        .value_counts()
        .sort_index()
    )

    return {
        str(year): int(count)
        for year, count in counts.items()
    }


def _hour_distribution(dataframe):
    """시간별 사고 건수와 비율 반환"""
    if "time_num" not in dataframe.columns:
        return {}

    series = pd.to_numeric(
        dataframe["time_num"],
        errors="coerce",
    ).dropna()

    if series.empty:
        return {}

    counts = series.astype(int).value_counts().sort_index()
    total = int(counts.sum())

    return {
        f"{int(hour):02d}시": {
            "count": int(count),
            "ratio_percent": round(count / total * 100, 1),
        }
        for hour, count in counts.items()
    }


def _severity_rate_table(
    dataframe,
    category_column,
    min_count=10,
    top_n=12,
):
    """범주별 사망·중상사고 비율을 계산해 비교 가능한 표로 반환"""
    if (
        category_column not in dataframe.columns
        or "acdnt_gae_dc" not in dataframe.columns
    ):
        return {}

    temp = dataframe[[category_column, "acdnt_gae_dc"]].copy()
    temp[category_column] = temp[category_column].astype(str).str.strip()
    temp["acdnt_gae_dc"] = temp["acdnt_gae_dc"].astype(str).str.strip()

    temp = temp[
        temp[category_column].ne("")
        & temp[category_column].str.lower().ne("nan")
        & temp[category_column].str.lower().ne("none")
    ]

    if temp.empty:
        return {}

    rows = []
    for label, group in temp.groupby(category_column, observed=True):
        total = int(len(group))
        if total < min_count:
            continue

        fatal = int(group["acdnt_gae_dc"].eq("사망사고").sum())
        serious = int(
            group["acdnt_gae_dc"].isin(["사망사고", "중상사고"]).sum()
        )

        rows.append(
            {
                "label": str(label),
                "total_count": total,
                "fatal_count": fatal,
                "fatal_rate_percent": round(fatal / total * 100, 2),
                "serious_or_fatal_count": serious,
                "serious_or_fatal_rate_percent": round(
                    serious / total * 100,
                    2,
                ),
            }
        )

    rows.sort(
        key=lambda item: (
            item["serious_or_fatal_rate_percent"],
            item["total_count"],
        ),
        reverse=True,
    )

    return {
        item["label"]: {
            key: value
            for key, value in item.items()
            if key != "label"
        }
        for item in rows[:top_n]
    }


def _top_combinations(
    dataframe,
    columns,
    top_n=12,
    min_count=3,
):
    """여러 범주의 결합 빈도를 상위 순으로 반환"""
    if any(column not in dataframe.columns for column in columns):
        return []

    temp = dataframe[list(columns)].copy()
    for column in columns:
        temp[column] = temp[column].astype(str).str.strip()

    valid_mask = pd.Series(True, index=temp.index)
    for column in columns:
        valid_mask &= (
            temp[column].ne("")
            & temp[column].str.lower().ne("nan")
            & temp[column].str.lower().ne("none")
        )

    temp = temp[valid_mask]
    if temp.empty:
        return []

    counts = (
        temp.groupby(list(columns), observed=True)
        .size()
        .sort_values(ascending=False)
    )
    total = int(len(temp))

    result = []
    for labels, count in counts.items():
        if int(count) < min_count:
            continue

        if not isinstance(labels, tuple):
            labels = (labels,)

        result.append(
            {
                "combination": {
                    column: str(label)
                    for column, label in zip(columns, labels)
                },
                "count": int(count),
                "ratio_percent": round(int(count) / total * 100, 1),
            }
        )

        if len(result) >= top_n:
            break

    return result


def _time_band_summary(dataframe):
    """실무적으로 해석하기 쉬운 시간대 구간별 사고·중대사고 집계"""
    if "time_num" not in dataframe.columns:
        return {}

    temp = dataframe.copy()
    temp["time_num"] = pd.to_numeric(temp["time_num"], errors="coerce")
    temp = temp[temp["time_num"].notna()].copy()

    if temp.empty:
        return {}

    bins = [-1, 5, 9, 15, 19, 23]
    labels = [
        "심야·새벽(00~05시)",
        "출근시간(06~09시)",
        "주간(10~15시)",
        "퇴근시간(16~19시)",
        "야간(20~23시)",
    ]

    temp["time_band"] = pd.cut(
        temp["time_num"],
        bins=bins,
        labels=labels,
        include_lowest=True,
    )

    result = {}
    total_all = int(len(temp))

    for label in labels:
        group = temp[temp["time_band"] == label]
        if group.empty:
            continue

        count = int(len(group))
        item = {
            "count": count,
            "ratio_percent": round(count / total_all * 100, 1),
        }

        if "acdnt_gae_dc" in group.columns:
            severity = group["acdnt_gae_dc"].astype(str)
            serious = int(
                severity.isin(["사망사고", "중상사고"]).sum()
            )
            item["serious_or_fatal_count"] = serious
            item["serious_or_fatal_rate_percent"] = round(
                serious / count * 100,
                1,
            )

        result[label] = item

    return result


def _hotspot_summary(
    top_hotspot_dataframe,
    source_dataframe,
):
    """사고다발지점별로 주소와 반경 내 세부 패턴을 함께 반환"""
    if top_hotspot_dataframe is None or top_hotspot_dataframe.empty:
        return []

    source = source_dataframe.reset_index(drop=True)
    weekday_short_map = {
        "월요일": "월",
        "화요일": "화",
        "수요일": "수",
        "목요일": "목",
        "금요일": "금",
        "토요일": "토",
        "일요일": "일",
    }

    result = []
    for rank, (_, row) in enumerate(
        top_hotspot_dataframe.iterrows(),
        start=1,
    ):
        location_name = row.get("legaldong_name", "")
        if pd.isna(location_name):
            location_name = ""

        address_info = get_address_info(
            latitude=row["latitude"],
            longitude=row["longitude"],
            fallback=str(location_name),
        )
        jibun_address = str(address_info.get("jibun_address", "")).strip()
        road_address = str(address_info.get("road_address", "")).strip()

        center_address = (
            jibun_address
            if jibun_address not in {"", "확인 불가", "주소 확인 불가"}
            else road_address
        )

        nearby_indices = row.get("nearby_indices", [])
        nearby = source.iloc[list(nearby_indices)].copy()
        nearby_count = int(len(nearby))

        fatal_count = 0
        serious_count = 0
        if "acdnt_gae_dc" in nearby.columns:
            severity = nearby["acdnt_gae_dc"].astype(str)
            fatal_count = int(severity.eq("사망사고").sum())
            serious_count = int(
                severity.isin(["사망사고", "중상사고"]).sum()
            )

        weekday_distribution = _distribution_dict(
            nearby,
            "dfk_dc",
            top_n=3,
        )
        weekday_distribution = {
            weekday_short_map.get(label, label): value
            for label, value in weekday_distribution.items()
        }

        result.append(
            {
                "rank": rank,
                "center_address": center_address or str(location_name),
                "nearby_accident_count": nearby_count,
                "fatal_accident_count": fatal_count,
                "serious_or_fatal_count": serious_count,
                "serious_or_fatal_rate_percent": round(
                    serious_count / nearby_count * 100,
                    1,
                ) if nearby_count else 0,
                "top_hours": _distribution_dict(
                    nearby,
                    "occrrnc_time_dc",
                    top_n=3,
                ),
                "top_weekdays": weekday_distribution,
                "top_accident_types": _distribution_dict(
                    nearby,
                    "acdnt_hdc",
                    top_n=3,
                ),
                "top_violations": _distribution_dict(
                    nearby,
                    "lrg_violt_1_dc",
                    top_n=3,
                ),
                "top_offending_vehicles": _distribution_dict(
                    nearby,
                    "wrngdo_vhcle_asort_dc",
                    top_n=3,
                ),
                "top_damaged_vehicles": _distribution_dict(
                    nearby,
                    "dmge_vhcle_asort_dc",
                    top_n=3,
                ),
            }
        )

    return result


def make_ai_analysis_package(
    target_df,
    top_hotspot_dataframe,
    selected_filter_info,
    hotspot_radius_m,
):
    """
    현재 필터 결과를 AI가 비교·추론하기 좋은 집계 패키지로 생성.
    모든 수치 계산은 Python이 수행하며 AI는 패턴 해석만 담당.
    """
    total_accidents = int(len(target_df))

    if "dprs_cnt" in target_df.columns:
        total_deaths = int(
            pd.to_numeric(
                target_df["dprs_cnt"],
                errors="coerce",
            )
            .fillna(0)
            .sum()
        )
    else:
        total_deaths = 0

    if "acdnt_gae_dc" in target_df.columns:
        severity_series = target_df["acdnt_gae_dc"].astype(str)
        fatal_accidents = int(severity_series.eq("사망사고").sum())
        serious_accidents = int(severity_series.eq("중상사고").sum())
        serious_or_fatal = int(
            severity_series.isin(["사망사고", "중상사고"]).sum()
        )
    else:
        fatal_accidents = 0
        serious_accidents = 0
        serious_or_fatal = 0

    package = {
        "metadata": {
            "purpose": "대전경찰청 교통사고 인사이트 분석",
            "data_scope": "현재 지도 필터에 해당하는 사고의 비식별 집계결과",
            "privacy_note": (
                "접수번호, 개인식별정보, 개별 사고 좌표와 원본 사고행은 "
                "외부 API에 포함하지 않음"
            ),
            "interpretation_rule": (
                "Python이 건수·비율·교차표를 계산하고 AI는 비교, "
                "패턴 탐색, 실무적 시사점 정리를 수행"
            ),
        },
        "selected_filters": selected_filter_info,
        "overview": {
            "total_accidents": total_accidents,
            "fatal_accidents": fatal_accidents,
            "serious_accidents": serious_accidents,
            "serious_or_fatal_accidents": serious_or_fatal,
            "serious_or_fatal_rate_percent": round(
                serious_or_fatal / total_accidents * 100,
                2,
            ) if total_accidents else 0,
            "total_deaths": total_deaths,
        },
        "basic_distributions": {
            "by_year": _year_distribution(target_df),
            "by_hour": _hour_distribution(target_df),
            "by_time_band": _time_band_summary(target_df),
            "by_weekday": _distribution_dict(target_df, "dfk_dc"),
            "by_accident_severity": _distribution_dict(
                target_df,
                "acdnt_gae_dc",
            ),
            "by_accident_type": _distribution_dict(
                target_df,
                "acdnt_hdc",
            ),
            "by_weather": _distribution_dict(
                target_df,
                "wether_sttus_dc",
            ),
            "by_violation": _distribution_dict(
                target_df,
                "lrg_violt_1_dc",
                top_n=12,
            ),
            "by_offending_vehicle": _distribution_dict(
                target_df,
                "wrngdo_vhcle_asort_dc",
                top_n=12,
            ),
            "by_damaged_vehicle": _distribution_dict(
                target_df,
                "dmge_vhcle_asort_dc",
                top_n=12,
            ),
            "by_fatal_type": _distribution_dict(
                target_df,
                "fatal_type",
                top_n=12,
            ),
            "by_fatal_age_group": _distribution_dict(
                target_df,
                "fatal_age_group",
                top_n=12,
            ),
        },
        "severity_rate_comparisons": {
            "by_hour": _severity_rate_table(
                target_df,
                "occrrnc_time_dc",
                min_count=10,
                top_n=24,
            ),
            "by_weekday": _severity_rate_table(
                target_df,
                "dfk_dc",
                min_count=10,
                top_n=7,
            ),
            "by_accident_type": _severity_rate_table(
                target_df,
                "acdnt_hdc",
                min_count=10,
            ),
            "by_violation": _severity_rate_table(
                target_df,
                "lrg_violt_1_dc",
                min_count=10,
            ),
            "by_weather": _severity_rate_table(
                target_df,
                "wether_sttus_dc",
                min_count=10,
            ),
            "by_offending_vehicle": _severity_rate_table(
                target_df,
                "wrngdo_vhcle_asort_dc",
                min_count=10,
            ),
            "by_damaged_vehicle": _severity_rate_table(
                target_df,
                "dmge_vhcle_asort_dc",
                min_count=10,
            ),
        },
        "cross_analyses": {
            "hour_x_accident_type": _cross_table_dict(
                target_df,
                "occrrnc_time_dc",
                "acdnt_hdc",
                max_rows=24,
                max_columns=5,
            ),
            "weekday_x_accident_type": _cross_table_dict(
                target_df,
                "dfk_dc",
                "acdnt_hdc",
                max_rows=7,
                max_columns=5,
            ),
            "violation_x_severity": _cross_table_dict(
                target_df,
                "lrg_violt_1_dc",
                "acdnt_gae_dc",
                max_rows=12,
                max_columns=6,
            ),
            "offending_x_damaged_vehicle": _cross_table_dict(
                target_df,
                "wrngdo_vhcle_asort_dc",
                "dmge_vhcle_asort_dc",
                max_rows=10,
                max_columns=10,
            ),
            "fatal_type_x_hour": _cross_table_dict(
                target_df,
                "fatal_type",
                "occrrnc_time_dc",
                max_rows=10,
                max_columns=24,
            ),
        },
        "dominant_combinations": {
            "hour_accident_type_violation": _top_combinations(
                target_df,
                ["occrrnc_time_dc", "acdnt_hdc", "lrg_violt_1_dc"],
                top_n=12,
                min_count=3,
            ),
            "offending_damaged_accident_type": _top_combinations(
                target_df,
                [
                    "wrngdo_vhcle_asort_dc",
                    "dmge_vhcle_asort_dc",
                    "acdnt_hdc",
                ],
                top_n=12,
                min_count=3,
            ),
            "weekday_hour_accident_type": _top_combinations(
                target_df,
                ["dfk_dc", "occrrnc_time_dc", "acdnt_hdc"],
                top_n=12,
                min_count=3,
            ),
        },
        "hotspots": {
            "analysis_radius_m": int(hotspot_radius_m),
            "ranked_locations": _hotspot_summary(
                top_hotspot_dataframe,
                target_df,
            ),
        },
        "analysis_limitations": [
            "선택 조건에 따른 기술통계로 인과관계를 직접 증명하지 않음",
            "교통량과 노출량 자료가 없어 단순 건수를 위험률로 해석할 수 없음",
            "도로구조, 신호현시, 시야, 공사 여부 등 현장정보는 포함되지 않음",
            "비율 비교 시 표본이 작은 범주는 변동성이 크므로 신중히 해석해야 함",
            "시설개선이나 단속대책 확정 전 현장점검과 관계기관 협의가 필요함",
        ],
    }

    return package


def build_ai_report_prompt(analysis_json):
    """비교·교차분석을 자연스러운 서술형 보고서로 작성하는 프롬프트"""
    return f"""
당신은 대한민국 경찰의 교통사고 데이터를 검토하는 선임 교통안전 분석관이다.
아래 JSON에는 Python으로 계산한 분포, 중대사고 비율, 교차표, 결합패턴,
사고다발지점별 세부 집계가 들어 있다.

목표는 통계를 항목별로 옮겨 적는 것이 아니라,
관리자가 놓치기 쉬운 패턴과 우선 대응과제를 설득력 있는 보고서 문장으로 풀어내는 것이다.

[절대 원칙]
1. JSON에 없는 사실·수치·도로형태·교통량·신호운영·사고원인을 만들지 않는다.
2. 통계적 연관을 원인으로 단정하지 않는다.
3. 원인 가능성을 언급할 때는 '가능성이 있다', '가설로 검토할 수 있다',
   '현장 확인이 필요하다'와 같이 불확실성을 분명히 표현한다.
4. 단순히 '가장 많다'는 서술은 최소화하고, 전체 비중과 중대사고 비율의 차이,
   시간·요일·차종·법규위반의 결합, 지점별 차이를 우선 분석한다.
5. 건수가 많은 범주와 중대사고 비율이 높은 범주를 명확히 구분한다.
6. 근거가 약하거나 표본이 작은 경우 해당 문단 안에서 자연스럽게 한계를 함께 밝힌다.
7. 대책은 데이터에서 확인된 패턴과 직접 연결하고, 일반론적 문구를 반복하지 않는다.
8. 문체는 대한민국 경찰 내부 관리자 보고용으로 전문적이되,
   지나치게 기계적이거나 단편적인 항목 나열이 되지 않도록 한다.

[문체 지침]
- '발견/근거/실무적 의미/주의' 같은 고정된 네 개의 꼬리표를 반복하지 않는다.
- 각 인사이트는 1~3개의 자연스러운 문단으로 작성한다.
- 첫 문장에서 핵심 판단을 제시하고, 이어지는 문장에서 수치 근거와 비교를 설명한다.
- 마지막 문장에서는 경찰 관리상 의미 또는 추가 확인 필요사항을 자연스럽게 연결한다.
- 모든 인사이트의 문장 구조를 똑같이 반복하지 말고, 비교형·대조형·전환형·요약형 문구를 다양하게 사용한다.
- 수치는 판단을 뒷받침할 때만 제시하고, 이미 제시한 수치를 불필요하게 반복하지 않는다.
- '중점 관리가 필요하다'만 반복하지 말고, 순찰·단속·보행자 보호·현장점검·시간대 조정 등
  구체적인 관리 방향으로 풀어 쓴다.

[분석할 핵심 질문]
- 전체 사고에서 흔한 특성과 사망·중상사고에서 두드러지는 특성은 어떻게 다른가?
- 단순 건수는 적지만 중대사고 비율이 상대적으로 높은 시간·요일·사고종별·법규위반은 무엇인가?
- 시간×사고종별×법규위반 또는 차종 결합에서 반복되는 패턴은 무엇인가?
- 사고다발지점들은 동일한 유형인가, 지점마다 서로 다른 프로파일을 보이는가?
- 경찰이 즉시 할 수 있는 조치와 현장 확인 후 결정할 조치를 어떻게 구분해야 하는가?

[출력 형식]
# AI 교통사고 인사이트 보고서

## 1. 분석 범위
선택 조건, 전체 사고, 사망·중상사고 규모를 짧은 서술형 문단으로 정리한다.

## 2. 핵심 인사이트
의미 있는 인사이트 4~6개를 선정한다.
각 항목은 다음 형식을 따른다.

### 인사이트 n. 핵심 판단을 담은 제목
고정된 하위 꼬리표 없이 1~3개 문단으로 자연스럽게 설명한다.
문단 안에는 반드시 다음 요소가 포함되어야 한다.
- 비교 또는 결합패턴
- 관련 건수·비율 등 수치 근거
- 경찰 관리 관점의 의미
- 인과 단정 방지 또는 표본·노출량 한계

예시 문체:
'차대사람 사고는 전체 사고에서 차지하는 비중은 상대적으로 크지 않지만,
중대사고로 이어지는 비율은 차대차보다 뚜렷하게 높다. 차대사람 806건 중 281건이
사망·중상사고로 분류되어 중대사고율이 34.9%였고, 차대차는 3,347건 중 539건으로
16.1%였다. 따라서 전체 건수 중심의 관리만으로는 보행자 사고의 위험성을 충분히
반영하기 어렵고, 보행자 보호활동은 발생 빈도보다 중대성에 더 큰 비중을 두어야 한다.
다만 교통량과 보행량이 포함되지 않은 자료이므로 실제 위험률로 단정하기보다는
취약 시간대와 지점에 대한 현장점검 근거로 활용하는 것이 적절하다.'

의미 있는 차이가 없다면 억지로 인사이트를 만들지 말고 그 사실을 밝힌다.

## 3. 사고다발지점 프로파일
각 지점을 주소와 건수만 나열하지 말고, 지점별 특징을 짧은 서술형 문단으로 비교한다.
대표 시간·요일·사고종별·법규위반·차종 조합을 전체 분석대상과 대비하고,
그 차이가 현장점검 우선순위에 어떤 의미가 있는지 설명한다.
도로형태 등 JSON에 없는 사항은 단정하지 않는다.

## 4. 우선 대응과제
다음 세 단계로 구분한다.
1. 즉시 시행 가능
2. 현장 확인 후 시행
3. 중기 관리

각 과제는 단순 명사 나열이 아니라 2~4문장으로 풀어 쓰고,
어떤 데이터 패턴에 근거한 조치인지 문장 안에 연결한다.

## 5. 관리자가 볼 핵심 결론
가장 중요한 판단 3개를 각각 완결된 문장으로 정리한다.
'사고가 많으니 주의해야 한다' 같은 일반론은 쓰지 않는다.

## 6. 분석 한계
JSON의 analysis_limitations를 반영해 짧은 서술형 문단 또는 3~5개 항목으로 정리한다.

[분석용 JSON]
{analysis_json}
""".strip()


def generate_ai_report(analysis_package):
    """OpenAI Responses API로 교통사고 인사이트 보고서 생성"""
    if OpenAI is None:
        raise RuntimeError(
            "openai 라이브러리가 설치되지 않았습니다. "
            "requirements.txt에 'openai'를 추가하세요."
        )

    try:
        api_key = st.secrets["OPENAI_API_KEY"]
    except KeyError as exc:
        raise RuntimeError(
            "Streamlit Secrets에 OPENAI_API_KEY를 등록하세요."
        ) from exc

    try:
        model_name = st.secrets.get(
            "OPENAI_MODEL",
            "gpt-4.1-mini",
        )
    except Exception:
        model_name = "gpt-4.1-mini"

    client = OpenAI(api_key=api_key)

    analysis_json = json.dumps(
        analysis_package,
        ensure_ascii=False,
        separators=(",", ":"),
    )

    prompt = build_ai_report_prompt(analysis_json)

    response = client.responses.create(
        model=model_name,
        input=prompt,
        max_output_tokens=3200,
    )

    return response.output_text, analysis_json

def make_filter_signature(selected_filter_info, target_df):
    """필터 변경 여부를 확인하기 위한 짧은 식별값"""
    signature_source = {
        "filters": selected_filter_info,
        "row_count": int(len(target_df)),
    }

    signature_text = json.dumps(
        signature_source,
        ensure_ascii=False,
        sort_keys=True,
        default=str,
    )

    return hashlib.sha256(
        signature_text.encode("utf-8")
    ).hexdigest()


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
    "관할 경찰서",
    station_options,
)

# -----------------------------------
# [순서 2] 발생 연월 선택
# accident_date 컬럼 활용
# 예) 2023년 3월 ~ 2024년 5월
# -----------------------------------

# Parquet에서 날짜형으로 정상 로드되도록 자료형 확인
df["accident_date"] = pd.to_datetime(
    df["accident_date"],
    errors="coerce",
)

# 실제 데이터에 존재하는 연월 목록 생성
year_month_options = (
    df["accident_date"]
    .dropna()
    .dt.to_period("M")
    .drop_duplicates()
    .sort_values()
    .tolist()
)

if year_month_options:
    start_year_month, end_year_month = (
        st.sidebar.select_slider(
            "발생연월 범위",
            options=year_month_options,
            value=(
                year_month_options[0],
                year_month_options[-1],
            ),
            format_func=lambda value: (
                f"{value.year}년 {value.month}월"
            ),
        )
    )

    st.sidebar.caption(
        f"선택 기간: "
        f"{start_year_month.year}년 "
        f"{start_year_month.month}월 ~ "
        f"{end_year_month.year}년 "
        f"{end_year_month.month}월"
    )

else:
    start_year_month = None
    end_year_month = None

    st.sidebar.warning(
        "선택 가능한 발생연월 데이터가 없습니다."
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
# [순서 6] 발생요일 선택
# 체크박스 복수선택
# 미선택 시 전체 요일
# -----------------------------------
weekday_order = [
    "월요일",
    "화요일",
    "수요일",
    "목요일",
    "금요일",
    "토요일",
    "일요일",
]

weekday_short_name = {
    "월요일": "월",
    "화요일": "화",
    "수요일": "수",
    "목요일": "목",
    "금요일": "금",
    "토요일": "토",
    "일요일": "일",
}

if "dfk_dc" in df.columns:
    available_weekdays = set(
        df["dfk_dc"]
        .dropna()
        .astype(str)
        .str.strip()
        .unique()
    )

    weekday_options = [
        weekday
        for weekday in weekday_order
        if weekday in available_weekdays
    ]

else:
    weekday_options = []

# 다른 사이드바 입력항목과 동일한 제목 형식
st.sidebar.write("발생요일")

# 7개 체크박스를 한 줄로 배치
weekday_cols = st.sidebar.columns(7)

selected_weekdays = []

for index, weekday in enumerate(weekday_options):
    with weekday_cols[index]:
        is_selected = st.checkbox(
            weekday_short_name[weekday],
            value=False,
            key=f"weekday_checkbox_{weekday}",
        )

    if is_selected:
        selected_weekdays.append(weekday)

if selected_weekdays:
    selected_weekday_labels = [
        weekday_short_name[weekday]
        for weekday in selected_weekdays
    ]

    st.sidebar.caption(
        "선택 요일: "
        + "·".join(selected_weekday_labels)
    )

else:
    st.sidebar.caption("선택 요일: 전체")

# ----------------------------------
st.sidebar.divider()

with st.sidebar.expander("🚗 차량 조건", expanded=False):
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

    selected_wrngdo = st.multiselect(
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

    selected_dmge = st.multiselect(
        "피해차량 차종 (복수 선택)",
        dmge_options,
        placeholder="전체 (미선택 시)",
    )


with st.sidebar.expander("🚨 사망사고 조건", expanded=False):
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

    selected_fatal_type = st.multiselect(
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

    selected_fatal_age = st.multiselect(
        "사망자 연령대 (복수 선택)",
        fatal_age_options,
        placeholder="전체 (미선택 시)",
    )


with st.sidebar.expander("🌦️ 환경·원인 조건", expanded=False):
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

    selected_wether = st.multiselect(
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

    selected_violt = st.multiselect(
        "법규위반유형 (복수 선택)",
        violt_options,
        placeholder="전체 (미선택 시)",
    )


st.sidebar.divider()
if st.sidebar.button("↺ 필터 초기화", use_container_width=True):
    st.session_state.clear()
    st.rerun()

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

# -----------------------------------
# [필터 2] 발생 연월
# 시작월 1일부터 종료월 마지막 날까지 포함
# -----------------------------------
if (
    start_year_month is not None
    and end_year_month is not None
    and "accident_date" in filtered_df.columns
):
    start_date = start_year_month.start_time
    end_date = end_year_month.end_time

    filtered_df = filtered_df[
        filtered_df["accident_date"].between(
            start_date,
            end_date,
            inclusive="both",
        )
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

# -----------------------------------
# [필터 6] 발생요일
# 하나 이상 선택한 경우에만 적용
# 미선택 시 전체 요일 유지
# -----------------------------------
if selected_weekdays and "dfk_dc" in filtered_df.columns:
    weekday_series = (
        filtered_df["dfk_dc"]
        .astype(str)
        .str.strip()
    )

    filtered_df = filtered_df[
        weekday_series.isin(selected_weekdays)
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

# ============================================================
# 상단 핵심지표(KPI)
# ============================================================
fatal_accident_count = 0
if "is_fatal" in filtered_df.columns:
    fatal_accident_count = int(
        pd.to_numeric(
            filtered_df["is_fatal"],
            errors="coerce",
        ).fillna(0).eq(1).sum()
    )
elif "acdnt_gae_dc" in filtered_df.columns:
    fatal_accident_count = int(
        filtered_df["acdnt_gae_dc"]
        .astype(str)
        .eq("사망사고")
        .sum()
    )

serious_accident_count = 0
if "acdnt_gae_dc" in filtered_df.columns:
    serious_accident_count = int(
        filtered_df["acdnt_gae_dc"]
        .astype(str)
        .eq("중상사고")
        .sum()
    )

kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)

with kpi_col1:
    st.metric("선택된 관할", selected_ps)

with kpi_col2:
    st.metric("총 사고", f"{len(filtered_df):,}건")

with kpi_col3:
    st.metric("사망사고", f"{fatal_accident_count:,}건")

with kpi_col4:
    st.metric("중상사고", f"{serious_accident_count:,}건")

# ============================================================
# KPI 아래 현재 검색조건 요약
# 발생연월 · 사고분류 · 사고종별 · 시간대 · 발생요일
# ============================================================

# -----------------------------------
# 발생연월
# -----------------------------------
if (
    start_year_month is not None
    and end_year_month is not None
):
    period_summary = (
        f"{start_year_month.year}년 "
        f"{start_year_month.month}월 – "
        f"{end_year_month.year}년 "
        f"{end_year_month.month}월"
    )
else:
    period_summary = "전체 기간"


# -----------------------------------
# 사고분류
# -----------------------------------
if selected_types:
    severity_short_map = {
        "사망사고": "사망",
        "중상사고": "중상",
        "경상사고": "경상",
        "부상신고사고": "부상신고",
    }

    severity_summary = "·".join(
        severity_short_map.get(
            value,
            value,
        )
        for value in selected_types
    )

else:
    severity_summary = "전체 사고"


# -----------------------------------
# 사고종별
# -----------------------------------
accident_type_summary = (
    selected_hdc
    if selected_hdc != "전체"
    else "전체 종별"
)


# -----------------------------------
# 시간대
# -----------------------------------
if start_time <= end_time:
    time_summary = (
        f"{start_time:02d}시 – "
        f"{end_time:02d}시"
    )

else:
    time_summary = (
        f"{start_time:02d}시 – "
        f"익일 {end_time:02d}시"
    )


# -----------------------------------
# 발생요일
# -----------------------------------
if selected_weekdays:
    weekday_summary = "·".join(
        weekday_short_name.get(
            value,
            value,
        )
        for value in selected_weekdays
    )

else:
    weekday_summary = "전체 요일"


# -----------------------------------
# 현재 검색조건 출력
# -----------------------------------
st.caption(
    f"🔎 현재 검색조건 │ "
    f"📅 {period_summary} │ "
    f"사고분류: {severity_summary} │ "
    f"사고종별: {accident_type_summary} │ "
    f"⏰ {time_summary} │ "
    f"요일: {weekday_summary}"
)


# ============================================================
# 페이지 탭
# ============================================================
map_tab, stats_tab = st.tabs(
    [
        "🗺️ 공간 분석",
        "📊 통계 대시보드",
    ]
)

with map_tab:
    st.subheader("🗺️ 교통사고 공간분석", anchor=False)

    # -----------------------------------
    # 지도 위 사고다발지점 표시 설정
    # 입력값은 적용 버튼을 눌렀을 때만 지도에 반영
    # 기본값: 반경 100m / 상위 5개
    # -----------------------------------
    if "hotspot_radius" not in st.session_state:
        st.session_state.hotspot_radius = 100

    if "hotspot_top_n" not in st.session_state:
        st.session_state.hotspot_top_n = 5

    st.markdown("입력한 반경(m)과 수에 따라, AI가 사고다발지역을 자동 탐색합니다.")

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

    st.markdown(
        """
        <div style="
            text-align: right;
            color: #6B7280;
            font-size: 0.82rem;
            margin-top: 2px;
            margin-bottom: 8px;
        ">
            ※ 레이어 선택창에서 사고다발지점·사망사고·히트맵·관할 경계를 개별적으로 켜고 끌 수 있습니다.
        </div>
        """,
        unsafe_allow_html=True,
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
# 반경 내 가해차량·피해차량·법규위반·사고상황 표시
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


def make_top3_rank_html(
    dataframe,
    column_name,
    total_count,
    label_map=None,
    separator="<br>",
    show_count=True,
):
    """
    지정 컬럼의 상위 3개 항목을 팝업용 HTML로 변환

    show_count=True
    → 1. 승용 63건 (75.0%)

    show_count=False
    → 월 (20.8%)
    """

    if column_name not in dataframe.columns:
        return (
            "<span style='color:#777777;'>"
            "집계 가능한 데이터 없음"
            "</span>"
        )

    series = (
        dataframe[column_name]
        .dropna()
        .astype(str)
        .str.strip()
    )

    series = series[
        (series != "")
        & (series.str.lower() != "nan")
        & (series.str.lower() != "none")
    ]

    top3 = series.value_counts().head(3)

    if top3.empty:
        return (
            "<span style='color:#777777;'>"
            "집계 가능한 데이터 없음"
            "</span>"
        )

    result = []

    for order, (label, item_count) in enumerate(
        top3.items(),
        start=1,
    ):
        display_label = (
            label_map.get(label, label)
            if label_map
            else label
        )

        ratio = (
            item_count / total_count * 100
            if total_count
            else 0
        )

        # 가해차량·피해차량·법규위반
        if show_count:
            item_html = (
                f"{order}. {display_label} "
                f"<span style='color:#000000; "
                f"font-weight:normal;'>"
                f"{int(item_count)}건 ({ratio:.1f}%)"
                f"</span>"
            )

        # 시간·요일
        else:
            item_html = (
                f"{display_label} "
                f"<span style='color:#000000; "
                f"font-weight:normal;'>"
                f"({ratio:.1f}%)"
                f"</span>"
            )

        result.append(item_html)

    return separator.join(result)


if not top_hotspot_df.empty:

    # get_top_hotspots() 함수에서 사용한 데이터와
    # 동일한 행 순서를 유지하기 위해 인덱스 초기화
    hotspot_source_df = filtered_df.reset_index(drop=True)

    weekday_short_map = {
        "월요일": "월",
        "화요일": "화",
        "수요일": "수",
        "목요일": "목",
        "금요일": "금",
        "토요일": "토",
        "일요일": "일",
    }

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
        # 중심좌표 상세주소 변환
        # 지번주소 우선, 없으면 도로명주소 사용
        # -----------------------------------
        address_info = get_address_info(
            latitude=lat,
            longitude=lon,
            fallback=location_name,
        )

        road_address = str(
            address_info.get("road_address", "")
        ).strip()

        jibun_address = str(
            address_info.get("jibun_address", "")
        ).strip()

        # 지번주소가 정상적으로 있으면 우선 사용
        if (
            jibun_address
            and jibun_address not in {
                "확인 불가",
                "주소 확인 불가",
                "nan",
                "None",
            }
        ):
            center_address = jibun_address

        # 지번주소가 없으면 도로명주소 사용
        elif (
            road_address
            and road_address not in {
                "확인 불가",
                "주소 확인 불가",
                "nan",
                "None",
            }
        ):
            center_address = road_address

        # 둘 다 없으면 데이터의 법정동명 사용
        else:
            center_address = (
                str(location_name).strip()
                if str(location_name).strip()
                else "주소 확인 불가"
            )






        # 해당 사고다발지점 반경 내 사고 추출
        nearby_indices = row["nearby_indices"]
        nearby_accidents = hotspot_source_df.iloc[
            nearby_indices
        ]

        # 가해차량·피해차량·법규위반 상위 3개
        vehicle_html = make_top3_rank_html(
            dataframe=nearby_accidents,
            column_name="wrngdo_vhcle_asort_dc",
            total_count=count,
        )

        damage_vehicle_html = make_top3_rank_html(
            dataframe=nearby_accidents,
            column_name="dmge_vhcle_asort_dc",
            total_count=count,
        )

        violation_html = make_top3_rank_html(
            dataframe=nearby_accidents,
            column_name="lrg_violt_1_dc",
            total_count=count,
        )
        # 사고상황: 시간·요일 상위 3개
        # 필터 결과가 3개 미만이면 실제 존재하는 항목만 표시
        time_html = make_top3_rank_html(
            dataframe=nearby_accidents,
            column_name="occrrnc_time_dc",
            total_count=count,
            separator=" &gt; ",
            show_count=False,
        )

        weekday_html = make_top3_rank_html(
            dataframe=nearby_accidents,
            column_name="dfk_dc",
            total_count=count,
            label_map=weekday_short_map,
            separator=" &gt; ",
            show_count=False,
        )

        # 사용자가 선택한 분석 반경 원 표시
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
                f"{center_address}<br>"
                f"반경 {hotspot_radius}m 내 사고 {count}건",
                sticky=True,
            ),
        ).add_to(hotspot_layer)

        # 사고다발지점 상세 팝업
        popup_html = f"""
        <div style="
            width:310px;
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

            <b>중심주소</b> : {center_address}<br>
            <b>분석반경</b> : {hotspot_radius}m<br>
            <b>사고건수</b> : {count}건

            <div style="
                margin-top:8px;
                padding-top:7px;
                border-top:1px solid #DDDDDD;
            ">
            <b>사고상황</b><br>
            1. 요일 : {weekday_html}<br>
            2. 시간 : {time_html}
            </div>            

            <div style="
                margin-top:8px;
                padding-top:7px;
                border-top:1px solid #DDDDDD;
            ">
                <b>가해차량</b><br>
                {vehicle_html}
            </div>

            <div style="
                margin-top:8px;
                padding-top:7px;
                border-top:1px solid #DDDDDD;
            ">
                <b>피해차량</b><br>
                {damage_vehicle_html}
            </div>

            <div style="
                margin-top:8px;
                padding-top:7px;
                border-top:1px solid #DDDDDD;
            ">
                <b>법규위반</b><br>
                {violation_html}
            </div>

        </div>
        """

        # 파란색 원형 순위 배지
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
                max_width=360,
            ),
            tooltip=folium.Tooltip(
                (
                    f"사고다발지점 {rank}위 · "
                    f"{center_address} · {count}건"
                ),
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

# ============================================================
# 5. 대시보드 상단 요약 및 페이지 탭
# ============================================================
st.markdown(
    """
    <style>
    /* 메인 화면 여백 */
    .block-container {
        padding-top: 3.2rem;
        padding-bottom: 2.5rem;
    }

    /* KPI 카드 */
    div[data-testid="stMetric"] {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 14px;
        padding: 14px 16px;
        box-shadow: 0 2px 8px rgba(15, 23, 42, 0.05);
    }
    div[data-testid="stMetric"] label {
        color: #64748B;
        font-size: 0.88rem;
    }
    div[data-testid="stMetric"] [data-testid="stMetricValue"] {
        color: #0F172A;
        font-weight: 700;
    }

    /* 페이지 탭 전체 영역 */
    div[data-baseweb="tab-list"] {
        gap: 12px;
        padding: 8px;
        margin-top: 8px;
        margin-bottom: 18px;
        border: 1px solid #CBD5E1;
        border-radius: 14px;
        background: #F1F5F9;
        box-shadow: 0 2px 8px rgba(15, 23, 42, 0.06);
    }

    /* 개별 탭 */
    button[data-baseweb="tab"] {
        min-height: 52px;
        padding: 12px 28px;
        border-radius: 10px;
        background: transparent;
        color: #475569;
        font-size: 1.08rem;
        font-weight: 700;
        transition: all 0.18s ease;
    }

    button[data-baseweb="tab"] p {
        font-size: 1.08rem;
        font-weight: 700;
    }

    button[data-baseweb="tab"]:hover {
        background: #E2E8F0;
        color: #1E3A8A;
    }

    /* 선택된 탭 */
    button[data-baseweb="tab"][aria-selected="true"] {
        background: #2563EB;
        color: #FFFFFF;
        box-shadow: 0 3px 10px rgba(37, 99, 235, 0.28);
    }

    button[data-baseweb="tab"][aria-selected="true"] p {
        color: #FFFFFF;
    }

    /* 기본 선택 밑줄은 숨김 */
    div[data-baseweb="tab-highlight"] {
        display: none;
    }

    /* 접기 영역 */
    div[data-testid="stExpander"] {
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        overflow: hidden;
        margin-bottom: 0.75rem;
        background-color: #FFFFFF;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

with map_tab:
    st_folium(
        m2,
        use_container_width=True,
        height=840,
        returned_objects=[],
    )

    st.markdown("---")

    # ============================================================
    # 5-1. 생성형 AI 교통사고 분석 보고서
    # - 현재 지도에 현출된 filtered_df의 집계 결과만 JSON으로 전송
    # - 버튼을 누를 때만 API 호출
    # ============================================================

    st.subheader("🤖 AI 교통사고 분석", anchor=False)

    # 현재 선택된 필터를 보고서 JSON에 함께 기록
    selected_filter_info = {
        "police_station": selected_ps,
        "year_month_range": [
        (
            f"{start_year_month.year}-{start_year_month.month:02d}"
            if start_year_month is not None
            else "전체"
        ),
        (
            f"{end_year_month.year}-{end_year_month.month:02d}"
            if end_year_month is not None
            else "전체"
        ),
    ],

        "accident_severity": selected_types if selected_types else ["전체"],
        "accident_type": selected_hdc,
        "time_range": (
            f"{start_time:02d}시~{end_time:02d}시"
            if start_time <= end_time
            else f"{start_time:02d}시~익일 {end_time:02d}시"
        ),
        "weekday": (
            [
                weekday_short_name[weekday]
                for weekday in selected_weekdays
            ]
            if selected_weekdays
            else ["전체"]
        ),
        "offending_vehicle": selected_wrngdo if selected_wrngdo else ["전체"],
        "damaged_vehicle": selected_dmge if selected_dmge else ["전체"],
        "fatal_type": (
            selected_fatal_type
            if selected_fatal_type
            else ["전체"]
        ),
        "fatal_age_group": (
            selected_fatal_age
            if selected_fatal_age
            else ["전체"]
        ),
        "weather": selected_wether if selected_wether else ["전체"],
        "violation": selected_violt if selected_violt else ["전체"],
    }

    current_filter_signature = make_filter_signature(
        selected_filter_info,
        filtered_df,
    )

    # 필터가 바뀌었으면 이전 조건에서 만든 보고서임을 명확히 표시
    report_is_current = (
        st.session_state.get("ai_report_signature")
        == current_filter_signature
    )

    generate_report_button = st.button(
        "📄 현재 조건 AI 상세 분석 보고서 생성",
        type="primary",
        use_container_width=True,
        disabled=filtered_df.empty,
    )

    # 현재 필터 조건의 분석용 JSON 패키지 생성
    analysis_package_preview = None

    if not filtered_df.empty:
        analysis_package_preview = make_ai_analysis_package(
            target_df=filtered_df,
            top_hotspot_dataframe=top_hotspot_df,
            selected_filter_info=selected_filter_info,
            hotspot_radius_m=hotspot_radius,
        )


    if generate_report_button:
        try:
            with st.spinner(
                "현재 조건의 사고 통계를 집계하고 AI 보고서를 작성하고 있습니다."
            ):
                ai_report, _ = generate_ai_report(
                    analysis_package_preview
                )

            st.session_state["ai_report"] = ai_report
            st.session_state["ai_report_signature"] = (
                current_filter_signature
            )
            st.session_state["ai_report_filters"] = (
                selected_filter_info
            )
            report_is_current = True

        except Exception as error:
            st.error("AI 분석 보고서 생성 중 오류가 발생했습니다.")
            st.code(str(error))

    if st.session_state.get("ai_report"):
        if not report_is_current:
            st.warning(
                "아래 보고서는 현재와 다른 필터 조건에서 생성되었습니다. "
                "현재 조건의 보고서가 필요하면 생성 버튼을 다시 누르세요."
            )

        with st.expander(
            "📑 AI 상세 분석 보고서",
            expanded=report_is_current,
        ):
            st.markdown(st.session_state["ai_report"])

            st.caption(
                "생성형 AI 결과는 의사결정 보조자료이며, "
                "최종 활용 전 담당자의 통계·현장 확인이 필요합니다."
            )

with stats_tab:
    # ============================================================
    # 6. 하단 레이아웃: 분석 통계 배치
    # ============================================================

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

    CHART_COLOR_SCALES = {
        # 차량 분석
        "offending_vehicle": [
            [0.0, "#DBEAFE"],
            [0.5, "#60A5FA"],
            [1.0, "#1D4ED8"],
        ],
        "damaged_vehicle": [
            [0.0, "#CCFBF1"],
            [0.5, "#2DD4BF"],
            [1.0, "#0F766E"],
        ],

        # 사망사고 분석
        "fatal_type": [
            [0.0, "#FEE2E2"],
            [0.5, "#F87171"],
            [1.0, "#B91C1C"],
        ],
        "fatal_age": [
            [0.0, "#F3E8FF"],
            [0.5, "#C084FC"],
            [1.0, "#7E22CE"],
        ],

        # 상황별 사고 분석
        "weekday": [
            [0.0, "#DCFCE7"],
            [0.5, "#4ADE80"],
            [1.0, "#15803D"],
        ],
        "time": [
            [0.0, "#E0E7FF"],
            [0.5, "#818CF8"],
            [1.0, "#3730A3"],
        ],
        "violation": [
            [0.0, "#FEF3C7"],   # 연한 Amber
            [0.5, "#FBBF24"],   # Amber 400
            [1.0, "#B45309"],   # Amber 700
        ],
    }

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
        # 가해차량·피해차량 그래프에서 '기타불명' 제외
        # ========================================================
        with st.expander("🚘 차량 분석", expanded=True):

            row2_col1, row2_col2 = st.columns(2)

            # --------------------------------------------------------
            # 가해차량 차종
            # --------------------------------------------------------
            with row2_col1:
                st.markdown("**🚗 가해차량 차종**")

                # 결측값·문자열 결측값·기타불명 제외
                wrngdo_series = (
                    filtered_df["wrngdo_vhcle_asort_dc"]
                    .dropna()
                    .astype(str)
                    .str.strip()
                )

                wrngdo_series = wrngdo_series[
                    (wrngdo_series != "")
                    & (~wrngdo_series.str.lower().isin(
                        ["nan", "none"]
                    ))
                    & (wrngdo_series != "기타불명")
                ]

                wrngdo_counts = (
                    wrngdo_series
                    .value_counts()
                    .rename_axis("차종")
                    .reset_index(name="사고건수")
                    .sort_values(
                        "사고건수",
                        ascending=False,
                    )
                )

                if not wrngdo_counts.empty:
                    fig_wrngdo = px.bar(
                        wrngdo_counts,
                        x="사고건수",
                        y="차종",
                        orientation="h",
                        text="사고건수",
                        color="사고건수",
                        color_continuous_scale=(
                            CHART_COLOR_SCALES[
                                "offending_vehicle"
                            ]
                        ),
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

                    fig_wrngdo.update_coloraxes(
                        showscale=False,
                    )

                    fig_wrngdo = apply_common_chart_style(
                        fig_wrngdo,
                        height=max(
                            330,
                            len(wrngdo_counts) * 36,
                        ),
                        horizontal=True,
                    )

                    # 건수가 가장 많은 차종을 위쪽에 표시
                    fig_wrngdo.update_yaxes(
                        autorange="reversed",
                        categoryorder="array",
                        categoryarray=(
                            wrngdo_counts["차종"].tolist()
                        ),
                    )

                    render_plotly_chart(fig_wrngdo)

                else:
                    st.info(
                        "선택된 조건에 해당하는 "
                        "가해차량 차종 데이터가 없습니다."
                    )

            # --------------------------------------------------------
            # 피해차량 차종
            # --------------------------------------------------------
            with row2_col2:
                st.markdown("**🚙 피해차량 차종**")

                # 결측값·문자열 결측값·기타불명 제외
                dmge_series = (
                    filtered_df["dmge_vhcle_asort_dc"]
                    .dropna()
                    .astype(str)
                    .str.strip()
                )

                dmge_series = dmge_series[
                    (dmge_series != "")
                    & (~dmge_series.str.lower().isin(
                        ["nan", "none"]
                    ))
                    & (dmge_series != "기타불명")
                ]

                dmge_counts = (
                    dmge_series
                    .value_counts()
                    .rename_axis("차종")
                    .reset_index(name="사고건수")
                    .sort_values(
                        "사고건수",
                        ascending=False,
                    )
                )

                if not dmge_counts.empty:
                    fig_dmge = px.bar(
                        dmge_counts,
                        x="사고건수",
                        y="차종",
                        orientation="h",
                        text="사고건수",
                        color="사고건수",
                        color_continuous_scale=(
                            CHART_COLOR_SCALES[
                                "damaged_vehicle"
                            ]
                        ),
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

                    fig_dmge.update_coloraxes(
                        showscale=False,
                    )

                    fig_dmge = apply_common_chart_style(
                        fig_dmge,
                        height=max(
                            330,
                            len(dmge_counts) * 36,
                        ),
                        horizontal=True,
                    )

                    # 건수가 가장 많은 차종을 위쪽에 표시
                    fig_dmge.update_yaxes(
                        autorange="reversed",
                        categoryorder="array",
                        categoryarray=(
                            dmge_counts["차종"].tolist()
                        ),
                    )

                    render_plotly_chart(fig_dmge)

                else:
                    st.info(
                        "선택된 조건에 해당하는 "
                        "피해차량 차종 데이터가 없습니다."
                    )


        # ========================================================
        # 세 번째 줄: 사망사고 분석
        # 사망자 유형 그래프에서 '기타불명' 제외
        # ========================================================
        with st.expander("🚨 사망사고 분석", expanded=True):

            row3_col1, row3_col2 = st.columns(2)

            # --------------------------------------------------------
            # 사망자 유형
            # --------------------------------------------------------
            with row3_col1:
                st.markdown("**🚨 사망자 유형**")

                # 결측값·문자열 결측값·기타불명 제외
                fatal_type_series = (
                    filtered_df["fatal_type"]
                    .dropna()
                    .astype(str)
                    .str.strip()
                )

                fatal_type_series = fatal_type_series[
                    (fatal_type_series != "")
                    & (~fatal_type_series.str.lower().isin(
                        ["nan", "none"]
                    ))
                    & (fatal_type_series != "기타불명")
                ]

                fatal_type_counts = (
                    fatal_type_series
                    .value_counts()
                    .rename_axis("사망자 유형")
                    .reset_index(name="사망사고 건수")
                    .sort_values(
                        "사망사고 건수",
                        ascending=False,
                    )
                )

                if not fatal_type_counts.empty:
                    fig_fatal_type = px.bar(
                        fatal_type_counts,
                        x="사망사고 건수",
                        y="사망자 유형",
                        orientation="h",
                        text="사망사고 건수",
                        color="사망사고 건수",
                        color_continuous_scale=(
                            CHART_COLOR_SCALES[
                                "fatal_type"
                            ]
                        ),
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

                    fig_fatal_type.update_coloraxes(
                        showscale=False,
                    )

                    fig_fatal_type = apply_common_chart_style(
                        fig_fatal_type,
                        height=max(
                            330,
                            len(fatal_type_counts) * 38,
                        ),
                        horizontal=True,
                    )

                    # 건수가 가장 많은 유형부터 위쪽에 표시
                    fig_fatal_type.update_yaxes(
                        autorange="reversed",
                        categoryorder="array",
                        categoryarray=(
                            fatal_type_counts[
                                "사망자 유형"
                            ].tolist()
                        ),
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

                # --------------------------------------------------------
                # 사망자 연령대 순서
                # 실제 데이터 구간 기준
                # --------------------------------------------------------
                fatal_age_order = [
                    "20세 이하",
                    "21-30세",
                    "31-40세",
                    "41-50세",
                    "51-60세",
                    "61-64세",
                    "65세 이상",
                ]

                # 실제 결측값과 문자열 결측값 제거
                fatal_age_series = (
                    filtered_df["fatal_age_group"]
                    .dropna()
                    .astype(str)
                    .str.strip()
                )

                fatal_age_series = fatal_age_series[
                    (fatal_age_series != "")
                    & (~fatal_age_series.str.lower().isin(
                        ["nan", "none"]
                    ))
                ]

                # 지정한 연령대 순서대로 집계
                fatal_age_counts = (
                    fatal_age_series
                    .value_counts()
                    .reindex(
                        fatal_age_order,
                        fill_value=0,
                    )
                    .rename_axis("사망자 연령대")
                    .reset_index(name="사망사고 건수")
                )

                # 사고가 0건인 연령대는 그래프에서 제외
                fatal_age_counts = fatal_age_counts[
                    fatal_age_counts[
                        "사망사고 건수"
                    ] > 0
                ]

                if not fatal_age_counts.empty:
                    fig_fatal_age = px.bar(
                        fatal_age_counts,
                        x="사망사고 건수",
                        y="사망자 연령대",
                        orientation="h",
                        text="사망사고 건수",
                        color="사망사고 건수",
                        color_continuous_scale=(
                            CHART_COLOR_SCALES[
                                "fatal_age"
                            ]
                        ),
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

                    fig_fatal_age.update_coloraxes(
                        showscale=False,
                    )

                    fig_fatal_age = apply_common_chart_style(
                        fig_fatal_age,
                        height=max(
                            330,
                            len(fatal_age_counts) * 38,
                        ),
                        horizontal=True,
                    )

                    # 낮은 연령대가 위쪽부터 표시되도록 순서 고정
                    fig_fatal_age.update_yaxes(
                        autorange="reversed",
                        categoryorder="array",
                        categoryarray=(
                            fatal_age_counts[
                                "사망자 연령대"
                            ].tolist()
                        ),
                    )

                    render_plotly_chart(fig_fatal_age)

                else:
                    st.info(
                        "선택된 조건에 해당하는 "
                        "사망자 연령대 데이터가 없습니다."
                    )


        with st.expander(
            "⏱️ 상황별 사고 분석",
            expanded=True,
        ):

            # ========================================================
            # 네 번째 줄: 상황별 사고 분석
            # 요일별 → 시간대별 → 법규위반별
            # ========================================================

            row4_col1, row4_col2, row4_col3 = st.columns(3)

            # --------------------------------------------------------
            # 요일별
            # --------------------------------------------------------
            with row4_col1:
                st.markdown("**📅 요일별**")

                weekday_order = [
                    "월요일",
                    "화요일",
                    "수요일",
                    "목요일",
                    "금요일",
                    "토요일",
                    "일요일",
                ]

                weekday_short_name = {
                    "월요일": "월",
                    "화요일": "화",
                    "수요일": "수",
                    "목요일": "목",
                    "금요일": "금",
                    "토요일": "토",
                    "일요일": "일",
                }

                weekday_counts = (
                    filtered_df["dfk_dc"]
                    .dropna()
                    .astype(str)
                    .str.strip()
                    .value_counts()
                    .reindex(weekday_order, fill_value=0)
                    .rename_axis("요일원본")
                    .reset_index(name="사고건수")
                )

                # 화면에는 한 글자로 표시
                weekday_counts["요일"] = (
                    weekday_counts["요일원본"]
                    .map(weekday_short_name)
                )

                # 데이터에 존재하지 않는 요일은 제외
                weekday_counts = weekday_counts[
                    weekday_counts["사고건수"] > 0
                ]

                if not weekday_counts.empty:
                    fig_weekday = px.bar(
                        weekday_counts,
                        x="요일",
                        y="사고건수",
                        text="사고건수",
                        color="사고건수",
                        color_continuous_scale=(
                            CHART_COLOR_SCALES["weekday"]
                        ),
                    )

                    fig_weekday.update_traces(
                        texttemplate="%{text:,}",
                        textposition="outside",
                        textfont=dict(size=12),
                        cliponaxis=False,
                        marker_line_width=0,
                        hovertemplate=(
                            "<b>%{x}요일</b><br>"
                            "사고 건수: %{y:,}건"
                            "<extra></extra>"
                        ),
                    )

                    fig_weekday.update_xaxes(
                        categoryorder="array",
                        categoryarray=[
                            weekday_short_name[weekday]
                            for weekday in weekday_order
                        ],
                    )

                    fig_weekday.update_coloraxes(
                        showscale=False
                    )

                    fig_weekday = apply_common_chart_style(
                        fig_weekday,
                        height=350,
                    )

                    render_plotly_chart(fig_weekday)

                else:
                    st.info(
                        "선택된 조건에 해당하는 "
                        "요일 데이터가 없습니다."
                    )


            # --------------------------------------------------------
            # 시간대별
            # --------------------------------------------------------
            with row4_col2:
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

                else:
                    time_counts = (
                        time_counts
                        .sort_values("시간")
                    )

                time_counts["시간표시"] = (
                    time_counts["시간"]
                    .astype(int)
                    .map(lambda value: f"{value:02d}시")
                )

                if not time_counts.empty:
                    fig_time = px.bar(
                        time_counts,
                        x="시간표시",
                        y="사고건수",
                        text="사고건수",
                        color="사고건수",
                        color_continuous_scale=(
                            CHART_COLOR_SCALES["time"]
                        ),
                    )

                    fig_time.update_traces(
                        texttemplate="%{text:,}",
                        textposition="outside",
                        textfont=dict(size=11),
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
                        categoryarray=(
                            time_counts["시간표시"]
                            .tolist()
                        ),
                        tickangle=-45,
                    )

                    fig_time.update_coloraxes(
                        showscale=False
                    )

                    fig_time = apply_common_chart_style(
                        fig_time,
                        height=350,
                    )

                    render_plotly_chart(fig_time)

                else:
                    st.info(
                        "선택된 조건에 해당하는 "
                        "시간대 데이터가 없습니다."
                    )


            # --------------------------------------------------------
            # 법규위반별
            # --------------------------------------------------------
            with row4_col3:
                st.markdown("**⚖️ 법규위반별**")

                violt_counts = (
                    filtered_df["lrg_violt_1_dc"]
                    .dropna()
                    .value_counts()
                    .rename_axis("법규위반유형")
                    .reset_index(name="사고건수")
                )

                # 기타를 가장 아래에 표시하기 위한 정렬
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

                if not violt_counts.empty:
                    fig_violt = px.bar(
                        violt_counts,
                        x="사고건수",
                        y="법규위반유형",
                        orientation="h",
                        text="사고건수",
                        color="사고건수",
                        color_continuous_scale=(
                            CHART_COLOR_SCALES["violation"]
                        ),
                    )

                    fig_violt.update_traces(
                        texttemplate="%{text:,}",
                        textposition="outside",
                        textfont=dict(size=12),
                        cliponaxis=False,
                        marker_line_width=0,
                        hovertemplate=(
                            "<b>%{y}</b><br>"
                            "사고 건수: %{x:,}건"
                            "<extra></extra>"
                        ),
                    )

                    fig_violt.update_coloraxes(
                        showscale=False
                    )

                    fig_violt = apply_common_chart_style(
                        fig_violt,
                        height=max(
                            350,
                            len(violt_counts) * 35,
                        ),
                        horizontal=True,
                    )

                    render_plotly_chart(fig_violt)

                else:
                    st.info(
                        "선택된 조건에 해당하는 "
                        "법규위반 데이터가 없습니다."
                    )

    else:
        st.warning("현재 필터 조건에 해당하는 사고 데이터가 없습니다.")

