import geopandas as gpd
import hashlib
import json
import urllib.error
import urllib.parse
import urllib.request
import re
import html

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
# 0. Streamlit 페이지 기본 설정
# 반드시 다른 Streamlit 명령보다 먼저 실행
# -----------------------------------
st.set_page_config(
    page_title="TAAP - AI",
    page_icon="🗺️",
    layout="wide",
)

# 메인 제목은 필터링 결과가 확정된 뒤
# 현재 검색조건과 함께 본문 최상단에 렌더링한다.


# ============================================================
# 공통 UI 디자인
# - 사이드바 배경·그룹 카드·버튼
# - KPI 아래 현재 검색조건 텍스트
# ============================================================
st.markdown(
    """
    <style>
    [data-testid="stSidebar"] {
        background: linear-gradient(
            180deg,
            #F8FAFC 0%,
            #F1F5F9 100%
        );
        border-right: 1px solid #E2E8F0;
    }

    [data-testid="stSidebar"]
    [data-testid="stSidebarContent"] {
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

    /* ==========================================
    KPI 아래 현재 검색조건
    박스 없이 작은 보조 텍스트로 표시
    ========================================== */
    .search-condition-text {
        display: flex;
        align-items: center;
        justify-content: flex-end;   /* ← 추가: 전체 오른쪽 정렬 */
        flex-wrap: wrap;
        gap: 6px;
        margin-top: 8px;
        margin-bottom: 18px;
        padding-left: 2px;
        color: #64748B;
        font-size: 0.86rem;
        line-height: 1.6;
    }

    .search-condition-title {
        color: #475569;
        font-weight: 700;
        margin-right: 3px;
    }

    .search-condition-item {
        white-space: nowrap;
    }

    .search-condition-divider {
        color: #CBD5E1;
        margin: 0 3px;
    }

    /* 사이드바 접기 그룹 카드 */
    [data-testid="stSidebar"]
    [data-testid="stExpander"] {
        border: 1px solid #D7E0EA !important;
        border-radius: 12px !important;
        background-color: #FFFFFF !important;
        margin: 9px 0 !important;
        overflow: hidden !important;
        box-shadow:
            0 2px 5px rgba(15, 23, 42, 0.05) !important;
    }

    [data-testid="stSidebar"]
    [data-testid="stExpander"] details {
        border: 0 !important;
        background: transparent !important;
    }

    [data-testid="stSidebar"]
    [data-testid="stExpander"] summary {
        padding: 0.35rem 0.25rem !important;
        font-weight: 700 !important;
        color: #334155 !important;
    }

    [data-testid="stSidebar"]
    [data-testid="stExpander"] summary:hover {
        color: #1D4ED8 !important;
        background-color: #F8FAFC !important;
    }

    /* ============================================================
    GIS 공간분석: 숫자 입력 항목 라벨 글자 확대
    - 분석 반경 (m)
    - 사고다발지역 수
    ============================================================ */
    [data-testid="stMain"]
    div[data-testid="stForm"]
    div[data-testid="stNumberInput"] label,

    [data-testid="stMain"]
    div[data-testid="stForm"]
    div[data-testid="stNumberInput"] label p {
        color: #334155 !important;
        font-size: 1.5rem !important;
        line-height: 1.5 !important;
        font-weight: 700 !important;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# 사고 데이터 불러오기 (Parquet 파일 로드 & 캐싱)
# - 원본 데이터 정제와 category 변환은 Parquet 저장 전에 완료
# - Streamlit에서는 재치환하지 않고 필요한 파생 컬럼만 생성
@st.cache_data
def load_data():
    df = pd.read_parquet("정제완료(21~26).parquet")

    # 시간대 필터링용 숫자형 변환
    # 결측/비정상 시간은 0시로 강제 치환하지 않고 NaN으로 유지
    # → 0시가 포함된 검색에서 결측시간 사고가 잘못 포함되는 문제 방지
    if "occrrnc_time_dc" in df.columns:
        df["time_num"] = (
            df["occrrnc_time_dc"]
            .astype(str)
            .str.replace("시", "", regex=False)
        )
        df["time_num"] = pd.to_numeric(
            df["time_num"],
            errors="coerce",
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

    # 원본 필터 결과의 위치번호를 보존한 뒤
    # 좌표가 유효한 사고만 공간연산에 사용
    df_temp = target_df.reset_index(drop=True).copy()
    df_temp["_source_position"] = np.arange(len(df_temp))

    df_temp["latitude"] = pd.to_numeric(
        df_temp["latitude"],
        errors="coerce",
    )
    df_temp["longitude"] = pd.to_numeric(
        df_temp["longitude"],
        errors="coerce",
    )

    df_temp = df_temp.dropna(
        subset=["latitude", "longitude"]
    ).reset_index(drop=True)

    if df_temp.empty:
        return pd.DataFrame()

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

    # 반경 내 사고의 원본 위치 인덱스
    # 팝업·AI 분석에서 filtered_df.iloc로 다시 추출할 수 있도록 변환
    source_positions = df_temp["_source_position"].to_numpy()
    df_temp["nearby_indices"] = [
        source_positions[indices]
        for indices in nearby_indices_array
    ]

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
    """실무적으로 해석하기 쉬운 시간대 구간별 사고·사고 집계"""
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


def _compact_comparison_profile(dataframe, label):
    """
    선택집단과 비교하기 위한 압축 기준 프로필.
    토큰을 과도하게 늘리지 않도록 핵심 분포와 사고율만 포함한다.
    """
    if dataframe is None or dataframe.empty:
        return {
            "label": label,
            "total_accidents": 0,
            "note": "비교 가능한 사고 데이터가 없음",
        }

    total = int(len(dataframe))
    severity = (
        dataframe["acdnt_gae_dc"].astype(str)
        if "acdnt_gae_dc" in dataframe.columns
        else pd.Series(dtype="object")
    )

    fatal = int(severity.eq("사망사고").sum()) if not severity.empty else 0
    serious_or_fatal = int(
        severity.isin(["사망사고", "중상사고"]).sum()
    ) if not severity.empty else 0

    return {
        "label": label,
        "total_accidents": total,
        "fatal_accidents": fatal,
        "serious_or_fatal_accidents": serious_or_fatal,
        "serious_or_fatal_rate_percent": round(
            serious_or_fatal / total * 100,
            2,
        ) if total else 0,
        "distributions": {
            "by_hour": _hour_distribution(dataframe),
            "by_weekday": _distribution_dict(dataframe, "dfk_dc"),
            "by_accident_type": _distribution_dict(
                dataframe,
                "acdnt_hdc",
            ),
            "by_violation": _distribution_dict(
                dataframe,
                "lrg_violt_1_dc",
                top_n=8,
            ),
            "by_offending_vehicle": _distribution_dict(
                dataframe,
                "wrngdo_vhcle_asort_dc",
                top_n=8,
            ),
            "by_damaged_vehicle": _distribution_dict(
                dataframe,
                "dmge_vhcle_asort_dc",
                top_n=8,
            ),
        },
        "severity_rate_comparisons": {
            "by_hour": _severity_rate_table(
                dataframe,
                "occrrnc_time_dc",
                min_count=10,
                top_n=24,
            ),
            "by_accident_type": _severity_rate_table(
                dataframe,
                "acdnt_hdc",
                min_count=10,
                top_n=6,
            ),
            "by_violation": _severity_rate_table(
                dataframe,
                "lrg_violt_1_dc",
                min_count=10,
                top_n=8,
            ),
        },
    }


def make_ai_analysis_package(
    target_df,
    top_hotspot_dataframe,
    selected_filter_info,
    hotspot_radius_m,
    station_reference_df=None,
    city_reference_df=None,
):
    """
    현재 필터 결과를 AI가 비교·추론하기 좋은 집계 패키지로 생성한다.

    원칙
    - 모든 건수·비율 계산은 Python이 수행
    - 선택집단에는 상세 교차분석을 제공
    - 비교집단에는 토큰 절약을 위해 압축된 기준 통계만 제공
    - AI는 계산이 아니라 비교·패턴해석·대안평가를 담당
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
            "purpose": "대전경찰청 교통사고 의사결정 지원 분석",
            "data_scope": "현재 지도 필터에 해당하는 사고의 비식별 집계결과",
            "privacy_note": (
                "접수번호, 개인식별정보, 개별 사고 좌표와 원본 사고행은 "
                "외부 API에 포함하지 않음"
            ),
            "analysis_architecture": (
                "Python이 사실·건수·비율·교차표를 계산하고, "
                "AI는 선택집단과 비교집단의 차이, 결합패턴, 실무적 의미와 "
                "대응대안을 판단함"
            ),
        },
        "selected_filters": selected_filter_info,
        "selected_population": {
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
                    top_n=10,
                ),
                "by_offending_vehicle": _distribution_dict(
                    target_df,
                    "wrngdo_vhcle_asort_dc",
                    top_n=10,
                ),
                "by_offending_driver_age_group": _distribution_dict(
                    target_df,
                    "acdnt_age_1_dc",
                    top_n=10,
                ),
                "by_damaged_vehicle": _distribution_dict(
                    target_df,
                    "dmge_vhcle_asort_dc",
                    top_n=10,
                ),
                "by_victim_age_group": _distribution_dict(
                    target_df,
                    "acdnt_age_2_dc",
                    top_n=10,
                ),
                "by_fatal_type": _distribution_dict(
                    target_df,
                    "fatal_type",
                    top_n=10,
                ),
                "by_fatal_age_group": _distribution_dict(
                    target_df,
                    "fatal_age_group",
                    top_n=10,
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
                    max_rows=10,
                    max_columns=6,
                ),
                "offending_x_damaged_vehicle": _cross_table_dict(
                    target_df,
                    "wrngdo_vhcle_asort_dc",
                    "dmge_vhcle_asort_dc",
                    max_rows=8,
                    max_columns=8,
                ),
                "fatal_type_x_hour": _cross_table_dict(
                    target_df,
                    "fatal_type",
                    "occrrnc_time_dc",
                    max_rows=8,
                    max_columns=24,
                ),
            },
            "dominant_combinations": {
                "hour_accident_type_violation": _top_combinations(
                    target_df,
                    ["occrrnc_time_dc", "acdnt_hdc", "lrg_violt_1_dc"],
                    top_n=10,
                    min_count=3,
                ),
                "offending_damaged_accident_type": _top_combinations(
                    target_df,
                    [
                        "wrngdo_vhcle_asort_dc",
                        "dmge_vhcle_asort_dc",
                        "acdnt_hdc",
                    ],
                    top_n=10,
                    min_count=3,
                ),
                "weekday_hour_accident_type": _top_combinations(
                    target_df,
                    ["dfk_dc", "occrrnc_time_dc", "acdnt_hdc"],
                    top_n=10,
                    min_count=3,
                ),
            },
        },
        "reference_populations": {
            "same_period_selected_station_all_accidents": (
                _compact_comparison_profile(
                    station_reference_df,
                    "동일 기간 선택 관할 전체 사고",
                )
                if station_reference_df is not None
                else None
            ),
            "same_period_daejeon_all_accidents": (
                _compact_comparison_profile(
                    city_reference_df,
                    "동일 기간 대전 전체 사고",
                )
                if city_reference_df is not None
                else None
            ),
            "comparison_rule": (
                "비교집단은 사용자가 선택한 사고종별·차종·요일·시간 등 세부필터를 "
                "적용하지 않은 동일 기간 기준집단이다. 선택집단의 특이성을 판단하기 위한 "
                "기준선으로만 사용한다."
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
            "선택 조건에 따른 관찰자료이므로 인과관계를 직접 증명하지 않음",
            "교통량·보행량·주행거리 등 노출량 자료가 없어 단순 건수를 위험률로 해석할 수 없음",
            "도로구조, 신호현시, 실제 속도, 시야, 공사 여부 등 현장정보는 포함되지 않음",
            "음주·졸음·주의분산 등 원인변수가 데이터에 없다면 원인으로 확정할 수 없음",
            "비율 비교 시 표본이 작은 범주는 변동성이 크므로 신중히 해석해야 함",
            "시설개선이나 단속대책 확정 전 현장점검과 관계기관 협의가 필요함",
        ],
    }

    return package


def build_ai_prompt(report_type, analysis_json, web_enabled=False):
    """
    분석 목적별 프롬프트 생성.
    모델이 곧바로 문장을 채우기보다 비교·검증·대안평가를 거쳐 최종 답을 작성하도록 설계한다.
    """

    common_rules = f"""
당신은 대한민국 경찰의 교통안전 정책과 교통사고 분석을 지원하는 선임 분석관이다.
제공된 JSON은 Python이 계산한 비식별 교통사고 집계자료이며, selected_population은 현재 선택집단,
reference_populations는 선택집단의 특이성을 판단하기 위한 동일기간 기준집단이다.

[핵심 분석원칙]
1. 교통사고 건수·비율·사망사고 및 중상사고 비율·교차표에 관한 사실은 반드시 JSON 값만 사용한다.
2. 숫자를 새로 추정하거나 JSON에 없는 통계를 만들어내지 않는다.
3. 단순히 '가장 많다'는 이유만으로 인사이트라고 판단하지 않는다.
4. 선택집단과 기준집단을 비교하여 상대적으로 과대표현되거나 중요도가 높은 특성을 우선 찾는다.
5. 사고빈도와 사고심각도를 구분한다. '사고가 많은 시간'과 '발생 시 상해정도가 높은 시간'은 다른 관리대상일 수 있다.
6. 변수 간 결합패턴과 조건부 차이를 탐색하되 관찰자료만으로 인과관계 또는 상관관계가 입증되었다고 표현하지 않는다.
7. 안전운전불이행은 다른 구체적 위반을 적용하기 어려운 경우 보충적으로 적용되는 특성이 있으므로 단순 최다항목이라는 이유만으로 핵심 원인이라고 해석하지 않는다.
8. 표본이 작거나 차이가 미미하거나 기준집단에서도 동일한 현상이면 중요한 인사이트로 과장하지 않는다.
9. 대책은 '단속 강화', '홍보 강화' 같은 구호가 아니라 대상·시간·장소·방법·확인지표를 가능한 범위에서 구체화한다.
10. 데이터로 확인할 수 없는 도로형태·신호운영·속도·음주·졸음·교통량 등은 사실처럼 단정하지 않는다.
11. 분석할 때 내부적으로 다음 순서를 따른다: 이상패턴 탐색 → 기준집단 비교 → 결합패턴 확인 → 다른 설명 가능성 검토 → 경찰 개입 가능성 평가 → 최종 판단.
12. 의미 있는 판단이 적으면 억지로 개수를 채우지 않는다.
13. 답변에는 Markdown 취소선 문법(~~텍스트~~)이나 수정 전·후 표현을 남기지 않는다.
14. 문장을 수정할 필요가 있으면 최종적으로 확정된 문장만 출력한다.
15. 본문의 세부 설명·근거·실행사항 등을 나열할 때는 각 항목 앞에 하이픈(-)을 사용한다.
16. 하위 항목을 다시 원형 또는 중첩 bullet 목록으로 만들지 않는다.


[외부자료 사용]
웹검색 사용 가능 여부: {"가능" if web_enabled else "사용하지 않음"}
""".strip()

    if web_enabled:
        common_rules += """
16. 외부자료는 내부 통계의 숫자를 보충하거나 변경하기 위해 사용하지 않는다.
17. 사고특성의 일반적 설명, 효과적인 개입수단, 연구결과, 타 기관 사례를 검증할 필요가 있을 때만 웹검색을 사용한다.
18. 검색 시 경찰청, 한국도로교통공단, 국토교통부, 한국교통안전공단, 정부·지자체, 공공연구기관, 학술논문 등 신뢰도 높은 1차·공식 자료를 우선한다.
19. 외부연구에서 확인된 일반적 위험요인과 현재 대전 데이터에서 직접 확인된 사실을 명확히 구분한다.
20. 외부자료를 실제로 사용한 문장에는 출처가 드러나도록 인용을 유지하고, 존재하지 않는 기관·연구·사례를 만들지 않는다.
"""
    else:
        common_rules += """
21. 이번 분석에서는 외부사례나 연구를 사실처럼 인용하지 않는다. JSON과 일반적인 분석 논리에 집중한다.
22. 데이터에 없는 원인 설명이 필요하면 '가능한 설명이지만 현재 자료로 확인할 수 없음'이라고 한계를 표시한다.
"""

    if report_type == "insight":
        task_prompt = """
[분석 목적]
현재 선택집단에서 실무자가 통계표만 보고 놓치기 쉬운 '의사결정 가치가 있는 차이'를 발굴한다.
단순 순위보다 기준집단 대비 차이, 사고빈도와 사망사고 및 중상사고 등 비율의 불일치, 시간·요일·사고종별·위반·차종의 결합패턴,
그리고 경찰활동의 우선순위를 바꿀 수 있는 발견을 우선한다.

[출력 형식]
# AI 핵심 인사이트

## 분석 범위
선택조건과 분석규모를 간결하게 설명한다.

## 핵심 인사이트
의사결정 가치가 있는 항목만 최대 5개 작성한다. 2개만 의미 있으면 2개만 작성한다.

### 인사이트 n. 판단을 담은 제목
각 인사이트는 다음 흐름으로 1~3문단 작성한다.
- 무엇이 관찰되었는지
- 기준집단 또는 다른 범주와 비교하면 왜 특이한지
- 결합패턴상 어떤 추가 해석이 가능한지
- 경찰활동의 우선순위를 어떻게 바꿀 수 있는지
- 데이터만으로 확정할 수 없는 부분은 무엇인지

'상관성이 확인되었다', '원인이다' 같은 표현은 자료가 이를 직접 입증하지 않는 한 사용하지 않는다.

## 실무자가 주목할 결론
가장 중요한 판단을 최대 3개만 완결된 문장으로 정리한다.
""".strip()

    elif report_type == "hotspot":
        task_prompt = """
[분석 목적]
사고다발지점별 사고 프로파일을 비교하여 모든 지점에 같은 대책을 적용하지 않도록 관리 우선순위를 제시한다.

[출력 형식]
# 사고다발지점 AI 진단

## 분석 개요
분석 반경과 지점 수, 전체 분석규모를 간단히 정리한다.

## 지점별 진단
ranked_locations 순서대로 작성하되 단순 통계 복사는 피한다.

### TOP n. 중심주소
- 해당 지점에서 가장 구별되는 사고특성
- 시간·요일·사고종별·위반·차종이 함께 만드는 패턴
- 바로 경찰활동으로 연결 가능한 사항
- 현장 확인 전에는 결론낼 수 없는 시설·환경사항

특징이 뚜렷하지 않으면 명확히 그렇게 표현한다.

## 지점 간 비교와 관리유형
지점들을 공통유형 또는 서로 다른 관리유형으로 묶을 수 있는지 판단하고,
경력배치·순찰·단속·현장점검의 우선순위를 제시한다.

## 현장점검 체크포인트
데이터로 확인할 수 없지만 대책 확정 전에 확인해야 할 사항을 5개 이내로 제시한다.
""".strip()

    elif report_type == "strategy":
        task_prompt = """
[분석 목적]
현재 선택집단에서 발견된 위험을 실제 경찰 교통안전활동으로 전환한다.
필요한 경우 웹검색으로 효과적인 개입방법과 공신력 있는 선행사례·연구를 확인한다.

[출력 형식]
# 맞춤형 교통안전 대응전략

## 1. 전략 판단
사고가 '많은 것'과 '상해정도가 심각해지는 것'을 구분하고, 가장 먼저 개입해야 할 위험집단·시간·지점을 선정한다.
선정 이유는 선택집단과 기준집단의 차이로 설명한다.

## 2. 분야별 대응과제
### 🚔 경력배치 및 순찰
### ⚖️ 단속활동
### 📢 맞춤형 교육·홍보
### 🚦 시설·환경 현장점검

각 분야에서 데이터상 근거, 실행방법, 기대효과 등을 중심으로 검토한다.
외부 연구·사례를 사용했다면 해당 자료가 왜 현재 데이터에 적용 가능한지까지 설명한다.

## 3. 시행 우선순위
- 즉시 시행 / 현장 확인 후 시행 / 중기 관리 등 적절한 수준으로 구분하여 우선순위 및 그 근거를 제시한다.

각 단계에는 너무 많은 과제를 나열하지 말고 효과성과 실행가능성이 높은 과제를 우선 배치한다.

""".strip()

    elif report_type == "police_report":
        task_prompt = """
[분석 목적]
현재 조건의 교통사고 분석결과를 경찰 내부 검토·지휘보고에
실제로 활용할 수 있는 보고서로 작성한다.

이번 보고서에서는 File Search를 통해 제공되는 기존 경찰 보고서를
단순 참고문헌이 아니라 '보고서 작성 형식의 직접적인 모범자료'로 사용한다.
소제목 중 '현황 및 문제점', '향후 계획'의 비중은 줄이고 '추진 방안'의 비율이 60% 이상이 되도록 조정한다.
경찰은 집약된 보고서를 선호하므로, '현황 및 문제점'과 '향후 계획'에서 꼭 필요한 사항이 아니면 제외한다.
보고서의 글자수는 공백을 포함하여 3,000자 이내로 한다.

[개조식 계층 표기 원칙]
- 1단계 항목은 반드시 'ㅁ'으로 시작하고 들여쓰기하지 않는다.
- 2단계 항목은 반드시 'ㅇ'으로 시작하며 1칸 들여쓴다.
- 3단계 항목은 반드시 '-'로 시작하며 2칸 들여쓴다.
- 세부항목에 '*', '•', '·' 기호를 사용하지 않는다.
- 계층은 'ㅁ → ㅇ → -' 순서로 통일한다.

[기존 경찰 보고서 참고 원칙]

1. 최종 보고서를 작성하기 전에 File Search를 이용하여
   등록된 기존 경찰 보고서의 실제 작성방식을 충분히 확인한다.

2. 특히 다음 요소를 기존 보고서에서 직접 파악하고 최대한 유사하게 구현한다.
   - 제목과 소제목 구성
   - 'ㅁ', 'ㅇ', '-' 등을 이용한 계층 구조
   - 항목별 들여쓰기 방식
   - 한 항목의 문장 길이
   - 경찰 내부보고 특유의 간결한 개조식 표현
   - 현황·문제점에서 대책으로 이어지는 논리적 전개
   - 통계와 판단을 한 문장 안에서 연결하는 방식
   - 추진방안 및 향후계획의 구성방식
   - 실제 보고서에서 반복적으로 나타나는 표현과 문체

3. 단순히 이 프롬프트의 형식만 따르지 말고,
   File Search에서 확인한 실제 경찰 보고서의 공통적인 형식을
   최종 작성형식에 우선적으로 반영한다.

4. 다만 기존 경찰 보고서는 '형식과 문체의 참고자료'일 뿐
   현재 보고서의 사실자료가 아니다.

5. 따라서 기존 보고서에 포함된 다음 정보는
   현재 보고서의 사실처럼 가져오지 않는다.
   - 과거 교통사고 통계
   - 특정 사건·사고
   - 특정 장소
   - 기존 정책의 시행 여부
   - 과거 추진실적
   - 특정 인물 또는 기관 내부 현황

6. 현재 교통사고에 관한 수치와 통계적 사실은
   반드시 제공된 JSON만을 기준으로 한다.

7. 외부 연구결과·정책·타 기관 사례 등은
   필요한 경우 Web Search로 확인된 자료만 사실근거로 사용한다.

8. 기존 경찰 보고서의 문장을 그대로 장문 복사하지 말고,
   그 작성방식을 학습하여 현재 분석결과에 맞는 새로운 문장을 작성한다.


[보고서 내용 작성 원칙]

단순한 통계 나열이 아니라
'왜 이 문제를 우선 관리해야 하는지'
→ '어떤 활동을 할 것인지'
→ '어떻게 효과를 확인할 것인지'
가 자연스럽게 연결되도록 작성한다.

선택집단과 기준집단을 비교하여
단순히 사고가 많은 현상과 상대적으로 특이한 현상을 구분한다.

사고빈도와 중상자 이상 사고비율 등 위험성이 서로 다른 경우
이를 구분하여 관리방향을 제시한다.



[기본 내용]

다음 사항은 반드시 검토하되,
최종 항목명과 배열은 기존 경찰 보고서에서 확인한
실제 보고형식을 가능한 한 우선하여 적용한다.

ㅁ 현황 및 문제점
  ㅇ 선택집단에서 정책적 가치가 높은 문제를 우선 제시
  ㅇ 기준집단과 비교해 실제로 특이한 현상인지 검토
  ㅇ 사고빈도·사망사고 및 중상사고의 비율·시간·요일·사고종별·위반·차종 등의
     결합패턴을 필요한 범위에서 제시

ㅁ 추진 방안
  ㅇ 교통경력 배치 및 순찰, 단속활동
    - 필요한 경우 교통외근 차량순찰·거점관리와
      기동대 등 도보지원근무 역할을 구분
    - 법규위반·시간·사고종별·차종·다발지점을 연결해 구체화

  ㅇ 맞춤형 교육·홍보활동
    - 교통사고 현황을 토대로 대상·전달내용·방법을 구체적으로 제안

  ㅇ 시설개선 필요부분 검토
    - 사고패턴에 따른 현장점검 항목을 먼저 제시
    - 현장 확인 후 검토 가능한 개선방향을 제안

  ㅇ 관계기관 협업
    - 꼭 필요한 경우에만 지자체·도로관리청 등과의 역할을 제안. 반드시 넣을 필요는 없음.

ㅁ 향후 계획
  ㅇ 추진 방안의 내용 중 시급성 등을 고려, 우선순위가 높은 것부터 추진일정을 구체적으로 제시 


[최종 문체 규칙]

1. 기존 경찰 보고서의 실제 개조식 문체를 최우선으로 참고한다.
2. 한 항목에 너무 많은 설명을 넣지 않는다.
3. 같은 통계를 여러 항목에서 반복하지 않는다.
4. 수치는 판단에 필요한 경우에만 정확히 병기한다.
5. '강화 필요', '적극 추진' 같은 추상적인 표현만 사용하지 말고
   가능한 경우 대상·시간·장소·방법을 구체화한다.
6. 결재선·문서번호·수신처·시행일자는 임의로 만들지 않는다.
7. Markdown 표는 사용하지 않는다.
8. Markdown 취소선(~~)이나 수정 흔적은 사용하지 않는다.
9. 최종 출력에는 완성된 보고서만 제시한다.
10. 
""".strip()

    else:
        raise ValueError(f"지원하지 않는 교통안전 실무보고서 유형입니다: {report_type}")

    return f"""
{common_rules}

{task_prompt}

[분석용 JSON]
{analysis_json}
""".strip()



def _safe_usage_value(obj, name, default=0):
    """OpenAI SDK 객체/딕셔너리 어느 형태에서도 usage 값을 안전하게 읽는다."""
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _format_response_with_clickable_citations(response):
    """
    Web Search URL annotation을 Streamlit Markdown에서 클릭 가능한 출처 링크로 변환한다.
    검색을 사용하지 않은 응답은 response.output_text를 그대로 반환한다.
    """
    try:
        output_items = response.output or []
    except Exception:
        return getattr(response, "output_text", "") or ""

    rendered_parts = []
    all_sources = []

    for item in output_items:
        item_type = (
            item.get("type")
            if isinstance(item, dict)
            else getattr(item, "type", "")
        )
        if item_type != "message":
            continue

        content_items = (
            item.get("content", [])
            if isinstance(item, dict)
            else getattr(item, "content", [])
        ) or []

        for content in content_items:
            content_type = (
                content.get("type")
                if isinstance(content, dict)
                else getattr(content, "type", "")
            )
            if content_type != "output_text":
                continue

            text_value = (
                content.get("text", "")
                if isinstance(content, dict)
                else getattr(content, "text", "")
            ) or ""

            annotations = (
                content.get("annotations", [])
                if isinstance(content, dict)
                else getattr(content, "annotations", [])
            ) or []

            citation_groups = {}
            for annotation in annotations:
                annotation_type = (
                    annotation.get("type")
                    if isinstance(annotation, dict)
                    else getattr(annotation, "type", "")
                )
                if annotation_type != "url_citation":
                    continue

                end_index = (
                    annotation.get("end_index")
                    if isinstance(annotation, dict)
                    else getattr(annotation, "end_index", None)
                )
                url = (
                    annotation.get("url", "")
                    if isinstance(annotation, dict)
                    else getattr(annotation, "url", "")
                ) or ""
                title = (
                    annotation.get("title", "")
                    if isinstance(annotation, dict)
                    else getattr(annotation, "title", "")
                ) or url

                if end_index is None or not url:
                    continue

                end_index = max(0, min(int(end_index), len(text_value)))
                citation_groups.setdefault(end_index, [])
                if url not in [source[0] for source in citation_groups[end_index]]:
                    citation_groups[end_index].append((url, title))

                if url not in [source[0] for source in all_sources]:
                    all_sources.append((url, title))

            # 뒤에서부터 링크를 삽입해야 annotation 문자위치가 어긋나지 않는다.
            formatted_text = text_value
            for end_index in sorted(citation_groups.keys(), reverse=True):
                links = " ".join(
                    f"[출처{idx}]({url})"
                    for idx, (url, _) in enumerate(
                        citation_groups[end_index],
                        start=1,
                    )
                )
                formatted_text = (
                    formatted_text[:end_index]
                    + f" {links}"
                    + formatted_text[end_index:]
                )

            rendered_parts.append(formatted_text)

    if not rendered_parts:
        return getattr(response, "output_text", "") or ""

    rendered_text = "\n\n".join(rendered_parts)

    if all_sources:
        source_lines = ["\n\n---\n\n### 웹 검색 참고 출처"]
        for index, (url, title) in enumerate(all_sources, start=1):
            safe_title = str(title).replace("\n", " ").strip() or url
            source_lines.append(f"{index}. [{safe_title}]({url})")
        rendered_text += "\n".join(source_lines)

    return rendered_text


def _count_web_search_calls(response):
    """Responses API 출력에서 실제 web_search 호출 횟수를 계산한다."""
    try:
        output_items = response.output or []
    except Exception:
        return 0

    count = 0
    for item in output_items:
        item_type = (
            item.get("type")
            if isinstance(item, dict)
            else getattr(item, "type", "")
        )
        if item_type in {"web_search_call", "web_search"}:
            count += 1
    return count


def _estimate_ai_cost_usd(model_name, usage, web_search_calls=0):
    """
    화면 확인용 예상비용.
    2026-08 현재 공개 표준요금 기준이며 실제 청구액과 차이가 날 수 있다.
    """
    model_prices = {
        "gpt-5.6": {"input": 5.0, "cached": 0.5, "output": 30.0},
        "gpt-5.6-sol": {"input": 5.0, "cached": 0.5, "output": 30.0},
        "gpt-5.6-terra": {"input": 2.0, "cached": 0.2, "output": 12.0},
        "gpt-5.6-luna": {"input": 0.2, "cached": 0.02, "output": 1.2},
    }

    price = model_prices.get(model_name)
    if not price or usage is None:
        return None

    input_tokens = int(_safe_usage_value(usage, "input_tokens", 0) or 0)
    output_tokens = int(_safe_usage_value(usage, "output_tokens", 0) or 0)

    input_details = _safe_usage_value(usage, "input_tokens_details", None)
    cached_tokens = int(
        _safe_usage_value(input_details, "cached_tokens", 0) or 0
    )
    uncached_tokens = max(input_tokens - cached_tokens, 0)

    token_cost = (
        uncached_tokens / 1_000_000 * price["input"]
        + cached_tokens / 1_000_000 * price["cached"]
        + output_tokens / 1_000_000 * price["output"]
    )

    # Web search: $10 / 1,000 calls = $0.01 / call.
    # 검색 콘텐츠 토큰은 usage 입력토큰에 포함되어 모델 입력요금으로 계산된다.
    web_tool_cost = web_search_calls * 0.01

    return round(token_cost + web_tool_cost, 4)


def generate_ai_report(analysis_package, report_type):
    """
    OpenAI Responses API로 목적별 AI 분석을 생성한다.

    - 핵심 인사이트: Terra / medium / 웹검색 없음
    - 다발지점 진단: Terra / medium / 웹검색 없음
    - 대응전략: Terra / high / 필요 시 웹검색
    - 교통안전 실무보고서: Terra / high / 필요 시 웹검색
    """
    if OpenAI is None:
        raise RuntimeError(
            "openai 라이브러리가 설치되지 않았습니다. "
            "requirements.txt에 최신 'openai'를 추가하세요."
        )

    try:
        api_key = st.secrets["OPENAI_API_KEY"]
    except KeyError as exc:
        raise RuntimeError(
            "Streamlit Secrets에 OPENAI_API_KEY를 등록하세요."
        ) from exc

    report_configs = {
        "insight": {
            "model": "gpt-5.6-terra",
            "reasoning_effort": "medium",
            "web_search": False,
            "max_output_tokens": 12000,
            "secret_key": "OPENAI_MODEL_INSIGHT",
        },
        "hotspot": {
            "model": "gpt-5.6-terra",
            "reasoning_effort": "medium",
            "web_search": False,
            "max_output_tokens": 12000,
            "secret_key": "OPENAI_MODEL_HOTSPOT",
        },
        "strategy": {
            "model": "gpt-5.6-terra",
            "reasoning_effort": "high",
            "web_search": True,
            "max_output_tokens": 18000,
            "secret_key": "OPENAI_MODEL_STRATEGY",
        },
        "police_report": {
            "model": "gpt-5.6-terra",
            "reasoning_effort": "high",
            "web_search": True,
            "max_output_tokens": 22000,
            "secret_key": "OPENAI_MODEL_REPORT",
        },
    }

    if report_type not in report_configs:
        raise ValueError(f"지원하지 않는 교통안전 실무보고서 유형입니다: {report_type}")

    config = report_configs[report_type].copy()

    model_name = config["model"]

    client = OpenAI(api_key=api_key)

    analysis_json = json.dumps(
        analysis_package,
        ensure_ascii=False,
        separators=(",", ":"),
    )

    prompt = build_ai_prompt(
        report_type=report_type,
        analysis_json=analysis_json,
        web_enabled=config["web_search"],
    )

# ============================================================
# OpenAI Responses API 요청 구성
# ============================================================

    request_kwargs = {
        "model": model_name,
        "input": prompt,
        "reasoning": {
            "effort": config["reasoning_effort"],
        },
        "max_output_tokens": config["max_output_tokens"],
    }


    # ============================================================
    # AI 도구 구성
    #
    # - 핵심 인사이트 : 도구 없음
    # - 다발지점 진단 : 도구 없음
    # - 대응전략      : Web Search
    # - 교통안전 실무보고서     : Web Search + 경찰보고서 File Search
    # ============================================================

    tools = []


    # ------------------------------------------------------------
    # Web Search
    # 대응전략 / 교통안전 실무보고서에서만 사용
    # ------------------------------------------------------------
    if config["web_search"]:
        tools.append(
            {
                "type": "web_search",
            }
        )


    # ------------------------------------------------------------
    # 경찰보고서 File Search
    # 교통안전 실무보고서(police_report)에서만 사용
    # ------------------------------------------------------------
    if report_type == "police_report":

        try:
            report_vector_store_id = st.secrets.get(
                "OPENAI_REPORT_VECTOR_STORE_ID",
                "",
            )
        except Exception:
            report_vector_store_id = ""

        if report_vector_store_id:

            tools.append(
                {
                    "type": "file_search",
                    "vector_store_ids": [
                        report_vector_store_id
                    ],
                    # 너무 많은 문서 조각을 가져오지 않도록 제한
                    "max_num_results": 10,
                }
            )

        else:

            raise RuntimeError(
                "교통안전 실무보고서용 Vector Store ID가 없습니다. "
                "Streamlit Secrets에 "
                "OPENAI_REPORT_VECTOR_STORE_ID를 등록하세요."
            )


    # ------------------------------------------------------------
    # 사용할 도구가 있을 때만 Responses API에 전달
    # ------------------------------------------------------------
    if tools:
        request_kwargs["tools"] = tools


    # ============================================================
    # AI 응답 생성
    # ============================================================

    response = client.responses.create(
        **request_kwargs
    )

    if getattr(response, "status", None) == "incomplete":
        incomplete_details = getattr(response, "incomplete_details", None)
        reason = getattr(incomplete_details, "reason", "알 수 없음")
        if not response.output_text:
            raise RuntimeError(
                f"AI 응답이 완성되기 전에 종료되었습니다. 사유: {reason}"
            )

    usage = getattr(response, "usage", None)
    web_search_calls = _count_web_search_calls(response)

    output_details = _safe_usage_value(
        usage,
        "output_tokens_details",
        None,
    )

    usage_info = {
        "model": model_name,
        "reasoning_effort": config["reasoning_effort"],
        "web_search_enabled": config["web_search"],
        "web_search_calls": int(web_search_calls),
        "input_tokens": int(
            _safe_usage_value(usage, "input_tokens", 0) or 0
        ),
        "cached_input_tokens": int(
            _safe_usage_value(
                _safe_usage_value(usage, "input_tokens_details", None),
                "cached_tokens",
                0,
            ) or 0
        ),
        "output_tokens": int(
            _safe_usage_value(usage, "output_tokens", 0) or 0
        ),
        "reasoning_tokens": int(
            _safe_usage_value(
                output_details,
                "reasoning_tokens",
                0,
            ) or 0
        ),
        "total_tokens": int(
            _safe_usage_value(usage, "total_tokens", 0) or 0
        ),
    }

    usage_info["estimated_cost_usd"] = _estimate_ai_cost_usd(
        model_name=model_name,
        usage=usage,
        web_search_calls=web_search_calls,
    )

    # ============================================================
    # AI 응답 최종 표시문 정리
    # - Web Search 출처 링크 변환
    # - Markdown 취소선 오작동 방지
    # - 시간범위의 ~를 Markdown과 충돌하지 않는 ∼로 변환
    # ============================================================

    display_text = _format_response_with_clickable_citations(response)

    # ------------------------------------------------------------
    # Markdown 취소선 방지
    #
    # Streamlit에서는 단일 ~도 문맥에 따라 취소선으로 해석될 수 있음
    #
    # 예)
    # 12~13시 → 12∼13시
    # 17~18시 → 17∼18시
    # 22시~06시 → 22시∼06시
    #
    # ∼(U+223C)은 화면상 물결표와 거의 동일하지만
    # Markdown 취소선 문법으로 해석되지 않음
    # ------------------------------------------------------------
    display_text = display_text.replace("~", "∼")

    return display_text, analysis_json, usage_info

def format_police_report_display(report_text, suppress_first_title=False):
    """
    AI 경찰보고서의 화면 표시용 형식을 정규화한다.

    - 보고서 제목: 크게 표시
    - 1단계(ㅁ): 소제목으로 강조
    - 2단계(ㅇ): 1칸 들여쓰기
    - 3단계(-): 4칸 들여쓰기

    교통안전 실무보고서는 unsafe_allow_html=True로 표시하므로 Markdown 제목을
    HTML 요소로 직접 변환하여 Streamlit Cloud에서도 크기가 확실히 적용되도록 한다.
    """
    if not report_text:
        return report_text

    formatted_lines = []
    first_content_seen = False
    section_seen = False
    first_title_suppressed = False

    for raw_line in str(report_text).splitlines():
        stripped = raw_line.strip()

        if not stripped:
            formatted_lines.append("")
            continue

        # Markdown 제목을 HTML 제목 클래스로 변환
        if stripped.startswith("### "):
            content = html.escape(stripped[4:].strip())
            formatted_lines.append(
                f'<div class="police-report-subtitle">{content}</div>'
            )
            first_content_seen = True
            continue

        if stripped.startswith("## "):
            content = html.escape(stripped[3:].strip())
            if suppress_first_title and not first_title_suppressed:
                first_title_suppressed = True
                first_content_seen = True
                continue
            formatted_lines.append(
                f'<div class="police-report-title">{content}</div>'
            )
            first_content_seen = True
            continue

        if stripped.startswith("# "):
            content = html.escape(stripped[2:].strip())
            if suppress_first_title and not first_title_suppressed:
                first_title_suppressed = True
                first_content_seen = True
                continue
            formatted_lines.append(
                f'<div class="police-report-title">{content}</div>'
            )
            first_content_seen = True
            continue

        # 1단계: 원문의 ㅁ 표시는 제거하고 번호형 소제목으로 표시
        if stripped.startswith(("ㅁ", "□", "■", "▪")):
            content = stripped.lstrip("ㅁ□■▪ \t").strip()
            formatted_lines.append(
                f'<div class="police-report-subtitle">{html.escape(content)}</div>'
            )
            first_content_seen = True
            section_seen = True
            continue

        # 2단계: 기호와 본문을 분리해 줄바꿈 이후에도 본문 첫 글자에 맞춤
        if stripped.startswith(("ㅇ", "○", "◦")):
            content = stripped.lstrip("ㅇ○◦ \t").strip()
            formatted_lines.append(
                '<div class="police-report-level2">'
                '<span class="police-report-level2-bullet">ㅇ</span>'
                f'<span class="police-report-level2-text">{html.escape(content)}</span>'
                '</div>'
            )
            first_content_seen = True
            continue

        # 3단계: 기호와 본문을 분리해 줄바꿈 이후에도 본문 첫 글자에 맞춤
        if stripped.startswith(("-", "*", "•", "·")):
            content = stripped.lstrip("-*•· \t").strip()
            formatted_lines.append(
                '<div class="police-report-level3">'
                '<span class="police-report-level3-bullet">-</span>'
                f'<span class="police-report-level3-text">{html.escape(content)}</span>'
                '</div>'
            )
            first_content_seen = True
            continue

        # 첫 일반 문장이 ㅁ 항목보다 앞에 있으면 보고서 제목으로 처리
        if not first_content_seen and not section_seen:
            formatted_lines.append(
                f'<div class="police-report-title">{html.escape(stripped)}</div>'
            )
            first_content_seen = True
            continue

        # 그 밖의 일반 문장은 본문으로 표시
        formatted_lines.append(
            f'<div class="police-report-body">{html.escape(stripped)}</div>'
        )
        first_content_seen = True

    return "\n".join(formatted_lines)


def strip_first_markdown_title(report_text):
    """브리핑 표지와 중복되는 첫 Markdown 제목만 제거한다."""
    lines = str(report_text or "").splitlines()
    for index, line in enumerate(lines):
        if line.strip().startswith("# "):
            del lines[index]
            break
    return "\n".join(lines).lstrip()


def extract_ai_report_summary(report_text, report_type):
    """생성 결과에서 브리핑 상단에 표시할 대표 판단 한 문장을 찾는다."""
    lines = [line.strip() for line in str(report_text or "").splitlines()]
    preferred_sections = {
        "insight": ("실무자가 주목할 결론", "핵심 인사이트"),
        "hotspot": ("지점 간 비교", "관리유형", "지점별 진단"),
        "strategy": ("전략 판단", "시행 우선순위"),
        "police_report": ("현황 및 문제점", "추진 방안"),
    }

    start_index = 0
    for section_name in preferred_sections.get(report_type, ()):
        matched_index = next(
            (
                index for index, line in enumerate(lines)
                if section_name in line.lstrip("#ㅁ□■▪ 0123456789.")
            ),
            None,
        )
        if matched_index is not None:
            start_index = matched_index + 1
            break

    candidates = lines[start_index:] + lines[:start_index]
    for line in candidates:
        cleaned = re.sub(r"^[#ㅁ□■▪ㅇ○◦\-*•·\s]+", "", line).strip()
        if not cleaned or len(cleaned) < 18:
            continue
        if cleaned.startswith(("분석 범위", "분석 개요", "핵심 인사이트")):
            continue
        return cleaned[:240]

    return "현재 선택조건의 교통사고 자료를 바탕으로 주요 특성과 대응 방향을 분석했습니다."


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



# ============================================================
# 0-3. AI 자연어 조건검색
# - 사용자의 자연어를 기존 12개 필터 값으로 변환
# - GPT는 '해석'만 담당하고 실제 필터링은 기존 Streamlit 로직이 수행
# - 모호하거나 지원하지 않는 표현은 임의 적용하지 않고 확인 메시지 반환
# ============================================================

def parse_natural_language_filters(user_query, filter_catalog):
    """
    자연어 검색문을 현재 지도 필터 구조에 맞는 JSON으로 변환한다.

    설계 원칙
    - gpt-5.6-luna + Structured Outputs 사용
    - 자연어 → 필터값 변환만 수행
    - 데이터 계산/검색은 기존 Python 코드가 수행
    - 명시되지 않은 조건은 전체로 처리
    - 모호한 표현은 status='need_clarification'으로 반환
    """
    if OpenAI is None:
        raise RuntimeError(
            "openai 라이브러리가 설치되지 않았습니다. "
            "requirements.txt에 최신 'openai'를 추가하세요."
        )

    try:
        api_key = st.secrets["OPENAI_API_KEY"]
    except KeyError as exc:
        raise RuntimeError(
            "Streamlit Secrets에 OPENAI_API_KEY를 등록하세요."
        ) from exc

    client = OpenAI(api_key=api_key)

    schema = {
        "type": "object",
        "properties": {
            "status": {
                "type": "string",
                "enum": ["ready", "need_clarification"],
            },
            "clarification_question": {
                "type": "string",
            },
            "interpretation_summary": {
                "type": "string",
            },
            "station": {
                "type": ["string", "null"],
            },
            "start_year_month": {
                "type": ["string", "null"],
            },
            "end_year_month": {
                "type": ["string", "null"],
            },
            "accident_severity": {
                "type": "array",
                "items": {"type": "string"},
            },
            "accident_type": {
                "type": ["string", "null"],
            },
            "start_time": {
                "type": ["integer", "null"],
                "minimum": 0,
                "maximum": 23,
            },
            "end_time": {
                "type": ["integer", "null"],
                "minimum": 0,
                "maximum": 23,
            },
            "weekdays": {
                "type": "array",
                "items": {"type": "string"},
            },
            "offending_vehicles": {
                "type": "array",
                "items": {"type": "string"},
            },
            "offending_driver_age_groups": {
                "type": "array",
                "items": {"type": "string"},
            },
            "damaged_vehicles": {
                "type": "array",
                "items": {"type": "string"},
            },
            "victim_age_groups": {
                "type": "array",
                "items": {"type": "string"},
            },
            "fatal_types": {
                "type": "array",
                "items": {"type": "string"},
            },
            "fatal_age_groups": {
                "type": "array",
                "items": {"type": "string"},
            },
            "weather": {
                "type": "array",
                "items": {"type": "string"},
            },
            "violations": {
                "type": "array",
                "items": {"type": "string"},
            },
            "hotspot_requested": {
                "type": "boolean",
            },
            "hotspot_radius": {
                "type": ["integer", "null"],
                "minimum": 50,
                "maximum": 300,
            },
            "hotspot_top_n": {
                "type": ["integer", "null"],
                "minimum": 1,
                "maximum": 10,
            },
        },
        "required": [
            "status",
            "clarification_question",
            "interpretation_summary",
            "station",
            "start_year_month",
            "end_year_month",
            "accident_severity",
            "accident_type",
            "start_time",
            "end_time",
            "weekdays",
            "offending_vehicles",
            "offending_driver_age_groups",
            "damaged_vehicles",
            "victim_age_groups",
            "fatal_types",
            "fatal_age_groups",
            "weather",
            "violations",
            "hotspot_requested",
            "hotspot_radius",
            "hotspot_top_n",
        ],
        "additionalProperties": False,
    }

    available_period = filter_catalog.get("available_period", [])
    latest_available_period = (
        str(available_period[-1])
        if available_period and available_period[-1]
        else "데이터의 최신 발생연월"
    )

    try:
        latest_period_for_example = pd.Period(
            latest_available_period,
            freq="M",
        )
        recent_three_year_start = (
            latest_period_for_example - 35
        ).strftime("%Y-%m")
    except Exception:
        recent_three_year_start = "최신월에서 35개월 전"

    rules = f"""
[경찰 교통사고 자연어 검색 규칙]

1. '보행자 사고'는 피해차량=보행자로 해석한다.
   - '보행자 사망사고' = 피해차량 보행자 + 사고분류 사망사고
   - '보행자 중상사고' = 피해차량 보행자 + 사고분류 중상사고

2. 그 외 'OOO 사고'에서 OOO가 차종이면 가해차량=OOO로 해석한다.
   - 예: 이륜차 사고 = 가해차량 이륜
   - 예: 화물차 사망사고 = 가해차량 화물 + 사고분류 사망사고
   - 예: 승용차 중상사고 = 가해차량 승용 + 사고분류 중상사고

3. 차종 동의어를 실제 필터값으로 정규화한다.
   - 승용차→승용, 승합차→승합, 화물차→화물
   - 오토바이/이륜차→이륜
   - 원동기장치자전거/원동기→원동기
   - 개인형이동장치/퍼스널모빌리티/PM→PM
   - 자전거→자전거, ATV→ATV

4. 시간 표현
   - 심야 = 00시~03시
   - 주간 = 06시~17시
   - 야간/밤 = 18시~익일 05시
   - 사용자가 구체적인 시작·종료시간을 직접 말하면 그 시간이 위 규칙보다 우선한다.
   - 자정을 넘는 범위도 그대로 start_time > end_time 형태로 반환한다.

5. 요일 표현
   - 평일 = 월요일, 화요일, 수요일, 목요일, 금요일
   - 주말 = 토요일, 일요일
   - 사용자가 특정 요일을 말하면 해당 요일만 반환한다.

6. 기간 표현
   - '25년', '25년도', '2025년' = start_year_month="2025-01",
     end_year_month="2025-12"
   - '24년' = start_year_month="2024-01",
     end_year_month="2024-12"
   - '최근 N년'은 available_period의 마지막 값인
     {latest_available_period}을 끝단으로 정확히 N×12개월을 계산한다.
   - 시작월은 최신월에서 (N×12-1)개월 전이다.
     예: 현재 데이터에서 최근 3년 =
     start_year_month="{recent_three_year_start}",
     end_year_month="{latest_available_period}"
   - 사용자가 특정 월을 말하면 해당 월을 YYYY-MM 형식으로 반환한다.
   - start_year_month와 end_year_month는 반드시 YYYY-MM 형식으로 반환한다.

7. '위험지역', '사고 많은 곳', '사고다발지역', '사고다발지점'은
   hotspot_requested=true로 해석한다.
   다른 필터조건이 함께 있으면 그 조건을 먼저 적용한 뒤 사고다발지점을 탐색한다.

8. 날씨 표현
   - 사용자의 문장에서 날씨 상태를 의미하는 표현이 있으면 weather 필터에 반드시 반영한다.
   - '비가 오는 날', '비 오는 날', '비오는 날', '우천', '강우' → weather=["비"]
   - '눈이 오는 날', '눈 오는 날', '눈오는 날', '강설' → weather=["눈"]
   - '안개 낀 날', '안개가 낀 날', '안개 발생', '안개' → weather=["안개"]
   - '흐린 날', '흐림', '흐린 날씨' → weather=["흐림"]
   - '맑은 날', '맑음', '화창한 날' → weather=["맑음"]
   - 여러 날씨가 함께 명시되면 해당 값을 모두 weather 배열에 넣는다.
   - 단, '악천후 사고'처럼 구체적인 날씨 상태가 명시되지 않은 표현은
     자동으로 범위를 정하지 않고 status='need_clarification'으로 한다.
     이 경우 어떤 날씨(예: 비, 눈, 안개)를 포함할지 구체적으로 입력하도록 요청한다.

9. 다음 표현은 별도 규칙으로 만들지 않는다.
   - 차량 사고
   - 인명피해 사고
   - 법규위반 사고
   사용자가 이런 표현만으로 검색하면 정확한 조건을 확인하도록 요청한다.

10. 명시되지 않은 필터는 null 또는 빈 배열로 반환한다.
    기존 화면의 이전 필터값을 추측하거나 유지하려 하지 않는다.

11. 지원되는 필터 범위를 넘어서는 조건은 임의로 다른 필터에 끼워 맞추지 않는다.
    정확한 필터가 없으면 status='need_clarification'으로 하고 이유를 짧게 안내한다.

12. 사용자가 '사망자 유형'이나 '사망자 연령'을 명시적으로 말한 경우에만
    fatal_types/fatal_age_groups를 사용한다.
    '보행자 사망사고'는 fatal_types가 아니라 피해차량=보행자로 처리한다.

13. 경찰 실무에서 사용하지 않는 사고분류 용어를 임의로 만들어내지 않는다.

14. 고령자·어린이 관련 연령대 표현

   가. 고령 보행자
   - '고령보행자', '고령 보행자', '고령자 보행사고',
     '노인 보행자', '어르신 보행사고' 등
     고령자와 보행자가 연결된 표현은 다음 두 조건을 함께 적용한다.
   - damaged_vehicles=["보행자"]
   - victim_age_groups=["65세 이상"]
   - 이 경우 offending_driver_age_groups에는 값을 넣지 않는다.

   나. 고령 운전자
   - '고령운전자', '고령 운전자', '고령운전',
     '노인 운전자', '어르신 운전자' 등
     고령자와 운전자가 연결된 표현은
     offending_driver_age_groups=["65세 이상"]으로 해석한다.
   - 이 경우 고령이라는 이유만으로 victim_age_groups에는 값을 넣지 않는다.

   다. 그 밖의 고령 관련 표현
   - '고령'이 피해자·보행자·운전자 중 누구를 의미하는지
     주변 문맥을 종합하여 판단한다.
   - 문맥상 피해자 또는 보행자를 의미하면
     victim_age_groups=["65세 이상"]으로 반환한다.
   - 문맥상 운전자를 의미하면
     offending_driver_age_groups=["65세 이상"]으로 반환한다.
   - 피해자와 운전자 중 어느 쪽인지 합리적으로 판단할 수 없으면
     임의로 가해운전자로 설정하지 말고
     status="need_clarification"으로 반환한다.

   라. 어린이 관련 표현
   - 사용자 문장에 '어린이'가 포함된 경우
     victim_age_groups=["12세 이하"]로 반환한다.
   - '어린이 보행자', '어린이 보행사고'는
     damaged_vehicles=["보행자"]와
     victim_age_groups=["12세 이하"]를 함께 적용한다.
   - 어린이를 가해운전자 연령대로 해석하지 않는다.

   마. 사용자가 운전자 또는 피해자의 구체적인 연령대를 명시하면
   실제 사용 가능한 각 연령대 필터값 범위 안에서 반환한다.

15. 사고다발지점 탐색조건
   - 사용자가 분석반경을 명시하면 hotspot_radius에 미터 단위 정수로 반환한다.
   - 예: '반경 150m', '150미터 반경', '분석반경 150미터'
     → hotspot_radius=150
   - 사용자가 사고다발지점 개수를 명시하면 hotspot_top_n에 정수로 반환한다.
   - 예: '사고다발지점 3개', '상위 3곳', 'TOP 3'
     → hotspot_top_n=3
   - 사고다발지점 탐색을 요청했지만 반경이나 개수를 명시하지 않은 경우
     hotspot_radius=null, hotspot_top_n=null로 반환한다.
   - 반경 또는 사고다발지점 개수를 명시한 문장은
     hotspot_requested=true로 해석한다.
""".strip()

    catalog_text = json.dumps(
        filter_catalog,
        ensure_ascii=False,
        indent=2,
        default=str,
    )

    prompt = f"""
당신은 경찰 교통사고 분석시스템의 자연어 조건검색 해석기다.
사용자가 입력한 문장을 아래 시스템의 실제 필터값으로 변환하라.

중요:
- 설명문을 작성하지 말고 지정된 JSON Schema에 맞는 값만 반환한다.
- 범주형 필터값은 반드시 아래 [실제 사용 가능한 필터값] 중에서 선택한다.
- start_year_month와 end_year_month는 예외로,
  available_period가 나타내는 시작월~종료월 범위 안에서
  YYYY-MM 형식의 임의 월을 반환할 수 있다.
- 명확하면 status='ready'.
- 모호하거나 지원할 수 없는 조건이면 status='need_clarification'.
- clarification_question은 ready일 때 빈 문자열로 한다.
- interpretation_summary에는 화면에 보여줄 짧은 한국어 해석결과를 작성한다.

{rules}

[실제 사용 가능한 필터값]
{catalog_text}

[사용자 입력]
{user_query}
""".strip()

    response = client.responses.create(
        model="gpt-4.1-mini",
        input=prompt,
        max_output_tokens=1800,
        text={
            "format": {
                "type": "json_schema",
                "name": "traffic_filter_parser",
                "strict": True,
                "schema": schema,
            }
        },
    )

    output_text = getattr(response, "output_text", "") or ""
    if not output_text.strip():
        raise RuntimeError("AI 자연어 검색 결과가 비어 있습니다.")

    parsed_result = json.loads(output_text)

    parsed_result = _supplement_weather_from_query(
        parsed_result=parsed_result,
        user_query=user_query,
        filter_catalog=filter_catalog,
    )

    parsed_result = _supplement_age_context_from_query(
        parsed_result=parsed_result,
        user_query=user_query,
        filter_catalog=filter_catalog,
    )

    parsed_result = _supplement_recent_period_from_query(
        parsed_result=parsed_result,
        user_query=user_query,
        filter_catalog=filter_catalog,
    )

    return parsed_result


def _supplement_weather_from_query(
    parsed_result,
    user_query,
    filter_catalog,
):
    """
    자연어 문장에 명백한 날씨 표현이 있는데 GPT가 놓친 경우 보정한다.

    - 명확한 날씨 표현만 결정론적으로 매핑
    - '악천후'는 기존 규칙대로 사용자 확인 대상으로 남김
    - 실제 데이터에 존재하는 날씨 필터값만 반영
    """
    query = str(user_query).strip()
    if not query:
        return parsed_result

    # 악천후는 구체 범위를 임의로 정하지 않음
    if "악천후" in query:
        return parsed_result

    weather_phrase_map = {
        "비": [
            "비가 오는 날",
            "비 오는 날",
            "비오는 날",
            "비가 올 때",
            "비 올 때",
            "우천",
            "강우",
        ],
        "눈": [
            "눈이 오는 날",
            "눈 오는 날",
            "눈오는 날",
            "눈이 올 때",
            "눈 올 때",
            "강설",
        ],
        "안개": [
            "안개 낀 날",
            "안개가 낀 날",
            "안개 발생",
            "안개",
        ],
        "흐림": [
            "흐린 날",
            "흐린 날씨",
            "흐림",
        ],
        "맑음": [
            "맑은 날",
            "맑은 날씨",
            "맑음",
            "화창한 날",
        ],
    }

    allowed_weather = {
        str(value)
        for value in filter_catalog.get("weather", [])
    }

    detected = []

    for weather_value, phrases in weather_phrase_map.items():
        if weather_value not in allowed_weather:
            continue

        if any(phrase in query for phrase in phrases):
            detected.append(weather_value)

    if not detected:
        return parsed_result

    current_weather = parsed_result.get("weather", []) or []
    merged_weather = []

    for value in list(current_weather) + detected:
        if (
            str(value) in allowed_weather
            and value not in merged_weather
        ):
            merged_weather.append(value)

    parsed_result["weather"] = merged_weather

    # 명백한 날씨 표현을 인식했으므로,
    # 다른 모호성이 없다면 날씨 누락만으로 확인질문을 할 필요 없음
    clarification = str(
        parsed_result.get("clarification_question", "")
    ).strip()

    weather_only_clarification_terms = [
        "날씨",
        "비",
        "눈",
        "안개",
        "흐림",
        "맑음",
    ]

    if (
        parsed_result.get("status") == "need_clarification"
        and clarification
        and any(
            term in clarification
            for term in weather_only_clarification_terms
        )
        and "악천후" not in query
    ):
        parsed_result["status"] = "ready"
        parsed_result["clarification_question"] = ""

    return parsed_result


def _supplement_age_context_from_query(
    parsed_result,
    user_query,
    filter_catalog,
):
    """
    명확한 고령 보행자·고령 운전자·어린이 표현을 결정론적으로 보정한다.

    원칙:
    - 고령 보행자: 피해차종 보행자 + 피해자 65세 이상
    - 고령 운전자: 가해운전자 65세 이상
    - 어린이: 피해자 12세 이하
    - 그 밖의 모호한 고령 표현은 AI의 문맥 판단 결과를 유지
    """
    query = re.sub(r"\s+", " ", str(user_query).strip())

    if not query:
        return parsed_result

    allowed_offending_ages = {
        str(value)
        for value in filter_catalog.get(
            "offending_driver_age_groups",
            [],
        )
    }
    allowed_victim_ages = {
        str(value)
        for value in filter_catalog.get("victim_age_groups", [])
    }
    allowed_damaged_vehicles = {
        str(value)
        for value in filter_catalog.get("damaged_vehicles", [])
    }

    is_elderly_pedestrian = bool(
        re.search(
            r"(?:고령|고령자|노인|어르신)\s*(?:보행자|보행인|보행사고|행인)",
            query,
        )
    )
    is_elderly_driver = bool(
        re.search(
            r"(?:고령|고령자|노인|어르신)\s*(?:운전자|운전)",
            query,
        )
    )

    if is_elderly_pedestrian:
        if "보행자" in allowed_damaged_vehicles:
            parsed_result["damaged_vehicles"] = ["보행자"]
        if "65세 이상" in allowed_victim_ages:
            parsed_result["victim_age_groups"] = ["65세 이상"]
        parsed_result["offending_driver_age_groups"] = []
    elif is_elderly_driver:
        if "65세 이상" in allowed_offending_ages:
            parsed_result["offending_driver_age_groups"] = ["65세 이상"]

    if "어린이" in query:
        if "12세 이하" in allowed_victim_ages:
            parsed_result["victim_age_groups"] = ["12세 이하"]
        parsed_result["offending_driver_age_groups"] = []

        if (
            re.search(r"어린이\s*(?:보행자|보행인|보행사고)", query)
            and "보행자" in allowed_damaged_vehicles
        ):
            parsed_result["damaged_vehicles"] = ["보행자"]

    return parsed_result


def _supplement_recent_period_from_query(
    parsed_result,
    user_query,
    filter_catalog,
):
    """'최근 N년'을 데이터의 최신 발생월 기준 N×12개월로 보정한다."""
    query = re.sub(r"\s+", " ", str(user_query).strip())
    match = re.search(r"최근\s*(\d+)\s*년(?:간)?", query)

    if not match:
        return parsed_result

    recent_years = int(match.group(1))
    if recent_years < 1:
        return parsed_result

    available_period = filter_catalog.get("available_period", [])
    if not available_period or not available_period[-1]:
        return parsed_result

    try:
        latest_period = pd.Period(
            str(available_period[-1]),
            freq="M",
        )
        earliest_available_period = (
            pd.Period(str(available_period[0]), freq="M")
            if available_period[0]
            else latest_period
        )
    except Exception:
        return parsed_result

    requested_start_period = latest_period - (recent_years * 12 - 1)
    applied_start_period = max(
        earliest_available_period,
        requested_start_period,
    )

    parsed_result["start_year_month"] = applied_start_period.strftime(
        "%Y-%m"
    )
    parsed_result["end_year_month"] = latest_period.strftime("%Y-%m")

    return parsed_result

def _validated_list(values, allowed_values):
    """AI가 반환한 배열을 현재 실제 필터값 범위 안으로 제한한다."""
    allowed = {str(value) for value in allowed_values}
    return [
        value
        for value in values
        if str(value) in allowed
    ]


def _queue_natural_language_filters(parsed_result, filter_catalog):
    """
    AI 해석결과를 다음 Streamlit 실행에서 적용하도록 대기열에 저장한다.
    기존 필터는 모두 초기화하고 자연어에서 명시된 조건만 적용한다.
    """
    if parsed_result.get("status") != "ready":
        return

    pending = {
        "station": (
            parsed_result.get("station")
            if parsed_result.get("station") in filter_catalog["stations"]
            else None
        ),
        "start_year_month": parsed_result.get("start_year_month"),
        "end_year_month": parsed_result.get("end_year_month"),
        "accident_severity": _validated_list(
            parsed_result.get("accident_severity", []),
            filter_catalog["accident_severity"],
        ),
        "accident_type": (
            parsed_result.get("accident_type")
            if parsed_result.get("accident_type")
            in filter_catalog["accident_types"]
            else None
        ),
        "start_time": parsed_result.get("start_time"),
        "end_time": parsed_result.get("end_time"),
        "weekdays": _validated_list(
            parsed_result.get("weekdays", []),
            filter_catalog["weekdays"],
        ),
        "offending_vehicles": _validated_list(
            parsed_result.get("offending_vehicles", []),
            filter_catalog["offending_vehicles"],
        ),
        "offending_driver_age_groups": _validated_list(
            parsed_result.get("offending_driver_age_groups", []),
            filter_catalog["offending_driver_age_groups"],
        ),
        "damaged_vehicles": _validated_list(
            parsed_result.get("damaged_vehicles", []),
            filter_catalog["damaged_vehicles"],
        ),
        "victim_age_groups": _validated_list(
            parsed_result.get("victim_age_groups", []),
            filter_catalog["victim_age_groups"],
        ),
        "fatal_types": _validated_list(
            parsed_result.get("fatal_types", []),
            filter_catalog["fatal_types"],
        ),
        "fatal_age_groups": _validated_list(
            parsed_result.get("fatal_age_groups", []),
            filter_catalog["fatal_age_groups"],
        ),
        "weather": _validated_list(
            parsed_result.get("weather", []),
            filter_catalog["weather"],
        ),
        "violations": _validated_list(
            parsed_result.get("violations", []),
            filter_catalog["violations"],
        ),
        "hotspot_requested": bool(
            parsed_result.get("hotspot_requested", False)
            or parsed_result.get("hotspot_radius") is not None
            or parsed_result.get("hotspot_top_n") is not None
        ),
        "hotspot_radius": parsed_result.get("hotspot_radius"),
        "hotspot_top_n": parsed_result.get("hotspot_top_n"),
        "interpretation_summary": (
            parsed_result.get("interpretation_summary", "")
        ),
    }

    st.session_state["_pending_natural_filters"] = pending


def _apply_pending_natural_filter_state():
    """
    사이드바 위젯이 생성되기 전에 대기 중인 자연어 필터를 session_state에 적용한다.
    발생연월은 year_month_options 생성 후 별도로 적용한다.
    """
    pending = st.session_state.pop(
        "_pending_natural_filters",
        None,
    )

    if not pending:
        return

    # ========================================================
    # 새 자연어 검색 시작 시 기존 검색조건 완전 초기화
    #
    # 원칙
    # - 이전 사이드바 조건을 이어받지 않음
    # - 새 문장에 명시된 조건만 다시 적용
    # - 명시되지 않은 조건은 전체/기본값으로 복귀
    # ========================================================

    # 위젯이 생성되기 전에 이전 상태를 제거
    filter_keys = [
        "filter_station",
        "filter_severity",
        "filter_accident_type",
        "start_time",
        "end_time",
        "filter_offending_vehicle",
        "filter_offending_driver_age",
        "filter_damaged_vehicle",
        "filter_victim_age",
        "filter_fatal_type",
        "filter_fatal_age",
        "filter_weather",
        "filter_violation",
    ]

    for key in filter_keys:
        st.session_state.pop(key, None)

    # 체크박스 요일도 모두 초기화
    for weekday in [
        "월요일",
        "화요일",
        "수요일",
        "목요일",
        "금요일",
        "토요일",
        "일요일",
    ]:
        st.session_state[
            f"weekday_checkbox_{weekday}"
        ] = False

    # --------------------------------------------------------
    # 명시되지 않은 조건의 기본값
    # --------------------------------------------------------
    st.session_state["filter_station"] = "전체"
    st.session_state["filter_severity"] = []
    st.session_state["filter_accident_type"] = "전체"
    st.session_state["start_time"] = 0
    st.session_state["end_time"] = 23
    st.session_state["filter_offending_vehicle"] = []
    st.session_state["filter_offending_driver_age"] = []
    st.session_state["filter_damaged_vehicle"] = []
    st.session_state["filter_victim_age"] = []    
    st.session_state["filter_fatal_type"] = []
    st.session_state["filter_fatal_age"] = []
    st.session_state["filter_weather"] = []
    st.session_state["filter_violation"] = []

    # 사고다발지점 설정도 새 검색마다 기본값으로 초기화
    st.session_state["hotspot_radius"] = 100
    st.session_state["hotspot_top_n"] = 10
    st.session_state.pop(
        "natural_search_hotspot_requested",
        None,
    )

    # --------------------------------------------------------
    # 자연어 검색에서 명시된 조건만 덮어쓰기
    # --------------------------------------------------------
    station = pending.get("station")
    if station:
        st.session_state["filter_station"] = station

    st.session_state["filter_severity"] = pending.get(
        "accident_severity",
        [],
    )

    accident_type = pending.get("accident_type")
    if accident_type:
        st.session_state["filter_accident_type"] = accident_type

    start_time_value = pending.get("start_time")
    end_time_value = pending.get("end_time")

    if start_time_value is not None:
        st.session_state["start_time"] = int(start_time_value)

    if end_time_value is not None:
        st.session_state["end_time"] = int(end_time_value)

    selected_weekdays_pending = set(
        pending.get("weekdays", [])
    )
    for weekday in [
        "월요일",
        "화요일",
        "수요일",
        "목요일",
        "금요일",
        "토요일",
        "일요일",
    ]:
        st.session_state[
            f"weekday_checkbox_{weekday}"
        ] = weekday in selected_weekdays_pending

    st.session_state["filter_offending_vehicle"] = pending.get(
        "offending_vehicles",
        [],
    )
    st.session_state["filter_offending_driver_age"] = pending.get(
        "offending_driver_age_groups",
        [],
    )
    st.session_state["filter_damaged_vehicle"] = pending.get(
        "damaged_vehicles",
        [],
    )
    st.session_state["filter_victim_age"] = pending.get(
        "victim_age_groups",
        [],
    )
    st.session_state["filter_fatal_type"] = pending.get(
        "fatal_types",
        [],
    )
    st.session_state["filter_fatal_age"] = pending.get(
        "fatal_age_groups",
        [],
    )
    st.session_state["filter_weather"] = pending.get(
        "weather",
        [],
    )
    st.session_state["filter_violation"] = pending.get(
        "violations",
        [],
    )

    # 연월은 pd.Period 목록이 만들어진 뒤 적용
    st.session_state["_pending_natural_period"] = {
        "start": pending.get("start_year_month"),
        "end": pending.get("end_year_month"),
    }

    summary_text = pending.get(
        "interpretation_summary",
        "",
    )
    summary_text = str(summary_text).replace("~", "∼")

    st.session_state["natural_search_last_summary"] = summary_text

    # --------------------------------------------------------
    # 사고다발지점 탐색 설정
    # - 새 자연어 검색마다 기본값 100m / 5개로 초기화
    # - 자연어에서 반경/개수를 명시하면 해당 값으로 변경
    # --------------------------------------------------------
    hotspot_radius_value = pending.get("hotspot_radius")
    if hotspot_radius_value is not None:
        hotspot_radius_value = int(hotspot_radius_value)
        if 50 <= hotspot_radius_value <= 300:
            st.session_state["hotspot_radius"] = hotspot_radius_value

    hotspot_top_n_value = pending.get("hotspot_top_n")
    if hotspot_top_n_value is not None:
        hotspot_top_n_value = int(hotspot_top_n_value)
        if 1 <= hotspot_top_n_value <= 10:
            st.session_state["hotspot_top_n"] = hotspot_top_n_value

    if pending.get("hotspot_requested"):
        st.session_state["natural_search_hotspot_requested"] = True
    else:
        st.session_state.pop(
            "natural_search_hotspot_requested",
            None,
        )



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

# 경찰서별 전체 교통사고 좌표의 사전 산출 중심값
# 관할 선택 시 통합 대시보드 지도와 GIS 분석 지도가 함께 사용한다.
POLICE_STATION_MAP_CENTERS = {
    "대덕": [36.366806, 127.423534],
    "동부": [36.335746, 127.440486],
    "둔산": [36.351509, 127.378119],
    "서부": [36.319485, 127.374153],
    "유성": [36.361749, 127.339306],
    "중부": [36.322427, 127.410541],
}

# 자연어 검색으로 대기 중인 필터가 있으면 사이드바 생성 전에 적용
_apply_pending_natural_filter_state()

# -----------------------------------
# 2. 스트림릿 사이드바 필터 구현
# -----------------------------------
# AI 자연어 검색은 사용자가 가장 먼저 만나는 사이드바 상단에 배치
st.sidebar.markdown(
    """
    <div class="sidebar-ai-search-title">
        <span>✨ AI 자연어 조건검색</span>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    with st.form(
        "natural_language_filter_form",
        clear_on_submit=False,
    ):
        natural_query_col, natural_apply_col = st.columns(
            [4.4, 1.25],
            gap="small",
        )
        with natural_query_col:
            natural_filter_query = st.text_input(
                "원하는 교통사고 검색조건을 문장으로 입력하세요.",
                key="natural_filter_query",
                placeholder="예: 최근 3년간 야간 보행자 사고",
                label_visibility="collapsed",
            )
        with natural_apply_col:
            natural_filter_submit = st.form_submit_button(
                "적용",
                use_container_width=True,
            )

st.sidebar.header("🔍 분석 필터 설정")


# -----------------------------------
# 사이드바 핵심 필터 제목 공통 스타일
# - 발생요일과 동일한 굵은 제목 형식
# - 위젯과 제목 사이 간격을 작게 유지
# -----------------------------------
def sidebar_filter_title(title_html, margin_bottom=0):
    st.sidebar.markdown(
        f"""
        <div style="
            font-size:1rem;
            font-weight:700;
            color:#F8FBFF;
            margin-top:2px;
            margin-bottom:{margin_bottom}px;
            line-height:1.35;
        "
        >
            {title_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


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

sidebar_filter_title("관할 경찰서", margin_bottom=14)

selected_ps = st.sidebar.selectbox(
    "관할 경찰서",
    station_options,
    key="filter_station",
    label_visibility="collapsed",
)

# -----------------------------------
# [순서 2] 발생 연월 선택
# - AI 자연어 검색 시에만 슬라이더 위젯을 새 key로 재생성
# - 평상시에는 같은 key를 유지하여 사용자의 마우스 조작값을 그대로 보존
# - 실제 필터는 select_slider의 반환값을 직접 사용
# -----------------------------------

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

sidebar_filter_title("발생연월", margin_bottom=14)

start_year_month = None
end_year_month = None

if year_month_options:

    # ========================================================
    # 1. 슬라이더 세대(version) 및 기본범위 초기화
    # ========================================================
    if "year_month_slider_version" not in st.session_state:
        st.session_state["year_month_slider_version"] = 0

    if "year_month_slider_default" not in st.session_state:
        st.session_state["year_month_slider_default"] = (
            year_month_options[0],
            year_month_options[-1],
        )

    # ========================================================
    # 2. 자연어 검색에서 전달된 기간 확인
    #    → 이때만 slider version을 증가시켜 새 위젯 생성
    # ========================================================
    pending_period = st.session_state.pop(
        "_pending_natural_period",
        None,
    )

    if pending_period is not None:
        period_start_text = pending_period.get("start")
        period_end_text = pending_period.get("end")

        try:
            natural_start_period = (
                pd.Period(period_start_text, freq="M")
                if period_start_text
                else year_month_options[0]
            )
        except Exception:
            natural_start_period = year_month_options[0]

        try:
            natural_end_period = (
                pd.Period(period_end_text, freq="M")
                if period_end_text
                else year_month_options[-1]
            )
        except Exception:
            natural_end_period = year_month_options[-1]

        # 실제 데이터 범위를 벗어나지 않도록 제한
        natural_start_period = max(
            year_month_options[0],
            min(natural_start_period, year_month_options[-1]),
        )
        natural_end_period = max(
            year_month_options[0],
            min(natural_end_period, year_month_options[-1]),
        )

        if natural_start_period > natural_end_period:
            natural_start_period, natural_end_period = (
                natural_end_period,
                natural_start_period,
            )

        # 실제 options에 없는 월은 가장 가까운 월로 보정
        def _nearest_available_period(target_period):
            return min(
                year_month_options,
                key=lambda option: abs(
                    option.ordinal - target_period.ordinal
                ),
            )

        natural_start_period = _nearest_available_period(
            natural_start_period
        )
        natural_end_period = _nearest_available_period(
            natural_end_period
        )

        # AI가 지정한 범위를 다음 슬라이더의 기본값으로 저장
        st.session_state["year_month_slider_default"] = (
            natural_start_period,
            natural_end_period,
        )

        # 핵심: 새 key를 쓰도록 세대 증가
        st.session_state["year_month_slider_version"] += 1

    # ========================================================
    # 3. 현재 기본값 안전성 검사
    # ========================================================
    slider_default = st.session_state.get(
        "year_month_slider_default",
        (
            year_month_options[0],
            year_month_options[-1],
        ),
    )

    default_is_valid = (
        isinstance(slider_default, (tuple, list))
        and len(slider_default) == 2
        and slider_default[0] in year_month_options
        and slider_default[1] in year_month_options
    )

    if not default_is_valid:
        slider_default = (
            year_month_options[0],
            year_month_options[-1],
        )
        st.session_state["year_month_slider_default"] = slider_default

    # ========================================================
    # 4. 발생연월 범위 슬라이더
    #
    # - AI 검색 직후: version 증가 → 새 key → AI 범위로 생성
    # - 사용자가 마우스로 변경: version 유지 → 같은 위젯 state 유지
    # ========================================================
    slider_version = int(
        st.session_state["year_month_slider_version"]
    )
    slider_key = f"filter_year_month_slider_{slider_version}"

    selected_year_month_range = st.sidebar.select_slider(
        "발생연월 범위",
        options=year_month_options,
        value=tuple(slider_default),
        key=slider_key,
        format_func=lambda value: (
            f"{value.year}년 {value.month}월"
        ),
        label_visibility="collapsed",
    )

    # ========================================================
    # 5. 슬라이더 반환값을 실제 필터에 직접 사용
    # ========================================================
    if (
        isinstance(selected_year_month_range, (tuple, list))
        and len(selected_year_month_range) == 2
    ):
        start_year_month = selected_year_month_range[0]
        end_year_month = selected_year_month_range[1]
    else:
        # 방어적 처리
        start_year_month = selected_year_month_range
        end_year_month = selected_year_month_range

    # 현재 선택범위를 다음 재실행의 기본값으로 저장
    # 같은 key가 존재하는 동안에는 widget state가 우선하므로
    # 사용자의 마우스 조작을 덮어쓰지 않는다.
    st.session_state["year_month_slider_default"] = (
        start_year_month,
        end_year_month,
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

sidebar_filter_title(
    "부상정도 <span style='font-weight:400;'>(복수선택)</span>",
    margin_bottom=14,
)

selected_types = st.sidebar.multiselect(
    "사고분류 (복수 선택)",
    type_options,
    key="filter_severity",
    placeholder="전체",
    label_visibility="collapsed",
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

sidebar_filter_title("사고종별", margin_bottom=14)

selected_hdc = st.sidebar.selectbox(
    "사고종별",
    hdc_options,
    key="filter_accident_type",
    label_visibility="collapsed",
)

# -----------------------------------
# [순서 5] 시간범위 선택
# 시작·종료시간을 한 행 2열로 배치
# 시작시간이 종료시간보다 크면 자정을 넘는 시간대로 처리
# -----------------------------------
sidebar_filter_title(
    "시간범위 <span style='font-weight:400;'>(시작·종료시간)</span>",
    margin_bottom=14,
)

time_options = list(range(24))

if "start_time" not in st.session_state:
    st.session_state["start_time"] = 0

if "end_time" not in st.session_state:
    st.session_state["end_time"] = 23

time_col1, time_col2 = st.sidebar.columns(2)

with time_col1:
    start_time = st.selectbox(
        "시작시간",
        options=time_options,
        format_func=lambda x: f"{x:02d}시",
        key="start_time",
        label_visibility="collapsed",
    )

with time_col2:
    end_time = st.selectbox(
        "종료시간",
        options=time_options,
        format_func=lambda x: f"{x:02d}시",
        key="end_time",
        label_visibility="collapsed",
    )

# 선택창 바로 아래 자정 포함 검색 안내
st.sidebar.markdown(
    """
    <div style="
        margin-top:-9px;
        margin-bottom:9px;
        color:#64748B;
        font-size:0.78rem;
        line-height:1.35;
    "
    >
        ※ 검색 시 '분'을 제거 (0시는 0시~0시59분)
    </div>
    """,
    unsafe_allow_html=True,
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

# 발생요일 제목
sidebar_filter_title("발생요일", margin_bottom=-10)

# 7개 체크박스를 한 줄로 배치
weekday_cols = st.sidebar.columns(7)

selected_weekdays = []

for index, weekday in enumerate(weekday_options):
    with weekday_cols[index]:
        checkbox_key = f"weekday_checkbox_{weekday}"

        if checkbox_key not in st.session_state:
            st.session_state[checkbox_key] = False

        is_selected = st.checkbox(
            weekday_short_name[weekday],
            key=checkbox_key,
        )

    if is_selected:
        selected_weekdays.append(weekday)

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
        key="filter_offending_vehicle",
        placeholder="전체 (미선택 시)",
    )

    # -----------------------------------
    # 가해운전자 연령대 선택
    # acdnt_age_1_dc 컬럼값 사용
    # - 기본 연령구간은 낮은 연령부터 정렬
    # - 그 외 실제 데이터값이 있으면 뒤에 추가
    # -----------------------------------
    offending_driver_age_order = [
        "20세 이하",
        "21-30세",
        "31-40세",
        "41-50세",
        "51-60세",
        "61-64세",
        "65세 이상",
    ]

    if "acdnt_age_1_dc" in df.columns:
        offending_driver_age_values = (
            df["acdnt_age_1_dc"]
            .dropna()
            .astype(str)
            .str.strip()
        )

        offending_driver_age_values = set(
            offending_driver_age_values[
                (offending_driver_age_values != "")
                & (
                    offending_driver_age_values
                    .str.lower()
                    .ne("nan")
                )
            ].unique()
        )

        offending_driver_age_options = [
            age_group
            for age_group in offending_driver_age_order
            if age_group in offending_driver_age_values
        ]

        # 정해진 기본 순서에 없는 값도 실제 데이터에 존재하면 누락하지 않음
        extra_age_values = sorted(
            offending_driver_age_values
            - set(offending_driver_age_order)
        )
        offending_driver_age_options.extend(
            extra_age_values
        )
    else:
        offending_driver_age_options = []

    selected_offending_driver_age = st.multiselect(
        "가해운전자 연령대 (복수 선택)",
        offending_driver_age_options,
        key="filter_offending_driver_age",
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
        key="filter_damaged_vehicle",
        placeholder="전체 (미선택 시)",
    )

    # -----------------------------------
    # 피해자 연령대 선택
    # acdnt_age_2_dc 컬럼값 사용
    # - 기본 연령구간은 낮은 연령부터 정렬
    # - 그 외 실제 데이터값이 있으면 뒤에 추가
    # -----------------------------------
    victim_age_order = [
        "12세 이하",
        "13-20세",
        "20세 이하",
        "21-30세",
        "31-40세",
        "41-50세",
        "51-60세",
        "61-64세",
        "65세 이상",
    ]

    if "acdnt_age_2_dc" in df.columns:
        victim_age_values = (
            df["acdnt_age_2_dc"]
            .dropna()
            .astype(str)
            .str.strip()
        )

        victim_age_values = set(
            victim_age_values[
                (victim_age_values != "")
                & victim_age_values.str.lower().ne("nan")
            ].unique()
        )

        victim_age_options = [
            age_group
            for age_group in victim_age_order
            if age_group in victim_age_values
        ]

        # 기본 순서에 없는 실제 데이터값도 누락하지 않음
        extra_victim_age_values = sorted(
            victim_age_values - set(victim_age_order)
        )
        victim_age_options.extend(extra_victim_age_values)

    else:
        victim_age_options = []

    selected_victim_age = st.multiselect(
        "피해자 연령대 (복수 선택)",
        victim_age_options,
        key="filter_victim_age",
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
        key="filter_fatal_type",
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
        key="filter_fatal_age",
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
        key="filter_weather",
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
        key="filter_violation",
        placeholder="전체 (미선택 시)",
    )


# ============================================================
# 필터 초기화
# - st.session_state.clear()를 사용하지 않고 필터 관련 키만
#   명시적으로 기본값으로 되돌린다.
# - 버튼 on_click 콜백은 다음 화면 렌더링 전에 실행되므로
#   이미 생성된 위젯 상태와 충돌하지 않는다.
# ============================================================
def reset_all_filters():
    """사이드바 필터와 자연어 검색조건을 기본값으로 초기화"""

    # --------------------------------------------------------
    # 핵심 필터
    # --------------------------------------------------------
    st.session_state["filter_station"] = "전체"

    if year_month_options:
        default_year_month_range = (
            year_month_options[0],
            year_month_options[-1],
        )

        st.session_state["year_month_slider_default"] = (
            default_year_month_range
        )

        # 새 key로 슬라이더를 재생성하여 화면도 전체기간으로 즉시 초기화
        st.session_state["year_month_slider_version"] = (
            int(st.session_state.get("year_month_slider_version", 0)) + 1
        )

    else:
        st.session_state.pop(
            "year_month_slider_default",
            None,
        )
        st.session_state.pop(
            "year_month_slider_version",
            None,
        )

    st.session_state["filter_severity"] = []
    st.session_state["filter_accident_type"] = "전체"

    st.session_state["start_time"] = 0
    st.session_state["end_time"] = 23

    # --------------------------------------------------------
    # 발생요일
    # --------------------------------------------------------
    for weekday in [
        "월요일",
        "화요일",
        "수요일",
        "목요일",
        "금요일",
        "토요일",
        "일요일",
    ]:
        st.session_state[
            f"weekday_checkbox_{weekday}"
        ] = False

    # --------------------------------------------------------
    # 차량 조건
    # --------------------------------------------------------
    st.session_state["filter_offending_vehicle"] = []
    st.session_state["filter_offending_driver_age"] = []
    st.session_state["filter_damaged_vehicle"] = []
    st.session_state["filter_victim_age"] = []

    # --------------------------------------------------------
    # 사망사고 조건
    # --------------------------------------------------------
    st.session_state["filter_fatal_type"] = []
    st.session_state["filter_fatal_age"] = []

    # --------------------------------------------------------
    # 환경·원인 조건
    # --------------------------------------------------------
    st.session_state["filter_weather"] = []
    st.session_state["filter_violation"] = []

    # --------------------------------------------------------
    # 사고다발지점 설정
    # --------------------------------------------------------
    st.session_state["hotspot_radius"] = 100
    st.session_state["hotspot_top_n"] = 10
    st.session_state.pop(
        "natural_search_hotspot_requested",
        None,
    )

    # --------------------------------------------------------
    # 자연어 검색 관련 잔여 상태 제거
    # 이전 자연어 조건이 다음 rerun에서 다시 적용되는 것을 방지
    # --------------------------------------------------------
    st.session_state.pop(
        "_pending_natural_filters",
        None,
    )
    st.session_state.pop(
        "_pending_natural_period",
        None,
    )
    st.session_state.pop(
        "natural_search_last_summary",
        None,
    )
    st.session_state.pop(
        "natural_filter_query",
        None,
    )


st.sidebar.divider()

st.sidebar.button(
    "↺ 필터 초기화",
    use_container_width=True,
    on_click=reset_all_filters,
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

    # 필터 변경 때마다 경계 중심을 다시 계산하지 않고, 전체 사고자료에서
    # 미리 산출한 관할별 중심좌표를 두 지도에 공통으로 적용한다.
    map_center = POLICE_STATION_MAP_CENTERS.get(selected_ps, center)
    zoom_level = 13

else:
    filtered_gdf = gdf
    filtered_boundary = ps_boundary
    map_center = center
    # 대전 전역도 초기 화면에서 한 단계 더 가깝게 표시
    zoom_level = 12

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

# [필터 7] 가해운전자 연령대
if (
    selected_offending_driver_age
    and "acdnt_age_1_dc" in filtered_df.columns
):
    offending_driver_age_series = (
        filtered_df["acdnt_age_1_dc"]
        .astype(str)
        .str.strip()
    )

    filtered_df = filtered_df[
        offending_driver_age_series.isin(
            selected_offending_driver_age
        )
    ]

# [필터 8] 피해차량 차종
if (
    selected_dmge
    and "dmge_vhcle_asort_dc" in filtered_df.columns
):
    filtered_df = filtered_df[
        filtered_df["dmge_vhcle_asort_dc"].isin(
            selected_dmge
        )
    ]

# [필터 9] 피해자 연령대
if (
    selected_victim_age
    and "acdnt_age_2_dc" in filtered_df.columns
):
    victim_age_series = (
        filtered_df["acdnt_age_2_dc"]
        .astype(str)
        .str.strip()
    )

    filtered_df = filtered_df[
        victim_age_series.isin(selected_victim_age)
    ]

# [필터 10] 사망자 유형
if selected_fatal_type and "fatal_type" in filtered_df.columns:
    filtered_df = filtered_df[
        filtered_df["fatal_type"].isin(
            selected_fatal_type
        )
    ]

# [필터 11] 사망자 연령대
if (
    selected_fatal_age
    and "fatal_age_group" in filtered_df.columns
):
    filtered_df = filtered_df[
        filtered_df["fatal_age_group"].isin(
            selected_fatal_age
        )
    ]

# [필터 12] 날씨
if (
    selected_wether
    and "wether_sttus_dc" in filtered_df.columns
):
    filtered_df = filtered_df[
        filtered_df["wether_sttus_dc"].isin(
            selected_wether
        )
    ]

# [필터 13] 법규위반유형
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
# KPI 현재 검색조건 요약
# 발생연월 · 사고분류 · 사고종별 · 시간대 · 발생요일
# ============================================================
if start_year_month is not None and end_year_month is not None:
    period_summary = (
        f"{start_year_month.year}년 {start_year_month.month}월~"
        f"{end_year_month.year}년 {end_year_month.month}월"
    )
else:
    period_summary = "전체 기간"

if selected_types:
    severity_short_map = {
        "사망사고": "사망",
        "중상사고": "중상",
        "경상사고": "경상",
        "부상신고사고": "부상신고",
    }
    severity_summary = "·".join(
        severity_short_map.get(value, value)
        for value in selected_types
    )
else:
    severity_summary = "전체 사고"

accident_type_summary = (
    selected_hdc
    if selected_hdc != "전체"
    else "전체 종별"
)

if start_time <= end_time:
    time_summary = f"{start_time:02d}시~{end_time:02d}시"
else:
    time_summary = f"{start_time:02d}시~익일 {end_time:02d}시"

if selected_weekdays:
    weekday_summary = "·".join(
        weekday_short_name.get(value, value)
        for value in selected_weekdays
    )
else:
    weekday_summary = "전체 요일"

# ============================================================
# 상단 통합 브랜드 헤더 + 화면 선택 메뉴
# - 제목/부제와 4개 화면 선택 버튼을 하나의 박스 안에 배치
# ============================================================
# 구버전 Streamlit 호환: st.container()의 key 인자는 사용하지 않는다.
# 헤더 전용 CSS 범위는 아래 HTML의 .taap-header-marker로 식별한다.
header_shell = st.container()

header_title_col, header_spacer_col, header_nav_col = header_shell.columns(
    [1.60, 0.10, 1.30],
    gap="small",
)

with header_title_col:
    st.markdown(
        """
<div class="taap-brand-panel-v2">
<span class="taap-header-marker" aria-hidden="true"></span>
<div class="taap-brand-primary-v2">
<div class="taap-brand-icon-v2">🗺️</div>
<div class="taap-brand-name-v2"><span class="taap-brand-name-main-v2">TAAP</span><span class="taap-brand-name-ai-v2">-AI</span></div>
</div>
<div class="taap-brand-subtitles-v2">
<div class="taap-brand-subtitle-en-v2"><span class="taap-brand-initial-v2">T</span>raffic <span class="taap-brand-initial-v2">A</span>ccident <span class="taap-brand-initial-v2">A</span>nalysis <span class="taap-brand-initial-v2">P</span>latform</div>
<div class="taap-brand-subtitle-ko-v2">교통사고 분석 및 AI 의사결정 지원 플랫폼</div>
</div>
</div>
        """,
        unsafe_allow_html=True,
    )

HEADER_NAV_PAGES = [
    ("dashboard", "대시보드"),
    ("gis", "GIS분석"),
    ("statistics", "통계분석"),
    ("ai_report", "AI 리포트"),
]

if st.session_state.get("active_analysis_page") not in {
    page_name for _, page_name in HEADER_NAV_PAGES
}:
    st.session_state["active_analysis_page"] = "대시보드"


def select_analysis_page(page_name):
    """상단 메뉴 버튼 클릭 시 현재 분석 화면을 변경한다."""
    st.session_state["active_analysis_page"] = page_name


with header_nav_col:
    # 제목 영역과 분리된 상단 메뉴 전용 식별자
    st.markdown(
        '<span class="taap-nav-marker-v3" aria-hidden="true"></span>',
        unsafe_allow_html=True,
    )

    nav_columns = st.columns(4, gap="small")
    for nav_column, (page_key, page_name) in zip(
        nav_columns,
        HEADER_NAV_PAGES,
    ):
        with nav_column:
            is_active_page = (
                st.session_state["active_analysis_page"] == page_name
            )
            st.button(
                page_name,
                key=f"header_nav_{page_key}",
                type="primary" if is_active_page else "secondary",
                use_container_width=True,
                on_click=select_analysis_page,
                args=(page_name,),
            )

selected_page = st.session_state["active_analysis_page"]

# 헤더와 KPI 사이의 실제 레이아웃 간격
# CSS 부모 선택자에 의존하지 않아 Streamlit 버전별로 동일하게 적용된다.
st.markdown(
    '<div class="taap-header-kpi-spacer" aria-hidden="true"></div>',
    unsafe_allow_html=True,
)

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

kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4, gap="medium")

with kpi_col1:
    st.markdown(
        f"""
        <div class="taap-kpi-card kpi-station">
            <div class="taap-kpi-label">선택된 관할</div>
            <div class="taap-kpi-value">{selected_ps}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with kpi_col2:
    st.markdown(
        f"""
        <div class="taap-kpi-card kpi-total">
            <div class="taap-kpi-label">총 사고</div>
            <div class="taap-kpi-value"><span class="taap-kpi-number">{len(filtered_df):,}</span><span class="taap-kpi-unit">건</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with kpi_col3:
    st.markdown(
        f"""
        <div class="taap-kpi-card kpi-fatal">
            <div class="taap-kpi-label">사망사고</div>
            <div class="taap-kpi-value"><span class="taap-kpi-number">{fatal_accident_count:,}</span><span class="taap-kpi-unit">건</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with kpi_col4:
    st.markdown(
        f"""
        <div class="taap-kpi-card kpi-serious">
            <div class="taap-kpi-label">중상사고</div>
            <div class="taap-kpi-value"><span class="taap-kpi-number">{serious_accident_count:,}</span><span class="taap-kpi-unit">건</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown(
    f"""
    <div class="taap-condition-strip">
        <span><b>분석기간</b> {period_summary}</span>
        <span><b>시간대</b> {time_summary}</span>
        <span><b>사고종별</b> {accident_type_summary}</span>
        <span><b>요일</b> {weekday_summary}</span>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# AI 자연어 조건검색 처리
# - 입력 UI는 사이드바 상단에 배치
# - 사용자의 문장을 기존 사이드바 검색조건으로 자동 변환
# ============================================================

if natural_filter_submit:
    if not natural_filter_query.strip():
        st.warning("검색할 조건을 문장으로 입력해 주세요.")
    else:
        # 현재 화면에서 실제 선택 가능한 값만 AI에 제공
        natural_filter_catalog = {
            "stations": station_options,
            "accident_severity": type_options,
            "accident_types": hdc_options,
            "weekdays": weekday_order,
            "offending_vehicles": wrngdo_options,
            "offending_driver_age_groups": offending_driver_age_options,
            "damaged_vehicles": dmge_options,
            "victim_age_groups": victim_age_options,
            "fatal_types": fatal_type_options,
            "fatal_age_groups": fatal_age_options,
            "weather": wether_options,
            "violations": violt_options,
            "available_period": [
                (
                    f"{year_month_options[0].year}-"
                    f"{year_month_options[0].month:02d}"
                    if year_month_options
                    else None
                ),
                (
                    f"{year_month_options[-1].year}-"
                    f"{year_month_options[-1].month:02d}"
                    if year_month_options
                    else None
                ),
            ],
        }

        try:
            with st.spinner(
                "입력한 문장에서 검색조건을 해석하고 있습니다."
            ):
                parsed_natural_filter = parse_natural_language_filters(
                    natural_filter_query.strip(),
                    natural_filter_catalog,
                )

            if (
                parsed_natural_filter.get("status")
                == "need_clarification"
            ):
                st.warning(
                    parsed_natural_filter.get(
                        "clarification_question",
                        "검색조건이 정확하지 않습니다. 조금 더 구체적으로 입력해 주세요.",
                    )
                )

                interpretation_text = parsed_natural_filter.get(
                    "interpretation_summary",
                    "",
                )
                if interpretation_text:
                    st.caption(
                        f"AI 해석: {interpretation_text}"
                    )

            else:
                _queue_natural_language_filters(
                    parsed_natural_filter,
                    natural_filter_catalog,
                )
                st.rerun()

        except Exception as error:
            st.error(
                "AI 자연어 조건검색 중 오류가 발생했습니다."
            )
            st.code(
                f"{type(error).__name__}: {error}"
            )

if st.session_state.get(
    "natural_search_hotspot_requested",
    False,
):
    hotspot_info_parts = []

    if "hotspot_radius" in st.session_state:
        hotspot_info_parts.append(
            f"반경 {int(st.session_state['hotspot_radius'])}m"
        )

    if "hotspot_top_n" in st.session_state:
        hotspot_info_parts.append(
            f"상위 {int(st.session_state['hotspot_top_n'])}개소"
        )

    hotspot_info_text = (
        " / ".join(hotspot_info_parts)
        if hotspot_info_parts
        else "현재 사고다발지점 설정"
    )

    st.info(
        "AI가 입력조건을 반영해 교통사고를 필터링하였습니다. "
        f"{hotspot_info_text} 기준으로 GIS 분석 탭에서 결과를 확인하세요."
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
# 지도 레이어 생성# ============================================================
# 화면 선택은 제목 오른쪽에서 수행한다.
# 실제 st.tabs를 사용하지 않아 숨겨진 Leaflet 지도 초기화 문제를 방지한다.

# 사고다발지점 계산에 필요한 값은 어느 화면에서도 항상 정의
if "hotspot_radius" not in st.session_state:
    st.session_state.hotspot_radius = 100

if "hotspot_top_n" not in st.session_state:
    st.session_state.hotspot_top_n = 10

hotspot_radius = int(st.session_state.hotspot_radius)
hotspot_top_n = int(st.session_state.hotspot_top_n)


if selected_page == "GIS분석":
    # ========================================================
    # 사고다발지점 탐색 설정
    # - GIS 분석 탭 내부에 배치
    # - 레이어 선택 안내문보다 위에 표시
    # ========================================================
    gis_settings_expander = st.expander(
        "🔎 사고다발지점 탐색 및 설정",
        expanded=True,
    )
    gis_settings_expander.markdown(
        """
    <span class="gis-hotspot-marker" aria-hidden="true"></span>
    <div class="gis-hotspot-guide">반경과 탐색 지점 수를 설정하면 해당 조건에 따라 사고다발지점을 자동으로 탐색합니다.</div>
        """,
        unsafe_allow_html=True,
    )

    with gis_settings_expander.form("hotspot_settings_form"):
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
            st.write("")
            st.write("")

            hotspot_apply = st.form_submit_button(
                "설정 적용",
                use_container_width=True,
            )

    if hotspot_apply:
        st.session_state.hotspot_radius = int(radius_input)
        st.session_state.hotspot_top_n = int(top_n_input)

    hotspot_radius = st.session_state.hotspot_radius
    hotspot_top_n = st.session_state.hotspot_top_n

    # -----------------------------------
    # 지도 위 사고다발지점 표시 설정
    # 입력값은 적용 버튼을 눌렀을 때만 지도에 반영
    # 기본값: 반경 100m / 상위 5개
    # -------------
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

            <div style="
                margin-top:8px;
                padding-top:7px;
                border-top:1px solid #DDDDDD;
            ">
            <b>사고상황</b><br>
            1. 요일 : {weekday_html}<br>
            2. 시간 : {time_html}

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
    ].dropna(
        subset=["latitude", "longitude"]
    )

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
            <b>발생일자</b> : {row['acdnt_year']}년 {row['acdnt_month']} {row['acdnt_day']}<br>
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

    /* =====================================================
       메인 화면 여백
       ===================================================== */
    .block-container {
        padding-top: 4rem;
        padding-bottom: 2.5rem;
    }

    /* =====================================================
       KPI 카드
       ===================================================== */
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

    div[data-testid="stMetric"]
    [data-testid="stMetricValue"] {
        color: #0F172A;
        font-weight: 700;
    }

    /* =====================================================
       페이지 탭: 파일철 인덱스 스타일
       로컬·Streamlit Community Cloud 공통 대응
       ===================================================== */

    /* 탭 전체 영역 */
    div[data-testid="stTabs"] {
        margin-top: 12px;
    }

    /* 탭 인덱스가 놓이는 상단 레일 */
    div[data-testid="stTabs"] [role="tablist"],
    div[data-testid="stTabs"] [data-baseweb="tab-list"] {
        display: flex !important;
        gap: 8px !important;
        align-items: flex-end !important;

        padding: 12px 14px 0 14px !important;
        margin: 0 !important;

        border: 1px solid #CBD5E1 !important;
        border-bottom: 0 !important;
        border-radius: 16px 16px 0 0 !important;

        background-color: #E8EEF6 !important;

        box-shadow:
            0 -1px 0 rgba(15, 23, 42, 0.02) !important;

        overflow: visible !important;
    }

    /* 모든 파일철 인덱스 탭 */
    div[data-testid="stTabs"] [role="tab"],
    div[data-testid="stTabs"] [data-baseweb="tab"] {
        position: relative !important;
        z-index: 1 !important;

        flex: 0 0 auto !important;
        min-height: 52px !important;

        padding: 11px 26px !important;
        margin: 0 !important;

        border: 1px solid #B8C4D4 !important;
        border-bottom: 1px solid #9AAAC0 !important;
        border-radius: 12px 12px 0 0 !important;

        background-color: #D7E0EC !important;
        color: #475569 !important;

        font-size: 1.05rem !important;
        font-weight: 700 !important;

        box-shadow:
            inset 0 -2px 3px rgba(15, 23, 42, 0.04) !important;

        transform: none !important;

        transition:
            background-color 0.15s ease,
            color 0.15s ease,
            transform 0.15s ease !important;
    }

    /* 탭 내부 글자와 아이콘 */
    div[data-testid="stTabs"] [role="tab"] *,
    div[data-testid="stTabs"] [data-baseweb="tab"] * {
        margin: 0 !important;
        padding: 0 !important;

        color: inherit !important;
        font-size: 1.05rem !important;
        font-weight: 700 !important;

        white-space: nowrap !important;
    }

    /* 마우스를 올린 탭 */
    div[data-testid="stTabs"] [role="tab"]:hover,
    div[data-testid="stTabs"] [data-baseweb="tab"]:hover {
        background-color: #E2E8F0 !important;
        color: #1E3A8A !important;
        transform: translateY(-2px) !important;
    }

    /* 선택된 파일철 인덱스 탭 */
    div[data-testid="stTabs"]
    [role="tab"][aria-selected="true"],
    div[data-testid="stTabs"]
    [data-baseweb="tab"][aria-selected="true"] {
        z-index: 5 !important;

        margin-bottom: -1px !important;

        border-color: #94A3B8 !important;
        border-bottom-color: #FFFFFF !important;

        background-color: #FFFFFF !important;
        color: #1D4ED8 !important;

        box-shadow:
            0 -3px 8px rgba(15, 23, 42, 0.09) !important;

        transform: translateY(-3px) !important;
    }

    /* 선택된 탭 내부 글자와 아이콘 */
    div[data-testid="stTabs"]
    [role="tab"][aria-selected="true"] *,
    div[data-testid="stTabs"]
    [data-baseweb="tab"][aria-selected="true"] * {
        color: #1D4ED8 !important;
        font-weight: 800 !important;
    }

    /* Streamlit 기본 선택 밑줄 제거 */
    div[data-testid="stTabs"]
    [data-baseweb="tab-highlight"],
    div[data-testid="stTabs"]
    [data-testid="stTabsTabHighlight"] {
        display: none !important;
    }

    /* =====================================================
       탭 아래 본문: 흰색 노트 페이지
       ===================================================== */
    div[data-testid="stTabs"] [role="tabpanel"],
    div[data-testid="stTabs"] [data-baseweb="tab-panel"] {
        position: relative !important;
        z-index: 2 !important;

        min-height: 120px !important;

        padding: 22px 20px 26px 20px !important;
        margin-top: 0 !important;

        border: 1px solid #94A3B8 !important;
        border-radius: 0 14px 14px 14px !important;

        background-color: #FFFFFF !important;

        box-shadow:
            0 5px 16px rgba(15, 23, 42, 0.08) !important;
    }

    /* =====================================================
       작은 화면 대응
       ===================================================== */
    @media (max-width: 768px) {

        div[data-testid="stTabs"] [role="tablist"],
        div[data-testid="stTabs"] [data-baseweb="tab-list"] {
            gap: 5px !important;

            padding-left: 8px !important;
            padding-right: 8px !important;

            overflow-x: auto !important;
            overflow-y: visible !important;
        }

        div[data-testid="stTabs"] [role="tab"],
        div[data-testid="stTabs"] [data-baseweb="tab"] {
            min-height: 46px !important;
            padding: 9px 15px !important;
        }

        div[data-testid="stTabs"] [role="tab"] *,
        div[data-testid="stTabs"] [data-baseweb="tab"] * {
            font-size: 0.92rem !important;
        }

        div[data-testid="stTabs"] [role="tabpanel"],
        div[data-testid="stTabs"] [data-baseweb="tab-panel"] {
            padding: 16px 10px 20px 10px !important;
        }
    }

    /* =====================================================
       접기 영역
       ===================================================== */
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


st.markdown(
    """
    <style>
    /* CSS만 담긴 Streamlit 요소가 빈 세로 공간을 차지하지 않도록 정리 */
    [data-testid="stMarkdownContainer"]:has(style) {
        display: none;
    }

    .block-container {
        padding-top: 1.35rem !important;
        background: transparent !important;
    }

    [data-testid="stMainBlockContainer"] {
        background: transparent !important;
    }

    [data-testid="stAppViewContainer"] {
        background: #EEF3F8;
    }

    /* Streamlit 기본 상단 헤더·Deploy·더보기 메뉴 숨김 */
    header,
    header[data-testid="stHeader"] {
        display: none !important;
        height: 0 !important;
        min-height: 0 !important;
        background: transparent !important;
    }

    [data-testid="stToolbar"],
    [data-testid="stDecoration"],
    [data-testid="stStatusWidget"],
    [data-testid="stAppDeployButton"],
    [data-testid="stMainMenu"],
    .stAppDeployButton,
    .stDeployButton,
    #MainMenu {
        display: none !important;
        visibility: hidden !important;
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0B3558 0%, #102A43 100%) !important;
        border-right: 1px solid #173F63 !important;
    }

    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] p {
        color: #E7F0F8 !important;
    }

    [data-testid="stSidebar"] hr {
        border-color: rgba(255, 255, 255, 0.16) !important;
    }

    [data-testid="stSidebar"] [data-baseweb="select"] > div {
        color: #F8FBFF !important;
        background: rgba(255, 255, 255, 0.09) !important;
        border-color: rgba(255, 255, 255, 0.22) !important;
        border-radius: 9px !important;
    }

    [data-testid="stSidebar"] [data-testid="stExpander"] {
        border-color: rgba(255, 255, 255, 0.17) !important;
        background: rgba(255, 255, 255, 0.055) !important;
    }

    [data-testid="stSidebar"] [data-testid="stExpander"] summary {
        color: #F1F7FD !important;
        background: transparent !important;
    }

    [data-testid="stSidebar"] .stButton > button {
        border-color: #5F91BE !important;
        color: #FFFFFF !important;
        background: linear-gradient(135deg, #2563EB, #1D4ED8) !important;
    }

    [data-testid="stSidebar"] div[data-testid="stForm"] {
        padding: 9px !important;
        border: 1px solid rgba(255, 255, 255, 0.18) !important;
        border-radius: 12px !important;
        background: rgba(57, 105, 145, 0.24) !important;
        box-shadow: none !important;
    }

    [data-testid="stSidebar"] div[data-testid="stForm"] input {
        color: #102A43 !important;
        background: #F8FBFF !important;
    }

    [data-testid="stSidebar"] div[data-testid="stForm"] button {
        min-height: 38px;
        border: 0 !important;
        color: #FFFFFF !important;
        background: linear-gradient(135deg, #2563EB, #1D4ED8) !important;
        font-weight: 800 !important;
    }

    .sidebar-ai-search-title {
        display: flex;
        flex-direction: column;
        gap: 2px;
        margin: 2px 0 8px;
        color: #FFFFFF !important;
        font-size: 1rem;
        font-weight: 850;
    }

    .sidebar-ai-search-title small {
        color: #BCD3E8 !important;
        font-size: 0.76rem;
        font-weight: 500;
    }

    .sidebar-direct-filter-title {
        margin: 16px 0 9px;
        padding-top: 13px;
        border-top: 1px solid rgba(255, 255, 255, 0.16);
        color: #D7E8F7 !important;
        font-size: 0.82rem;
        font-weight: 800;
        letter-spacing: 0.04em;
    }

    /* 화면 선택 메뉴에만 적용되는 독립형 내비게이션 */
    section.main div[data-testid="stRadio"],
    /* 본문 expander 제목 */
    [data-testid="stMain"]
    div[data-testid="stExpander"]
    > details > summary p {
        display: block !important;
        color: #173F63 !important;
        font-size: 2rem !important;
        line-height: 1.4 !important;
        font-weight: 850 !important;
        letter-spacing: -0.02em !important;
    }

    section.main div[data-testid="stRadio"] > div,
    [data-testid="stMain"] div[data-testid="stRadio"] > div {
        display: flex;
        flex-wrap: nowrap;
        gap: 8px;
        padding: 0;
    }

    section.main div[data-testid="stRadio"] label,
    [data-testid="stMain"] div[data-testid="stRadio"] label {
        min-height: 40px;
        padding: 8px 20px !important;
        border: 1px solid #CBD8E6;
        border-radius: 10px;
        color: #31516F;
        background: #FFFFFF;
        font-weight: 800;
        box-shadow: 0 2px 7px rgba(16, 42, 67, 0.05);
        transition: background-color 0.15s ease, color 0.15s ease, transform 0.15s ease;
    }

    section.main div[data-testid="stRadio"] label:hover,
    [data-testid="stMain"] div[data-testid="stRadio"] label:hover {
        border-color: #8EA9C2;
        color: #102A43;
        background: #F8FBFF;
        transform: translateY(-1px);
    }

    section.main div[data-testid="stRadio"] label > div:first-child,
    [data-testid="stMain"] div[data-testid="stRadio"] label > div:first-child {
        display: none;
    }

    section.main div[data-testid="stRadio"] label:has(input:checked),
    [data-testid="stMain"] div[data-testid="stRadio"] label:has(input:checked) {
        border-color: #102A43;
        color: #FFFFFF !important;
        background: linear-gradient(135deg, #0B3558, #102A43);
        box-shadow: 0 5px 13px rgba(16, 42, 67, 0.18);
    }

    section.main div[data-testid="stRadio"] label:has(input:checked) p,
    [data-testid="stMain"] div[data-testid="stRadio"] label:has(input:checked) p {
        color: #FFFFFF !important;
    }

    /* AI 메뉴는 청록색 포인트로 별도 강조 */
    section.main div[data-testid="stRadio"] label:last-child,
    [data-testid="stMain"] div[data-testid="stRadio"] label:last-child {
        border-color: #75C9C7;
        color: #087E7D;
    }

    section.main div[data-testid="stRadio"] label:last-child:has(input:checked),
    [data-testid="stMain"] div[data-testid="stRadio"] label:last-child:has(input:checked) {
        border-color: #0EA5A4;
        color: #FFFFFF !important;
        background: linear-gradient(135deg, #087E7D, #0EA5A4);
    }

    /* =====================================================
       발표용 상단 시스템 헤더
       - key/border 인자에 의존하지 않음
       - 제목과 4개 화면 메뉴가 들어있는 동일 horizontal block만 지정
       ===================================================== */
    [data-testid="stMain"] div[data-testid="stHorizontalBlock"]:has(.taap-header-marker) {
        align-items: center !important;
        padding: 13px 18px !important;
        margin-bottom: 13px !important;
        border: 1px solid #1C4B70 !important;
        border-radius: 16px !important;
        background: linear-gradient(110deg, #0A2740 0%, #0D385A 58%, #125273 100%) !important;
        box-shadow: 0 9px 24px rgba(8, 35, 58, 0.18) !important;
    }

    .taap-header-marker {
        display: none;
    }

    /* 헤더 안 화면 선택 메뉴: 오른쪽 정렬 */
    [data-testid="stMain"] div[data-testid="stHorizontalBlock"]:has(.taap-header-marker)
    div[data-testid="stRadio"] {
        width: 100% !important;
        margin: 0 !important;
    }

    [data-testid="stMain"] div[data-testid="stHorizontalBlock"]:has(.taap-header-marker)
    div[data-testid="stRadio"] > div {
        width: 100% !important;
        display: flex !important;
        justify-content: flex-end !important;
        flex-wrap: nowrap !important;
        gap: 8px !important;
    }

    [data-testid="stMain"] div[data-testid="stHorizontalBlock"]:has(.taap-header-marker)
    div[data-testid="stRadio"] label {
        min-height: 42px !important;
        padding: 8px 17px !important;
        border: 1px solid rgba(219, 234, 254, 0.28) !important;
        border-radius: 10px !important;
        color: #E8F1F8 !important;
        background: rgba(255, 255, 255, 0.07) !important;
        box-shadow: none !important;
        backdrop-filter: blur(3px);
    }

    [data-testid="stMain"] div[data-testid="stHorizontalBlock"]:has(.taap-header-marker)
    div[data-testid="stRadio"] label p {
        color: #E8F1F8 !important;
        font-weight: 800 !important;
        white-space: nowrap !important;
    }

    [data-testid="stMain"] div[data-testid="stHorizontalBlock"]:has(.taap-header-marker)
    div[data-testid="stRadio"] label:hover {
        border-color: rgba(255, 255, 255, 0.58) !important;
        background: rgba(255, 255, 255, 0.13) !important;
        transform: translateY(-1px);
    }

    [data-testid="stMain"] div[data-testid="stHorizontalBlock"]:has(.taap-header-marker)
    div[data-testid="stRadio"] label:has(input:checked) {
        border-color: #7DB4FF !important;
        color: #FFFFFF !important;
        background: linear-gradient(135deg, #2563EB, #1D4ED8) !important;
        box-shadow: 0 5px 14px rgba(37, 99, 235, 0.34) !important;
    }

    [data-testid="stMain"] div[data-testid="stHorizontalBlock"]:has(.taap-header-marker)
    div[data-testid="stRadio"] label:has(input:checked) p {
        color: #FFFFFF !important;
    }

    /* AI 메뉴도 헤더 안에서는 동일한 메뉴 체계를 사용 */
    [data-testid="stMain"] div[data-testid="stHorizontalBlock"]:has(.taap-header-marker)
    div[data-testid="stRadio"] label:last-child {
        border-color: rgba(94, 234, 212, 0.42) !important;
    }

    [data-testid="stMain"] div[data-testid="stHorizontalBlock"]:has(.taap-header-marker)
    div[data-testid="stRadio"] label:last-child:has(input:checked) {
        border-color: #6EE7E2 !important;
        background: linear-gradient(135deg, #0F8B8D, #0EA5A4) !important;
        box-shadow: 0 5px 14px rgba(14, 165, 164, 0.30) !important;
    }

    /* =====================================================
       본문 expander: 제목 문구는 숨기고 접기/펼치기 화살표만 유지
       - 사이드바 expander에는 적용하지 않음
       ===================================================== */
    [data-testid="stMain"] div[data-testid="stExpander"] > details {
        border: 1px solid #D9E3ED !important;
        border-radius: 14px !important;
        background: #FFFFFF !important;
        box-shadow: 0 3px 12px rgba(15, 39, 71, 0.045) !important;
        overflow: hidden !important;
    }

    [data-testid="stMain"] div[data-testid="stExpander"] > details > summary {
        min-height: 28px !important;
        padding: 4px 10px !important;
        background: #F8FBFE !important;
        border-bottom: 1px solid #EDF2F7 !important;
    }

    [data-testid="stMain"] div[data-testid="stExpander"] > details > summary p {
        display: none !important;
    }

    [data-testid="stMain"] div[data-testid="stExpander"] > details > summary:hover {
        background: #F1F6FB !important;
    }

    .taap-main-header {
        display: flex;
        align-items: center;
        min-height: 72px;
        padding: 3px 5px;
        margin: 0;
        border: 0;
        border-radius: 0;
        background: transparent;
        box-shadow: none;
    }

    .taap-title-group {
        display: flex;
        align-items: center;
        gap: 11px;
        min-width: 270px;
        padding-right: 18px;
        border-right: 1px solid rgba(219, 234, 254, 0.28);
    }

    .taap-title-icon {
        display: grid;
        place-items: center;
        width: 52px;
        height: 52px;
        flex: 0 0 52px;
        border-radius: 12px;
        background: linear-gradient(145deg, #2563EB, #0EA5A4);
        box-shadow: 0 6px 14px rgba(37, 99, 235, 0.20);
        font-size: 1.65rem;
    }

    .taap-title-text {
        color: #FFFFFF;
        font-size: 2.75rem;
        line-height: 1;
        font-weight: 900;
        letter-spacing: -0.05em;
        white-space: nowrap;
    }

    .taap-subtitle-group {
        min-width: 330px;
        padding-left: 18px;
    }

    .taap-subtitle-en {
        color: #DBEAFE;
        font-size: 1.18rem;
        font-weight: 800;
    }

    .taap-subtitle-ko {
        margin-top: 4px;
        color: #C7D8E8;
        font-size: 1.04rem;
        font-weight: 700;
    }

    .taap-condition-strip {
        display: flex;
        align-items: center;
        justify-content: flex-end;
        flex-wrap: wrap;
        gap: 5px 0;
        margin: 7px 2px 9px;
        color: #64748B;
        font-size: 1.02rem;
        line-height: 1.4;
    }

    .taap-condition-strip span {
        padding: 0 10px;
        border-right: 1px solid #C7D5E4;
        white-space: nowrap;
    }

    .taap-condition-strip span:last-child {
        padding-right: 0;
        border-right: 0;
    }

    .taap-condition-strip b {
        margin-right: 5px;
        color: #173F63;
        font-weight: 850;
    }

    .taap-condition-group {
        display: flex;
        justify-content: flex-end;
        flex-wrap: wrap;
        gap: 6px 0;
        margin-left: auto;
        color: #64748B;
        font-size: 0.78rem;
    }

    .taap-condition-group span {
        padding: 0 9px;
        border-right: 1px solid #D5E0EB;
        white-space: nowrap;
    }

    .taap-condition-group span:last-child {
        padding-right: 0;
        border-right: 0;
    }

    .taap-condition-group b {
        margin-right: 4px;
        color: #173F63;
        font-weight: 850;
    }

    .dashboard-section-title {
        display: flex;
        align-items: baseline;
        justify-content: space-between;
        min-height: 38px;
        padding: 3px 4px 8px 4px;
        margin-bottom: 2px;
        border-bottom: 0;
    }

    .dashboard-section-title span {
        color: #173F63;
        font-size: 1.10rem;
        font-weight: 900;
        letter-spacing: -0.025em;
    }

    .dashboard-section-title small {
        color: #71869A;
        font-size: 0.76rem;
        font-weight: 600;
    }

    .dashboard-chart-title {
        min-height: 27px;
        color: #173F63;
        font-size: 1rem;
        font-weight: 900;
        letter-spacing: -0.025em;
    }

    .dashboard-summary-panel {
        min-height: 536px;
        padding: 18px 17px;
        border: 1px solid #D6E1EC;
        border-radius: 13px;
        background: linear-gradient(180deg, #F7FAFD 0%, #EEF5FA 100%);
        box-sizing: border-box;
    }

    .dashboard-summary-kicker {
        margin-bottom: 9px;
        color: #0E7490;
        font-size: 0.9rem;
        font-weight: 900;
        letter-spacing: 0.02em;
    }

    .dashboard-summary-lead {
        margin-bottom: 14px;
        padding-bottom: 14px;
        border-bottom: 1px solid #D8E4EE;
        color: #294A66;
        font-size: 1.2rem;
        line-height: 1.65;
    }

    .dashboard-summary-lead b {
        color: #0B3558;
        font-weight: 900;
    }

    .dashboard-summary-item {
        display: flex;
        flex-direction: column;
        gap: 3px;
        margin-bottom: 10px;
        padding: 9px 11px;
        border: 1px solid #DEE8F1;
        border-radius: 9px;
        background: rgba(255, 255, 255, 0.78);
    }

    .dashboard-summary-item span {
        color: #71869A;
        font-size: 0.74rem;
        font-weight: 700;
    }

    .dashboard-summary-item b {
        color: #173F63;
        font-size: 1.0rem;
        font-weight: 900;
    }

    .dashboard-summary-note {
        margin-top: 13px;
        padding: 7px 12px;
        border-left: 3px solid #0EA5A4;
        border-radius: 0 8px 8px 0;
        background: #E8F7F6;
        color: #416278;
        font-size: 0.9rem;
        line-height: 1.65;
    }

    /* 본문 expander는 라벨 자체도 공백으로 변경. 헤더 높이는 화살표만 남도록 축소 */
    [data-testid="stMain"] div[data-testid="stExpander"] > details > summary {
        min-height: 24px !important;
        height: 24px !important;
        padding: 1px 8px !important;
        border-bottom: 0 !important;
        background: #FFFFFF !important;
    }

    [data-testid="stMain"] div[data-testid="stExpander"] > details > summary [data-testid="stMarkdownContainer"],
    [data-testid="stMain"] div[data-testid="stExpander"] > details > summary p,
    [data-testid="stMain"] div[data-testid="stExpander"] > details > summary span:not([data-testid]) {
        font-size: 0 !important;
        line-height: 0 !important;
    }

    /* 대시보드 지도/그래프 섹션 제목 아래 선 제거 */
    .dashboard-section-title {
        border-bottom: 0 !important;
    }

    /* 본문은 한 단계 진한 블루그레이, 흰색 카드 대비 강화 */
    [data-testid="stAppViewContainer"] {
        background: #E8EEF5 !important;
    }

    /* 자연어 검색: 좁고 한 줄로 */
    [data-testid="stSidebar"] .sidebar-ai-search-title {
        margin-bottom: 6px !important;
    }
    [data-testid="stSidebar"] div[data-testid="stForm"] {
        padding: 7px !important;
    }
    [data-testid="stSidebar"] div[data-testid="stForm"] [data-testid="stHorizontalBlock"] {
        gap: 6px !important;
        align-items: center !important;
    }
    [data-testid="stSidebar"] div[data-testid="stForm"] input {
        min-height: 46px !important;
        height: 46px !important;
        padding: 0 11px !important;
        font-size: 1.1rem !important;
        line-height: 46px !important;
        font-weight: 400 !important;
        box-sizing: border-box !important;
    }
    [data-testid="stSidebar"] div[data-testid="stForm"] [data-baseweb="input"],
    [data-testid="stSidebar"] div[data-testid="stForm"] [data-baseweb="base-input"] {
        min-height: 46px !important;
        height: 46px !important;
        display: flex !important;
        align-items: center !important;
    }
    /* 자연어 입력창 아래의 'Press Enter to submit form' 안내만 숨김 */
    [data-testid="stSidebar"] div[data-testid="stForm"]
    div[data-testid="InputInstructions"],
    [data-testid="stSidebar"] div[data-testid="stForm"]
    div[data-testid="stInputInstructions"],
    [data-testid="stSidebar"] div[data-testid="stForm"]
    div[data-testid="stTextInput"] small {
        display: none !important;
    }
    [data-testid="stSidebar"] div[data-testid="stForm"] button {
        min-height: 46px !important;
        height: 46px !important;
        padding: 0 7px !important;
        font-size: 0.90rem !important;
        white-space: nowrap !important;
    }

    /* 사이드바 select / multiselect 선택값은 항상 흰색 */
    [data-testid="stSidebar"] [data-baseweb="select"] span,
    [data-testid="stSidebar"] [data-baseweb="select"] div,
    [data-testid="stSidebar"] [data-baseweb="tag"] span,
    [data-testid="stSidebar"] [data-baseweb="tag"] div {
        color: #FFFFFF !important;
    }

    /* selectbox / multiselect 우측 드롭다운 아이콘 */
    [data-testid="stSidebar"] [data-baseweb="select"] svg {
        fill: #FFFFFF !important;
        color: #FFFFFF !important;
    }

    /* 사이드바 하단 조건 expander: hover 시에도 네이비 + 흰색 유지 */
    [data-testid="stSidebar"] [data-testid="stExpander"] summary:hover {
        background: rgba(61, 119, 161, 0.32) !important;
        color: #FFFFFF !important;
    }
    [data-testid="stSidebar"] [data-testid="stExpander"] summary:hover p,
    [data-testid="stSidebar"] [data-testid="stExpander"] summary:hover span,
    [data-testid="stSidebar"] [data-testid="stExpander"] summary:hover svg {
        color: #FFFFFF !important;
        fill: #FFFFFF !important;
    }

    /* =====================================================
       TAAP-AI 최종 단일 우선순위 UI 규칙
       - 기준 색상은 map_re7의 밝은 본문 + 네이비 사이드바 유지
       - 헤더, KPI, AI 분석, GIS 간격만 이 블록에서 최종 제어
       ===================================================== */

    /* 상단 제목과 메뉴: 한 배경으로 묶지 않고 5개 독립 박스로 고정 */
    [data-testid="stMain"] div[data-testid="stHorizontalBlock"]:has(.taap-header-marker) {
        gap: 10px !important;
        align-items: stretch !important;
        margin: 0 !important;
        padding: 0 !important;
        border: 0 !important;
        border-radius: 0 !important;
        background: transparent !important;
        box-shadow: none !important;
        overflow: visible !important;
    }

    [data-testid="stMain"] div[data-testid="stHorizontalBlock"]:has(.taap-header-marker)
    > div[data-testid="column"] {
        background: transparent !important;
    }

    .taap-main-header {
    width: 100% !important;
    height: 116px !important;
    min-height: 116px !important;

    display: flex !important;
    align-items: center !important;

    gap: 0 !important;
    margin: 0 !important;
    padding: 17px 20px !important;

    border: 1px solid #245474 !important;
    border-radius: 16px !important;

    background: linear-gradient(
        145deg,
        #082B46 0%,
        #0D3E5C 58%,
        #12516D 100%
    ) !important;

    box-shadow:
        0 9px 23px rgba(16, 42, 67, 0.21) !important;

    box-sizing: border-box !important;
    overflow: hidden !important;

    font-family:
        "Pretendard",
        "Noto Sans KR",
        "Malgun Gothic",
        "Apple SD Gothic Neo",
        Arial,
        sans-serif !important;
}


/* 지도 아이콘 + TAAP-AI */
.taap-main-header .taap-title-group {
    min-width: 285px !important;

    display: flex !important;
    align-items: center !important;

    gap: 14px !important;
    padding-right: 22px !important;

    border-right:
        1px solid rgba(191, 219, 254, 0.34) !important;
}


/* 제목 왼쪽 지도 아이콘 */
.taap-main-header .taap-title-icon {
    width: 62px !important;
    height: 62px !important;
    flex: 0 0 62px !important;

    display: grid !important;
    place-items: center !important;

    border: 1px solid rgba(255, 255, 255, 0.13) !important;
    border-radius: 15px !important;

    background: linear-gradient(
        145deg,
        #2563EB,
        #0EA5A4
    ) !important;

    box-shadow:
        0 7px 18px rgba(14, 165, 164, 0.28) !important;

    font-size: 1.8rem !important;
}


    /* TAAP-AI 전체 제목 */
    .taap-main-header .taap-title-text {
        display: flex !important;
        align-items: baseline !important;

        font-family:
            Arial,
            "Pretendard",
            "Noto Sans KR",
            sans-serif !important;

        font-size: 3.15rem !important;
        line-height: 1 !important;
        font-weight: 900 !important;

        letter-spacing: -0.01em !important;
        white-space: nowrap !important;
    }


    /* TAAP- 부분 */
    .taap-title-main {
        color: #FFFFFF !important;
    }


    /* AI 부분만 청록색으로 강조 */
    .taap-title-ai {
        color: #5EEAD4 !important;

        text-shadow:
            0 0 14px rgba(94, 234, 212, 0.20) !important;
    }


    /* 오른쪽 영문·한글 부제 묶음 */
    .taap-main-header .taap-subtitle-group {
        min-width: 0 !important;

        display: flex !important;
        flex-direction: column !important;
        justify-content: center !important;

        padding-left: 22px !important;

        overflow: hidden !important;
    }


    /* 영문 부제 */
    .taap-main-header .taap-subtitle-en {
        margin: 0 !important;

        color: #DCECF7 !important;

        font-size: 1.10rem !important;
        line-height: 1.25 !important;
        font-weight: 800 !important;

        letter-spacing: -0.018em !important;
        white-space: nowrap !important;
    }


    /* 한글 부제 */
    .taap-main-header .taap-subtitle-ko {
        margin-top: 7px !important;

        color: #AFCBDD !important;

        font-size: 0.96rem !important;
        line-height: 1.25 !important;
        font-weight: 700 !important;

        letter-spacing: -0.025em !important;
        white-space: nowrap !important;
    }

    [data-testid="stMain"] div[data-testid="stHorizontalBlock"]:has(.taap-header-marker)
    div[data-testid="stRadio"] {
        width: 100% !important;
        height: 112px !important;
        min-height: 112px !important;
        display: flex !important;
        align-items: center !important;
        justify-content: flex-end !important;
        margin: 0 !important;
        padding: 8px 0 0 !important;
        border: 0 !important;
        border-radius: 0 !important;
        background: transparent !important;
        box-shadow: none !important;
    }

    [data-testid="stMain"] div[data-testid="stHorizontalBlock"]:has(.taap-header-marker)
    div[data-testid="stRadio"] > div,
    [data-testid="stMain"] div[data-testid="stHorizontalBlock"]:has(.taap-header-marker)
    div[data-testid="stRadio"] div[role="radiogroup"] {
        width: 100% !important;
        height: 58px !important;
        display: flex !important;
        align-items: center !important;
        justify-content: flex-end !important;
        flex-wrap: nowrap !important;
        gap: 11px !important;
        padding: 0 !important;
        margin-left: auto !important;
    }

    [data-testid="stMain"] div[data-testid="stHorizontalBlock"]:has(.taap-header-marker)
    div[data-testid="stRadio"] label {
        min-width: 116px !important;
        height: 58px !important;
        min-height: 58px !important;
        flex: 0 0 auto !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        padding: 9px 16px !important;
        border: 1px solid #234E70 !important;
        border-radius: 14px !important;
        color: #EAF3FA !important;
        background: linear-gradient(145deg, #0A2C47 0%, #114867 100%) !important;
        box-shadow: 0 7px 18px rgba(16, 42, 67, 0.18) !important;
        box-sizing: border-box !important;
    }

    [data-testid="stMain"] div[data-testid="stHorizontalBlock"]:has(.taap-header-marker)
    div[data-testid="stRadio"] label p {
        color: #EAF3FA !important;
        font-size: 0.94rem !important;
        font-weight: 850 !important;
        text-align: center !important;
        white-space: nowrap !important;
    }

    [data-testid="stMain"] div[data-testid="stHorizontalBlock"]:has(.taap-header-marker)
    div[data-testid="stRadio"] label:hover {
        border-color: #6E9FC3 !important;
        background: linear-gradient(145deg, #123B59 0%, #176080 100%) !important;
        transform: translateY(-1px) !important;
    }

    [data-testid="stAppViewContainer"] [data-testid="stMain"]
    div[data-testid="stHorizontalBlock"]:has(.taap-header-marker)
    div[data-testid="stRadio"] label:has(input:checked),
    [data-testid="stAppViewContainer"] [data-testid="stMain"]
    div[data-testid="stHorizontalBlock"]:has(.taap-header-marker)
    div[data-testid="stRadio"] label:has([aria-checked="true"]),
    [data-testid="stAppViewContainer"] [data-testid="stMain"]
    div[data-testid="stHorizontalBlock"]:has(.taap-header-marker)
    div[data-testid="stRadio"] label[data-checked="true"] {
        color: #FFFFFF !important;
        border-color: #5EEAD4 !important;
        background: linear-gradient(135deg, #0F8B8D 0%, #0EA5A4 100%) !important;
        box-shadow: 0 8px 20px rgba(14, 165, 164, 0.32) !important;
    }

    [data-testid="stAppViewContainer"] [data-testid="stMain"]
    div[data-testid="stHorizontalBlock"]:has(.taap-header-marker)
    div[data-testid="stRadio"] label:has(input:checked) p,
    [data-testid="stAppViewContainer"] [data-testid="stMain"]
    div[data-testid="stHorizontalBlock"]:has(.taap-header-marker)
    div[data-testid="stRadio"] label:has([aria-checked="true"]) p,
    [data-testid="stAppViewContainer"] [data-testid="stMain"]
    div[data-testid="stHorizontalBlock"]:has(.taap-header-marker)
    div[data-testid="stRadio"] label[data-checked="true"] p {
        color: #FFFFFF !important;
    }

    /* KPI: 흰 카드에 의미별 상단선과 숫자색 적용 */
    .taap-kpi-card {
        position: relative;
        min-height: 101px;
        padding: 18px 16px 13px;
        border: 1px solid #D7E1EA;
        border-radius: 14px;
        background: #FFFFFF;
        box-shadow: 0 4px 13px rgba(15, 42, 67, 0.08);
        box-sizing: border-box;
        overflow: hidden;
    }

    /* 실제 HTML 간격 요소: 제목과 KPI 사이 18px 확보 */
    .taap-header-kpi-spacer {
        display: block !important;
        width: 100% !important;
        height: 28px !important;
        min-height: 28px !important;
    }

    /* 발생연월 슬라이더 아래의 전체 범위 시작·종료 눈금 숨김 */
    [data-testid="stSidebar"] [data-testid="stTickBar"],
    [data-testid="stSidebar"] [data-testid="stSliderTickBar"] {
        display: none !important;
    }

    .taap-kpi-card::before {
        content: "";
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 5px;
        background: var(--kpi-color);
    }

    .taap-kpi-label {
        margin-bottom: 8px;
        color: #64748B;
        font-size: 1.0rem;
        font-weight: 700;
    }

    .taap-kpi-value {
        color: var(--kpi-color);
        font-size: 2.2rem;
        line-height: 1.05;
        font-weight: 900;
        letter-spacing: -0.035em;
    }

    .taap-kpi-number {
        font-size: 1em !important;
        font-weight: 900 !important;
    }

    .taap-kpi-unit {
        display: inline-block !important;
        margin-left: 2px !important;
        font-size: 0.58em !important;
        line-height: 1 !important;
        font-weight: 800 !important;
        vertical-align: 0.10em !important;
        letter-spacing: -0.02em !important;
    }

    .kpi-station { --kpi-color: #2563EB; }
    .kpi-total   { --kpi-color: #475569; }
    .kpi-fatal   { --kpi-color: #C62828; }
    .kpi-serious { --kpi-color: #D97706; }

    /* GIS 사고다발지점 설정: 바깥 카드는 투명, 실제 입력부만 옅게 구분 */
    [data-testid="stMain"] div[data-testid="stExpander"]:has(.gis-hotspot-marker),
    section.main div[data-testid="stExpander"]:has(.gis-hotspot-marker) {
        margin: 0 0 0.45rem !important;
        padding: 0 !important;
        border: 0 !important;
        background: transparent !important;
        box-shadow: none !important;
    }

    [data-testid="stMain"] div[data-testid="stExpander"]:has(.gis-hotspot-marker) > details,
    section.main div[data-testid="stExpander"]:has(.gis-hotspot-marker) > details {
        border: 0 !important;
        border-radius: 0 !important;
        background: transparent !important;
        box-shadow: none !important;
    }

    [data-testid="stMain"] div[data-testid="stExpander"]:has(.gis-hotspot-marker) summary,
    section.main div[data-testid="stExpander"]:has(.gis-hotspot-marker) summary {
        min-height: 48px !important;
        padding: 5px 2px 8px !important;
        color: #173F5F !important;
        background: transparent !important;
    }

    [data-testid="stMain"] div[data-testid="stExpander"]:has(.gis-hotspot-marker) summary p,
    section.main div[data-testid="stExpander"]:has(.gis-hotspot-marker) summary p {
        color: #173F5F !important;
        font-size: 1.55rem !important;
        line-height: 1.35 !important;
        font-weight: 850 !important;
    }

    [data-testid="stMain"] div[data-testid="stExpander"]:has(.gis-hotspot-marker) div[data-testid="stExpanderDetails"],
    section.main div[data-testid="stExpander"]:has(.gis-hotspot-marker) div[data-testid="stExpanderDetails"],
    [data-testid="stMain"] div[data-testid="stExpander"]:has(.gis-hotspot-marker) > details > div,
    section.main div[data-testid="stExpander"]:has(.gis-hotspot-marker) > details > div {
        padding: 0 0 7px !important;
        border: 0 !important;
        background: transparent !important;
        box-shadow: none !important;
    }

    .gis-hotspot-marker {
        display: none !important;
    }

    /* 사고다발지점 설정 안내문 */
    .gis-hotspot-guide {
        margin: 0 0 10px !important;
        color: #52697A !important;
        font-size: 1rem !important;
        line-height: 1.45 !important;
        font-weight: 650 !important;
    }

    /* 설정 입력 영역: 본문보다 진한 청회색 띠 */
    [data-testid="stMain"]
    div[data-testid="stExpander"]:has(.gis-hotspot-marker)
    div[data-testid="stForm"],
    section.main
    div[data-testid="stExpander"]:has(.gis-hotspot-marker)
    div[data-testid="stForm"] {
        padding: 13px 14px 11px !important;
        border-top: 1px solid #B7CBD8 !important;
        border-right: 1px solid #B7CBD8 !important;
        border-bottom: 1px solid #B7CBD8 !important;
        border-left: 4px solid #0F8B8D !important;
        border-radius: 10px !important;
        background: #DDEAF1 !important;
        box-shadow: 0 2px 7px rgba(16, 42, 67, 0.06) !important;
    }

    /* 분석 반경·사고다발지역 수 라벨 */
    [data-testid="stMain"]
    div[data-testid="stExpander"]:has(.gis-hotspot-marker)
    div[data-testid="stForm"] label,
    section.main
    div[data-testid="stExpander"]:has(.gis-hotspot-marker)
    div[data-testid="stForm"] label {
        color: #173F5F !important;
        font-weight: 750 !important;
    }

    /* 숫자 입력칸은 흰색으로 분리 */
    [data-testid="stMain"]
    div[data-testid="stExpander"]:has(.gis-hotspot-marker)
    div[data-testid="stNumberInput"] > div,
    section.main
    div[data-testid="stExpander"]:has(.gis-hotspot-marker)
    div[data-testid="stNumberInput"] > div {
        border-color: #C2D2DD !important;
        background: #FFFFFF !important;
    }

    /* 설정 적용 버튼 강조 */
    [data-testid="stMain"]
    div[data-testid="stExpander"]:has(.gis-hotspot-marker)
    div[data-testid="stForm"] .stButton > button,
    section.main
    div[data-testid="stExpander"]:has(.gis-hotspot-marker)
    div[data-testid="stForm"] .stButton > button {
        color: #FFFFFF !important;
        border: 1px solid #0F8B8D !important;
        background: linear-gradient(
            135deg,
            #0F7F82 0%,
            #12A5A2 100%
        ) !important;
        font-weight: 800 !important;
        box-shadow: 0 3px 8px rgba(15, 139, 141, 0.22) !important;
    }

    [data-testid="stMain"]
    div[data-testid="stExpander"]:has(.gis-hotspot-marker)
    div[data-testid="stForm"] .stButton > button:hover,
    section.main
    div[data-testid="stExpander"]:has(.gis-hotspot-marker)
    div[data-testid="stForm"] .stButton > button:hover {
        border-color: #0B6F72 !important;
        background: linear-gradient(
            135deg,
            #0B7376 0%,
            #0F918F 100%
        ) !important;
    }

    /* AI 분석 허브 */
    .ai-analysis-hero {
        display: flex;
        align-items: center;
        gap: 16px;
        margin: 7px 0 18px;
        padding: 22px 24px;
        border: 1px solid #BCD4E5;
        border-radius: 16px;
        background: linear-gradient(115deg, #F8FBFE 0%, #EDF5FA 58%, #E6F5F3 100%);
        box-shadow: 0 6px 18px rgba(15, 55, 82, 0.08);
    }

    .ai-analysis-hero-icon {
        width: 62px;
        height: 62px;
        flex: 0 0 62px;
        display: grid;
        place-items: center;
        border-radius: 15px;
        color: #FFFFFF;
        background: linear-gradient(145deg, #2563EB, #0EA5A4);
        box-shadow: 0 7px 16px rgba(37, 99, 235, 0.22);
        font-size: 1.12rem;
        font-weight: 900;
    }

    .ai-analysis-hero-copy { flex: 1; min-width: 0; }

    .ai-analysis-eyebrow {
        margin-bottom: 6px;
        color: #0E7490;
        font-size: 1.02rem !important;
        line-height: 1.3;
        font-weight: 900;
        letter-spacing: 0.04em;
    }

    .ai-analysis-hero h2 {
        margin: 0 !important;
        color: #102A43 !important;
        font-size: 1.82rem !important;
        font-weight: 900 !important;
        letter-spacing: -0.035em !important;
    }

    .ai-analysis-hero p {
        margin: 6px 0 0 !important;
        color: #5B7185 !important;
        font-size: 1.05rem !important;
        line-height: 1.55 !important;
        font-weight: 600 !important;
    }

    .ai-analysis-status {
        min-width: 168px;
        padding: 12px 16px 13px;
        border: 1px solid #B9DCD7;
        border-radius: 11px;
        color: #477064;
        background: rgba(255, 255, 255, 0.72);
        font-size: 0.94rem;
        line-height: 1.35;
        text-align: center;
    }

    .ai-analysis-status .ai-status-dot {
        display: inline-block;
        width: 7px;
        height: 7px;
        margin-right: 5px;
        border-radius: 50%;
        background: #10B981;
        box-shadow: 0 0 0 3px rgba(16, 185, 129, 0.13);
    }

    .ai-status-ready {
        color: #477064 !important;
        font-size: 0.88rem !important;
        font-weight: 700 !important;
        white-space: nowrap !important;
    }

    .ai-status-label {
        margin-top: 5px !important;
        color: #647B88 !important;
        font-size: 0.82rem !important;
        font-weight: 700 !important;
    }

    .ai-status-value {
        margin-top: 1px !important;
        color: #0F5F59;
        font-size: 1.22rem;
        line-height: 1.15 !important;
        font-weight: 900;
        white-space: nowrap !important;
    }

    .ai-status-number {
        font-size: 1.12em !important;
        font-weight: 900 !important;
    }

    .ai-status-unit {
        margin-left: 1px !important;
        font-size: 0.66em !important;
        font-weight: 800 !important;
        vertical-align: 0.08em !important;
    }

    .ai-feature-card {
        height: auto;
        min-height: 0;
        margin-bottom: -2px;
        min-height: 168px;
        padding: 18px 17px 19px;
        border: 1px solid #D5E0EA;
        border-top: 4px solid var(--ai-card-color);
        border-radius: 14px 14px 8px 8px;
        background: #FFFFFF;
        box-shadow: 0 5px 15px rgba(15, 42, 67, 0.07);
        box-sizing: border-box;
    }

    .ai-feature-heading {
        display: flex;
        align-items: center;
        gap: 8px;
        margin-bottom: 10px;
        white-space: nowrap;
    }

    .ai-feature-icon {
        width: 36px;
        height: 36px;
        flex: 0 0 36px;
        display: grid;
        place-items: center;
        border-radius: 10px;
        background: color-mix(in srgb, var(--ai-card-color) 12%, white);
        font-size: 1.05rem;
    }

    .ai-feature-title {
        margin-bottom: 0;
        color: var(--ai-card-color);
        font-size: 1.24rem;
        line-height: 1.25;
        font-weight: 900;
    }

    .ai-feature-description {
        color: #667A8C;
        font-size: 1.02rem;
        line-height: 1.62;
        word-break: keep-all;
    }

    .ai-feature-insight       { --ai-card-color: #2563EB; }
    .ai-feature-hotspot       { --ai-card-color: #7C3AED; }
    .ai-feature-strategy      { --ai-card-color: #0F8B8D; }
    .ai-feature-police_report { --ai-card-color: #334E68; }

    [data-testid="stMain"] div[data-testid="column"]:has(.ai-feature-card)
    .stButton > button {
        min-height: 48px !important;
        border-radius: 0 0 10px 10px !important;
        border-color: #C4D3DF !important;
        color: #173F63 !important;
        background: #F5F8FB !important;
        font-weight: 850 !important;
        font-size: 1.02rem !important;
        box-shadow: 0 4px 12px rgba(15, 42, 67, 0.06) !important;
    }

    [data-testid="stMain"] div[data-testid="column"]:has(.ai-feature-card)
    .stButton > button:hover {
        color: #FFFFFF !important;
        border-color: #1D4ED8 !important;
        background: linear-gradient(135deg, #2563EB, #1D4ED8) !important;
    }

    /* 확대된 제목 박스와 메뉴의 세로 중심을 맞춤 */
    [data-testid="stMain"] div[data-testid="stRadio"] {
        min-height: 116px !important;

        display: flex !important;
        align-items: center !important;
        justify-content: flex-end !important;
    }
    
    
    /* 메인 화면 선택 메뉴의 선택색을 AI 메뉴와 동일한 청록색으로 강제 통일 */
    [data-testid="stMain"] div[data-testid="stRadio"] label:has(input:checked) {
        color: #FFFFFF !important;
        border-color: #0EA5A4 !important;
        background: linear-gradient(135deg, #0F8B8D 0%, #0EA5A4 100%) !important;
        box-shadow: 0 6px 16px rgba(14, 165, 164, 0.28) !important;
    }

    [data-testid="stMain"] div[data-testid="stRadio"] label:has(input:checked) p {
        color: #FFFFFF !important;
    }

    /* Folium 컴포넌트 아래 기본 흰색 여백 제거 */
    [data-testid="stMain"] div[data-testid="stCustomComponentV1"] {
        margin-bottom: 0 !important;
        padding-bottom: 0 !important;
        background: transparent !important;
    }

    [data-testid="stMain"] div[data-testid="stCustomComponentV1"] iframe {
        display: block !important;
        margin-bottom: 0 !important;
        background: transparent !important;
    }

    /* =====================================================
       오류 방지용 브랜드 헤더 V2
       - 과거 taap-main-header 계열과 완전히 분리된 전용 클래스
       ===================================================== */
    .taap-brand-panel-v2 {
        width: 100% !important;
        height: 112px !important;
        min-height: 112px !important;
        display: flex !important;
        align-items: center !important;
        margin: 0 !important;
        padding: 17px 20px !important;
        border: 1px solid #245474 !important;
        border-radius: 16px !important;
        background: linear-gradient(145deg, #082B46 0%, #0D3E5C 58%, #12516D 100%) !important;
        box-shadow: 0 9px 23px rgba(16, 42, 67, 0.21) !important;
        box-sizing: border-box !important;
        overflow: hidden !important;
        font-family: "Pretendard", "Noto Sans KR", "Malgun Gothic", Arial, sans-serif !important;
    }

    .taap-brand-primary-v2 {
        min-width: 275px !important;
        display: flex !important;
        align-items: center !important;
        gap: 14px !important;
        padding-right: 22px !important;
        border-right: 1px solid rgba(191, 219, 254, 0.34) !important;
    }

    .taap-brand-icon-v2 {
        width: 60px !important;
        height: 60px !important;
        flex: 0 0 60px !important;
        display: grid !important;
        place-items: center !important;
        border: 1px solid rgba(255, 255, 255, 0.13) !important;
        border-radius: 15px !important;
        color: #FFFFFF !important;
        background: linear-gradient(145deg, #2563EB, #0EA5A4) !important;
        box-shadow: 0 7px 18px rgba(14, 165, 164, 0.28) !important;
        font-size: 1.75rem !important;
    }

    .taap-brand-name-v2 {
        display: flex !important;
        align-items: baseline !important;
        font-family: Arial, "Pretendard", sans-serif !important;
        font-size: 3rem !important;
        line-height: 1 !important;
        font-weight: 900 !important;
        letter-spacing: -0.055em !important;
        white-space: nowrap !important;
    }

    .taap-brand-name-main-v2 {
        color: #38BDF8 !important;
        text-shadow: 0 0 15px rgba(56, 189, 248, 0.20) !important;
    }
    .taap-brand-name-ai-v2 {
        color: #FFFFFF !important;
        text-shadow: none !important;
    }

    .taap-brand-subtitles-v2 {
        min-width: 0 !important;
        display: flex !important;
        flex-direction: column !important;
        justify-content: center !important;
        padding-left: 22px !important;
        overflow: hidden !important;
    }

    .taap-brand-subtitle-en-v2 {
        margin: 0 !important;
        color: #FFFFFF !important;
        font-size: 1.3rem !important;
        line-height: 1.25 !important;
        font-weight: 800 !important;
        letter-spacing: 0.05em !important;
        white-space: nowrap !important;
    }

    .taap-brand-initial-v2 {
        color: #38BDF8 !important;
        font-weight: 950 !important;
        text-shadow: 0 0 12px rgba(56, 189, 248, 0.20) !important;
    }

    .taap-brand-subtitle-ko-v2 {
        margin-top: 7px !important;
        color: #FDE68A !important;
        font-size: 1.3rem !important;
        line-height: 1.25 !important;
        font-weight: 700 !important;
        letter-spacing: 0.07em !important;
        white-space: nowrap !important;
    }

    /* =====================================================
       상단 4개 화면 전환 버튼 V3
       - 제목과 독립된 영역에만 적용
       - 선택 버튼은 공통 청록색
       ===================================================== */
    .taap-nav-marker-v3 {
        display: none !important;
    }

    [data-testid="stMain"]
    div[data-testid="stElementContainer"]:has(.taap-nav-marker-v3) {
        display: none !important;
    }

    [data-testid="stMain"]
    :is(div[data-testid="stColumn"], div[data-testid="column"]):has(.taap-nav-marker-v3)
    > div[data-testid="stVerticalBlock"] {
        padding-top: 18px !important;
    }

    [data-testid="stMain"]
    :is(div[data-testid="stColumn"], div[data-testid="column"]):has(.taap-nav-marker-v3)
    div[data-testid="stHorizontalBlock"] {
        gap: 6px !important;
        margin-left: auto !important;
        padding: 7px !important;
        border: 1px solid #C5D4DF !important;
        border-radius: 14px !important;
        background: #DCE6EE !important;
        box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.78), 0 4px 12px rgba(15, 42, 67, 0.07) !important;
    }

    [data-testid="stMain"]
    :is(div[data-testid="stColumn"], div[data-testid="column"]):has(.taap-nav-marker-v3)
    .stButton > button {
        min-height: 48px !important;
        height: 48px !important;
        padding: 7px 12px !important;
        border: 1px solid #CBD9E3 !important;
        border-radius: 10px !important;
        color: #26445C !important;
        background: #EEF4F7 !important;
        box-shadow: 0 1px 3px rgba(15, 42, 67, 0.06) !important;
        font-size: 0.94rem !important;
        font-weight: 800 !important;
        white-space: nowrap !important;
    }

    [data-testid="stMain"]
    :is(div[data-testid="stColumn"], div[data-testid="column"]):has(.taap-nav-marker-v3)
    .stButton > button:hover {
        color: #176F72 !important;
        border-color: #9CCFCD !important;
        background: #D9EFEE !important;
        transform: translateY(-1px) !important;
    }

    [data-testid="stMain"]
    :is(div[data-testid="stColumn"], div[data-testid="column"]):has(.taap-nav-marker-v3)
    button[kind="primary"],
    [data-testid="stMain"]
    :is(div[data-testid="stColumn"], div[data-testid="column"]):has(.taap-nav-marker-v3)
    button[data-testid="stBaseButton-primary"] {
        color: #FFFFFF !important;
        border-color: #138F91 !important;
        background: linear-gradient(135deg, #117F82 0%, #13A0A0 100%) !important;
        box-shadow: 0 6px 14px rgba(14, 165, 164, 0.28) !important;
    }

    [data-testid="stMain"]
    :is(div[data-testid="stColumn"], div[data-testid="column"]):has(.taap-nav-marker-v3)
    button[kind="primary"] p,
    [data-testid="stMain"]
    :is(div[data-testid="stColumn"], div[data-testid="column"]):has(.taap-nav-marker-v3)
    button[data-testid="stBaseButton-primary"] p {
        color: #FFFFFF !important;
    }

    /* =====================================================
       상단 메뉴 최종 규칙
       - 내부 열 마커가 아니라 제목 마커가 있는 최상위 헤더에서 직접 탐색
       - 현재 Streamlit 기본 primary 빨간색보다 우선 적용
       ===================================================== */
    [data-testid="stMain"]
    div[data-testid="stHorizontalBlock"]:has(.taap-header-marker)
    div[data-testid="stHorizontalBlock"]:has(.stButton) {
        gap: 6px !important;
        margin-top: 18px !important;
        padding: 7px !important;
        border: 1px solid #C5D4DF !important;
        border-radius: 14px !important;
        background: #DCE6EE !important;
        box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.82), 0 4px 12px rgba(15, 42, 67, 0.07) !important;
    }

    [data-testid="stMain"]
    div[data-testid="stHorizontalBlock"]:has(.taap-header-marker)
    .stButton > button {
        width: 100% !important;
        min-height: 48px !important;
        height: 48px !important;
        padding: 7px 12px !important;
        border: 1px solid #CBD9E3 !important;
        border-radius: 10px !important;
        color: #26445C !important;
        background: #EEF4F7 !important;
        box-shadow: 0 1px 3px rgba(15, 42, 67, 0.06) !important;
        font-size: 0.94rem !important;
        font-weight: 800 !important;
        white-space: nowrap !important;
    }

    [data-testid="stMain"]
    div[data-testid="stHorizontalBlock"]:has(.taap-header-marker)
    .stButton > button:hover {
        color: #176F72 !important;
        border-color: #9CCFCD !important;
        background: #D9EFEE !important;
        transform: translateY(-1px) !important;
    }

    [data-testid="stAppViewContainer"] [data-testid="stMain"]
    div[data-testid="stHorizontalBlock"]:has(.taap-header-marker)
    button[data-testid="stBaseButton-primary"],
    [data-testid="stAppViewContainer"] [data-testid="stMain"]
    div[data-testid="stHorizontalBlock"]:has(.taap-header-marker)
    button[kind="primary"] {
        color: #FFFFFF !important;
        border-color: #138F91 !important;
        background: linear-gradient(135deg, #117F82 0%, #13A0A0 100%) !important;
        box-shadow: 0 6px 14px rgba(14, 165, 164, 0.30) !important;
    }

    [data-testid="stAppViewContainer"] [data-testid="stMain"]
    div[data-testid="stHorizontalBlock"]:has(.taap-header-marker)
    button[data-testid="stBaseButton-primary"] p,
    [data-testid="stAppViewContainer"] [data-testid="stMain"]
    div[data-testid="stHorizontalBlock"]:has(.taap-header-marker)
    button[kind="primary"] p {
        color: #FFFFFF !important;
    }

    /* 발생연월 슬라이더는 Streamlit 기본 빨간 계열을 그대로 사용한다. */

    /* 사이드바의 3개 조건 Expander 제목을 일반 필터 라벨과 같은 크기로 통일 */
    [data-testid="stSidebar"] [data-testid="stExpander"] summary p,
    [data-testid="stSidebar"] [data-testid="stExpander"] summary span {
        font-size: 1rem !important;
        line-height: 1.35 !important;
        font-weight: 700 !important;
        letter-spacing: -0.015em !important;
    }

    @media (max-width: 1100px) {
        .taap-condition-group { display: none; }
        .dashboard-summary-panel { min-height: auto; }
        .taap-brand-subtitle-en-v2 { font-size: 0.88rem !important; }
        .taap-brand-subtitle-ko-v2 { font-size: 0.78rem !important; }
    }

    @media (max-width: 780px) {
        .taap-subtitle-group { display: none; }
        .taap-title-group { border-right: 0; }
        .taap-brand-subtitles-v2 { display: none !important; }
        .taap-brand-primary-v2 { border-right: 0 !important; }
    }

    /* =====================================================
       상단 화면 선택 메뉴 V4 — 최종 우선순위
       - 구버전: section.main
       - Community Cloud: stAppViewContainer
       - 기존 stMain / radio 규칙과 무관하게 버튼 열을 직접 지정
       ===================================================== */
    section.main div[data-testid="column"]:has(.taap-nav-marker-v3),
    section.main div[data-testid="stColumn"]:has(.taap-nav-marker-v3),
    [data-testid="stAppViewContainer"] div[data-testid="column"]:has(.taap-nav-marker-v3),
    [data-testid="stAppViewContainer"] div[data-testid="stColumn"]:has(.taap-nav-marker-v3) {
        min-height: 112px !important;
        display: flex !important;
        align-items: center !important;
        padding: 17px 0 !important;
        border: 0 !important;
        border-radius: 0 !important;
        background: transparent !important;
        box-shadow: none !important;
        box-sizing: border-box !important;
    }

    section.main div[data-testid="column"]:has(.taap-nav-marker-v3) > div,
    section.main div[data-testid="stColumn"]:has(.taap-nav-marker-v3) > div,
    [data-testid="stAppViewContainer"] div[data-testid="column"]:has(.taap-nav-marker-v3) > div,
    [data-testid="stAppViewContainer"] div[data-testid="stColumn"]:has(.taap-nav-marker-v3) > div {
        width: 100% !important;
    }

    section.main div[data-testid="column"]:has(.taap-nav-marker-v3) div[data-testid="stHorizontalBlock"],
    section.main div[data-testid="stColumn"]:has(.taap-nav-marker-v3) div[data-testid="stHorizontalBlock"],
    [data-testid="stAppViewContainer"] div[data-testid="column"]:has(.taap-nav-marker-v3) div[data-testid="stHorizontalBlock"],
    [data-testid="stAppViewContainer"] div[data-testid="stColumn"]:has(.taap-nav-marker-v3) div[data-testid="stHorizontalBlock"] {
        width: 100% !important;
        gap: 8px !important;
        margin: 0 !important;
        padding: 0 !important;
        border: 0 !important;
        background: transparent !important;
        box-shadow: none !important;
    }

    section.main div[data-testid="column"]:has(.taap-nav-marker-v3) .stButton > button,
    section.main div[data-testid="stColumn"]:has(.taap-nav-marker-v3) .stButton > button,
    [data-testid="stAppViewContainer"] div[data-testid="column"]:has(.taap-nav-marker-v3) .stButton > button,
    [data-testid="stAppViewContainer"] div[data-testid="stColumn"]:has(.taap-nav-marker-v3) .stButton > button {
        width: 100% !important;
        min-height: 50px !important;
        height: 50px !important;
        padding: 8px 9px !important;
        border: 1px solid #94A3B8 !important;
        border-radius: 10px !important;
        color: #FFFFFF !important;
        background: linear-gradient(145deg, #64748B 0%, #475569 100%) !important;
        box-shadow: 0 4px 10px rgba(16, 42, 67, 0.14) !important;
        font-size: 1.3rem !important;
        font-weight: 850 !important;
        white-space: nowrap !important;
    }

    section.main div[data-testid="column"]:has(.taap-nav-marker-v3) .stButton > button p,
    section.main div[data-testid="stColumn"]:has(.taap-nav-marker-v3) .stButton > button p,
    [data-testid="stAppViewContainer"] div[data-testid="column"]:has(.taap-nav-marker-v3) .stButton > button p,
    [data-testid="stAppViewContainer"] div[data-testid="stColumn"]:has(.taap-nav-marker-v3) .stButton > button p {
        color: inherit !important;
        font-size: inherit !important;
        font-weight: inherit !important;
    }

    section.main div[data-testid="column"]:has(.taap-nav-marker-v3) .stButton > button:hover,
    section.main div[data-testid="stColumn"]:has(.taap-nav-marker-v3) .stButton > button:hover,
    [data-testid="stAppViewContainer"] div[data-testid="column"]:has(.taap-nav-marker-v3) .stButton > button:hover,
    [data-testid="stAppViewContainer"] div[data-testid="stColumn"]:has(.taap-nav-marker-v3) .stButton > button:hover {
        color: #FFFFFF !important;
        border-color: #CBD5E1 !important;
        background: linear-gradient(145deg, #718096 0%, #526477 100%) !important;
        transform: translateY(-1px) !important;
        box-shadow: 0 6px 13px rgba(16, 42, 67, 0.18) !important;
    }

    section.main div[data-testid="column"]:has(.taap-nav-marker-v3) button[kind="primary"],
    section.main div[data-testid="stColumn"]:has(.taap-nav-marker-v3) button[kind="primary"],
    [data-testid="stAppViewContainer"] div[data-testid="column"]:has(.taap-nav-marker-v3) button[kind="primary"],
    [data-testid="stAppViewContainer"] div[data-testid="stColumn"]:has(.taap-nav-marker-v3) button[kind="primary"],
    section.main div[data-testid="column"]:has(.taap-nav-marker-v3) button[data-testid="stBaseButton-primary"],
    section.main div[data-testid="stColumn"]:has(.taap-nav-marker-v3) button[data-testid="stBaseButton-primary"],
    [data-testid="stAppViewContainer"] div[data-testid="column"]:has(.taap-nav-marker-v3) button[data-testid="stBaseButton-primary"],
    [data-testid="stAppViewContainer"] div[data-testid="stColumn"]:has(.taap-nav-marker-v3) button[data-testid="stBaseButton-primary"] {
        color: #FFFFFF !important;
        border-color: #5EEAD4 !important;
        background: linear-gradient(135deg, #0F8B8D 0%, #12AAA7 100%) !important;
        box-shadow: 0 6px 16px rgba(14, 165, 164, 0.35) !important;
    }

    /* GIS 지도 위 레이어 안내문 */
    .gis-map-guide {
        margin-top: -1.8rem !important;
        margin-bottom: 0rem !important;
        text-align: right;
        color: #6B7280;
        font-size: 0.82rem;
        line-height: 1.3;
    }  

    /* 안내문을 감싸는 Streamlit 바깥 컨테이너의 아래 여백 축소 */
    [data-testid="stElementContainer"]:has(.gis-map-guide),
    .element-container:has(.gis-map-guide) {
        margin-bottom: -1.4rem !important;
    }  
    
    </style>
    """,
    unsafe_allow_html=True,
)



# ============================================================
# 통합 대시보드
# - 기존 GIS 공간분석 지도(m2)는 수정하거나 재사용하지 않음
# - 대시보드용 히트맵 지도는 별도의 Folium 객체로 생성
# ============================================================
if selected_page == "대시보드":

    def apply_overview_chart_style(
        fig,
        height=236,
        show_legend=False,
    ):
        """통합 대시보드 전용 Plotly 공통 디자인"""
        fig.update_layout(
            height=height,
            margin=dict(l=24, r=18, t=10, b=28),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(
                family='"Malgun Gothic", "Apple SD Gothic Neo", sans-serif',
                size=16,
                color="#334155",
            ),
            showlegend=show_legend,
            hoverlabel=dict(
                bgcolor="#FFFFFF",
                bordercolor="#CBD5E1",
                font=dict(size=14, color="#1E293B"),
            ),
        )
        fig.update_xaxes(
            title_text="",
            showline=True,
            linewidth=1,
            linecolor="#CBD5E1",
            showgrid=False,
            tickfont=dict(size=15),
            automargin=True,
            zeroline=False,
        )
        fig.update_yaxes(
            title_text="",
            gridcolor="#E5E7EB",
            tickfont=dict(size=15),
            automargin=True,
            zeroline=False,
        )
        return fig

    if filtered_df.empty:
        st.warning("현재 필터 조건에 해당하는 사고 데이터가 없습니다.")

    else:
        # ----------------------------------------------------
        # 대시보드 전용 히트맵 지도
        # 배경지도와 히트맵 외의 마커·경계·레이어는 추가하지 않음
        # ----------------------------------------------------
        dashboard_heatmap = folium.Map(
            location=map_center,
            zoom_start=zoom_level,
            tiles=None,
        )

        folium.TileLayer(
            tiles="OpenStreetMap",
            control=False,
        ).add_to(dashboard_heatmap)

        if heat_data:
            HeatMap(
                data=heat_data,
                radius=15,
                blur=18,
                min_opacity=0.25,
            ).add_to(dashboard_heatmap)

        # ----------------------------------------------------
        # 현재 조건 자동 요약
        # - 별도의 API 호출 없이 필터 결과에서 즉시 계산
        # - 지도 오른쪽에서 현재 분석집단의 특징을 한눈에 제시
        # ----------------------------------------------------
        dashboard_year_counts = (
            filtered_df["acdnt_year"]
            .value_counts()
            .sort_index()
            .rename_axis("연도")
            .reset_index(name="사고건수")
        )

        # 선택 기간 전체에서 같은 월을 합산하여 최다 발생월 산출
        dashboard_month_values = pd.Series(dtype="float64")
        if "acdnt_month" in filtered_df.columns:
            # '01월', '1월', 숫자형 값에서 월 숫자만 안전하게 추출
            dashboard_month_values = pd.to_numeric(
                filtered_df["acdnt_month"]
                .astype("string")
                .str.extract(r"(\d{1,2})", expand=False),
                errors="coerce",
            )

        # 월 컬럼이 없거나 해석 가능한 값이 하나도 없으면 발생일자로 보완
        if (
            dashboard_month_values.dropna().empty
            and "accident_date" in filtered_df.columns
        ):
            dashboard_month_values = pd.to_datetime(
                filtered_df["accident_date"],
                errors="coerce",
            ).dt.month

        dashboard_month_values = dashboard_month_values.dropna()

        dashboard_month_values = dashboard_month_values.astype(int)
        dashboard_month_values = dashboard_month_values[
            dashboard_month_values.between(1, 12)
        ]

        if dashboard_month_values.empty:
            peak_month_name = "해당 없음"
            peak_month_count = 0
        else:
            dashboard_month_counts = (
                dashboard_month_values
                .value_counts()
                .sort_index()
            )
            peak_month_number = int(dashboard_month_counts.idxmax())
            peak_month_name = f"{peak_month_number}월"
            peak_month_count = int(
                dashboard_month_counts.loc[peak_month_number]
            )

        dashboard_hdc_order = ["차대차", "차대사람", "차량단독"]
        dashboard_hdc_counts = (
            filtered_df["acdnt_hdc"]
            .value_counts()
            .reindex(dashboard_hdc_order, fill_value=0)
            .rename_axis("사고종별")
            .reset_index(name="사고건수")
        )

        # 필터 결과 중 최다 사망자 유형
        # 사이드바의 '사망자 유형 (복수 선택)'과 동일한 fatal_type 컬럼 사용
        dashboard_fatal_type_counts = pd.Series(dtype="int64")
        if "fatal_type" in filtered_df.columns:
            fatal_type_values = (
                filtered_df["fatal_type"]
                .dropna()
                .astype(str)
                .str.strip()
            )
            fatal_type_values = fatal_type_values[
                ~fatal_type_values.isin(["", "nan", "None"])
            ]
            dashboard_fatal_type_counts = fatal_type_values.value_counts()

        if dashboard_fatal_type_counts.empty:
            peak_fatal_type_name = "해당 없음"
            peak_fatal_type_count = 0
        else:
            peak_fatal_type_name = str(
                dashboard_fatal_type_counts.index[0]
            )
            peak_fatal_type_count = int(
                dashboard_fatal_type_counts.iloc[0]
            )

        dashboard_hour_counts = (
            pd.to_numeric(filtered_df["time_num"], errors="coerce")
            .dropna()
            .astype(int)
            .value_counts()
            .reindex(range(24), fill_value=0)
            .rename_axis("시간")
            .reset_index(name="사고건수")
        )

        dashboard_weekday_order = [
            "월요일", "화요일", "수요일", "목요일",
            "금요일", "토요일", "일요일",
        ]
        dashboard_weekday_counts = (
            filtered_df["dfk_dc"]
            .value_counts()
            .reindex(dashboard_weekday_order, fill_value=0)
            .rename_axis("요일")
            .reset_index(name="사고건수")
        )
        dashboard_weekday_counts["표시요일"] = (
            dashboard_weekday_counts["요일"].map(weekday_short_name)
        )

        peak_year_row = dashboard_year_counts.loc[
            dashboard_year_counts["사고건수"].idxmax()
        ]
        peak_hour_row = dashboard_hour_counts.loc[
            dashboard_hour_counts["사고건수"].idxmax()
        ]
        peak_weekday_row = dashboard_weekday_counts.loc[
            dashboard_weekday_counts["사고건수"].idxmax()
        ]
        severe_total = fatal_accident_count + serious_accident_count
        severe_ratio = severe_total / len(filtered_df) * 100
        mapped_count = len(heat_data)
        mapped_ratio = mapped_count / len(filtered_df) * 100

        dashboard_overview_expander = st.expander(
            " ",
            expanded=True,
        )

        # 지도와 현재 통계 요약 전체 영역의 좌우 여백
        overview_left_space, overview_content, overview_right_space = (
            dashboard_overview_expander.columns(
                [0.012, 0.976, 0.012],
                gap="small",
            )
        )

        # 지도와 통계 요약 사이의 기존 간격은 그대로 유지
        with overview_content:
            dashboard_map_col, dashboard_summary_col = st.columns(
                [1.72, 0.78],
                gap="medium",
            )
        # ----------------------------------------------------
        # 왼쪽: 히트맵 지도
        # ----------------------------------------------------
        with dashboard_map_col:
            st.markdown(
                """
                <div class="dashboard-section-title">
                    <span>🗺️ 교통사고 분포 지도</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

            with dashboard_map_col:
                st_folium(
                    dashboard_heatmap,
                    use_container_width=True,
                    height=580,
                    returned_objects=[],
                    key="dashboard_heatmap_only",
                )

        # ----------------------------------------------------
        # 오른쪽: 현재 조건 자동 요약
        # ----------------------------------------------------
        with dashboard_summary_col:
            st.markdown(
                """
                <div class="dashboard-section-title">
                    <span>🔎 현재 통계 요약</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.markdown(
                f"""
                <div class="dashboard-summary-panel">
                    <div class="dashboard-summary-kicker">검색된 교통사고 분석</div>
                <div class="dashboard-summary-lead">
                    총 <b>{len(filtered_df):,}건</b> 중<br>
                    사망·중상사고는
                    <b>{severe_total:,}건({severe_ratio:.1f}%)</b>입니다.
                </div>
                    <div class="dashboard-summary-item">
                        <span>최다 사망유형</span>
                        <b>{peak_fatal_type_name} · {peak_fatal_type_count:,}건</b>
                    </div>
                    <div class="dashboard-summary-item">
                        <span>집중 시간대</span>
                        <b>{int(peak_hour_row['시간']):02d}시 · {int(peak_hour_row['사고건수']):,}건</b>
                    </div>
                    <div class="dashboard-summary-item">
                        <span>최다 발생요일</span>
                        <b>{peak_weekday_row['요일']} · {int(peak_weekday_row['사고건수']):,}건</b>
                    </div>
                    <div class="dashboard-summary-item">
                        <span>최다 발생연도</span>
                        <b>{int(peak_year_row['연도'])}년 · {int(peak_year_row['사고건수']):,}건</b>
                    </div>
                    <div class="dashboard-summary-item">
                        <span>최다 발생월</span>
                        <b>{peak_month_name} · {peak_month_count:,}건</b>
                    </div>
                    <div class="dashboard-summary-note">
                        심층 분석은 ‘AI 리포트’에서 생성할 수 있습니다.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # ----------------------------------------------------
        # 지도 아래: 4개 그래프를 별도 expander에 가로 4열 배치
        # ----------------------------------------------------
        dashboard_charts_expander = st.expander(" ", expanded=True)


        chart_col1, chart_col2, chart_col3, chart_col4 = (
            dashboard_charts_expander.columns(4, gap="small")
        )

        with chart_col1:
            st.markdown(
                "<div class='dashboard-chart-title'>발생연도별 추이</div>",
                unsafe_allow_html=True,
            )
        fig_dashboard_year = px.line(
            dashboard_year_counts,
            x="연도",
            y="사고건수",
            markers=True,
        )
        fig_dashboard_year.update_traces(
            line=dict(color="#2563EB", width=3),
            marker=dict(
                size=7,
                color="#FFFFFF",
                line=dict(color="#2563EB", width=2),
            ),
            hovertemplate=(
                "<b>%{x}년</b><br>사고 건수: %{y:,}건"
                "<extra></extra>"
            ),
        )
        fig_dashboard_year.update_xaxes(dtick=1)
        fig_dashboard_year = apply_overview_chart_style(
            fig_dashboard_year,
        )
        chart_col1.plotly_chart(
            fig_dashboard_year,
            use_container_width=True,
            config={"displayModeBar": False},
            key="dashboard_year_chart",
        )

        with chart_col2:
            st.markdown(
                "<div class='dashboard-chart-title'>사고종별</div>",
                unsafe_allow_html=True,
            )
        fig_dashboard_hdc = px.pie(
            dashboard_hdc_counts,
            names="사고종별",
            values="사고건수",
            hole=0.62,
            color="사고종별",
            color_discrete_map={
                "차대차": "#2563EB",
                "차대사람": "#0EA5A4",
                "차량단독": "#64748B",
            },
        )
        
        fig_dashboard_hdc.update_traces(
            textinfo="percent",
            textfont_size=16,
            marker=dict(
                line=dict(color="#FFFFFF", width=2)
            ),
            hovertemplate=(
                "<b>%{label}</b><br>%{value:,}건 · %{percent}"
                "<extra></extra>"
            ),
        )


        fig_dashboard_hdc.update_layout(
            annotations=[
                dict(
                    text=(
                        f"<b>{len(filtered_df):,}</b>"
                        "<br><span style='font-size:10px'>전체 사고</span>"
                    ),
                    x=0.5,
                    y=0.5,
                    showarrow=False,
                    font=dict(size=16, color="#0F172A"),
                )
            ],
            legend=dict(
                orientation="h",
                y=-0.12,
                x=0.5,
                xanchor="center",
                font=dict(size=16),
                title=None,
            ),
        )
        fig_dashboard_hdc = apply_overview_chart_style(
            fig_dashboard_hdc,
            show_legend=True,
        )
        chart_col2.plotly_chart(
            fig_dashboard_hdc,
            use_container_width=True,
            config={"displayModeBar": False},
            key="dashboard_hdc_chart",
        )

        with chart_col3:
            st.markdown(
                "<div class='dashboard-chart-title'>시간대별 사고</div>",
                unsafe_allow_html=True,
            )
        fig_dashboard_hour = px.bar(
            dashboard_hour_counts,
            x="시간",
            y="사고건수",
            color_discrete_sequence=["#0EA5A4"],
        )
        fig_dashboard_hour.update_traces(
            hovertemplate=(
                "<b>%{x}시</b><br>사고 건수: %{y:,}건"
                "<extra></extra>"
            )
        )

        fig_dashboard_hour.update_xaxes(
            tickmode="array",
            tickvals=[3, 9, 15, 21],
            ticktext=["03시", "09시", "15시", "21시"],
            tickangle=0,
            tickfont=dict(size=14),
        )

        fig_dashboard_hour = apply_overview_chart_style(
            fig_dashboard_hour,
        )
        chart_col3.plotly_chart(
            fig_dashboard_hour,
            use_container_width=True,
            config={"displayModeBar": False},
            key="dashboard_hour_chart",
        )

        with chart_col4:
            st.markdown(
                "<div class='dashboard-chart-title'>요일별 사고</div>",
                unsafe_allow_html=True,
            )
        fig_dashboard_weekday = px.bar(
            dashboard_weekday_counts,
            x="표시요일",
            y="사고건수",
            color_discrete_sequence=["#E3A008"],
        )
        fig_dashboard_weekday.update_traces(
            hovertemplate=(
                "<b>%{x}요일</b><br>사고 건수: %{y:,}건"
                "<extra></extra>"
            )
        )
        fig_dashboard_weekday = apply_overview_chart_style(
            fig_dashboard_weekday,
        )
        chart_col4.plotly_chart(
            fig_dashboard_weekday,
            use_container_width=True,
            config={"displayModeBar": False},
            key="dashboard_weekday_chart",
        )


if selected_page == "GIS분석":

    # GIS 설정 카드와 지도 사이 간격 축소
    st.markdown(
        """
        <div class="gis-map-guide">
            ※ 레이어 선택창에서 사고다발지점·사망사고·히트맵·관할 경계를
            개별적으로 켜고 끌 수 있습니다.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st_folium(
        m2,
        use_container_width=True,
        height=840,
        returned_objects=[],
    )


if selected_page == "AI 리포트":
    # ============================================================
    # 5-1. 목적별 생성형 AI 교통사고 분석
    # - 선택집단 + 동일기간 기준집단 비교
    # - 목적별 모델/Reasoning/Web Search 차등 적용
    # - 호출별 토큰·예상비용 확인
    # ============================================================

    st.markdown(
        f"""
        <section class="ai-analysis-hero">
            <div class="ai-analysis-hero-icon">AI</div>
            <div class="ai-analysis-hero-copy">
                <div class="ai-analysis-eyebrow">TAAP-AI DECISION SUPPORT</div>
                <h2>AI 교통사고 분석</h2>
                <p>현재 적용된 조건과 사고 데이터를 바탕으로 목적별 분석을 수행합니다.</p>
            </div>
            <div class="ai-analysis-status">
                <div class="ai-status-ready"><span class="ai-status-dot"></span> 분석 준비 완료</div>
                <div class="ai-status-label">분석 대상</div>
                <div class="ai-status-value"><span class="ai-status-number">{len(filtered_df):,}</span><span class="ai-status-unit">건</span></div>
            </div>
        </section>
        """,
        unsafe_allow_html=True,
    )

    # 현재 선택된 필터를 분석 JSON에 함께 기록
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
            [weekday_short_name[weekday] for weekday in selected_weekdays]
            if selected_weekdays
            else ["전체"]
        ),
        "offending_vehicle": selected_wrngdo if selected_wrngdo else ["전체"],
        "offending_driver_age_group": (
            selected_offending_driver_age
            if selected_offending_driver_age
            else ["전체"]
        ),
        "damaged_vehicle": selected_dmge if selected_dmge else ["전체"],
        "victim_age_group": (
            selected_victim_age
            if selected_victim_age
            else ["전체"]
        ),
        "fatal_type": selected_fatal_type if selected_fatal_type else ["전체"],
        "fatal_age_group": selected_fatal_age if selected_fatal_age else ["전체"],
        "weather": selected_wether if selected_wether else ["전체"],
        "violation": selected_violt if selected_violt else ["전체"],
    }

    current_filter_signature = make_filter_signature(
        selected_filter_info,
        filtered_df,
    )

    # ------------------------------------------------------------
    # 동일기간 비교집단 생성
    # 세부필터는 적용하지 않고 기간만 동일하게 맞춘다.
    # - 선택 관할 전체 사고
    # - 대전 전체 사고
    # ------------------------------------------------------------
    comparison_period_df = df.copy()

    if (
        start_year_month is not None
        and end_year_month is not None
        and "accident_date" in comparison_period_df.columns
    ):
        comparison_start_date = start_year_month.start_time
        comparison_end_date = end_year_month.end_time

        comparison_period_df = comparison_period_df[
            comparison_period_df["accident_date"].between(
                comparison_start_date,
                comparison_end_date,
                inclusive="both",
            )
        ].copy()

    city_reference_df = comparison_period_df

    if selected_ps != "전체":
        station_reference_df = comparison_period_df[
            comparison_period_df["관할"] == selected_ps
        ].copy()
    else:
        # '전체'를 선택한 경우 두 기준집단이 동일하므로 중복 전송하지 않는다.
        station_reference_df = None

    # 현재 조건의 분석용 JSON 패키지
    analysis_package_preview = None

    if not filtered_df.empty:
        analysis_package_preview = make_ai_analysis_package(
            target_df=filtered_df,
            top_hotspot_dataframe=top_hotspot_df,
            selected_filter_info=selected_filter_info,
            hotspot_radius_m=hotspot_radius,
            station_reference_df=station_reference_df,
            city_reference_df=city_reference_df,
        )

    # ------------------------------------------------------------
    # AI 분석 유형 정의
    # ------------------------------------------------------------
    ai_report_types = {
        "insight": {
            "card_title": "핵심 인사이트",
            "button": "🔍 인사이트 보고서 작성",
            "title": "🔍 AI 핵심 인사이트",
            "description": (
                "필터링된 통계를 동기간의 대전청 전체 교통사고와 비교해 "
                "의사결정 시 고려할 패턴을 찾습니다."
            ),
            "spinner": (
                "선택집단과 기준집단을 비교하여 핵심 패턴을 분석하고 있습니다."
            ),
        },
        "hotspot": {
            "card_title": "다발지점 진단",
            "button": "📍 진단보고서 작성",
            "title": "📍 사고다발지점 AI 진단",
            "description": (
                "도출된 사고다발지점별 교통사고를 프로파일링해 "
                "관리 우선순위를 진단합니다."
            ),
            "spinner": "사고다발지점별 특성과 차이를 분석하고 있습니다.",
        },
        "strategy": {
            "card_title": "맞춤형 대응전략",
            "button": "🎯 대응전략 보고서 작성",
            "title": "🎯 맞춤형 교통안전 대응전략",
            "description": (
                "데이터상 위험요소를 세부적으로 검토하고 "
                "실제 경찰활동과 연계할 전략을 발굴합니다."
            ),
            "spinner": (
                "교통사고 패턴을 대응과제로 전환하고 참고할 전문자료 등을 검토하고 있습니다."
            ),
        },
        "police_report": {
            "card_title": "교통안전 실무보고서",
            "button": "📄 실무보고서 작성",
            "title": "📄 교통사고 분석 및 대응방향 보고",
            "description": (
                "검색된 통계자료를 외부 전문자료 등과 비교분석해 "
                "실무용으로 활용할 수 있는 보고서를 작성합니다."
            ),
            "spinner": (
                "필터링된 통계를 분석해 경찰 형식의 교통안전대책을 작성하고 있습니다."
            ),
        },
    }

     # ============================================================
    # 교통안전 실무보고서 선택 및 생성
    #
    # 동작 방식
    # 1. 버튼 클릭 즉시 선택 상태 저장
    # 2. 생성할 보고서 유형도 pending 상태로 저장
    # 3. Streamlit 재실행 후 pending 값을 읽어 자동 생성
    #
    # → 버튼은 한 번만 클릭
    # → 선택한 버튼도 즉시 primary 색상으로 변경
    # → 같은 클릭으로 AI 생성까지 자동 실행
    # ============================================================

    def queue_ai_report(report_type):
        """
        교통안전 실무보고서 버튼 클릭 콜백

        - 클릭한 보고서 유형을 현재 선택 상태로 저장
        - 실제 AI 생성을 pending 상태로 저장
        - 기존 결과 선택창 상태와는 직접 연결하지 않음

        생성 완료 후 별도 rerun으로 새 보고서를 확정 표시한다.
        """

        st.session_state["selected_ai_report_type"] = report_type
        st.session_state["pending_ai_report_type"] = report_type


    # ------------------------------------------------------------
    # 4개 AI 분석 버튼
    # ------------------------------------------------------------
    button_columns = st.columns(4)

    for column, (report_type, report_info) in zip(
        button_columns,
        ai_report_types.items(),
    ):
        with column:

            st.markdown(
                f"""
                <div class="ai-feature-card ai-feature-{report_type}">
                    <div class="ai-feature-heading">
                        <span class="ai-feature-icon">{report_info['button'].split()[0]}</span>
                        <span class="ai-feature-title">{report_info['card_title']}</span>
                    </div>
                    <div class="ai-feature-description">
                        {report_info['description']}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.button(
                report_info["button"],
                key=f"generate_ai_{report_type}",
                type=(
                    "primary"
                    if st.session_state.get(
                        "selected_ai_report_type"
                    ) == report_type
                    else "secondary"
                ),
                use_container_width=True,
                disabled=filtered_df.empty,
                on_click=queue_ai_report,
                args=(report_type,),
            )

    # ------------------------------------------------------------
    # 필터 결과가 없는 경우
    # ------------------------------------------------------------
    if filtered_df.empty:
        st.warning(
            "현재 필터 조건에 해당하는 사고 데이터가 없어 "
            "AI 분석을 생성할 수 없습니다."
        )


    # ============================================================
    # 대기 중인 교통안전 실무보고서 생성
    #
    # pop()을 사용하여 생성 시작과 동시에 pending 상태 제거
    # → Streamlit 재실행 시 같은 API가 자동으로 재호출되는 것을 방지
    # ============================================================

    # ------------------------------------------------------------
    # 기존 생성 결과 선택창 버전
    # - 새 교통안전 실무보고서 생성 시마다 key를 변경해 오래된 브라우저 위젯 상태와 분리
    # ------------------------------------------------------------
    if "existing_report_selector_version" not in st.session_state:
        st.session_state["existing_report_selector_version"] = 0


    clicked_report_type = st.session_state.pop(
        "pending_ai_report_type",
        None,
    )


    if clicked_report_type is not None:

        report_info = ai_report_types[clicked_report_type]
        report_generation_succeeded = False

        try:

            with st.spinner(report_info["spinner"]):

                ai_result, _, ai_usage = generate_ai_report(
                    analysis_package=analysis_package_preview,
                    report_type=clicked_report_type,
                )

            # ----------------------------------------------------
            # 생성 결과 저장
            # ----------------------------------------------------
            st.session_state[
                f"ai_result_{clicked_report_type}"
            ] = ai_result

            st.session_state[
                f"ai_usage_{clicked_report_type}"
            ] = ai_usage

            st.session_state[
                f"ai_signature_{clicked_report_type}"
            ] = current_filter_signature

            st.session_state[
                f"ai_filters_{clicked_report_type}"
            ] = selected_filter_info

            # ----------------------------------------------------
            # 방금 생성한 보고서를 최우선 표시 대상으로 확정
            # ----------------------------------------------------
            st.session_state["selected_ai_report_type"] = (
                clicked_report_type
            )

            # ----------------------------------------------------
            # 기존 생성 결과 selectbox를 새 key로 재생성
            # - 긴 AI 호출 중 브라우저에 남아 있던 이전 선택값과 분리
            # ----------------------------------------------------
            st.session_state["existing_report_selector_version"] = (
                int(
                    st.session_state.get(
                        "existing_report_selector_version",
                        0,
                    )
                )
                + 1
            )

            report_generation_succeeded = True

        except Exception as error:

            # ----------------------------------------------------
            # 오류 표시
            # 단순 오류문구뿐 아니라 실제 Exception 종류와
            # 상세 메시지를 화면에서 바로 확인할 수 있도록 표시
            # ----------------------------------------------------
            st.error(
                f"{report_info['title']} 생성 중 오류가 발생했습니다."
            )

            st.code(
                f"{type(error).__name__}: {error}"
            )

        # --------------------------------------------------------
        # 생성 완료 후 새 Session State로 다시 렌더링
        # - try/except 밖에서 호출하여 Streamlit rerun 제어 예외와 충돌 방지
        # - 같은 실행 사이클에서 오래된 selectbox 상태가 재적용되는 것을 방지
        # --------------------------------------------------------
        if report_generation_succeeded:
            st.rerun()
    # ------------------------------------------------------------
    # 생성된 AI 결과 표시
    # ------------------------------------------------------------
    selected_ai_report_type = st.session_state.get("selected_ai_report_type")

    if (
        selected_ai_report_type in ai_report_types
        and st.session_state.get(f"ai_result_{selected_ai_report_type}")
    ):
        selected_info = ai_report_types[selected_ai_report_type]

        selected_report_is_current = (
            st.session_state.get(f"ai_signature_{selected_ai_report_type}")
            == current_filter_signature
        )

        st.divider()

        if not selected_report_is_current:
            st.warning(
                "아래 결과는 현재와 다른 필터 조건에서 생성되었습니다. "
                "현재 조건의 분석이 필요하면 해당 버튼을 다시 누르세요."
            )

        with st.expander(" ", expanded=True):

            # --------------------------------------------------------
            # 교통안전 실무보고서 본문 가독성 개선
            # - 제목/소제목 크기는 기존 Streamlit Markdown 유지
            # - 일반 본문과 목록 글자만 확대
            # --------------------------------------------------------

            # 이 Expander에만 정책 브리핑형 보고서 디자인을 적용하기 위한 마커
            st.markdown(
                '<span class="ai-report-document-marker" aria-hidden="true"></span>',
                unsafe_allow_html=True,
            )

            st.markdown(
                """
                <style>

                /* ---------------------------------------------------------
                AI 생성 결과 전용 정책 브리핑 문서 스타일
                - 다른 Expander에는 영향을 주지 않음
                --------------------------------------------------------- */
                .ai-report-document-marker {
                    display: none !important;
                }

                [data-testid="stMain"]
                div[data-testid="stExpander"]:has(.ai-report-document-marker)
                > details {
                    overflow: visible !important;
                    border: 1px solid #D1DDE6 !important;
                    border-radius: 18px !important;
                    background: linear-gradient(145deg, #E7EDF2 0%, #EEF3F6 100%) !important;
                    box-shadow: 0 8px 24px rgba(15, 42, 67, 0.07) !important;
                }

                [data-testid="stMain"]
                div[data-testid="stExpander"]:has(.ai-report-document-marker)
                > details > div {
                    width: calc(100% - 44px) !important;
                    max-width: 900px !important;
                    margin: 8px auto 28px !important;
                    padding: 18px 60px 44px !important;
                    border: 1px solid #D9E1E6 !important;
                    border-radius: 4px !important;
                    background: #FCFCFA !important;
                    box-shadow: 0 12px 30px rgba(29, 49, 64, 0.10) !important;
                    box-sizing: border-box !important;
                }

                /* 실제 Streamlit Expander 본문 구조에 직접 좌우 여백 적용 */
                [data-testid="stMain"]
                div[data-testid="stExpander"]:has(.ai-report-document-marker)
                div[data-testid="stExpanderDetails"] {
                    width: calc(100% - 56px) !important;
                    max-width: 980px !important;
                    margin: 10px auto 30px !important;
                    padding: 18px 64px 48px !important;
                    border: 1px solid #D9E1E6 !important;
                    border-radius: 5px !important;
                    background: #FCFCFA !important;
                    box-shadow: 0 12px 30px rgba(29, 49, 64, 0.10) !important;
                    box-sizing: border-box !important;
                }

                .ai-brief-header {
                    margin: 0 -60px 30px !important;
                    padding: 30px 60px 28px !important;
                    border-bottom: 1px solid #DCE5EC !important;
                    background: linear-gradient(115deg, #F7FAFB 0%, #ECF5F4 100%) !important;
                }

                .ai-brief-eyebrow {
                    margin-bottom: 9px !important;
                    color: #0F8B8D !important;
                    font-size: 0.78rem !important;
                    line-height: 1.2 !important;
                    font-weight: 900 !important;
                    letter-spacing: 0.13em !important;
                }

                .ai-brief-title {
                    color: #102A43 !important;
                    font-size: 2.15rem !important;
                    line-height: 1.28 !important;
                    font-weight: 900 !important;
                    letter-spacing: -0.04em !important;
                }

                .ai-brief-meta {
                    display: flex !important;
                    flex-wrap: wrap !important;
                    gap: 7px !important;
                    margin-top: 15px !important;
                }

                .ai-brief-meta span {
                    padding: 5px 9px !important;
                    border: 1px solid #D5E2E9 !important;
                    border-radius: 999px !important;
                    color: #587083 !important;
                    background: rgba(255, 255, 255, 0.76) !important;
                    font-size: 1.32rem !important;
                    font-weight: 700 !important;
                }

                .ai-brief-kpis {
                    display: grid !important;
                    grid-template-columns: repeat(3, minmax(0, 1fr)) !important;
                    gap: 10px !important;
                    margin: 0 0 18px !important;
                }

                .ai-brief-kpi {
                    padding: 14px 16px !important;
                    border: 1px solid #D9E4EA !important;
                    border-radius: 12px !important;
                    background: #FAFCFD !important;
                }

                .ai-brief-kpi-label {
                    color: #738596 !important;
                    font-size: 1.28rem !important;
                    font-weight: 750 !important;
                }

                .ai-brief-kpi-value {
                    margin-top: 4px !important;
                    color: #173F5F !important;
                    font-size: 2rem !important;
                    font-weight: 900 !important;
                    white-space: nowrap !important;
                    overflow: hidden !important;
                    text-overflow: ellipsis !important;
                }

                .ai-brief-kpi-number {
                    font-size: 1em !important;
                    font-weight: 900 !important;
                }

                .ai-brief-kpi-unit {
                    display: inline-block !important;
                    margin-left: 2px !important;
                    font-size: 0.58em !important;
                    line-height: 1 !important;
                    font-weight: 800 !important;
                    vertical-align: 0.10em !important;
                }

                .ai-brief-summary {
                    margin: 0 0 27px !important;
                    padding: 17px 20px !important;
                    border-left: 4px solid #0F8B8D !important;
                    border-radius: 0 12px 12px 0 !important;
                    color: #29485F !important;
                    background: #EFF8F7 !important;
                    font-size: 1.18rem !important;
                    line-height: 1.7 !important;
                    font-weight: 650 !important;
                }

                .ai-brief-summary b {
                    display: block !important;
                    margin-bottom: 5px !important;
                    color: #0F766E !important;
                    font-size: 0.8rem !important;
                    letter-spacing: 0.08em !important;
                }

                [data-testid="stMain"]
                div[data-testid="stExpander"]:has(.ai-report-document-marker)
                div[data-testid="stMarkdownContainer"] {
                    color: #1E293B !important;
                }

                /* ---------------------------------------------------------
                교통안전 실무보고서 일반 본문
                --------------------------------------------------------- */
                div[data-testid="stExpander"]:has(.ai-report-document-marker)
                div[data-testid="stMarkdownContainer"] p {
                    font-family: "Noto Serif KR", "KoPub Batang", "문화체육관광부 바탕체", "문체부 바탕체", "휴먼명조", "HYMyeongJo-Extra", "Batang", serif !important;
                    font-size: 1.25rem !important;
                    line-height: 1.82 !important;
                    letter-spacing: -0.012em !important;
                    margin-top: 0.25rem !important;
                    margin-bottom: 0.78rem !important;
                }

                /* ---------------------------------------------------------
                교통안전 실무보고서 목록
                --------------------------------------------------------- */
                div[data-testid="stExpander"]:has(.ai-report-document-marker)
                div[data-testid="stMarkdownContainer"] li {
                    font-family: "Noto Serif KR", "KoPub Batang", "문화체육관광부 바탕체", "문체부 바탕체", "휴먼명조", "HYMyeongJo-Extra", "Batang", serif !important;
                    font-size: 1.25rem !important;
                    line-height: 1.78 !important;
                    letter-spacing: -0.012em !important;
                    margin-bottom: 0.34rem !important;
                }

                /* ---------------------------------------------------------
                일반 AI 결과의 Markdown 제목
                --------------------------------------------------------- */
                div[data-testid="stExpander"]:has(.ai-report-document-marker)
                div[data-testid="stMarkdownContainer"] h2 {
                    font-size: 1.82rem !important;
                    line-height: 1.35 !important;
                    font-weight: 800 !important;
                    margin-top: 1.8rem !important;
                    margin-bottom: 0.8rem !important;
                    padding: 0 0 9px !important;
                    border-left: 0 !important;
                    border-bottom: 1px solid #C9D8E1 !important;
                    border-radius: 0 !important;
                    color: #16324A !important;
                    background: transparent !important;
                }

                div[data-testid="stExpander"]:has(.ai-report-document-marker)
                div[data-testid="stMarkdownContainer"] h3 {
                    font-size: 1.55rem !important;
                    line-height: 1.4 !important;
                    font-weight: 750 !important;
                    margin-top: 1.4rem !important;
                    margin-bottom: 0.35rem !important;
                    padding-bottom: 0 !important;
                }

                /* ---------------------------------------------------------
                교통안전 실무보고서 전용 제목·소제목
                format_police_report_display()가 생성하는 HTML 클래스
                --------------------------------------------------------- */
                .police-report-title {
                    font-size: 1.75rem !important;
                    line-height: 1.35 !important;
                    font-weight: 800 !important;
                    color: #0F172A !important;
                    margin-top: 0.4rem !important;
                    margin-bottom: 1.1rem !important;
                    padding-bottom: 0.65rem !important;
                    border-bottom: 3px solid #0F8B8D !important;
                }

                .police-report-subtitle {
                    font-size: 1.72rem !important;
                    line-height: 1.45 !important;
                    font-weight: 750 !important;
                    color: #1E293B !important;
                    margin-top: 1.45rem !important;
                    margin-bottom: 0.55rem !important;
                    padding: 0 0 9px !important;
                    border-left: 0 !important;
                    border-bottom: 1px solid #C9D8E1 !important;
                    border-radius: 0 !important;
                    color: #16324A !important;
                    background: transparent !important;
                }

                .police-report-level2,
                .police-report-level3,
                .police-report-body {
                    font-family: "Noto Serif KR", "KoPub Batang", "문화체육관광부 바탕체", "문체부 바탕체", "휴먼명조", "HYMyeongJo-Extra", "Batang", serif !important;
                    font-size: 1.25rem !important;
                    line-height: 1.8 !important;
                    letter-spacing: -0.012em !important;
                    color: #1F2937 !important;
                }

                /* 새로운 ㅇ 항목 사이 간격 */
                .police-report-level2 {
                    margin-top: 1.3rem !important;
                    margin-bottom: 0.65rem !important;
                    line-height: 1.72 !important;
                }

                /* 새로운 - 항목 사이 간격 */
                .police-report-level3 {
                    margin-top: 1rem !important;
                    margin-bottom: 0.65rem !important;
                    line-height: 1.68 !important;
                }

                .police-report-body {
                    margin-top: 0.25rem !important;
                    margin-bottom: 0.7rem !important;
                }

                /* 소제목 바로 다음 문단의 위쪽 여백 최소화 */
                div[data-testid="stExpander"]:has(.ai-report-document-marker)
                div[data-testid="stMarkdownContainer"] h3 + p {
                    margin-top: 0 !important;
                }

                /* ---------------------------------------------------------
                정책 브리핑 V2 — 최종 우선순위
                - 실제 Streamlit 열로 좌우 여백을 확보
                - 한글·고딕 중심의 단일 디자인 언어 사용
                --------------------------------------------------------- */
                [data-testid="stMain"]
                div[data-testid="stExpander"]:has(.ai-report-document-marker)
                > details,
                section.main
                div[data-testid="stExpander"]:has(.ai-report-document-marker)
                > details {
                    background: #E8EEF4 !important;
                    border: 1px solid #C9D6E1 !important;
                    border-radius: 16px !important;
                    box-shadow: 0 8px 24px rgba(15, 42, 67, 0.08) !important;
                }

                [data-testid="stMain"]
                div[data-testid="stExpander"]:has(.ai-report-document-marker)
                > details > div,
                [data-testid="stMain"]
                div[data-testid="stExpander"]:has(.ai-report-document-marker)
                div[data-testid="stExpanderDetails"],
                section.main
                div[data-testid="stExpander"]:has(.ai-report-document-marker)
                > details > div,
                section.main
                div[data-testid="stExpander"]:has(.ai-report-document-marker)
                div[data-testid="stExpanderDetails"] {
                    width: 100% !important;
                    max-width: none !important;
                    margin: 0 !important;
                    padding: 26px 18px 38px !important;
                    border: 0 !important;
                    border-radius: 0 !important;
                    background: transparent !important;
                    box-shadow: none !important;
                    box-sizing: border-box !important;
                }

                div[data-testid="column"]:has(.ai-report-canvas-marker),
                div[data-testid="stColumn"]:has(.ai-report-canvas-marker) {
                    counter-reset: ai-brief-section !important;
                    padding: 0 46px 42px !important;
                    border: 1px solid #D8E1E8 !important;
                    border-radius: 8px !important;
                    background: #FFFFFF !important;
                    box-shadow: 0 12px 30px rgba(29, 49, 64, 0.10) !important;
                    box-sizing: border-box !important;
                    overflow: hidden !important;
                }

                .ai-report-canvas-marker {
                    display: none !important;
                }

                .ai-brief-header {
                    margin: 0 -46px 25px !important;
                    padding: 27px 46px 24px !important;
                    border-bottom: 1px solid #D9E4EB !important;
                    background: #F6FAFC !important;
                }

                .ai-brief-eyebrow {
                    margin-bottom: 8px !important;
                    color: #0F8B8D !important;
                    font-size: 1.32rem !important;
                    font-weight: 900 !important;
                    letter-spacing: 0.08em !important;
                }

                .ai-brief-title {
                    color: #102A43 !important;
                    font-family: "Pretendard", "Noto Sans KR", "Malgun Gothic", sans-serif !important;
                    font-size: 2.55rem !important;
                    line-height: 1.25 !important;
                    font-weight: 900 !important;
                }

                .ai-brief-kpis {
                    gap: 12px !important;
                    margin-bottom: 20px !important;
                }

                .ai-brief-kpi {
                    min-height: 88px !important;
                    padding: 14px 16px !important;
                    border-top: 3px solid #2E6F95 !important;
                    border-right: 1px solid #D9E4EA !important;
                    border-bottom: 1px solid #D9E4EA !important;
                    border-left: 1px solid #D9E4EA !important;
                    border-radius: 10px !important;
                    background: #FBFDFE !important;
                }

                .ai-brief-summary {
                    margin: 0 0 30px !important;
                    padding: 18px 21px !important;
                    border-left: 4px solid #0F8B8D !important;
                    border-radius: 0 10px 10px 0 !important;
                    color: #173F5F !important;
                    background: #EDF7F6 !important;
                    font-family: "Pretendard", "Noto Sans KR", "Malgun Gothic", sans-serif !important;
                    font-size: 1.62rem !important;
                    line-height: 1.65 !important;
                    font-weight: 700 !important;
                }

                .ai-brief-summary b {
                    color: #0F766E !important;
                    font-size: 1.34rem !important;
                    letter-spacing: 0.03em !important;
                }

                div[data-testid="stExpander"]:has(.ai-report-document-marker)
                div[data-testid="stMarkdownContainer"] p,
                div[data-testid="stExpander"]:has(.ai-report-document-marker)
                div[data-testid="stMarkdownContainer"] li,
                .police-report-level2,
                .police-report-level3,
                .police-report-body {
                    font-family: "Pretendard", "Noto Sans KR", "Malgun Gothic", sans-serif !important;
                    font-size: 1.64rem !important;
                    line-height: 1.72 !important;
                    letter-spacing: -0.018em !important;
                    color: #263746 !important;
                }

                div[data-testid="stExpander"]:has(.ai-report-document-marker)
                div[data-testid="stMarkdownContainer"] h2,
                .police-report-subtitle {
                    counter-increment: ai-brief-section !important;
                    margin: 2rem 0 0.85rem !important;
                    padding: 0 0 0 13px !important;
                    border-left: 4px solid #1E6A8D !important;
                    border-bottom: 0 !important;
                    color: #12354F !important;
                    background: transparent !important;
                    font-family: "Pretendard", "Noto Sans KR", "Malgun Gothic", sans-serif !important;
                    font-size: 2.05rem !important;
                    line-height: 1.35 !important;
                    font-weight: 900 !important;
                }

                div[data-testid="column"]:has(.ai-report-canvas-marker)
                div[data-testid="stMarkdownContainer"] h2::before,
                div[data-testid="stColumn"]:has(.ai-report-canvas-marker)
                div[data-testid="stMarkdownContainer"] h2::before,
                div[data-testid="column"]:has(.ai-report-canvas-marker)
                .police-report-subtitle::before,
                div[data-testid="stColumn"]:has(.ai-report-canvas-marker)
                .police-report-subtitle::before {
                    content: counter(ai-brief-section, decimal-leading-zero) "  " !important;
                    margin-right: 4px !important;
                    color: #0F8B8D !important;
                    font-size: 0.82em !important;
                    font-weight: 900 !important;
                }

                div[data-testid="column"]:has(.ai-report-canvas-marker)
                div[data-testid="stMarkdownContainer"] h3,
                div[data-testid="stColumn"]:has(.ai-report-canvas-marker)
                div[data-testid="stMarkdownContainer"] h3 {
                    font-size: 2.05rem !important;
                    line-height: 1.38 !important;
                    font-weight: 850 !important;
                }

                div[data-testid="column"]:has(.ai-report-canvas-marker)
                .police-report-title,
                div[data-testid="stColumn"]:has(.ai-report-canvas-marker)
                .police-report-title {
                    font-size: 2.25rem !important;
                    line-height: 1.35 !important;
                    font-weight: 900 !important;
                }

                div[data-testid="column"]:has(.ai-report-canvas-marker)
                [data-testid="stCaptionContainer"] p,
                div[data-testid="stColumn"]:has(.ai-report-canvas-marker)
                [data-testid="stCaptionContainer"] p {
                    font-size: 1.12rem !important;
                    line-height: 1.55 !important;
                }

                .police-report-level2 {
                    display: flex !important;
                    align-items: flex-start !important;
                    gap: 0.45rem !important;
                    margin: 0.42rem 0 0.25rem !important;
                    padding-left: 0.8rem !important;
                    border-left: 0 !important;
                    font-weight: 650 !important;
                }

                .police-report-level2-bullet {
                    display: block !important;
                    flex: 0 0 0.75rem !important;
                    color: #284B63 !important;
                    font-weight: 800 !important;
                    line-height: inherit !important;
                    text-align: center !important;
                }

                .police-report-level2-text {
                    display: block !important;
                    flex: 1 1 auto !important;
                    min-width: 0 !important;
                    line-height: inherit !important;
                }

                .police-report-level3 {
                    display: flex !important;
                    align-items: flex-start !important;
                    gap: 0.45rem !important;
                    margin: 0.35rem 0 !important;
                    padding-left: 2.2rem !important;
                    color: #425466 !important;
                    line-height: 1.82 !important;
                }

                .police-report-level3-bullet {
                    display: block !important;
                    flex: 0 0 0.65rem !important;
                    color: #526A7D !important;
                    font-weight: 700 !important;
                    line-height: inherit !important;
                    text-align: center !important;
                }

                .police-report-level3-text {
                    display: block !important;
                    flex: 1 1 auto !important;
                    min-width: 0 !important;
                    line-height: inherit !important;
                }

                /* 확대된 보고서 글자에 맞춘 세로 가독성 */
                div[data-testid="column"]:has(.ai-report-canvas-marker)
                div[data-testid="stMarkdownContainer"] p,
                div[data-testid="column"]:has(.ai-report-canvas-marker)
                div[data-testid="stMarkdownContainer"] li,
                div[data-testid="stColumn"]:has(.ai-report-canvas-marker)
                div[data-testid="stMarkdownContainer"] p,
                div[data-testid="stColumn"]:has(.ai-report-canvas-marker)
                div[data-testid="stMarkdownContainer"] li,
                .police-report-level2,
                .police-report-body {
                    line-height: 1.72 !important;
                }

                div[data-testid="column"]:has(.ai-report-canvas-marker)
                div[data-testid="stMarkdownContainer"] p,
                div[data-testid="stColumn"]:has(.ai-report-canvas-marker)
                div[data-testid="stMarkdownContainer"] p,
                .police-report-body {
                    margin-top: 0.4rem !important;
                    margin-bottom: 1rem !important;
                }

                div[data-testid="column"]:has(.ai-report-canvas-marker)
                div[data-testid="stMarkdownContainer"] li,
                div[data-testid="stColumn"]:has(.ai-report-canvas-marker)
                div[data-testid="stMarkdownContainer"] li {
                    margin-bottom: 0.65rem !important;
                }

                .police-report-level2 {
                    margin-top: 1.05rem !important;
                    margin-bottom: 0.62rem !important;
                    line-height: 1.72 !important;
                }

                /* 같은 항목 안의 줄은 가깝게, 새 '-' 항목은 문단처럼 분리 */
                .police-report-level3 {
                    margin-top: 0.78rem !important;
                    margin-bottom: 0.62rem !important;
                    line-height: 1.68 !important;
                }

                div[data-testid="column"]:has(.ai-report-canvas-marker)
                div[data-testid="stMarkdownContainer"] h2,
                div[data-testid="stColumn"]:has(.ai-report-canvas-marker)
                div[data-testid="stMarkdownContainer"] h2,
                .police-report-subtitle {
                    margin-top: 2.5rem !important;
                    margin-bottom: 1.15rem !important;
                    line-height: 1.45 !important;
                }

                div[data-testid="column"]:has(.ai-report-canvas-marker)
                div[data-testid="stMarkdownContainer"] h3,
                div[data-testid="stColumn"]:has(.ai-report-canvas-marker)
                div[data-testid="stMarkdownContainer"] h3 {
                    margin-top: 1.9rem !important;
                    margin-bottom: 0.75rem !important;
                    line-height: 1.48 !important;
                }

                .ai-brief-summary {
                    padding-top: 21px !important;
                    padding-bottom: 21px !important;
                    line-height: 1.78 !important;
                }

                .ai-brief-kpi-value {
                    line-height: 1.25 !important;
                }

                div[data-testid="column"]:has(.ai-report-canvas-marker)
                [data-testid="stCaptionContainer"] p,
                div[data-testid="stColumn"]:has(.ai-report-canvas-marker)
                [data-testid="stCaptionContainer"] p {
                    line-height: 1.7 !important;
                }

                @media (max-width: 780px) {
                    [data-testid="stMain"]
                    div[data-testid="stExpander"]:has(.ai-report-document-marker)
                    > details > div {
                        width: calc(100% - 20px) !important;
                        margin: 4px auto 18px !important;
                        padding: 12px 20px 28px !important;
                    }
                    [data-testid="stMain"]
                    div[data-testid="stExpander"]:has(.ai-report-document-marker)
                    div[data-testid="stExpanderDetails"] {
                        width: calc(100% - 20px) !important;
                        margin: 4px auto 18px !important;
                        padding: 12px 20px 28px !important;
                    }
                    div[data-testid="column"]:has(.ai-report-canvas-marker),
                    div[data-testid="stColumn"]:has(.ai-report-canvas-marker) {
                        padding: 0 20px 28px !important;
                    }
                    .ai-brief-header { margin: 0 -20px 20px !important; padding: 24px 20px !important; }
                    .ai-brief-kpis { grid-template-columns: 1fr !important; }
                }

                /* =====================================================
                교통안전 실무보고서 단어 단위 줄바꿈
                - 한 단어가 두 줄로 나뉘는 현상 방지
                - 너무 긴 URL·숫자 문자열만 예외적으로 줄바꿈
                ===================================================== */
                div[data-testid="column"]:has(.ai-report-canvas-marker)
                div[data-testid="stMarkdownContainer"] p,
                div[data-testid="column"]:has(.ai-report-canvas-marker)
                div[data-testid="stMarkdownContainer"] li,
                div[data-testid="stColumn"]:has(.ai-report-canvas-marker)
                div[data-testid="stMarkdownContainer"] p,
                div[data-testid="stColumn"]:has(.ai-report-canvas-marker)
                div[data-testid="stMarkdownContainer"] li,
                .police-report-title,
                .police-report-subtitle,
                .police-report-level2-text,
                .police-report-level3-text,
                .police-report-body,
                .ai-brief-summary,
                .ai-brief-title {
                    word-break: keep-all !important;
                    overflow-wrap: break-word !important;
                    line-break: strict !important;
                    white-space: normal !important;
                }                


                </style>
                """,
                unsafe_allow_html=True,
            )

            # 실제 Streamlit 열을 이용해 보고서의 좌우 여백을 안정적으로 확보한다.
            # CSS max-width에만 의존하지 않으므로 로컬·Cloud에서 같은 폭을 유지한다.
            report_margin_left, report_content_col, report_margin_right = st.columns(
                [0.9, 8.2, 0.9],
                gap="small",
            )

            with report_content_col:
                st.markdown(
                    '<span class="ai-report-canvas-marker" aria-hidden="true"></span>',
                    unsafe_allow_html=True,
                )

                ai_display_text = st.session_state[
                    f"ai_result_{selected_ai_report_type}"
                ]
    
                def _brief_mode_value(column_name, fallback="자료 없음", suffix=""):
                    if column_name not in filtered_df.columns:
                        return fallback
                    values = filtered_df[column_name].dropna().astype(str)
                    values = values[~values.isin(["", "nan", "None", "불명", "기타불명"])]
                    if values.empty:
                        return fallback
                    value = values.value_counts().index[0]
                    if suffix and suffix not in value:
                        value = f"{value}{suffix}"
                    return value
    
                report_title = re.sub(
                    r"^[^0-9A-Za-z가-힣]+",
                    "",
                    selected_info["title"],
                ).strip()
                peak_hour = _brief_mode_value("occrrnc_time_dc", suffix="시")
                top_fatal_type = _brief_mode_value("fatal_type")
                report_summary = extract_ai_report_summary(
                    ai_display_text,
                    selected_ai_report_type,
                )
    
                st.markdown(
                    f"""
                    <div class="ai-brief-header">
                        <div class="ai-brief-eyebrow">TAAP-AI 정책 브리핑</div>
                        <div class="ai-brief-title">{html.escape(report_title)}</div>
                        <div class="ai-brief-meta">
                            <span>{html.escape(str(selected_ps))} 관할</span>
                            <span>{html.escape(period_summary)}</span>
                            <span>{html.escape(accident_type_summary)}</span>
                        </div>
                    </div>
                    <div class="ai-brief-kpis">
                        <div class="ai-brief-kpi">
                            <div class="ai-brief-kpi-label">분석 사고</div>
                            <div class="ai-brief-kpi-value"><span class="ai-brief-kpi-number">{len(filtered_df):,}</span><span class="ai-brief-kpi-unit">건</span></div>
                        </div>
                        <div class="ai-brief-kpi">
                            <div class="ai-brief-kpi-label">최다 발생시간</div>
                            <div class="ai-brief-kpi-value">{html.escape(peak_hour)}</div>
                        </div>
                        <div class="ai-brief-kpi">
                            <div class="ai-brief-kpi-label">최다 사망유형</div>
                            <div class="ai-brief-kpi-value">{html.escape(top_fatal_type)}</div>
                        </div>
                    </div>
                    <div class="ai-brief-summary">
                        <b>핵심 판단</b>
                        {html.escape(report_summary)}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
    
                if selected_ai_report_type == "police_report":
                    st.markdown(
                        format_police_report_display(
                            ai_display_text,
                            suppress_first_title=True,
                        ),
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(strip_first_markdown_title(ai_display_text))
    
                st.caption(
                    "생성형 AI 결과는 의사결정 보조자료이며, "
                    "최종 활용 전 담당자의 통계·현장 확인이 필요합니다."
                )


        # ============================================================
        # 이미 생성한 다른 유형의 결과 열람
        # - 새 보고서 생성 때마다 selectbox key를 새로 만들어
        #   브라우저에 남은 과거 선택 상태가 현재 보고서를 덮어쓰지 못하도록 함
        # ============================================================
        available_report_types = [
            report_type
            for report_type in ai_report_types
            if st.session_state.get(
                f"ai_result_{report_type}"
            )
        ]

        if len(available_report_types) > 1:

            current_selected_type = st.session_state.get(
                "selected_ai_report_type"
            )

            if current_selected_type not in available_report_types:
                current_selected_type = available_report_types[0]
                st.session_state[
                    "selected_ai_report_type"
                ] = current_selected_type

            selector_version = int(
                st.session_state.get(
                    "existing_report_selector_version",
                    0,
                )
            )

            selector_key = (
                f"existing_ai_report_selector_{selector_version}"
            )

            default_index = available_report_types.index(
                current_selected_type
            )

            selected_existing_type = st.selectbox(
                "기존 생성 결과 보기",
                options=available_report_types,
                index=default_index,
                format_func=lambda value: (
                    ai_report_types[value]["button"]
                ),
                key=selector_key,
            )

            if (
                selected_existing_type
                != st.session_state.get(
                    "selected_ai_report_type"
                )
            ):
                st.session_state[
                    "selected_ai_report_type"
                ] = selected_existing_type
                st.rerun()

if selected_page == "통계분석":
    # ============================================================
    # 6. 하단 레이아웃: 분석 통계 배치
    # ============================================================

    # ============================================================
    # 6-1. 그래프 공통 디자인 설정
    # ============================================================

    # 일반 그래프용 색상
    COLOR_PRIMARY = "#2563EB"
    COLOR_SECONDARY = "#0EA5A4"
    COLOR_PURPLE = "#536E8A"
    COLOR_ORANGE = "#D99000"
    COLOR_FATAL = "#DC4C4C"
    COLOR_FATAL_LIGHT = "#F08A8A"
    COLOR_TEXT = "#172033"
    COLOR_GRID = "#E3EAF2"
    COLOR_MUTED = "#64748B"

    # 사고분류별 고정 색상
    ACCIDENT_CLASS_COLORS = {
        "사망사고": "#DC4C4C",
        "중상사고": "#D99000",
        "경상사고": "#2563EB",
        "부상신고사고": "#0EA5A4",
    }

    # 사고종별 고정 색상
    ACCIDENT_TYPE_COLORS = {
        "차대차": "#2563EB",
        "차대사람": "#0EA5A4",
        "차량단독": "#64748B",
    }

    # 차종 그래프 팔레트
    VEHICLE_COLORS = [
        "#2563EB",
        "#0EA5A4",
        "#102A43",
        "#4F7CAC",
        "#0891B2",
        "#536E8A",
        "#D99000",
        "#7895B2",
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

# ============================================================
# 통계 대시보드 연속형 색상
# - 건수가 많을수록 진한 색
# - 최솟값도 흰색에 묻히지 않도록 중간 밝기부터 시작
# ============================================================

    CHART_COLOR_SCALES = {
        # --------------------------------------------------------
        # 차량 분석
        # --------------------------------------------------------

        # 가해차량 : Blue
        "offending_vehicle": [
            [0.0, "#93C5FD"],   # Blue 300
            [0.5, "#3B82F6"],   # Blue 500
            [1.0, "#1D4ED8"],   # Blue 700
        ],

        # 피해차량 : Teal
        "damaged_vehicle": [
            [0.0, "#5EEAD4"],   # Teal 300
            [0.5, "#0EA5A4"],   # TAAP Teal
            [1.0, "#0F766E"],   # Teal 700
        ],

        # --------------------------------------------------------
        # 사망사고 분석
        # --------------------------------------------------------

        # 사망자 유형 : Red
        "fatal_type": [
            [0.0, "#F3A4A4"],
            [0.5, "#DC4C4C"],   # TAAP Fatal Red
            [1.0, "#B12C2C"],
        ],

        # 사망자 연령대 : Purple
        "fatal_age": [
            [0.0, "#D8B4FE"],   # Purple 300
            [0.5, "#A855F7"],   # Purple 500
            [1.0, "#6B21A8"],   # Purple 800
        ],

        # --------------------------------------------------------
        # 상황별 사고 분석
        # --------------------------------------------------------

        # 요일별 : Teal
        "weekday": [
            [0.0, "#7FD6D4"],
            [0.5, "#0EA5A4"],
            [1.0, "#0F6E73"],
        ],

        # 시간대별 : Blue
        "time": [
            [0.0, "#93C5FD"],
            [0.5, "#2563EB"],
            [1.0, "#173F8A"],
        ],

        # 법규위반별 : Amber
        "violation": [
            [0.0, "#F2D58A"],
            [0.5, "#D99000"],   # TAAP Amber
            [1.0, "#9A6500"],
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


    st.markdown("##### 📈 세부 항목별 교통사고 현황")


    # ============================================================
    # 6-3. 필터링 결과가 있는 경우
    # ============================================================

    if not filtered_df.empty:

        # ========================================================
        # 첫 번째 줄: 연도별 / 사고분류 / 사고종별
        # ========================================================

        accident_overview_expander = st.expander(
            " ",
            expanded=True,
        )

        row1_col1, row1_col2, row1_col3 = accident_overview_expander.columns(3)

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
            st.markdown("**📂 부상 정도**")

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

            # 차트 표시용 짧은 사고분류명
            accident_class_short_map = {
                "사망사고": "사망",
                "중상사고": "중상",
                "경상사고": "경상",
                "부상신고사고": "그 외",
            }

            type_counts["표시명"] = (
                type_counts["사고분류"]
                .map(accident_class_short_map)
                .fillna(type_counts["사고분류"])
            )

            fig_type = px.pie(
                type_counts,
                values="사고건수",
                names="표시명",
                hole=0.58,
                color="사고분류",
                color_discrete_map=ACCIDENT_CLASS_COLORS,
            )

            fig_type.update_traces(
                # --------------------------------------------------------
                # 도넛차트 기본 표시
                # 예: 경상사고 : 78.2%
                # --------------------------------------------------------
                textposition="outside",
                texttemplate="<b>%{label} :</b><br>%{percent:.1%}",
                textfont=dict(
                    size=16,
                    color=COLOR_TEXT,
                    family='"Malgun Gothic", sans-serif',
                ),
                automargin=True,
                domain=dict(
                    x=[0.14, 0.86],
                    y=[0.06, 0.94],
                ),

                marker=dict(
                    line=dict(
                        color="white",
                        width=2,
                    )
                ),

                # --------------------------------------------------------
                # 마우스 hover 팝업
                # 발생 연도별 추이 그래프와 동일한 형태로 정리
                # --------------------------------------------------------
                hovertemplate=(
                    "<b>%{label}</b><br>"
                    "사고 건수: %{value:,}건<br>"
                    "비율: %{percent:.1%}"
                    "<extra></extra>"
                ),
            )

            fig_type.update_layout(
                height=300,
                margin=dict(
                    l=48,
                    r=48,
                    t=10,
                    b=10,
                ),
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(
                    family='"Malgun Gothic", sans-serif',
                    size=13,
                    color=COLOR_TEXT,
                ),

                hoverlabel=dict(
                    bgcolor="white",
                    bordercolor="#CBD5E1",
                    font=dict(
                        size=14,
                        color=COLOR_TEXT,
                        family='"Malgun Gothic", sans-serif',
                    ),
                ),
                showlegend=False,
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
            st.markdown("**🚑 사고 종별**")

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
                # --------------------------------------------------------
                # 도넛차트 기본 표시
                # 예: 차대차 : 77.8%
                # --------------------------------------------------------
                textposition="outside",
                texttemplate="<b>%{label} :</b><br>%{percent:.1%}",
                textfont=dict(
                    size=16,
                    color=COLOR_TEXT,
                    family='"Malgun Gothic", sans-serif',
                ),
                automargin=True,
                domain=dict(
                    x=[0.14, 0.86],
                    y=[0.06, 0.94],
                ),

                marker=dict(
                    line=dict(
                        color="white",
                        width=2,
                    )
                ),

                # --------------------------------------------------------
                # 마우스 hover 팝업
                # --------------------------------------------------------
                hovertemplate=(
                    "<b>%{label}</b><br>"
                    "사고 건수: %{value:,}건<br>"
                    "비율: %{percent:.1%}"
                    "<extra></extra>"
                ),
            )

            fig_hdc.update_layout(
                height=300,
                margin=dict(
                    l=48,
                    r=48,
                    t=10,
                    b=10,
                ),
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(
                    family='"Malgun Gothic", sans-serif',
                    size=13,
                    color=COLOR_TEXT,
                ),
                hoverlabel=dict(
                    bgcolor="white",
                    bordercolor="#CBD5E1",
                    font=dict(
                        size=14,
                        color=COLOR_TEXT,
                        family='"Malgun Gothic", sans-serif',
                    ),
                ),                
                showlegend=False,
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

        with st.expander(" ", expanded=True):
            # ========================================================
            # 두 번째 줄: 차량 분석
            # ========================================================


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

                # 그래프 표시용 라벨만 문자열로 변환
                # Parquet category 컬럼에 새 범주를 직접 넣지 않아 Cloud 오류 방지
                wrngdo_counts["차종"] = (
                    wrngdo_counts["차종"]
                    .astype("string")
                    .replace({"기타불명": "기타"})
                )

                fig_wrngdo = px.bar(
                    wrngdo_counts,
                    x="사고건수",
                    y="차종",
                    orientation="h",
                    text="사고건수",
                    color="사고건수",
                    color_continuous_scale=(
                        CHART_COLOR_SCALES["offending_vehicle"]
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

                # 그래프 표시용 라벨만 문자열로 변환
                # category에 새 범주를 직접 추가하지 않음
                dmge_counts["차종"] = (
                    dmge_counts["차종"]
                    .astype("string")
                    .replace({"기타불명": "기타"})
                )

                fig_dmge = px.bar(
                    dmge_counts,
                    x="사고건수",
                    y="차종",
                    orientation="h",
                    text="사고건수",
                    color="사고건수",
                    color_continuous_scale=(
                        CHART_COLOR_SCALES["damaged_vehicle"]
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
        with st.expander(" ", expanded=True):
            # ========================================================
            # 세 번째 줄: 사망사고 분석
            # ========================================================


            row3_col1, row3_col2 = st.columns(2)


        # --------------------------------------------------------
        # 사망자 유형
        # --------------------------------------------------------
            with row3_col1:
                st.markdown("**🛑 사망자 유형**")

                fatal_type_counts = (
                    filtered_df["fatal_type"]
                    .dropna()
                    .value_counts()
                    .rename_axis("사망자 유형")
                    .reset_index(name="사망사고 건수")
                    .sort_values("사망사고 건수", ascending=False)
                )

                # 그래프 표시용 라벨만 문자열로 변환
                # category에 새 범주를 직접 추가하지 않음
                fatal_type_counts["사망자 유형"] = (
                    fatal_type_counts["사망자 유형"]
                    .astype("string")
                    .replace({"기타불명": "기타"})
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
                            CHART_COLOR_SCALES["fatal_type"]
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
                    "61-64세",
                    "65세 이상",
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
                        text="사망사고 건수",
                        color="사망사고 건수",
                        color_continuous_scale=(
                            CHART_COLOR_SCALES["fatal_age"]
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
                        height=max(330, len(fatal_age_counts) * 38),
                        horizontal=True,
                    )

                    render_plotly_chart(fig_fatal_age)

                else:
                    st.info(
                        "선택된 조건에 해당하는 "
                        "사망자 연령대 데이터가 없습니다."
                    )
        with st.expander(" ", expanded=True):
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
