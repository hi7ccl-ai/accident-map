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
# 0. Streamlit 페이지 기본 설정
# 반드시 다른 Streamlit 명령보다 먼저 실행
# -----------------------------------
st.set_page_config(
    page_title="Traffic Atlas AI",
    page_icon="🗺️",
    layout="wide",
)

# -----------------------------------
# 0-1. 메인 제목 및 부제목
# Streamlit 기본 제목 방식 사용
# -----------------------------------
st.title("🗺️ Traffic Accident Analysis Platform - AI")

st.markdown(
    """
    <div style="
        font-size: 1.15rem;
        color: #64748B;
        margin-top: -12px;
        margin-bottom: 16px;
        margin-left: 15px;

    ">
        교통사고 AI 분석 및 의사결정 지원시스템
    </div>
    """,
    unsafe_allow_html=True,
)

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
    </style>
    """,
    unsafe_allow_html=True,
)


# 사고 데이터 불러오기 (Parquet 파일 로드 & 캐싱)
# - 원본 데이터 정제와 category 변환은 Parquet 저장 전에 완료
# - Streamlit에서는 재치환하지 않고 필요한 파생 컬럼만 생성
@st.cache_data
def load_data():
    df = pd.read_parquet("정제완료(21~25).parquet")

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
                "by_damaged_vehicle": _distribution_dict(
                    target_df,
                    "dmge_vhcle_asort_dc",
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
        raise ValueError(f"지원하지 않는 AI 보고서 유형입니다: {report_type}")

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
    - AI 보고서: Terra / high / 필요 시 웹검색
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
        raise ValueError(f"지원하지 않는 AI 보고서 유형입니다: {report_type}")

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
    # - AI 보고서     : Web Search + 경찰보고서 File Search
    # ============================================================

    tools = []


    # ------------------------------------------------------------
    # Web Search
    # 대응전략 / AI 보고서에서만 사용
    # ------------------------------------------------------------
    if config["web_search"]:
        tools.append(
            {
                "type": "web_search",
            }
        )


    # ------------------------------------------------------------
    # 경찰보고서 File Search
    # AI 보고서(police_report)에서만 사용
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
                "AI 보고서용 Vector Store ID가 없습니다. "
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
            color:#31333F;
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
    label_visibility="collapsed",
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

sidebar_filter_title("발생연월", margin_bottom=14)

# 데이터가 비어 있거나 날짜 변환에 실패해도 이후 코드에서 안전하게 참조
start_year_month = None
end_year_month = None

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
            label_visibility="collapsed",
        )
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

sidebar_filter_title("사고분류 (복수선택)", margin_bottom=14)

selected_types = st.sidebar.multiselect(
    "사고분류 (복수 선택)",
    type_options,
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
time_col1, time_col2 = st.sidebar.columns(2)

with time_col1:
    start_time = st.selectbox(
        "시작시간",
        options=time_options,
        index=0,
        format_func=lambda x: f"{x:02d}시",
        key="start_time",
        label_visibility="collapsed",
    )

with time_col2:
    end_time = st.selectbox(
        "종료시간",
        options=time_options,
        index=23,
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
        ※ 자정을 포함한 검색 가능 (ex : 22시~06시)<br>
        ※ '분'을 제거한 시각 검색 (ex : 7시15분 → 7시)
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
        is_selected = st.checkbox(
            weekday_short_name[weekday],
            value=False,
            key=f"weekday_checkbox_{weekday}",
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

st.markdown(
    f"""
    <div class="search-condition-text">
        <span class="search-condition-title"> [현재 검색조건]</span>
        <span class="search-condition-item"> {period_summary}</span>
        <span class="search-condition-divider">|</span>
        <span class="search-condition-item">사고분류: {severity_summary}</span>
        <span class="search-condition-divider">|</span>
        <span class="search-condition-item">사고종별: {accident_type_summary}</span>
        <span class="search-condition-divider">|</span>
        <span class="search-condition-item"> {time_summary}</span>
        <span class="search-condition-divider">|</span>
        <span class="search-condition-item">요일: {weekday_summary}</span>
    </div>
    """,
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

kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)

with kpi_col1:
    st.metric("선택된 관할", selected_ps)

with kpi_col2:
    st.metric("총 사고", f"{len(filtered_df):,}건")

with kpi_col3:
    st.metric("사망사고", f"{fatal_accident_count:,}건")

with kpi_col4:
    st.metric("중상사고", f"{serious_accident_count:,}건")



if "hotspot_radius" not in st.session_state:
    st.session_state.hotspot_radius = 100

if "hotspot_top_n" not in st.session_state:
    st.session_state.hotspot_top_n = 5

st.markdown(
    """
    <div style="
        font-size: 1.08rem;
        color: #64748B;
        margin-top: 10px;
        margin-bottom: 10px;
        line-height: 1.6;
    ">
        <span style="
            color: #1D4ED8;
            font-weight: 700;
        ">
            🔎사고다발지점 탐색
        </span>
        <span>
            : 입력한 반경(m)과 수에 따라, AI가 사고다발지역을 자동 탐색합니다.
        </span>
    </div>
    """,
    unsafe_allow_html=True,
)

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
# 페이지 탭
# ============================================================
map_tab, stats_tab, ai_tab = st.tabs(
    [
        "🗺️ GIS 분석",
        "📊 통계 대시보드",
        "🤖 AI 리포트",
    ]
)

with map_tab:
    st.subheader("🗺️ 교통사고 공간분석", anchor=False)

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


with map_tab:

    st.markdown(
        """
        <div style="
            text-align: right;
            color: #6B7280;
            font-size: 0.82rem;
            margin-top: 2px;
            margin-bottom: 2px;
        ">
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


with ai_tab:
    # ============================================================
    # 5-1. 목적별 생성형 AI 교통사고 분석
    # - 선택집단 + 동일기간 기준집단 비교
    # - 목적별 모델/Reasoning/Web Search 차등 적용
    # - 호출별 토큰·예상비용 확인
    # ============================================================

    st.subheader("📈 AI 교통사고 분석", anchor=False)

    st.markdown(
        """
        현재 지도에 적용된 조건을 기준으로 AI 분석을 수행합니다.  
        """
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
        "damaged_vehicle": selected_dmge if selected_dmge else ["전체"],
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
            "button": "🔍 핵심 인사이트",
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
            "button": "📍 다발지점 진단",
            "title": "📍 사고다발지점 AI 진단",
            "description": (
                "도출된 사고다발지점별 교통사고를 프로파일링해 "
                "관리 우선순위를 진단합니다."
            ),
            "spinner": "사고다발지점별 특성과 차이를 분석하고 있습니다.",
        },
        "strategy": {
            "button": "🎯 대응전략",
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
            "button": "📄 AI 보고서",
            "title": "📄 교통사고 분석 및 대응방향 보고",
            "description": (
                "검색된 통계자료를 외부 전문자료 등과 비교분석해 "
                "실무용으로 활용할 수 있는 보고서를 작성합니다."
            ),
            "spinner": (
                "필터링된 통계를 분석해 경찰 형식의 AI 보고서를 작성하고 있습니다."
            ),
        },
    }

     # ============================================================
    # AI 보고서 선택 및 생성
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
        AI 보고서 버튼 클릭 콜백

        - 화면에서 선택된 버튼을 즉시 표시하기 위한 상태
        - 실제 AI 생성을 위한 대기 상태
        를 동시에 저장한다.
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

            st.caption(report_info["description"])


    # ------------------------------------------------------------
    # 필터 결과가 없는 경우
    # ------------------------------------------------------------
    if filtered_df.empty:
        st.warning(
            "현재 필터 조건에 해당하는 사고 데이터가 없어 "
            "AI 분석을 생성할 수 없습니다."
        )


    # ============================================================
    # 대기 중인 AI 보고서 생성
    #
    # pop()을 사용하여 생성 시작과 동시에 pending 상태 제거
    # → Streamlit 재실행 시 같은 API가 자동으로 재호출되는 것을 방지
    # ============================================================

    clicked_report_type = st.session_state.pop(
        "pending_ai_report_type",
        None,
    )


    if clicked_report_type is not None:

        report_info = ai_report_types[clicked_report_type]

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

        with st.expander(selected_info["title"], expanded=True):

            # --------------------------------------------------------
            # AI 보고서 본문 가독성 개선
            # - 제목/소제목 크기는 기존 Streamlit Markdown 유지
            # - 일반 본문과 목록 글자만 확대
            # --------------------------------------------------------

            st.markdown(
                """
                <style>

                /* ---------------------------------------------------------
                AI 보고서 일반 본문
                --------------------------------------------------------- */
                div[data-testid="stExpander"]
                div[data-testid="stMarkdownContainer"] p {
                    font-size: 1.2rem !important;
                    line-height: 1.85 !important;
                    margin-top: 0.25rem !important;
                    margin-bottom: 0.7rem !important;
                }

                /* ---------------------------------------------------------
                AI 보고서 목록
                --------------------------------------------------------- */
                div[data-testid="stExpander"]
                div[data-testid="stMarkdownContainer"] li {
                    font-size: 1.2rem !important;
                    line-height: 1.8 !important;
                    margin-bottom: 0.25rem !important;
                }

                /* ---------------------------------------------------------
                큰 항목 제목
                예: 1. 전략 판단
                Markdown ## → h2
                --------------------------------------------------------- */
                div[data-testid="stExpander"]
                div[data-testid="stMarkdownContainer"] h2 {
                    margin-top: 1.8rem !important;
                    margin-bottom: 0.8rem !important;
                }

                /* ---------------------------------------------------------
                작은 소제목
                예: 1) 우선 개입 판단
                Markdown ### → h3

                현재보다 위쪽 여백을 넓히고,
                제목 아래 여백은 좁게 조정
                --------------------------------------------------------- */
                div[data-testid="stExpander"]
                div[data-testid="stMarkdownContainer"] h3 {
                    margin-top: 1.4rem !important;
                    margin-bottom: 0.35rem !important;
                    padding-bottom: 0 !important;
                }

                /* 소제목 바로 다음 문단의 위쪽 여백 최소화 */
                div[data-testid="stExpander"]
                div[data-testid="stMarkdownContainer"] h3 + p {
                    margin-top: 0 !important;
                }

                </style>
                """,
                unsafe_allow_html=True,
            )

            ai_display_text = st.session_state[
                f"ai_result_{selected_ai_report_type}"
            ]

            st.markdown(ai_display_text)

            st.caption(
                "생성형 AI 결과는 의사결정 보조자료이며, "
                "최종 활용 전 담당자의 통계·현장 확인이 필요합니다."
            )


        # 이미 생성한 다른 유형의 결과를 API 재호출 없이 열람
        available_report_types = [
            report_type
            for report_type in ai_report_types
            if st.session_state.get(f"ai_result_{report_type}")
        ]

        if len(available_report_types) > 1:
            selected_existing_type = st.selectbox(
                "기존 생성 결과 보기",
                options=available_report_types,
                index=available_report_types.index(selected_ai_report_type),
                format_func=lambda value: ai_report_types[value]["button"],
                key="existing_ai_report_selector",
            )

            if selected_existing_type != selected_ai_report_type:
                st.session_state["selected_ai_report_type"] = selected_existing_type
                st.rerun()

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
            [0.5, "#14B8A6"],   # Teal 500
            [1.0, "#0F766E"],   # Teal 700
        ],

        # --------------------------------------------------------
        # 사망사고 분석
        # --------------------------------------------------------

        # 사망자 유형 : Red
        "fatal_type": [
            [0.0, "#FCA5A5"],   # Red 300
            [0.5, "#EF4444"],   # Red 500
            [1.0, "#B91C1C"],   # Red 700
        ],

        # 사망자 연령대 : Purple
        "fatal_age": [
            [0.0, "#D8B4FE"],   # Purple 300
            [0.5, "#A855F7"],   # Purple 500
            [1.0, "#7E22CE"],   # Purple 700
        ],

        # --------------------------------------------------------
        # 상황별 사고 분석
        # --------------------------------------------------------

        # 요일별 : Green
        "weekday": [
            [0.0, "#86EFAC"],   # Green 300
            [0.5, "#22C55E"],   # Green 500
            [1.0, "#15803D"],   # Green 700
        ],

        # 시간대별 : Indigo
        "time": [
            [0.0, "#A5B4FC"],   # Indigo 300
            [0.5, "#6366F1"],   # Indigo 500
            [1.0, "#3730A3"],   # Indigo 800
        ],

        # 법규위반별 : Amber
        "violation": [
            [0.0, "#FCD34D"],   # Amber 300
            [0.5, "#F59E0B"],   # Amber 500
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
            st.markdown("**📂 사고 분류**")

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

        with st.expander("🚘 차량 분석", expanded=True):
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
        with st.expander("🚨 사망사고 분석", expanded=True):
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
        with st.expander("⏱️ 상황별 사고 분석", expanded=True):
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

