import streamlit as st
import pandas as pd
import geopandas as gpd
import numpy as np

import plotly.express as px
import plotly.graph_objects as go
import altair as alt

import folium
import json

from streamlit_folium import st_folium

# ==================================================
# 페이지 설정
# ==================================================

st.set_page_config(
    page_title="열섬현상 대시보드",
    layout="wide"
)

# ==================================================
# 디자인 CSS
# ==================================================

st.markdown(
    """
    <style>
    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    .main-header {
        background: linear-gradient(135deg, #fff7ec 0%, #fef0d9 45%, #fdd49e 100%);
        padding: 28px 32px;
        border-radius: 18px;
        border: 1px solid #f3c27a;
        margin-bottom: 28px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
    }

    .main-title {
        font-size: 38px;
        font-weight: 850;
        color: #222222;
        margin-bottom: 8px;
    }

    .main-subtitle {
        font-size: 16px;
        color: #555555;
        line-height: 1.6;
    }

    .small-caption {
        color: #777777;
        font-size: 13px;
        margin-top: -8px;
        margin-bottom: 12px;
    }

    div[data-testid="stMetric"] {
        background-color: #ffffff;
        border: 1px solid #eeeeee;
        padding: 16px 18px;
        border-radius: 14px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.04);
    }

    div[data-testid="stMetricValue"] {
        font-size: 30px;
        font-weight: 750;
        color: #222222;
    }

    hr {
        margin-top: 2rem;
        margin-bottom: 2rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ==================================================
# 데이터 불러오기
# ==================================================

@st.cache_data
def load_data():
    df = pd.read_parquet(
        "data/monthly_from_hourly_master.parquet"
    )

    df["city"] = df["city"].astype(str).str.lower().str.strip()
    df["adm_cd"] = df["adm_cd"].astype(str).str.replace(".0", "", regex=False).str.strip()
    df["adm_nm"] = df["adm_nm"].astype(str).str.strip()
    df["year"] = df["year"].astype(int)
    df["month"] = df["month"].astype(int)

    df["date"] = pd.to_datetime(
        df["year"].astype(str) + "-" + df["month"].astype(str) + "-01"
    )

    return df


@st.cache_data
def load_hiri_top30():
    try:
        hiri = pd.read_csv(
            "data/22_HIRI_위험지역_TOP30.csv",
            encoding="utf-8-sig"
        )
        hiri["행정동코드"] = hiri["행정동코드"].astype(str).str.replace(".0", "", regex=False).str.strip()
        return hiri
    except Exception:
        return pd.DataFrame()

@st.cache_data
def load_model_summary():
    try:
        return pd.read_csv(
            "data/22_모델성능_요약표.csv",
            encoding="utf-8-sig"
        )
    except Exception:
        return pd.DataFrame()


@st.cache_data
def load_compression_compare():
    try:
        return pd.read_csv(
            "data/22_압축방식_비교표.csv",
            encoding="utf-8-sig"
        )
    except Exception:
        return pd.DataFrame()


@st.cache_data
def load_policy_effect():
    try:
        return pd.read_excel(
            "data/22_보고서용_결과표.xlsx",
            sheet_name="정책효과_TOP30"
        )
    except Exception:
        return pd.DataFrame()

@st.cache_data
def load_prediction_all_targets():
    try:
        return pd.read_csv(
            "data/dashboard_prediction_all_targets.csv",
            encoding="utf-8-sig"
        )
    except Exception:
        return pd.DataFrame()


@st.cache_data
def load_prediction_long_format():
    try:
        return pd.read_csv(
            "data/dashboard_prediction_long_format.csv",
            encoding="utf-8-sig"
        )
    except Exception:
        return pd.DataFrame()


@st.cache_data
def load_simulation_example():
    try:
        return pd.read_csv(
            "data/dashboard_simulation_example.csv",
            encoding="utf-8-sig"
        )
    except Exception:
        return pd.DataFrame()


@st.cache_data
def load_model_performance_all_targets():
    try:
        return pd.read_csv(
            "data/dashboard_model_performance_all_targets.csv",
            encoding="utf-8-sig"
        )
    except Exception:
        return pd.DataFrame()


@st.cache_data
def load_ppt_modeling_best_table():
    try:
        return pd.read_csv(
            "data/29_ppt_modeling_best_table.csv",
            encoding="utf-8-sig"
        )
    except Exception:
        return pd.DataFrame()

@st.cache_data
def load_geojson():
    with open(
        "data/seoul_busan_naju_adm_base.geojson",
        "r",
        encoding="utf-8-sig"
    ) as f:
        geojson_data = json.load(f)

    for feature in geojson_data["features"]:
        props = feature.get("properties", {})
        clean_props = {}

        for k, v in props.items():
            clean_key = str(k).replace("\ufeff", "").strip()

            if isinstance(v, str):
                clean_value = v.replace("\ufeff", "").strip()
            else:
                clean_value = v

            clean_props[clean_key] = clean_value

        feature["properties"] = clean_props

    return geojson_data


df = load_data()
hiri_top30 = load_hiri_top30()
prediction_all = load_prediction_all_targets()
prediction_long = load_prediction_long_format()
simulation_example = load_simulation_example()
model_performance_all = load_model_performance_all_targets()
ppt_modeling_best = load_ppt_modeling_best_table()
model_summary = load_model_summary()
compression_compare = load_compression_compare()
policy_effect = load_policy_effect()
geojson_data = load_geojson()

# ==================================================
# 보조 함수
# ==================================================

CITY_LABEL = {
    "seoul": "서울",
    "busan": "부산",
    "naju": "나주"
}

CITY_REVERSE = {
    "서울": "seoul",
    "부산": "busan",
    "나주": "naju"
}


def clean_text(value):
    return str(value).replace("\ufeff", "").strip()


def clean_adm_cd(value):
    value = clean_text(value)

    if value.endswith(".0"):
        value = value.replace(".0", "")

    return value


def minmax(series):
    if series.max() == series.min():
        return series * 0
    return (series - series.min()) / (series.max() - series.min())


def add_hiri_columns(data):
    data = data.copy()

    data["HIRI"] = (
        minmax(data["night_temp_mean"]) * 0.25
        + minmax(data["night_tropical_ratio"]) * 0.25
        + minmax(data["tropical_hour_ratio"]) * 0.20
        + minmax(data["impervious_ratio"]) * 0.10
        + minmax(data["building_density"]) * 0.10
        + (1 - minmax(data["green_ratio"])) * 0.10
    ) * 100

    data["risk_level"] = data["HIRI"].apply(get_hiri_level)
    data["risk_color"] = data["risk_level"].apply(get_risk_color)

    return data


def get_hiri_level(hiri):
    if hiri >= 75:
        return "매우 높음"
    elif hiri >= 50:
        return "높음"
    elif hiri >= 25:
        return "보통"
    else:
        return "낮음"


def get_risk_color(level):
    if level == "매우 높음":
        return "#d73027"
    elif level == "높음":
        return "#fc8d59"
    elif level == "보통":
        return "#fee08b"
    else:
        return "#91cf60"


def get_polygon_center(geometry):
    coords = geometry["coordinates"]

    if geometry["type"] == "Polygon":
        first_polygon = coords[0]
    elif geometry["type"] == "MultiPolygon":
        first_polygon = coords[0][0]
    else:
        return [36.5, 127.8]

    lon_list = [p[0] for p in first_polygon]
    lat_list = [p[1] for p in first_polygon]

    return [
        sum(lat_list) / len(lat_list),
        sum(lon_list) / len(lon_list)
    ]


def safe_percent_diff(value, avg):
    if pd.isna(value) or pd.isna(avg) or avg == 0:
        return None
    return ((value - avg) / avg) * 100


def safe_change_rate(first, last):
    if pd.isna(first) or pd.isna(last) or first == 0:
        return None
    return ((last - first) / first) * 100


def convert_df_to_csv(download_df):
    return download_df.to_csv(index=False, encoding="utf-8-sig")


def get_rank_in_city(data, adm_cd, column, ascending=False):
    temp = data.copy()
    temp["adm_cd_clean"] = temp["adm_cd"].apply(clean_adm_cd)
    temp = temp.sort_values(column, ascending=ascending).reset_index(drop=True)
    temp["rank"] = temp.index + 1

    selected = temp[temp["adm_cd_clean"] == clean_adm_cd(adm_cd)]

    if selected.empty:
        return None

    return int(selected["rank"].iloc[0])


def get_percentile_text(data, value, column, high_is_bad=True):
    if column not in data.columns or pd.isna(value):
        return None

    valid = data[column].dropna()

    if len(valid) == 0:
        return None

    percentile = (valid <= value).mean() * 100

    if high_is_bad:
        upper_percent = 100 - percentile
        return f"상위 {upper_percent:.1f}%"
    else:
        lower_percent = percentile
        return f"하위 {lower_percent:.1f}%"


def add_risk_legend(map_object):
    legend_html = """
    <div style="
        position: fixed;
        bottom: 40px;
        left: 40px;
        width: 170px;
        background-color: white;
        border: 2px solid #555;
        z-index: 9999;
        font-size: 14px;
        padding: 12px;
        border-radius: 8px;
        box-shadow: 2px 2px 6px rgba(0,0,0,0.3);
    ">
    <b>월별 HIRI 위험등급</b><br>
    <span style="color:#91cf60;">■</span> 낮음<br>
    <span style="color:#fee08b;">■</span> 보통<br>
    <span style="color:#fc8d59;">■</span> 높음<br>
    <span style="color:#d73027;">■</span> 매우 높음
    </div>
    """
    map_object.get_root().html.add_child(folium.Element(legend_html))


df = add_hiri_columns(df)

# ==================================================
# 상단 헤더
# ==================================================

st.markdown(
    """
    <div class="main-header">
        <div class="main-title">🌡️ 도시 열섬현상 분석 대시보드</div>
        <div class="main-subtitle">
            서울·부산·나주 행정동 단위의 월별 열섬위험지수(HIRI), 공간 분포, 원인 지표, 시계열 변화와 도시 비교를 확인하는 분석형 대시보드입니다.<br>
            현재 대시보드는 시간별 자료를 월별로 집계한 최종 데이터셋을 기반으로 구성되었습니다.
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown("""
<div class='analysis-card'>
<h3>📊 분석 데이터 개요</h3>

<b>분석 지역</b><br>
서울 · 부산 · 나주<br><br>

<b>분석 단위</b><br>
행정동 약 650개<br><br>

<b>분석 기간</b><br>
2018 ~ 2024<br><br>

<b>활용 데이터</b><br>
ERA5 기상자료<br>
인구 데이터<br>
건물 밀도<br>
도로 밀도<br>
녹지 면적<br>
불투수면 비율

</div>
""", unsafe_allow_html=True)

# ==================================================
# HIRI 산정 방식
# ==================================================

st.markdown("## 🧭 HIRI 산정 방식")

st.markdown(
    """
    HIRI는 열 노출, 열 지속성, 도시 구조, 냉각 능력, 인구 노출을 통합하여
    행정동 단위의 열섬 위험도를 정량화한 지표입니다.
    """
)

st.image(
    "data/KakaoTalk_20260614_180649063_03.png",
    caption="HIRI 구성 요소 및 가중치",
    use_container_width=True
)

st.divider()

# ==================================================
# 사이드바
# ==================================================

city_display_options = [
    CITY_LABEL.get(c, c)
    for c in sorted(df["city"].dropna().unique())
]

city_display = st.sidebar.selectbox(
    "도시 선택",
    city_display_options
)

city = CITY_REVERSE.get(city_display, city_display)

city_df = df[df["city"] == city]

st.sidebar.markdown("---")

search_text = st.sidebar.text_input(
    "🔍 행정동 검색",
    placeholder="예: 화곡1동, 중앙동, 남평읍"
)

dong_list = sorted(city_df["adm_nm"].dropna().unique())

if search_text:
    searched_dong_list = [
        d for d in dong_list
        if search_text.strip() in d
    ]

    if searched_dong_list:
        dong_options = searched_dong_list
    else:
        st.sidebar.warning("검색 결과가 없습니다.")
        dong_options = dong_list
else:
    dong_options = dong_list

dong = st.sidebar.selectbox(
    "행정동 선택",
    dong_options
)

dong_df = city_df[city_df["adm_nm"] == dong].sort_values("date")

year = st.sidebar.selectbox(
    "연도 선택",
    sorted(dong_df["year"].dropna().unique())
)

year_df = dong_df[dong_df["year"] == year]

month = st.sidebar.selectbox(
    "월 선택",
    sorted(year_df["month"].dropna().unique())
)

row = year_df[year_df["month"] == month].iloc[0]

city_month_df = city_df[
    (city_df["year"] == year)
    & (city_df["month"] == month)
].copy()

# ==================================================
# 1. 선택 행정동 요약
# ==================================================

st.header(f"📍 {city_display} {dong} ({year}년 {month}월)")
st.markdown(
    '<div class="small-caption">선택한 행정동의 월별 핵심 열환경 지표와 HIRI 위험등급 요약입니다.</div>',
    unsafe_allow_html=True
)

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric("월평균기온", f"{row['temp_2m_c_mean']:.2f}℃")

with col2:
    st.metric("야간평균기온", f"{row['night_temp_mean']:.2f}℃")

with col3:
    st.metric("열대야비율", f"{row['night_tropical_ratio']:.3f}")

with col4:
    st.metric("월별 HIRI", f"{row['HIRI']:.1f}")

with col5:
    st.metric("위험등급", row["risk_level"])

st.divider()

# ==================================================
# 2. 도시 내 상대 순위
# ==================================================

st.subheader("🏅 도시 내 상대 순위")
st.markdown(
    '<div class="small-caption">선택한 도시·연도·월 기준으로 행정동의 상대적 위치를 보여줍니다.</div>',
    unsafe_allow_html=True
)

total_count = city_month_df["adm_cd"].nunique()

risk_rank = get_rank_in_city(
    city_month_df,
    row["adm_cd"],
    "HIRI",
    ascending=False
)

temp_rank = get_rank_in_city(
    city_month_df,
    row["adm_cd"],
    "temp_2m_c_mean",
    ascending=False
)

tropical_rank = get_rank_in_city(
    city_month_df,
    row["adm_cd"],
    "night_tropical_ratio",
    ascending=False
)

green_rank = get_rank_in_city(
    city_month_df,
    row["adm_cd"],
    "green_ratio",
    ascending=True
)

rank_col1, rank_col2, rank_col3, rank_col4 = st.columns(4)

with rank_col1:
    st.metric("HIRI 순위", f"{risk_rank}위 / {total_count}개")

with rank_col2:
    st.metric("평균기온 순위", f"{temp_rank}위 / {total_count}개")

with rank_col3:
    st.metric("열대야비율 순위", f"{tropical_rank}위 / {total_count}개")

with rank_col4:
    st.metric("녹지부족 순위", f"{green_rank}위 / {total_count}개")

st.caption("녹지부족 순위는 녹지비율이 낮을수록 높은 순위로 표시됩니다.")

st.divider()

# ==================================================
# 3. 선택 행정동 공간 분포 + HIRI 해석
# ==================================================

left_col, right_col = st.columns([1.35, 1])

with left_col:
    st.subheader("🗺️ 선택 행정동 공간 분포")

    try:
        selected_adm_cd = clean_adm_cd(row["adm_cd"])
        selected_adm_nm = clean_text(row["adm_nm"])
        selected_city_kr = CITY_LABEL.get(row["city"], row["city"])

        selected_features = []

        for feature in geojson_data["features"]:
            props = feature.get("properties", {})

            geo_adm_cd = clean_adm_cd(props.get("adm_cd", ""))
            geo_adm_nm = clean_text(props.get("adm_nm", ""))
            geo_city = clean_text(props.get("city", ""))

            if (
                geo_adm_cd == selected_adm_cd
                or (
                    geo_adm_nm == selected_adm_nm
                    and geo_city == selected_city_kr
                )
            ):
                selected_features.append(feature)

        if selected_features:
            selected_geojson = {
                "type": "FeatureCollection",
                "features": selected_features
            }

            center_lat, center_lon = get_polygon_center(
                selected_features[0]["geometry"]
            )

            m = folium.Map(
                location=[center_lat, center_lon],
                zoom_start=14,
                tiles="CartoDB positron"
            )

            folium.GeoJson(
                selected_geojson,
                tooltip=f"{city_display} {dong}",
                style_function=lambda x: {
                    "fillColor": row["risk_color"],
                    "color": "#222222",
                    "weight": 4,
                    "fillOpacity": 0.55
                }
            ).add_to(m)

    

            st_folium(m, width=760, height=520, key="selected_dong_map")

        else:
            st.warning("선택한 행정동을 GeoJSON에서 찾지 못했습니다.")

    except Exception as e:
        st.error(f"지도 로딩 오류: {e}")


with right_col:
    st.subheader("🔥 HIRI 해석")

    if row["risk_level"] in ["매우 높음", "높음"]:
        st.error(f"{dong}은(는) 월별 HIRI 기준 열섬 위험이 높은 지역입니다.")
    elif row["risk_level"] == "보통":
        st.warning(f"{dong}은(는) 일부 열섬 지표가 높은 편입니다.")
    else:
        st.success(f"{dong}은(는) 상대적으로 열섬 위험이 낮은 지역입니다.")

    st.markdown(
        """
        **월별 HIRI 산정 기준**

        - 야간평균기온
        - 야간 열대야비율
        - 전체 고온시간비율
        - 불투수면비율
        - 건물밀도
        - 녹지비율의 완화 효과

        위 변수들을 0에서 1 사이의 범위로 정규화한 뒤 가중합하여 0점에서 100점 사이의 지수로 환산했습니다.
        """
    )

st.divider()

# ==================================================
# 4. 선택 행정동 환경 지표
# ==================================================

st.subheader("🌳 선택 행정동 환경 지표")
st.markdown(
    '<div class="small-caption">열섬 발생과 관련된 주요 도시·토지피복 지표입니다.</div>',
    unsafe_allow_html=True
)

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("녹지비율", f"{row['green_ratio']:.3f}")

with col2:
    st.metric("불투수면비율", f"{row['impervious_ratio']:.3f}")

with col3:
    st.metric("도로비율", f"{row['road_ratio']:.3f}")

with col4:
    st.metric("건물밀도", f"{row['building_density']:.3f}")

st.divider()

# ==================================================
# 5. 도시 평균 대비 원인 지표 분석
# ==================================================

st.subheader("🔎 도시 평균 대비 원인 지표 분석")

city_avg = {
    "night_temp_mean": city_month_df["night_temp_mean"].mean(),
    "night_tropical_ratio": city_month_df["night_tropical_ratio"].mean(),
    "tropical_hour_ratio": city_month_df["tropical_hour_ratio"].mean(),
    "green_ratio": city_month_df["green_ratio"].mean(),
    "impervious_ratio": city_month_df["impervious_ratio"].mean(),
    "road_ratio": city_month_df["road_ratio"].mean(),
    "building_density": city_month_df["building_density"].mean(),
    "total_pop": city_month_df["total_pop"].mean()
}

cause_comments = []

variables = [
    ("night_temp_mean", "야간평균기온", "high_bad"),
    ("night_tropical_ratio", "열대야비율", "high_bad"),
    ("tropical_hour_ratio", "고온시간비율", "high_bad"),
    ("green_ratio", "녹지비율", "low_bad"),
    ("impervious_ratio", "불투수면비율", "high_bad"),
    ("road_ratio", "도로비율", "high_bad"),
    ("building_density", "건물밀도", "high_bad"),
    ("total_pop", "인구", "high_bad"),
]

for col, label, direction in variables:
    diff = safe_percent_diff(row[col], city_avg[col])

    if diff is None:
        continue

    if direction == "high_bad" and diff > 0:
        percentile = get_percentile_text(city_month_df, row[col], col, high_is_bad=True)
        cause_comments.append(
            f"• **{label}**이 도시 평균보다 **{diff:.1f}% 높고**, 도시 내 **{percentile}** 수준입니다."
        )

    if direction == "low_bad" and diff < 0:
        percentile = get_percentile_text(city_month_df, row[col], col, high_is_bad=False)
        cause_comments.append(
            f"• **{label}**이 도시 평균보다 **{abs(diff):.1f}% 낮고**, 도시 내 **{percentile}** 수준입니다."
        )

if cause_comments:
    st.info("선택한 행정동을 같은 도시·같은 연도·같은 월 평균과 비교해 해석했습니다.")
    for comment in cause_comments:
        st.markdown(comment)
else:
    st.success("선택한 행정동은 주요 열섬 원인 지표가 도시 평균과 유사하거나 상대적으로 양호한 수준입니다.")

st.divider()

# ==================================================
# 정책 시뮬레이션
# ==================================================

st.subheader("🧪 정책 시뮬레이션")
st.markdown(
    '<div class="small-caption">녹지비율, 불투수면비율, 도로비율, 건물밀도 변화에 따른 월별 HIRI 변화를 간단히 추정합니다.</div>',
    unsafe_allow_html=True
)

st.info(
    """
    이 시뮬레이션은 현재 대시보드에서 사용 중인 월별 HIRI 계산식을 바탕으로 한 즉석 재계산 방식입니다.
    실제 도시정책 효과를 확정적으로 예측하는 모델은 아니며, 변수 변화에 따른 상대적 방향성을 확인하는 용도입니다.
    """
)

sim_col1, sim_col2 = st.columns(2)

with sim_col1:
    green_change = st.slider(
        "녹지비율 변화율",
        min_value=-50,
        max_value=100,
        value=20,
        step=5,
        help="예: 20은 현재 녹지비율을 20% 증가시키는 것을 의미합니다."
    )

    impervious_change = st.slider(
        "불투수면비율 변화율",
        min_value=-50,
        max_value=100,
        value=-10,
        step=5,
        help="예: -10은 현재 불투수면비율을 10% 감소시키는 것을 의미합니다."
    )

with sim_col2:
    road_change = st.slider(
        "도로비율 변화율",
        min_value=-50,
        max_value=100,
        value=0,
        step=5
    )

    building_change = st.slider(
        "건물밀도 변화율",
        min_value=-50,
        max_value=100,
        value=0,
        step=5
    )


def apply_change(value, change_percent):
    changed = value * (1 + change_percent / 100)
    return max(changed, 0)


sim_row = row.copy()

sim_row["green_ratio"] = apply_change(row["green_ratio"], green_change)
sim_row["impervious_ratio"] = apply_change(row["impervious_ratio"], impervious_change)
sim_row["road_ratio"] = apply_change(row["road_ratio"], road_change)
sim_row["building_density"] = apply_change(row["building_density"], building_change)

# 비율형 변수는 1을 넘지 않도록 제한
for ratio_col in ["green_ratio", "impervious_ratio", "road_ratio"]:
    sim_row[ratio_col] = min(sim_row[ratio_col], 1)


# 현재 선택된 도시/연도/월 데이터 기준으로 정규화 범위 계산
def normalize_value(value, series):
    min_val = series.min()
    max_val = series.max()

    if max_val == min_val:
        return 0

    return (value - min_val) / (max_val - min_val)


def calculate_hiri_for_row(target_row, base_df):
    night_temp_score = normalize_value(
        target_row["night_temp_mean"],
        base_df["night_temp_mean"]
    )

    night_tropical_score = normalize_value(
        target_row["night_tropical_ratio"],
        base_df["night_tropical_ratio"]
    )

    tropical_hour_score = normalize_value(
        target_row["tropical_hour_ratio"],
        base_df["tropical_hour_ratio"]
    )

    impervious_score = normalize_value(
        target_row["impervious_ratio"],
        base_df["impervious_ratio"]
    )

    building_score = normalize_value(
        target_row["building_density"],
        base_df["building_density"]
    )

    green_score = 1 - normalize_value(
        target_row["green_ratio"],
        base_df["green_ratio"]
    )

    hiri_value = (
        night_temp_score * 0.25
        + night_tropical_score * 0.25
        + tropical_hour_score * 0.20
        + impervious_score * 0.10
        + building_score * 0.10
        + green_score * 0.10
    ) * 100

    return max(min(hiri_value, 100), 0)


current_hiri = row["HIRI"]
sim_hiri = calculate_hiri_for_row(sim_row, city_month_df)
hiri_diff = sim_hiri - current_hiri

current_level = get_hiri_level(current_hiri)
sim_level = get_hiri_level(sim_hiri)

st.markdown("#### 시뮬레이션 결과")

result_col1, result_col2, result_col3, result_col4 = st.columns(4)

with result_col1:
    st.metric("현재 HIRI", f"{current_hiri:.1f}")

with result_col2:
    st.metric("시뮬레이션 HIRI", f"{sim_hiri:.1f}", delta=f"{hiri_diff:.1f}")

with result_col3:
    st.metric("현재 위험등급", current_level)

with result_col4:
    st.metric("변경 후 위험등급", sim_level)

if hiri_diff < 0:
    st.success(f"시뮬레이션 결과 HIRI가 **{abs(hiri_diff):.1f}점 감소**하는 것으로 추정됩니다.")
elif hiri_diff > 0:
    st.warning(f"시뮬레이션 결과 HIRI가 **{hiri_diff:.1f}점 증가**하는 것으로 추정됩니다.")
else:
    st.info("시뮬레이션 전후 HIRI 변화가 거의 없습니다.")

sim_compare_df = pd.DataFrame({
    "지표": ["녹지비율", "불투수면비율", "도로비율", "건물밀도"],
    "현재값": [
        row["green_ratio"],
        row["impervious_ratio"],
        row["road_ratio"],
        row["building_density"]
    ],
    "변경 후": [
        sim_row["green_ratio"],
        sim_row["impervious_ratio"],
        sim_row["road_ratio"],
        sim_row["building_density"]
    ]
})

st.dataframe(
    sim_compare_df.round(4),
    use_container_width=True
)

st.divider()

# ==================================================
# 미래 3년 HIRI 예측
# ==================================================

import numpy as np

st.subheader("🔮 미래 3년 HIRI 예측")
st.markdown(
    '<div class="small-caption">선택 행정동의 과거 월별 HIRI 흐름을 바탕으로 향후 3년의 같은 월 HIRI를 단순 추세 방식으로 예측합니다.</div>',
    unsafe_allow_html=True
)

target_month_history = dong_df[dong_df["month"] == month].copy()
target_month_history = target_month_history.sort_values("year")

if len(target_month_history) >= 3:
    x = target_month_history["year"].values
    y = target_month_history["HIRI"].values

    coef = np.polyfit(x, y, 1)
    trend_model = np.poly1d(coef)

    future_years = [2025, 2026, 2027]

    pred_rows = []

    for future_year in future_years:
        pred_hiri = float(trend_model(future_year))
        pred_hiri = max(min(pred_hiri, 100), 0)

        pred_rows.append({
            "year": future_year,
            "month": month,
            "예측_HIRI": pred_hiri,
            "예측_위험등급": get_hiri_level(pred_hiri)
        })

    pred_df = pd.DataFrame(pred_rows)

    pred_col1, pred_col2, pred_col3 = st.columns(3)

    with pred_col1:
        st.metric("2025년 예측 HIRI", f"{pred_df.iloc[0]['예측_HIRI']:.1f}", pred_df.iloc[0]["예측_위험등급"])

    with pred_col2:
        st.metric("2026년 예측 HIRI", f"{pred_df.iloc[1]['예측_HIRI']:.1f}", pred_df.iloc[1]["예측_위험등급"])

    with pred_col3:
        st.metric("2027년 예측 HIRI", f"{pred_df.iloc[2]['예측_HIRI']:.1f}", pred_df.iloc[2]["예측_위험등급"])

    pred_chart_df = pd.concat([
        target_month_history[["year", "month", "HIRI"]].rename(columns={"HIRI": "HIRI"}),
        pred_df.rename(columns={"예측_HIRI": "HIRI"})[["year", "month", "HIRI"]]
    ])

    pred_chart_df["구분"] = [
        "실측" if y <= 2024 else "예측"
        for y in pred_chart_df["year"]
    ]

    pred_chart = (
        alt.Chart(pred_chart_df)
        .mark_line(point=True)
        .encode(
            x=alt.X("year:O", title="연도"),
            y=alt.Y("HIRI:Q", title="HIRI"),
            color=alt.Color("구분:N", title="구분"),
            tooltip=[
                alt.Tooltip("year:O", title="연도"),
                alt.Tooltip("month:O", title="월"),
                alt.Tooltip("HIRI:Q", title="HIRI", format=".1f"),
                alt.Tooltip("구분:N", title="구분")
            ]
        )
        .properties(height=330)
    )

    st.altair_chart(pred_chart, use_container_width=True)

    st.caption("현재 예측은 선형 추세 기반의 간단한 예측입니다. 최종 보고서에서는 탐색적 예측 결과로 해석하는 것이 좋습니다.")

else:
    st.warning("예측에 필요한 동일 월 과거 데이터가 부족합니다.")

st.divider()

# ==================================================
# 예측 기반 정책 시뮬레이션
# ==================================================

st.subheader("🧪 예측 기반 정책 시뮬레이션")
st.markdown(
    '<div class="small-caption">현재 선택한 정책 변화가 향후 HIRI 예측값에 어떤 영향을 줄 수 있는지 추정합니다.</div>',
    unsafe_allow_html=True
)

if len(target_month_history) >= 3:
    base_future_hiri = float(pred_df[pred_df["year"] == 2027]["예측_HIRI"].iloc[0])

    current_sim_hiri = calculate_hiri_for_row(sim_row, city_month_df)
    current_policy_effect = current_sim_hiri - current_hiri

    future_policy_hiri = base_future_hiri + current_policy_effect
    future_policy_hiri = max(min(future_policy_hiri, 100), 0)

    future_level = get_hiri_level(base_future_hiri)
    future_policy_level = get_hiri_level(future_policy_hiri)

    sim_future_col1, sim_future_col2, sim_future_col3 = st.columns(3)

    with sim_future_col1:
        st.metric("2027년 기준 예측 HIRI", f"{base_future_hiri:.1f}", future_level)

    with sim_future_col2:
        st.metric(
            "정책 적용 후 2027년 HIRI",
            f"{future_policy_hiri:.1f}",
            delta=f"{future_policy_hiri - base_future_hiri:.1f}"
        )

    with sim_future_col3:
        st.metric("정책 적용 후 위험등급", future_policy_level)

    if future_policy_hiri < base_future_hiri:
        st.success(f"정책 시뮬레이션 적용 시 2027년 HIRI가 약 **{base_future_hiri - future_policy_hiri:.1f}점 감소**할 수 있습니다.")
    elif future_policy_hiri > base_future_hiri:
        st.warning(f"정책 시뮬레이션 적용 시 2027년 HIRI가 약 **{future_policy_hiri - base_future_hiri:.1f}점 증가**할 수 있습니다.")
    else:
        st.info("정책 적용 전후의 2027년 예측 HIRI 변화가 거의 없습니다.")

else:
    st.warning("예측 기반 정책 시뮬레이션을 수행하기 위한 과거 데이터가 부족합니다.")

st.divider()

# ==================================================
# 6. 월별 시계열 변화
# ==================================================

st.subheader("📈 선택 행정동 월별 시계열 변화")

plot_df = dong_df.copy()

c1, c2 = st.columns(2)

with c1:
    st.write("월별 HIRI 변화")
    hiri_chart = (
        alt.Chart(plot_df)
        .mark_line(point=True)
        .encode(
            x=alt.X("date:T", title="시점"),
            y=alt.Y("HIRI:Q", title="HIRI"),
            tooltip=[
                alt.Tooltip("year:O", title="연도"),
                alt.Tooltip("month:O", title="월"),
                alt.Tooltip("HIRI:Q", title="HIRI", format=".1f")
            ]
        )
        .properties(height=300)
    )
    st.altair_chart(hiri_chart, use_container_width=True)

with c2:
    st.write("월별 평균기온 변화")
    temp_chart = (
        alt.Chart(plot_df)
        .mark_line(point=True)
        .encode(
            x=alt.X("date:T", title="시점"),
            y=alt.Y("temp_2m_c_mean:Q", title="평균기온(℃)"),
            tooltip=[
                alt.Tooltip("year:O", title="연도"),
                alt.Tooltip("month:O", title="월"),
                alt.Tooltip("temp_2m_c_mean:Q", title="평균기온", format=".2f")
            ]
        )
        .properties(height=300)
    )
    st.altair_chart(temp_chart, use_container_width=True)

st.divider()

# ==================================================
# 7. 사용자 선택형 그래프
# ==================================================

st.subheader("📊 사용자 선택형 그래프")

available_graph_columns = {
    "HIRI": "HIRI",
    "월평균기온": "temp_2m_c_mean",
    "야간평균기온": "night_temp_mean",
    "열대야비율": "night_tropical_ratio",
    "고온시간비율": "tropical_hour_ratio",
    "녹지비율": "green_ratio",
    "불투수면비율": "impervious_ratio",
    "도로비율": "road_ratio",
    "건물밀도": "building_density",
    "인구": "total_pop"
}

graph_label = st.selectbox(
    "그래프로 확인할 지표 선택",
    list(available_graph_columns.keys())
)

graph_col = available_graph_columns[graph_label]

custom_chart = (
    alt.Chart(plot_df)
    .mark_line(point=True)
    .encode(
        x=alt.X("date:T", title="시점"),
        y=alt.Y(f"{graph_col}:Q", title=graph_label),
        tooltip=[
            alt.Tooltip("year:O", title="연도"),
            alt.Tooltip("month:O", title="월"),
            alt.Tooltip(f"{graph_col}:Q", title=graph_label, format=".3f")
        ]
    )
    .properties(height=330)
)

st.altair_chart(custom_chart, use_container_width=True)

st.divider()

# ==================================================
# 8. 월별 추세 해석
# ==================================================

st.subheader("📊 월별 추세 해석")

first_row = plot_df.iloc[0]
last_row = plot_df.iloc[-1]

hiri_change = safe_change_rate(first_row["HIRI"], last_row["HIRI"])
temp_change = safe_change_rate(first_row["temp_2m_c_mean"], last_row["temp_2m_c_mean"])
tropical_change = safe_change_rate(first_row["night_tropical_ratio"], last_row["night_tropical_ratio"])

hiri_change = 0 if hiri_change is None else hiri_change
temp_change = 0 if temp_change is None else temp_change
tropical_change = 0 if tropical_change is None else tropical_change

st.info(
    f"""
    **{city_display} {dong}의 {int(first_row['year'])}년 {int(first_row['month'])}월 → {int(last_row['year'])}년 {int(last_row['month'])}월 변화**

    - HIRI 변화율: **{hiri_change:.1f}%**
    - 평균기온 변화율: **{temp_change:.1f}%**
    - 열대야비율 변화율: **{tropical_change:.1f}%**
    """
)

if hiri_change > 0 and tropical_change > 0:
    st.warning("HIRI와 열대야비율이 함께 증가해 열환경 부담이 커지는 흐름으로 해석할 수 있습니다.")
elif hiri_change > 0:
    st.warning("HIRI가 상승하는 흐름을 보이며, 열섬 위험도 관리가 필요합니다.")
else:
    st.success("선택 기간 동안 HIRI가 급격히 악화되지는 않은 것으로 보입니다.")

st.divider()

# ==================================================
# 9. 선택 도시 전체 HIRI 위험도 지도
# ==================================================

st.subheader(f"🗺️ {city_display} {year}년 {month}월 전체 행정동 HIRI 지도")
st.markdown(
    '<div class="small-caption">붉은색에 가까울수록 월별 HIRI가 높은 행정동입니다.</div>',
    unsafe_allow_html=True
)

risk_lookup = {
    clean_adm_cd(r["adm_cd"]): {
        "adm_nm": r["adm_nm"],
        "HIRI": r["HIRI"],
        "risk_level": r["risk_level"],
        "risk_color": r["risk_color"],
        "temp_2m_c_mean": r["temp_2m_c_mean"],
        "night_tropical_ratio": r["night_tropical_ratio"]
    }
    for _, r in city_month_df.iterrows()
}

city_features = []

for feature in geojson_data["features"]:
    props = feature.get("properties", {})

    geo_city = clean_text(props.get("city", ""))
    geo_adm_cd = clean_adm_cd(props.get("adm_cd", ""))

    if geo_city == city_display and geo_adm_cd in risk_lookup:
        feature["properties"]["HIRI"] = round(risk_lookup[geo_adm_cd]["HIRI"], 1)
        feature["properties"]["risk_level"] = risk_lookup[geo_adm_cd]["risk_level"]
        feature["properties"]["risk_color"] = risk_lookup[geo_adm_cd]["risk_color"]
        feature["properties"]["temp_2m_c_mean"] = round(risk_lookup[geo_adm_cd]["temp_2m_c_mean"], 2)
        feature["properties"]["night_tropical_ratio"] = round(risk_lookup[geo_adm_cd]["night_tropical_ratio"], 3)

        city_features.append(feature)

if city_features:
    city_geojson = {
        "type": "FeatureCollection",
        "features": city_features
    }

    center_points = [
        get_polygon_center(feature["geometry"])
        for feature in city_features
    ]

    map_center_lat = sum([p[0] for p in center_points]) / len(center_points)
    map_center_lon = sum([p[1] for p in center_points]) / len(center_points)

    city_map = folium.Map(
        location=[map_center_lat, map_center_lon],
        zoom_start=11,
        tiles="CartoDB dark_matter"
    )

    folium.GeoJson(
        city_geojson,
        style_function=lambda feature: {
            "fillColor": feature["properties"].get("risk_color", "#cccccc"),
            "color": "#f5f5f5",
            "weight": 0.8,
            "fillOpacity": 0.72
        },
        tooltip=folium.GeoJsonTooltip(
            fields=[
                "adm_nm",
                "HIRI",
                "risk_level",
                "temp_2m_c_mean",
                "night_tropical_ratio"
            ],
            aliases=[
                "행정동",
                "HIRI",
                "위험등급",
                "평균기온",
                "열대야비율"
            ],
            localize=True
        )
    ).add_to(city_map)

    add_risk_legend(city_map)

    st_folium(city_map, width=1200, height=560, key="city_hiri_map")

else:
    st.warning("선택한 도시/연도/월에 해당하는 지도 데이터를 찾지 못했습니다.")

st.divider()

# ==================================================
# 위험지역 변화 탐색 지도
# ==================================================

st.subheader("🕹️ 위험지역 변화 탐색 지도")
st.markdown(
    '<div class="small-caption">연도와 월을 바꿔가며 도시 전체 HIRI 공간 분포가 어떻게 달라지는지 확인할 수 있습니다.</div>',
    unsafe_allow_html=True
)

map_col1, map_col2 = st.columns(2)

with map_col1:
    explore_year = st.selectbox(
        "변화 지도 연도 선택",
        sorted(city_df["year"].dropna().unique()),
        index=sorted(city_df["year"].dropna().unique()).index(year)
    )

with map_col2:
    explore_month = st.selectbox(
        "변화 지도 월 선택",
        sorted(city_df["month"].dropna().unique()),
        index=sorted(city_df["month"].dropna().unique()).index(month)
    )

explore_df = city_df[
    (city_df["year"] == explore_year)
    & (city_df["month"] == explore_month)
].copy()

explore_lookup = {
    clean_adm_cd(r["adm_cd"]): {
        "adm_nm": r["adm_nm"],
        "HIRI": r["HIRI"],
        "risk_level": r["risk_level"],
        "risk_color": r["risk_color"],
        "temp_2m_c_mean": r["temp_2m_c_mean"],
        "night_tropical_ratio": r["night_tropical_ratio"]
    }
    for _, r in explore_df.iterrows()
}

explore_features = []

for feature in geojson_data["features"]:
    props = feature.get("properties", {})

    geo_city = clean_text(props.get("city", ""))
    geo_adm_cd = clean_adm_cd(props.get("adm_cd", ""))

    if geo_city == city_display and geo_adm_cd in explore_lookup:
        feature["properties"]["HIRI"] = round(explore_lookup[geo_adm_cd]["HIRI"], 1)
        feature["properties"]["risk_level"] = explore_lookup[geo_adm_cd]["risk_level"]
        feature["properties"]["risk_color"] = explore_lookup[geo_adm_cd]["risk_color"]
        feature["properties"]["temp_2m_c_mean"] = round(explore_lookup[geo_adm_cd]["temp_2m_c_mean"], 2)
        feature["properties"]["night_tropical_ratio"] = round(explore_lookup[geo_adm_cd]["night_tropical_ratio"], 3)

        explore_features.append(feature)

if explore_features:
    explore_geojson = {
        "type": "FeatureCollection",
        "features": explore_features
    }

    center_points = [
        get_polygon_center(feature["geometry"])
        for feature in explore_features
    ]

    explore_center_lat = sum([p[0] for p in center_points]) / len(center_points)
    explore_center_lon = sum([p[1] for p in center_points]) / len(center_points)

    explore_map = folium.Map(
        location=[explore_center_lat, explore_center_lon],
        zoom_start=11,
        tiles="CartoDB dark_matter"
    )

    folium.GeoJson(
        explore_geojson,
        style_function=lambda feature: {
            "fillColor": feature["properties"].get("risk_color", "#cccccc"),
            "color": "#f5f5f5",
            "weight": 0.8,
            "fillOpacity": 0.72
        },
        tooltip=folium.GeoJsonTooltip(
            fields=[
                "adm_nm",
                "HIRI",
                "risk_level",
                "temp_2m_c_mean",
                "night_tropical_ratio"
            ],
            aliases=[
                "행정동",
                "HIRI",
                "위험등급",
                "평균기온",
                "열대야비율"
            ],
            localize=True
        )
    ).add_to(explore_map)

    add_risk_legend(explore_map)

    st_folium(explore_map, width=1200, height=560, key="explore_hiri_map")

else:
    st.warning("선택한 연도/월에 해당하는 지도 데이터를 찾지 못했습니다.")

st.divider()

# ==================================================
# 10. HIRI TOP 10
# ==================================================

st.subheader(f"🏆 {city_display} {year}년 {month}월 HIRI TOP 10")

ranking_df = city_month_df.sort_values(
    by="HIRI",
    ascending=False
).head(10)

ranking_display_df = ranking_df[
    [
        "adm_nm",
        "HIRI",
        "risk_level",
        "temp_2m_c_mean",
        "night_temp_mean",
        "night_tropical_ratio",
        "tropical_hour_ratio",
        "green_ratio",
        "impervious_ratio",
        "road_ratio",
        "building_density",
        "total_pop"
    ]
]

st.dataframe(
    ranking_display_df.round(4),
    use_container_width=True
)

st.divider()

# ==================================================
# 11. 도시 비교
# ==================================================

st.subheader(f"🏙️ {year}년 {month}월 도시별 열섬 지표 비교")

compare_df = df[
    (df["year"] == year)
    & (df["month"] == month)
].copy()

city_compare = compare_df.groupby("city").agg(
    평균기온=("temp_2m_c_mean", "mean"),
    야간평균기온=("night_temp_mean", "mean"),
    열대야비율=("night_tropical_ratio", "mean"),
    평균HIRI=("HIRI", "mean"),
    평균녹지비율=("green_ratio", "mean"),
    평균불투수면비율=("impervious_ratio", "mean"),
    평균도로비율=("road_ratio", "mean"),
    평균건물밀도=("building_density", "mean")
).reset_index()

city_compare["도시"] = city_compare["city"].map(CITY_LABEL)

st.dataframe(
    city_compare.drop(columns=["city"]).round(4),
    use_container_width=True
)

chart_col1, chart_col2 = st.columns(2)

with chart_col1:
    st.markdown("#### 🌡️ 도시별 평균기온")

    temp_chart = (
        alt.Chart(city_compare)
        .mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6)
        .encode(
            x=alt.X("도시:N", title="도시", sort="-y"),
            y=alt.Y("평균기온:Q", title="평균기온(℃)"),
            tooltip=[
                alt.Tooltip("도시:N", title="도시"),
                alt.Tooltip("평균기온:Q", title="평균기온", format=".2f")
            ]
        )
        .properties(height=330)
    )

    temp_text = temp_chart.mark_text(
        align="center",
        baseline="bottom",
        dy=-5
    ).encode(
        text=alt.Text("평균기온:Q", format=".2f")
    )

    st.altair_chart(temp_chart + temp_text, use_container_width=True)

with chart_col2:
    st.markdown("#### 🔥 도시별 평균 HIRI")

    hiri_chart = (
        alt.Chart(city_compare)
        .mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6)
        .encode(
            x=alt.X("도시:N", title="도시", sort="-y"),
            y=alt.Y("평균HIRI:Q", title="평균 HIRI"),
            tooltip=[
                alt.Tooltip("도시:N", title="도시"),
                alt.Tooltip("평균HIRI:Q", title="평균 HIRI", format=".1f")
            ]
        )
        .properties(height=330)
    )

    hiri_text = hiri_chart.mark_text(
        align="center",
        baseline="bottom",
        dy=-5
    ).encode(
        text=alt.Text("평균HIRI:Q", format=".1f")
    )

    st.altair_chart(hiri_chart + hiri_text, use_container_width=True)

st.divider()

# ==================================================
# 예측 결과 요약
# ==================================================

st.subheader("🔮 예측 결과 요약")
st.markdown(
    '<div class="small-caption">행정동별 연간 HIRI 및 주요 열섬 지표의 실제값과 예측값을 비교한 결과입니다.</div>',
    unsafe_allow_html=True
)

if not prediction_all.empty:
    pred_city_df = prediction_all[prediction_all["city"] == city].copy()

    if "adm_nm" in pred_city_df.columns:
        pred_dong_df = pred_city_df[pred_city_df["adm_nm"] == dong].copy()
    else:
        pred_dong_df = pd.DataFrame()

    if not pred_dong_df.empty:
        latest_pred = pred_dong_df.sort_values("year").iloc[-1]

        pcol1, pcol2, pcol3, pcol4 = st.columns(4)

        with pcol1:
            st.metric(
                "실제 연간 HIRI",
                f"{latest_pred['HIRI_YEAR_FROM_DAILY_actual']:.1f}",
                latest_pred.get("HIRI_actual_grade", "")
            )

        with pcol2:
            st.metric(
                "예측 연간 HIRI",
                f"{latest_pred['HIRI_YEAR_FROM_DAILY_pred']:.1f}",
                latest_pred.get("HIRI_pred_grade", "")
            )

        with pcol3:
            st.metric(
                "예측 오차",
                f"{latest_pred['HIRI_YEAR_FROM_DAILY_error']:.1f}"
            )

        with pcol4:
            st.metric(
                "절대오차",
                f"{latest_pred['HIRI_YEAR_FROM_DAILY_abs_error']:.1f}"
            )

        pred_display_cols = [
            "city_kor",
            "adm_nm",
            "year",
            "night_temp_mean_actual",
            "night_temp_mean_pred",
            "tropical_hours_actual",
            "tropical_hours_pred",
            "night_tropical_hours_actual",
            "night_tropical_hours_pred",
            "HIRI_YEAR_FROM_DAILY_actual",
            "HIRI_YEAR_FROM_DAILY_pred",
            "HIRI_actual_grade",
            "HIRI_pred_grade"
        ]

        pred_display_cols = [
            c for c in pred_display_cols
            if c in pred_dong_df.columns
        ]

        st.dataframe(
            pred_dong_df[pred_display_cols].sort_values("year").round(3),
            use_container_width=True
        )

    else:
        st.info("선택한 행정동의 예측 결과가 없습니다.")

else:
    st.info("예측 결과 파일을 찾지 못했습니다.")

st.divider()

# ==================================================
# 실제값 vs 예측값 검증
# ==================================================

st.subheader("📈 실제값 vs 예측값 검증")
st.markdown(
    '<div class="small-caption">선택 행정동의 예측 대상별 실제값과 예측값을 비교합니다.</div>',
    unsafe_allow_html=True
)

if not prediction_long.empty:
    long_city_df = prediction_long[prediction_long["city"] == city].copy()
    long_dong_df = long_city_df[long_city_df["adm_nm"] == dong].copy()

    if not long_dong_df.empty:
        target_options = (
            long_dong_df[["target", "target_label"]]
            .drop_duplicates()
            .sort_values("target_label")
        )

        target_label_list = target_options["target_label"].tolist()

        selected_target_label = st.selectbox(
            "예측 검증 지표 선택",
            target_label_list
        )

        selected_target = target_options[
            target_options["target_label"] == selected_target_label
        ]["target"].iloc[0]

        target_df = long_dong_df[
            long_dong_df["target"] == selected_target
        ].sort_values("year")

        actual_df = target_df[["year", "actual"]].rename(
            columns={"actual": "value"}
        )
        actual_df["구분"] = "실제값"

        pred_df_line = target_df[["year", "prediction"]].rename(
            columns={"prediction": "value"}
        )
        pred_df_line["구분"] = "예측값"

        compare_line_df = pd.concat(
            [actual_df, pred_df_line],
            ignore_index=True
        )

        unit_text = target_df["unit"].iloc[0] if "unit" in target_df.columns else ""

        compare_chart = (
            alt.Chart(compare_line_df)
            .mark_line(point=True)
            .encode(
                x=alt.X("year:O", title="연도"),
                y=alt.Y("value:Q", title=f"{selected_target_label}({unit_text})"),
                color=alt.Color("구분:N", title="구분"),
                tooltip=[
                    alt.Tooltip("year:O", title="연도"),
                    alt.Tooltip("구분:N", title="구분"),
                    alt.Tooltip("value:Q", title=selected_target_label, format=".2f")
                ]
            )
            .properties(height=360)
        )

        st.altair_chart(compare_chart, use_container_width=True)

        error_mean = target_df["abs_error"].mean()

        st.info(
            f"선택한 지표의 평균 절대오차는 **{error_mean:.3f}{unit_text}**입니다."
        )

    else:
        st.info("선택한 행정동의 실제값-예측값 비교 데이터가 없습니다.")

else:
    st.info("실제값-예측값 검증 파일을 찾지 못했습니다.")

st.divider()

# ==================================================
# 모델 성능 요약
# ==================================================

st.subheader("🤖 예측 모델 성능 요약")
st.markdown(
    '<div class="small-caption">예측 대상별 최적 모델과 성능지표를 요약한 결과입니다.</div>',
    unsafe_allow_html=True
)

if not model_summary.empty:
    st.dataframe(
        model_summary.round(4),
        use_container_width=True
    )

    if "R2" in model_summary.columns and "예측대상" in model_summary.columns:
        model_chart = (
            alt.Chart(model_summary)
            .mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6)
            .encode(
                x=alt.X("예측대상:N", title="예측 대상", sort="-y"),
                y=alt.Y("R2:Q", title="R²"),
                tooltip=[
                    alt.Tooltip("예측대상:N", title="예측 대상"),
                    alt.Tooltip("최적모델:N", title="최적 모델"),
                    alt.Tooltip("R2:Q", title="R²", format=".4f"),
                    alt.Tooltip("MAE:Q", title="MAE", format=".4f"),
                    alt.Tooltip("RMSE:Q", title="RMSE", format=".4f")
                ]
            )
            .properties(height=330)
        )

        text = model_chart.mark_text(
            align="center",
            baseline="bottom",
            dy=-5
        ).encode(
            text=alt.Text("R2:Q", format=".2f")
        )

        st.altair_chart(model_chart + text, use_container_width=True)

else:
    st.info("모델 성능 요약 파일을 찾지 못했습니다.")

st.divider()

# ==================================================
# 데이터 압축방식 비교
# ==================================================

st.subheader("🧩 데이터 압축방식 비교")
st.markdown(
    '<div class="small-caption">월별·연도별·특성벡터 등 데이터 구성 방식별 예측 성능을 비교한 결과입니다.</div>',
    unsafe_allow_html=True
)

if not compression_compare.empty:
    st.dataframe(
        compression_compare.round(4),
        use_container_width=True
    )

    if "평균_R2" in compression_compare.columns and "압축방식" in compression_compare.columns:
        compression_chart = (
            alt.Chart(compression_compare)
            .mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6)
            .encode(
                x=alt.X("압축방식:N", title="압축방식", sort="-y"),
                y=alt.Y("평균_R2:Q", title="평균 R²"),
                tooltip=[
                    alt.Tooltip("압축방식:N", title="압축방식"),
                    alt.Tooltip("압축단위:N", title="압축단위"),
                    alt.Tooltip("평균_R2:Q", title="평균 R²", format=".4f"),
                    alt.Tooltip("평균_MAE:Q", title="평균 MAE", format=".4f"),
                    alt.Tooltip("평균_RMSE:Q", title="평균 RMSE", format=".4f")
                ]
            )
            .properties(height=330)
        )

        text = compression_chart.mark_text(
            align="center",
            baseline="bottom",
            dy=-5
        ).encode(
            text=alt.Text("평균_R2:Q", format=".2f")
        )

        st.altair_chart(compression_chart + text, use_container_width=True)

else:
    st.info("압축방식 비교 파일을 찾지 못했습니다.")

st.divider()

# ==================================================
# 다중 예측모델 성능 비교
# ==================================================

st.subheader("🤖 다중 예측모델 성능 비교")
st.markdown(
    '<div class="small-caption">연간 HIRI, 야간기온, 열대야시간 등 여러 예측 대상에 대한 모델 성능 비교 결과입니다.</div>',
    unsafe_allow_html=True
)

if not model_performance_all.empty:
    st.dataframe(
        model_performance_all.round(4),
        use_container_width=True
    )

    if "test_r2" in model_performance_all.columns:
        performance_best = (
            model_performance_all
            .sort_values("test_r2", ascending=False)
            .groupby("target_label")
            .head(1)
            .copy()
        )

        perf_chart = (
            alt.Chart(performance_best)
            .mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6)
            .encode(
                x=alt.X("target_label:N", title="예측 대상", sort="-y"),
                y=alt.Y("test_r2:Q", title="Test R²"),
                tooltip=[
                    alt.Tooltip("target_label:N", title="예측 대상"),
                    alt.Tooltip("model:N", title="모델"),
                    alt.Tooltip("test_r2:Q", title="Test R²", format=".4f"),
                    alt.Tooltip("test_mae:Q", title="MAE", format=".4f"),
                    alt.Tooltip("test_rmse:Q", title="RMSE", format=".4f")
                ]
            )
            .properties(height=360)
        )

        text = perf_chart.mark_text(
            align="center",
            baseline="bottom",
            dy=-5
        ).encode(
            text=alt.Text("test_r2:Q", format=".2f")
        )

        st.altair_chart(perf_chart + text, use_container_width=True)

else:
    st.info("다중 예측모델 성능 파일을 찾지 못했습니다.")

st.divider()

# ==================================================
# 최종 모델 선정 결과
# ==================================================

st.subheader("🏆 최종 모델 선정 결과")
st.markdown(
    '<div class="small-caption">여러 예측 단위와 변수 조합을 비교하여 최종 예측 모델을 선정한 결과입니다.</div>',
    unsafe_allow_html=True
)

if not ppt_modeling_best.empty:
    best_model_row = ppt_modeling_best.iloc[0]

    st.success(
        f"최종적으로 **{best_model_row['압축방식']}** 방식과 "
        f"**{best_model_row['최적모델']}** 모델이 가장 우수한 성능을 보여 "
        f"주요 예측 모델로 선정되었습니다."
    )

    best_summary_df = pd.DataFrame({
        "구분": ["최종 압축방식", "분석 단위", "예측 대상", "최적 모델", "Test R²"],
        "내용": [
            best_model_row["압축방식"],
            best_model_row["분석단위"],
            best_model_row["예측대상"],
            best_model_row["최적모델"],
            best_model_row["Test R² 평균±표준편차"]
        ]
    })

    st.dataframe(
        best_summary_df,
        use_container_width=True,
        hide_index=True
    )

else:
    st.info("최종 모델 선정 결과 파일을 찾지 못했습니다.")

st.divider()

# ==================================================
# 발표용 최적 모델 요약
# ==================================================

st.subheader("🏆 발표용 최적 모델 요약")
st.markdown(
    '<div class="small-caption">분석 단위와 예측 대상별 최적 모델을 요약한 표입니다.</div>',
    unsafe_allow_html=True
)

if not ppt_modeling_best.empty:
    st.dataframe(
        ppt_modeling_best,
        use_container_width=True
    )
else:
    st.info("발표용 모델 요약 파일을 찾지 못했습니다.")

st.divider()

# ==================================================
# 정책 시뮬레이션 예시 결과
# ==================================================

st.subheader("🌿 정책 시뮬레이션 예시 결과")
st.markdown(
    '<div class="small-caption">정책 적용 전후의 주요 열섬 지표 변화 예시입니다.</div>',
    unsafe_allow_html=True
)

if not simulation_example.empty:
    sim_ex = simulation_example.iloc[0]

    st.markdown(
        f"**예시 지역: {sim_ex['city_kor']} {sim_ex['adm_nm']} ({int(sim_ex['year'])}년)**"
    )

    scol1, scol2, scol3, scol4 = st.columns(4)

    with scol1:
        st.metric(
            "HIRI 변화",
            f"{sim_ex['HIRI_YEAR_FROM_DAILY_after']:.1f}",
            delta=f"{sim_ex['HIRI_YEAR_FROM_DAILY_change']:.1f}"
        )

    with scol2:
        st.metric(
            "야간평균기온 변화",
            f"{sim_ex['night_temp_mean_after']:.2f}℃",
            delta=f"{sim_ex['night_temp_mean_change']:.3f}℃"
        )

    with scol3:
        st.metric(
            "열대야시간 변화",
            f"{sim_ex['tropical_hours_after']:.1f}시간",
            delta=f"{sim_ex['tropical_hours_change']:.1f}시간"
        )

    with scol4:
        st.metric(
            "회복실패일수 변화",
            f"{sim_ex['recovery_failure_days_after']:.1f}일",
            delta=f"{sim_ex['recovery_failure_days_change']:.1f}일"
        )

    sim_change_df = pd.DataFrame({
        "지표": [
            "HIRI",
            "야간평균기온",
            "열대야시간",
            "야간열대야시간",
            "야간냉각량",
            "회복실패일수"
        ],
        "정책 전": [
            sim_ex["HIRI_YEAR_FROM_DAILY_before"],
            sim_ex["night_temp_mean_before"],
            sim_ex["tropical_hours_before"],
            sim_ex["night_tropical_hours_before"],
            sim_ex["cooling_amount_mean_before"],
            sim_ex["recovery_failure_days_before"]
        ],
        "정책 후": [
            sim_ex["HIRI_YEAR_FROM_DAILY_after"],
            sim_ex["night_temp_mean_after"],
            sim_ex["tropical_hours_after"],
            sim_ex["night_tropical_hours_after"],
            sim_ex["cooling_amount_mean_after"],
            sim_ex["recovery_failure_days_after"]
        ],
        "변화량": [
            sim_ex["HIRI_YEAR_FROM_DAILY_change"],
            sim_ex["night_temp_mean_change"],
            sim_ex["tropical_hours_change"],
            sim_ex["night_tropical_hours_change"],
            sim_ex["cooling_amount_mean_change"],
            sim_ex["recovery_failure_days_change"]
        ]
    })

    st.dataframe(
        sim_change_df.round(4),
        use_container_width=True
    )

else:
    st.info("정책 시뮬레이션 예시 파일을 찾지 못했습니다.")

st.divider()

# ==================================================
# 정책효과 TOP30
# ==================================================

st.subheader("🌿 정책효과 TOP30")
st.markdown(
    '<div class="small-caption">녹지 확대, 불투수면 감소, 건물밀도 조정 등 정책 적용 전후 HIRI 변화가 큰 지역을 정리한 결과입니다.</div>',
    unsafe_allow_html=True
)

if not policy_effect.empty:
    st.dataframe(
        policy_effect.round(4),
        use_container_width=True
    )

    possible_change_cols = [
        "HIRI_변화량",
        "hiri_변화량",
        "정책효과",
        "변화량"
    ]

    change_col = None
    for col in possible_change_cols:
        if col in policy_effect.columns:
            change_col = col
            break

    name_col = "행정동명" if "행정동명" in policy_effect.columns else None

    if change_col and name_col:
        policy_chart_df = policy_effect.head(10).copy()

        policy_chart = (
            alt.Chart(policy_chart_df)
            .mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6)
            .encode(
                x=alt.X(f"{name_col}:N", title="행정동", sort="-y"),
                y=alt.Y(f"{change_col}:Q", title="HIRI 변화량"),
                tooltip=[
                    alt.Tooltip(f"{name_col}:N", title="행정동"),
                    alt.Tooltip(f"{change_col}:Q", title="HIRI 변화량", format=".2f")
                ]
            )
            .properties(height=330)
        )

        st.altair_chart(policy_chart, use_container_width=True)

else:
    st.info("정책효과 TOP30 시트를 찾지 못했습니다.")

st.divider()

# ==================================================
# 12. HIRI 위험지역 TOP30
# ==================================================

st.subheader("🚨 연도별 HIRI 위험지역 TOP30")

if not hiri_top30.empty:
    st.caption("별도 산출된 연도별 HIRI 위험지역 TOP30 결과표입니다.")

    st.dataframe(
        hiri_top30.round(4),
        use_container_width=True
    )

else:
    st.info("HIRI TOP30 파일을 찾지 못했습니다.")

st.divider()

# ==================================================
# 13. 데이터 다운로드
# ==================================================

st.subheader("⬇️ 데이터 다운로드")

download_col1, download_col2, download_col3 = st.columns(3)

with download_col1:
    st.download_button(
        label="선택 행정동 월별 데이터 다운로드",
        data=convert_df_to_csv(dong_df),
        file_name=f"{city_display}_{dong}_monthly_data.csv",
        mime="text/csv"
    )

with download_col2:
    st.download_button(
        label="HIRI TOP10 다운로드",
        data=convert_df_to_csv(ranking_display_df),
        file_name=f"{city_display}_{year}_{month}_hiri_top10.csv",
        mime="text/csv"
    )

with download_col3:
    st.download_button(
        label="도시 비교 데이터 다운로드",
        data=convert_df_to_csv(city_compare),
        file_name=f"{year}_{month}_city_compare.csv",
        mime="text/csv"
    )

st.divider()

# ==================================================
# 14. 원본 데이터
# ==================================================

with st.expander("🧾 선택 행정동 전체 월별 데이터 보기"):
    st.dataframe(
        dong_df.round(4),
        use_container_width=True
    )

