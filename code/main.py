
import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path

# =====================================
# 1. 기본 설정
# =====================================

st.set_page_config(
    page_title="국내 애니메이션산업 데이터 분석",
    page_icon="🎬",
    layout="wide"
)

st.title("🎬 국내 애니메이션산업 데이터 분석")
st.write(
    "2020~2022년 국내 애니메이션산업의 "
    "지역별 매출액, 종사자 수, 사업체 수를 분석합니다."
)

st.divider()

# 데이터 파일 위치
DATA_DIR = Path("data")


# =====================================
# 2. CSV 데이터 불러오기
# =====================================

@st.cache_data
def load_csv(filename):
    path = DATA_DIR / filename

    if not path.exists():
        raise FileNotFoundError(
            f"데이터 파일을 찾을 수 없습니다: {path}"
        )

    try:
        return pd.read_csv(path, encoding="utf-8-sig")
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="cp949")


try:
    revenue_raw = load_csv("매출액_연도세로형.csv")
    employee_raw = load_csv("종사자수_연도세로형.csv")
    company_raw = load_csv(
        "애니메이션산업_지역별_연도별_사업체수_세로형(1).csv"
    )
    job_raw = load_csv("직무별_연도세로형.csv")

    st.success("4개의 데이터 파일을 성공적으로 불러왔습니다.")

except Exception as e:
    st.error(f"데이터 불러오기 오류: {e}")
    st.stop()


# =====================================
# 3. 데이터 전처리
# =====================================

def clean_regional_data(df, value_col):
    df = df.copy()

    required = ["지역", "COL2", "연도", "값"]

    if not all(col in df.columns for col in required):
        raise ValueError(
            f"지역별 데이터의 열 구성이 올바르지 않습니다: {df.columns.tolist()}"
        )

    # 반복된 제목 행 제거
    df["지역"] = df["지역"].astype(str).str.strip()
    df = df[df["지역"] != "지역"]

    # 지역명 통합
    df["지역명"] = df["COL2"].fillna("").astype(str).str.strip()

    df.loc[
        df["지역명"].isin(["", "nan", "None"]),
        "지역명"
    ] = df["지역"]

    # 지역명 표준화
    region_map = {
        "경기": "경기도",
        "강원": "강원도",
        "충북": "충청북도",
        "충남": "충청남도",
        "전북": "전라북도",
        "전남": "전라남도",
        "경북": "경상북도",
        "경남": "경상남도",
        "제주": "제주도"
    }

    df["지역명"] = df["지역명"].replace(region_map)

    # 합계 및 소계 제거
    excluded = [
        "합계",
        "소계",
        "7개시",
        "9개 도",
        "9개도"
    ]

    df = df[~df["지역명"].isin(excluded)]

    # 숫자형 변환
    df["연도"] = pd.to_numeric(
        df["연도"], errors="coerce"
    )

    df[value_col] = pd.to_numeric(
        df["값"], errors="coerce"
    )

    df = df.dropna(
        subset=["지역명", "연도", value_col]
    )

    # 중복된 지역-연도 자료 확인
    df = df.drop_duplicates(
        subset=["지역명", "연도"],
        keep="first"
    )

    return df[
        ["지역명", "연도", value_col]
    ]


def clean_company_data(df):
    df = df.copy()

    required = ["지역", "연도", "사업체수"]

    if not all(col in df.columns for col in required):
        raise ValueError(
            f"사업체 데이터의 열 구성이 올바르지 않습니다: {df.columns.tolist()}"
        )

    df["지역"] = df["지역"].astype(str).str.strip()

    region_map = {
        "경기": "경기도",
        "강원": "강원도",
        "충북": "충청북도",
        "충남": "충청남도",
        "전북": "전라북도",
        "전남": "전라남도",
        "경북": "경상북도",
        "경남": "경상남도",
        "제주": "제주도"
    }

    df["지역"] = df["지역"].replace(region_map)

    excluded = [
        "합계",
        "소계",
        "7개시",
        "9개 도",
        "9개도"
    ]

    df = df[~df["지역"].isin(excluded)]

    df["연도"] = pd.to_numeric(
        df["연도"], errors="coerce"
    )

    df["사업체수"] = pd.to_numeric(
        df["사업체수"], errors="coerce"
    )

    df = df.dropna(
        subset=["지역", "연도", "사업체수"]
    )

    df = df.drop_duplicates(
        subset=["지역", "연도"],
        keep="first"
    )

    return df.rename(
        columns={"지역": "지역명"}
    )[["지역명", "연도", "사업체수"]]


def clean_job_data(df):
    df = df.copy()

    required = ["직무별", "연도", "값"]

    if not all(col in df.columns for col in required):
        raise ValueError(
            f"직무별 데이터의 열 구성이 올바르지 않습니다: {df.columns.tolist()}"
        )

    # 반복된 제목 행 제거
    df["직무별"] = df["직무별"].astype(str).str.strip()
    df = df[df["직무별"] != "직무별"]

    df["연도"] = pd.to_numeric(
        df["연도"], errors="coerce"
    )

    df["값"] = pd.to_numeric(
        df["값"], errors="coerce"
    )

    df = df.dropna(
        subset=["직무별", "연도", "값"]
    )

    return df


try:
    revenue = clean_regional_data(
        revenue_raw, "매출액"
    )

    employee = clean_regional_data(
        employee_raw, "종사자수"
    )

    company = clean_company_data(company_raw)

    job = clean_job_data(job_raw)

except Exception as e:
    st.error(f"데이터 전처리 오류: {e}")
    st.stop()


# =====================================
# 4. 지역별 데이터 결합
# =====================================

data = pd.merge(
    revenue,
    employee,
    on=["지역명", "연도"],
    how="inner",
    validate="one_to_one"
)

data = pd.merge(
    data,
    company,
    on=["지역명", "연도"],
    how="inner",
    validate="one_to_one"
)

if data.empty:
    st.error(
        "지역명과 연도가 일치하는 데이터가 없습니다. "
        "각 CSV 파일의 지역명과 연도 형식을 확인해 주세요."
    )
    st.stop()

data = data.sort_values(
    ["연도", "지역명"]
).reset_index(drop=True)


# =====================================
# 5. 사이드바
# =====================================

st.sidebar.title("📊 분석 설정")

years = sorted(data["연도"].unique())

selected_year = st.sidebar.selectbox(
    "분석 연도 선택",
    years,
    index=len(years) - 1
)

selected_data = data[
    data["연도"] == selected_year
]


# =====================================
# 6. 주요 통계
# =====================================

st.header("1. 주요 통계")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "지역별 매출액 합계",
        f'{selected_data["매출액"].sum():,.0f}'
    )

with col2:
    st.metric(
        "지역별 종사자 수 합계",
        f'{selected_data["종사자수"].sum():,.0f}명'
    )

with col3:
    st.metric(
        "지역별 사업체 수 합계",
        f'{selected_data["사업체수"].sum():,.0f}개'
    )

st.caption(
    "개별 지역의 값을 합산한 수치입니다. "
    "매출액 단위는 원자료의 단위를 따릅니다."
)


# =====================================
# 7. 지역별 데이터 비교
# =====================================

st.divider()
st.header("2. 지역별 데이터 비교")

metric = st.selectbox(
    "비교할 항목",
    ["매출액", "종사자수", "사업체수"]
)

fig1 = px.bar(
    selected_data.sort_values(
        metric, ascending=False
    ),
    x="지역명",
    y=metric,
    color="지역명",
    text_auto=".2s",
    title=f"{selected_year}년 지역별 {metric} 비교",
    labels={
        "지역명": "지역",
        "매출액": "매출액",
        "종사자수": "종사자 수",
        "사업체수": "사업체 수"
    }
)

fig1.update_layout(
    showlegend=False,
    xaxis_title="지역",
    yaxis_title=metric
)

st.plotly_chart(
    fig1,
    use_container_width=True
)


# =====================================
# 8. 연도별 변화 추이
# =====================================

st.divider()
st.header("3. 연도별 변화 추이")

trend_metric = st.selectbox(
    "변화 추이를 확인할 항목",
    ["매출액", "종사자수", "사업체수"],
    key="trend"
)

fig2 = px.line(
    data,
    x="연도",
    y=trend_metric,
    color="지역명",
    markers=True,
    title=f"지역별 {trend_metric} 변화 추이",
    labels={
        "연도": "연도",
        "매출액": "매출액",
        "종사자수": "종사자 수",
        "사업체수": "사업체 수"
    }
)

st.plotly_chart(
    fig2,
    use_container_width=True
)


# =====================================
# 9. 매출액과 종사자 수의 관계
# =====================================

st.divider()
st.header("4. 매출액과 종사자 수의 관계")

fig3 = px.scatter(
    data,
    x="종사자수",
    y="매출액",
    color="연도",
    size="사업체수",
    hover_name="지역명",
    title="지역별 종사자 수와 매출액의 관계",
    labels={
        "종사자수": "종사자 수",
        "매출액": "매출액",
        "사업체수": "사업체 수"
    }
)

st.plotly_chart(
    fig3,
    use_container_width=True
)

st.write(
    "산점도를 통해 종사자 수와 매출액 사이의 관계를 "
    "확인할 수 있습니다. 다만 상관관계가 나타나더라도 "
    "그것만으로 인과관계를 입증할 수는 없습니다."
)


# =====================================
# 10. 데이터 분포 확인
# =====================================

st.divider()
st.header("5. 지역별 데이터 분포")

distribution_metric = st.selectbox(
    "분포를 확인할 항목",
    ["매출액", "종사자수", "사업체수"],
    key="distribution"
)

fig4 = px.box(
    data,
    x="연도",
    y=distribution_metric,
    points="all",
    title=f"연도별 지역 {distribution_metric} 분포",
    labels={
        "연도": "연도",
        "매출액": "매출액",
        "종사자수": "종사자 수",
        "사업체수": "사업체 수"
    }
)

st.plotly_chart(
    fig4,
    use_container_width=True
)


# =====================================
# 11. 직무별 종사자 현황
# =====================================

st.divider()
st.header("6. 직무별 종사자 현황")

job_years = sorted(job["연도"].unique())

selected_job_year = st.selectbox(
    "직무별 현황 연도 선택",
    job_years,
    index=len(job_years) - 1
)

job_selected = job[
    (job["연도"] == selected_job_year)
    & (job["직무별"] != "합계")
]

fig5 = px.bar(
    job_selected,
    x="직무별",
    y="값",
    color="직무별",
    text_auto=True,
    title=f"{selected_job_year}년 직무별 종사자 수",
    labels={
        "직무별": "직무",
        "값": "종사자 수"
    }
)

fig5.update_layout(
    showlegend=False,
    xaxis_title="직무",
    yaxis_title="종사자 수"
)

st.plotly_chart(
    fig5,
    use_container_width=True
)


# =====================================
# 12. 전처리된 데이터 확인
# =====================================

st.divider()
st.header("7. 전처리된 데이터 확인")

st.subheader("지역별 통합 데이터")

st.dataframe(
    data,
    use_container_width=True
)

st.download_button(
    label="지역별 통합 데이터 다운로드",
    data=data.to_csv(
        index=False,
        encoding="utf-8-sig"
    ),
    file_name="애니메이션산업_지역별_통합데이터.csv",
    mime="text/csv"
)

st.subheader("직무별 종사자 데이터")

st.dataframe(
    job,
    use_container_width=True
)

st.download_button(
    label="직무별 데이터 다운로드",
    data=job.to_csv(
        index=False,
        encoding="utf-8-sig"
    ),
    file_name="애니메이션산업_직무별_데이터.csv",
    mime="text/csv"
)


# =====================================
# 13. 데이터 정보
# =====================================

st.divider()
st.header("8. 데이터 정보")

info_col1, info_col2 = st.columns(2)

with info_col1:
    st.write("**지역별 통합 데이터**")
    st.write(f"- 행 개수: {len(data)}")
    st.write(f"- 열 개수: {len(data.columns)}")
    st.write(f"- 분석 연도: {years}")

with info_col2:
    st.write("**직무별 데이터**")
    st.write(f"- 행 개수: {len(job)}")
    st.write(f"- 열 개수: {len(job.columns)}")
    st.write(f"- 분석 직무 수: {job['직무별'].nunique()}")

st.success("데이터 분석이 완료되었습니다.")

