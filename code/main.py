
import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path

# ==========================================
# 1. 기본 설정
# ==========================================

st.set_page_config(
    page_title="애니메이션산업 데이터 분석",
    page_icon="🎬",
    layout="wide"
)

# GitHub 저장소 구조:
# data-project/
# ├── code/
# │   ├── main.py
# │   └── requirements.txt
# └── data/
#     ├── 매출액_연도세로형.csv
#     ├── 종사자수_연도세로형.csv
#     ├── 애니메이션산업_지역별_연도별_사업체수_세로형.csv
#     └── 직무별_연도세로형.csv

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

st.title("🎬 국내 애니메이션산업 데이터 분석")
st.write(
    "2020~2022년 국내 애니메이션산업의 지역별 매출액, "
    "종사자 수, 사업체 수 및 직무별 종사자 현황을 분석합니다."
)

st.info(
    "원본 데이터와 전처리 데이터를 비교하고 "
    "통계분석 및 시각화를 통해 데이터의 특징을 확인합니다."
)

st.divider()


# ==========================================
# 2. CSV 파일 불러오기
# ==========================================

@st.cache_data
def load_csv(filename):
    file_path = DATA_DIR / filename

    if not file_path.exists():
        raise FileNotFoundError(
            f"파일을 찾을 수 없습니다: {file_path}"
        )

    try:
        return pd.read_csv(
            file_path,
            encoding="utf-8-sig"
        )
    except UnicodeDecodeError:
        return pd.read_csv(
            file_path,
            encoding="cp949"
        )


try:
    revenue_raw = load_csv("매출액_연도세로형.csv")
    employee_raw = load_csv("종사자수_연도세로형.csv")

    company_raw = load_csv(
        "애니메이션산업_지역별_연도별_사업체수_세로형.csv"
    )

    job_raw = load_csv("직무별_연도세로형.csv")

    st.success("4개의 원본 데이터 파일을 불러왔습니다.")

except Exception as e:
    st.error(f"데이터 불러오기 오류: {e}")
    st.stop()


# ==========================================
# 3. 공통 설정
# ==========================================

REGION_MAP = {
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

AGGREGATE_NAMES = [
    "합계",
    "소계",
    "7개시",
    "9개 도",
    "9개도"
]


def standardize_region(value):
    if pd.isna(value):
        return None

    value = str(value).strip()

    if value in ["", "nan", "None"]:
        return None

    return REGION_MAP.get(value, value)


def get_statistics(values):
    values = pd.to_numeric(
        pd.Series(values),
        errors="coerce"
    ).dropna()

    if len(values) == 0:
        return {
            "데이터 개수": 0,
            "평균": None,
            "중앙값": None,
            "표준편차": None,
            "최솟값": None,
            "최댓값": None
        }

    return {
        "데이터 개수": len(values),
        "평균": values.mean(),
        "중앙값": values.median(),
        "표준편차": values.std(),
        "최솟값": values.min(),
        "최댓값": values.max()
    }


# ==========================================
# 4. 지역별 매출액 및 종사자 수 전처리
# ==========================================

def clean_regional_data(df, value_name):
    df = df.copy()

    required = ["지역", "COL2", "연도", "값"]

    for col in required:
        if col not in df.columns:
            raise ValueError(
                f"{value_name} 데이터에 {col} 열이 없습니다."
            )

    original_count = len(df)

    original_values = pd.to_numeric(
        df["값"], errors="coerce"
    ).dropna()

    # 반복된 제목 행 제거
    header_mask = (
        df["지역"].astype(str).str.strip() == "지역"
    )

    header_count = int(header_mask.sum())
    df = df[~header_mask].copy()

    # 실제 지역명 추출
    df["지역명"] = (
        df["COL2"].fillna("").astype(str).str.strip()
    )

    empty_mask = df["지역명"].isin(
        ["", "nan", "None"]
    )

    df.loc[empty_mask, "지역명"] = (
        df.loc[empty_mask, "지역"].astype(str).str.strip()
    )

    df["지역명"] = df["지역명"].apply(
        standardize_region
    )

    # 합계 및 소계 제거
    aggregate_mask = df["지역명"].isin(
        AGGREGATE_NAMES
    )

    aggregate_count = int(aggregate_mask.sum())
    df = df[~aggregate_mask].copy()

    # 숫자형 변환
    df["연도"] = pd.to_numeric(
        df["연도"], errors="coerce"
    )

    df[value_name] = pd.to_numeric(
        df["값"], errors="coerce"
    )

    invalid_count = int(
        df[value_name].isna().sum()
    )

    missing_count = int(
        df[
            ["지역명", "연도", value_name]
        ].isna().any(axis=1).sum()
    )

    df = df.dropna(
        subset=["지역명", "연도", value_name]
    )

    # 정확히 같은 중복 행만 제거
    before_duplicates = len(df)

    df = df.drop_duplicates(
        subset=["지역명", "연도", value_name]
    )

    duplicate_count = (
        before_duplicates - len(df)
    )

    # 동일 지역·연도에 서로 다른 값이 있는지 검사
    if df.duplicated(
        subset=["지역명", "연도"]
    ).any():
        raise ValueError(
            f"{value_name} 데이터에 동일 지역·연도의 "
            "중복 값이 있습니다. 원본 파일을 확인해 주세요."
        )

    log = {
        "원본 행 수": original_count,
        "반복 제목 행": header_count,
        "합계·소계 행": aggregate_count,
        "숫자 변환 불가 행": invalid_count,
        "필수값 누락 행": missing_count,
        "중복 행": duplicate_count,
        "전처리 후 행 수": len(df)
    }

    return (
        df[["지역명", "연도", value_name]].copy(),
        log,
        original_values,
        df[value_name]
    )


# ==========================================
# 5. 사업체 수 전처리
# ==========================================

def clean_company_data(df):
    df = df.copy()

    original_count = len(df)

    if "연도" not in df.columns:
        raise ValueError(
            "사업체 데이터에 연도 열이 없습니다."
        )

    if "사업체수" not in df.columns:
        raise ValueError(
            "사업체 데이터에 사업체수 열이 없습니다."
        )

    if "지역" in df.columns:
        region_column = "지역"
    elif "지역구분" in df.columns:
        region_column = "지역구분"
    else:
        raise ValueError(
            "사업체 데이터에서 지역 열을 찾을 수 없습니다."
        )

    original_values = pd.to_numeric(
        df["사업체수"], errors="coerce"
    ).dropna()

    # 반복된 제목 행 제거
    header_mask = (
        df[region_column].astype(str).str.strip() == "지역"
    )

    header_count = int(header_mask.sum())
    df = df[~header_mask].copy()

    # 지역명 표준화
    df["지역명"] = df[region_column].apply(
        standardize_region
    )

    # 합계 및 소계 제거
    aggregate_mask = df["지역명"].isin(
        AGGREGATE_NAMES
    )

    aggregate_count = int(aggregate_mask.sum())
    df = df[~aggregate_mask].copy()

    # 숫자형 변환
    df["연도"] = pd.to_numeric(
        df["연도"], errors="coerce"
    )

    df["사업체수"] = pd.to_numeric(
        df["사업체수"], errors="coerce"
    )

    invalid_count = int(
        df["사업체수"].isna().sum()
    )

    missing_count = int(
        df[
            ["지역명", "연도", "사업체수"]
        ].isna().any(axis=1).sum()
    )

    df = df.dropna(
        subset=["지역명", "연도", "사업체수"]
    )

    before_duplicates = len(df)

    df = df.drop_duplicates(
        subset=["지역명", "연도", "사업체수"]
    )

    duplicate_count = (
        before_duplicates - len(df)
    )

    if df.duplicated(
        subset=["지역명", "연도"]
    ).any():
        raise ValueError(
            "사업체 수 데이터에 동일 지역·연도의 "
            "중복 값이 있습니다."
        )

    log = {
        "원본 행 수": original_count,
        "반복 제목 행": header_count,
        "합계·소계 행": aggregate_count,
        "숫자 변환 불가 행": invalid_count,
        "필수값 누락 행": missing_count,
        "중복 행": duplicate_count,
        "전처리 후 행 수": len(df)
    }

    return (
        df[["지역명", "연도", "사업체수"]].copy(),
        log,
        original_values,
        df["사업체수"]
    )


# ==========================================
# 6. 직무별 종사자 수 전처리
# ==========================================

def clean_job_data(df):
    df = df.copy()

    original_count = len(df)

    required = ["직무별", "연도", "값"]

    for col in required:
        if col not in df.columns:
            raise ValueError(
                f"직무별 데이터에 {col} 열이 없습니다."
            )

    original_values = pd.to_numeric(
        df["값"], errors="coerce"
    ).dropna()

    # 반복된 제목 행 제거
    header_mask = (
        df["직무별"].astype(str).str.strip() == "직무별"
    )

    header_count = int(header_mask.sum())
    df = df[~header_mask].copy()

    # 숫자형 변환
    df["연도"] = pd.to_numeric(
        df["연도"], errors="coerce"
    )

    df["종사자수"] = pd.to_numeric(
        df["값"], errors="coerce"
    )

    invalid_count = int(
        df["종사자수"].isna().sum()
    )

    missing_count = int(
        df[
            ["직무별", "연도", "종사자수"]
        ].isna().any(axis=1).sum()
    )

    df = df.dropna(
        subset=["직무별", "연도", "종사자수"]
    )

    before_duplicates = len(df)

    df = df.drop_duplicates(
        subset=["직무별", "연도", "종사자수"]
    )

    duplicate_count = (
        before_duplicates - len(df)
    )

    if df.duplicated(
        subset=["직무별", "연도"]
    ).any():
        raise ValueError(
            "직무별 데이터에 동일 직무·연도의 "
            "중복 값이 있습니다."
        )

    log = {
        "원본 행 수": original_count,
        "반복 제목 행": header_count,
        "숫자 변환 불가 행": invalid_count,
        "필수값 누락 행": missing_count,
        "중복 행": duplicate_count,
        "전처리 후 행 수": len(df)
    }

    return (
        df[["직무별", "연도", "종사자수"]].copy(),
        log,
        original_values,
        df["종사자수"]
    )


# ==========================================
# 7. 전처리 실행
# ==========================================

try:
    revenue, revenue_log, revenue_before, revenue_after = (
        clean_regional_data(revenue_raw, "매출액")
    )

    employee, employee_log, employee_before, employee_after = (
        clean_regional_data(employee_raw, "종사자수")
    )

    company, company_log, company_before, company_after = (
        clean_company_data(company_raw)
    )

    job, job_log, job_before, job_after = (
        clean_job_data(job_raw)
    )

except Exception as e:
    st.error(f"전처리 오류: {e}")
    st.stop()


# ==========================================
# 8. 전처리 기록
# ==========================================

st.header("1. 원본 데이터와 전처리 과정")

logs = {
    "매출액": revenue_log,
    "종사자 수": employee_log,
    "사업체 수": company_log,
    "직무별 종사자 수": job_log
}

log_df = pd.DataFrame(logs).T.fillna(0)

st.subheader("전처리 과정 요약")

st.dataframe(
    log_df,
    use_container_width=True
)

st.write(
    "반복된 제목 행, 합계·소계 행, 숫자 변환이 불가능한 행과 "
    "중복 자료를 확인했습니다."
)


# ==========================================
# 9. 원본과 전처리 데이터 행 수 비교
# ==========================================

st.subheader("원본과 전처리 후 행 수 비교")

count_records = []

for name, log in logs.items():
    count_records.append({
        "데이터": name,
        "원본": log["원본 행 수"],
        "전처리 후": log["전처리 후 행 수"]
    })

count_df = pd.DataFrame(count_records)

count_long = count_df.melt(
    id_vars="데이터",
    var_name="구분",
    value_name="행 수"
)

fig_count = px.bar(
    count_long,
    x="데이터",
    y="행 수",
    color="구분",
    barmode="group",
    text_auto=True,
    title="원본과 전처리 후 데이터 행 수 비교"
)

st.plotly_chart(
    fig_count,
    use_container_width=True
)


# ==========================================
# 10. 원본과 전처리 통계 비교
# ==========================================

st.divider()
st.header("2. 원본과 전처리 데이터 통계 비교")

comparison_data = {
    "매출액": (revenue_before, revenue_after),
    "종사자 수": (employee_before, employee_after),
    "사업체 수": (company_before, company_after),
    "직무별 종사자 수": (job_before, job_after)
}

comparison_rows = []

for name, (before, after) in comparison_data.items():

    before_stats = get_statistics(before)
    after_stats = get_statistics(after)

    for stat_name in [
        "데이터 개수",
        "평균",
        "중앙값",
        "표준편차",
        "최솟값",
        "최댓값"
    ]:
        comparison_rows.append({
            "데이터": name,
            "통계 항목": stat_name,
            "원본": before_stats[stat_name],
            "전처리 후": after_stats[stat_name]
        })

comparison_df = pd.DataFrame(comparison_rows)

st.dataframe(
    comparison_df,
    use_container_width=True
)

st.caption(
    "원본과 전처리 데이터의 통계량을 비교하여 "
    "데이터 분포의 변화를 확인합니다."
)


# ==========================================
# 11. 원본과 전처리 데이터 분포 비교
# ==========================================

st.divider()
st.header("3. 전처리 전후 데이터 분포 비교")

distribution_choice = st.selectbox(
    "비교할 데이터",
    [
        "매출액",
        "종사자 수",
        "사업체 수",
        "직무별 종사자 수"
    ]
)

before_values, after_values = comparison_data[
    distribution_choice
]

box_data = pd.concat([
    pd.DataFrame({
        "구분": "원본 데이터",
        "값": before_values
    }),
    pd.DataFrame({
        "구분": "전처리 데이터",
        "값": after_values
    })
], ignore_index=True)

fig_box = px.box(
    box_data,
    x="구분",
    y="값",
    color="구분",
    points="all",
    title=f"{distribution_choice} 원본과 전처리 데이터 비교"
)

st.plotly_chart(
    fig_box,
    use_container_width=True
)


# ==========================================
# 12. 지역별 데이터 결합
# ==========================================

st.divider()
st.header("4. 지역별 데이터 통합")

merged = pd.merge(
    revenue,
    employee,
    on=["지역명", "연도"],
    how="outer",
    validate="one_to_one"
)

merged = pd.merge(
    merged,
    company,
    on=["지역명", "연도"],
    how="outer",
    validate="one_to_one"
)

# 세 자료가 모두 존재하는 행만 분석에 사용
data = merged.dropna(
    subset=["매출액", "종사자수", "사업체수"]
).copy()

unmatched = merged[
    merged[
        ["매출액", "종사자수", "사업체수"]
    ].isna().any(axis=1)
].copy()

st.write(
    f"전체 지역·연도 조합: {len(merged)}개"
)

st.write(
    f"세 자료가 모두 일치하는 조합: {len(data)}개"
)

st.write(
    f"일부 자료가 일치하지 않는 조합: {len(unmatched)}개"
)

if not unmatched.empty:
    with st.expander("일치하지 않은 데이터 확인"):
        st.dataframe(
            unmatched,
            use_container_width=True
        )

data = data.sort_values(
    ["연도", "지역명"]
).reset_index(drop=True)

if data.empty:
    st.error(
        "세 자료가 모두 일치하는 데이터가 없습니다. "
        "지역명과 연도를 확인해 주세요."
    )
    st.stop()

st.subheader("전처리된 지역별 통합 데이터")

st.dataframe(
    data,
    use_container_width=True
)


# ==========================================
# 13. 연도 선택 및 주요 통계
# ==========================================

st.divider()
st.header("5. 지역별 주요 통계")

years = sorted(data["연도"].unique())

selected_year = st.selectbox(
    "분석 연도 선택",
    years,
    index=len(years) - 1
)

selected_data = data[
    data["연도"] == selected_year
]

col1, col2, col3 = st.columns(3)

col1.metric(
    "지역별 매출액 합계",
    f'{selected_data["매출액"].sum():,.0f}'
)

col2.metric(
    "지역별 종사자 수 합계",
    f'{selected_data["종사자수"].sum():,.0f}명'
)

col3.metric(
    "지역별 사업체 수 합계",
    f'{selected_data["사업체수"].sum():,.0f}개'
)

st.caption(
    "개별 지역의 합계이며, 전국 합계와 집계 범위가 다를 수 있습니다."
)


# ==========================================
# 14. 지역별 비교 그래프
# ==========================================

st.divider()
st.header("6. 지역별 데이터 비교")

metric = st.selectbox(
    "비교할 항목",
    ["매출액", "종사자수", "사업체수"]
)

fig_region = px.bar(
    selected_data.sort_values(
        metric,
        ascending=False
    ),
    x="지역명",
    y=metric,
    color="지역명",
    text_auto=".2s",
    title=f"{selected_year}년 지역별 {metric} 비교"
)

fig_region.update_layout(
    showlegend=False,
    xaxis_title="지역",
    yaxis_title=metric
)

st.plotly_chart(
    fig_region,
    use_container_width=True
)


# ==========================================
# 15. 연도별 변화 추이
# ==========================================

st.divider()
st.header("7. 연도별 변화 추이")

trend_metric = st.selectbox(
    "변화 추이를 확인할 항목",
    ["매출액", "종사자수", "사업체수"],
    key="trend"
)

fig_trend = px.line(
    data,
    x="연도",
    y=trend_metric,
    color="지역명",
    markers=True,
    title=f"지역별 {trend_metric} 변화 추이"
)

st.plotly_chart(
    fig_trend,
    use_container_width=True
)


# ==========================================
# 16. 매출액과 종사자 수의 관계
# ==========================================

st.divider()
st.header("8. 매출액과 종사자 수의 관계")

fig_scatter = px.scatter(
    data,
    x="종사자수",
    y="매출액",
    color="연도",
    size="사업체수",
    hover_name="지역명",
    title="지역별 종사자 수와 매출액의 관계"
)

st.plotly_chart(
    fig_scatter,
    use_container_width=True
)


# ==========================================
# 17. 상관관계 분석 (SciPy 미사용)
# ==========================================

st.subheader("상관관계 통계 분석")

# 피어슨 상관계수
pearson = data["종사자수"].corr(
    data["매출액"],
    method="pearson"
)

# 스피어만 상관계수
# 각 변수에 순위를 부여한 후 피어슨 상관계수를 계산
rank_employee = data["종사자수"].rank(
    method="average"
)

rank_revenue = data["매출액"].rank(
    method="average"
)

spearman = rank_employee.corr(
    rank_revenue,
    method="pearson"
)

corr1, corr2 = st.columns(2)

corr1.metric(
    "피어슨 상관계수",
    f"{pearson:.3f}" if pd.notna(pearson) else "계산 불가"
)

corr2.metric(
    "스피어만 상관계수",
    f"{spearman:.3f}" if pd.notna(spearman) else "계산 불가"
)

st.write(
    "피어슨 상관계수는 두 변수의 선형 관계를, "
    "스피어만 상관계수는 두 변수의 순위 관계를 나타냅니다."
)

st.caption(
    "스피어만 상관계수는 각 변수에 평균 순위를 부여한 뒤 "
    "순위 간 피어슨 상관계수를 계산하는 방식으로 구했습니다. "
    "SciPy 라이브러리를 사용하지 않습니다."
)

st.warning(
    "상관관계는 두 변수의 관련성을 나타낼 뿐, "
    "인과관계를 증명하지는 않습니다. "
    "또한 지역별 자료를 합친 결과이므로 연도와 지역의 영향을 "
    "함께 고려해야 합니다."
)


# ==========================================
# 18. 연도별 지역 데이터 분포
# ==========================================

st.divider()
st.header("9. 지역별 데이터 분포")

distribution_metric = st.selectbox(
    "분포를 확인할 항목",
    ["매출액", "종사자수", "사업체수"],
    key="regional_distribution"
)

fig_distribution = px.box(
    data,
    x="연도",
    y=distribution_metric,
    points="all",
    title=f"연도별 지역 {distribution_metric} 분포"
)

st.plotly_chart(
    fig_distribution,
    use_container_width=True
)


# ==========================================
# 19. 직무별 종사자 현황
# ==========================================

st.divider()
st.header("10. 직무별 종사자 현황")

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

fig_job = px.bar(
    job_selected,
    x="직무별",
    y="종사자수",
    color="직무별",
    text_auto=True,
    title=f"{selected_job_year}년 직무별 종사자 수"
)

fig_job.update_layout(
    showlegend=False,
    xaxis_title="직무",
    yaxis_title="종사자 수"
)

st.plotly_chart(
    fig_job,
    use_container_width=True
)


# ==========================================
# 20. 전처리 데이터 다운로드
# ==========================================

st.divider()
st.header("11. 전처리 데이터 다운로드")

st.download_button(
    label="지역별 통합 데이터 다운로드",
    data=data.to_csv(
        index=False,
        encoding="utf-8-sig"
    ),
    file_name="애니메이션산업_지역별_통합데이터.csv",
    mime="text/csv"
)

st.download_button(
    label="직무별 데이터 다운로드",
    data=job.to_csv(
        index=False,
        encoding="utf-8-sig"
    ),
    file_name="애니메이션산업_직무별_전처리데이터.csv",
    mime="text/csv"
)


# ==========================================
# 21. 전처리 근거
# ==========================================

st.divider()
st.header("12. 전처리 근거 및 분석 결과")

st.markdown("""
### 데이터 전처리 기준

1. **반복된 제목 행 제거:** 데이터 중간에 반복된 열 제목 행을 제거했습니다.
2. **합계·소계 행 제거:** 개별 지역 분석에 적합하지 않은 집계 행을 지역별 분석에서 제외했습니다.
3. **지역명 표준화:** 경기와 경기도처럼 같은 지역을 다르게 표현한 경우 명칭을 통일했습니다.
4. **숫자형 변환:** 연도와 매출액, 종사자 수, 사업체 수를 숫자형으로 변환했습니다.
5. **결측값 확인:** 지역명이나 연도, 분석에 필요한 수치가 없는 행을 분석에서 제외했습니다.
6. **중복 자료 확인:** 동일 지역·연도에 서로 다른 수치가 존재하는 경우 임의로 삭제하지 않고 오류를 표시하도록 했습니다.
7. **직무별 합계 보존:** 직무별 자료의 합계는 원본 비교를 위해 보존하고 직무 구성 그래프에서는 제외했습니다.

### 전처리의 타당성 확인

원본과 전처리 데이터의 행 개수, 평균, 중앙값, 표준편차 및 상자그림을 비교하여 데이터 분포의 변화를 확인했습니다.

단, 통계량의 변화만으로 전처리가 적절하다고 판단할 수는 없습니다. 각 행이 어떤 의미를 가지는지와 원본 통계의 집계 기준도 함께 확인해야 합니다.

### 분석 시 유의사항

매출액과 종사자 수의 상관계수는 두 지표의 관계를 확인하는 자료입니다. 상관관계만으로 한 지표가 다른 지표의 원인이라고 결론 내릴 수는 없습니다.
""")

st.success("원본 데이터 비교 및 전처리 분석이 완료되었습니다.")
