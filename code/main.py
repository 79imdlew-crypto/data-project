
import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path

# =====================================================
# 1. 기본 설정
# =====================================================

st.set_page_config(
    page_title="국내 애니메이션산업 데이터 분석",
    page_icon="🎬",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

st.title("🎬 국내 애니메이션산업 데이터 분석")
st.write(
    "2020~2022년 국내 애니메이션산업의 지역별 매출액, "
    "종사자 수, 사업체 수 및 직무별 인력 구조를 분석합니다."
)

st.info(
    "원본 데이터 전처리부터 데이터 시각화, 통계분석, "
    "탐구 주제에 따른 해석까지 진행하는 분석 프로그램입니다."
)


# =====================================================
# 2. 데이터 파일 불러오기
# =====================================================

@st.cache_data
def load_csv(filename):
    path = DATA_DIR / filename

    if not path.exists():
        raise FileNotFoundError(
            f"파일을 찾을 수 없습니다: {path}"
        )

    try:
        return pd.read_csv(path, encoding="utf-8-sig")
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="cp949")


try:
    revenue_raw = load_csv("매출액_연도세로형.csv")
    employee_raw = load_csv("종사자수_연도세로형.csv")
    company_raw = load_csv(
        "애니메이션산업_지역별_연도별_사업체수_세로형.csv"
    )
    job_raw = load_csv("직무별_연도세로형.csv")

    st.success("4개의 원본 데이터 파일을 불러왔습니다.")

except Exception as e:
    st.error(f"데이터 파일 오류: {e}")
    st.write("현재 data 폴더의 파일 목록:")

    if DATA_DIR.exists():
        st.write([p.name for p in DATA_DIR.iterdir()])

    st.stop()


# =====================================================
# 3. 공통 함수
# =====================================================

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


def standardize_region(value):
    if pd.isna(value):
        return None

    value = str(value).strip()

    if value in ["", "nan", "None"]:
        return None

    return REGION_MAP.get(value, value)


def is_aggregate(value):
    if pd.isna(value):
        return True

    value = str(value).strip()

    # 합계 및 소계 등 집계 행 제외
    if "합계" in value or "소계" in value:
        return True

    if value in ["7개시", "9개 도", "9개도"]:
        return True

    return False


def numeric_values(series):
    return pd.to_numeric(
        series,
        errors="coerce"
    ).dropna()


def safe_corr(x, y, method="pearson"):
    temp = pd.DataFrame({
        "x": x,
        "y": y
    }).dropna()

    if len(temp) < 3:
        return None

    if temp["x"].nunique() < 2:
        return None

    if temp["y"].nunique() < 2:
        return None

    if method == "spearman":
        # SciPy 없이 스피어만 상관계수 계산
        x_rank = temp["x"].rank(method="average")
        y_rank = temp["y"].rank(method="average")

        return x_rank.corr(y_rank, method="pearson")

    return temp["x"].corr(
        temp["y"],
        method="pearson"
    )


def corr_description(value):
    if value is None or pd.isna(value):
        return "데이터가 부족하거나 값의 변동이 없어 상관관계를 계산할 수 없습니다."

    if value >= 0.7:
        return "강한 양의 상관관계"
    elif value >= 0.4:
        return "중간 정도의 양의 상관관계"
    elif value > 0:
        return "약한 양의 상관관계"
    elif value <= -0.7:
        return "강한 음의 상관관계"
    elif value <= -0.4:
        return "중간 정도의 음의 상관관계"
    elif value < 0:
        return "약한 음의 상관관계"

    return "뚜렷한 선형 상관관계가 거의 없음"


def format_number(value):
    if value is None or pd.isna(value):
        return "계산 불가"

    return f"{value:,.2f}"


def format_change(value):
    if value is None or pd.isna(value):
        return "계산 불가"

    if value > 0:
        return f"{value:.1f}% 증가"
    elif value < 0:
        return f"{abs(value):.1f}% 감소"

    return "변화 없음"


# =====================================================
# 4. 지역별 매출액 및 종사자 수 전처리
# =====================================================

def clean_regional_data(df, value_name):
    df = df.copy()

    required = ["지역", "COL2", "연도", "값"]

    for col in required:
        if col not in df.columns:
            raise ValueError(
                f"{value_name} 데이터에 {col} 열이 없습니다."
            )

    original_count = len(df)

    original_values = numeric_values(df["값"])

    # 반복 제목 행 제거
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

    # 집계 행 제거
    aggregate_mask = df["지역명"].apply(
        is_aggregate
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

    # 완전히 동일한 행만 제거
    before = len(df)

    df = df.drop_duplicates(
        subset=["지역명", "연도", value_name]
    )

    duplicate_count = before - len(df)

    # 같은 지역과 연도에 서로 다른 값이 있으면 오류
    if df.duplicated(
        subset=["지역명", "연도"]
    ).any():
        raise ValueError(
            f"{value_name}에 동일 지역·연도의 "
            "서로 다른 값이 있습니다."
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


# =====================================================
# 5. 사업체 수 전처리
# =====================================================

def clean_company_data(df):
    df = df.copy()

    original_count = len(df)

    region_column = (
        "지역" if "지역" in df.columns
        else "지역구분"
    )

    if region_column not in df.columns:
        raise ValueError("사업체 수 데이터의 지역 열을 찾을 수 없습니다.")

    if "연도" not in df.columns or "사업체수" not in df.columns:
        raise ValueError("사업체 수 데이터의 필수 열을 찾을 수 없습니다.")

    original_values = numeric_values(df["사업체수"])

    # 반복 제목 행 제거
    header_mask = (
        df[region_column].astype(str).str.strip() == "지역"
    )

    header_count = int(header_mask.sum())
    df = df[~header_mask].copy()

    df["지역명"] = df[region_column].apply(
        standardize_region
    )

    aggregate_mask = df["지역명"].apply(
        is_aggregate
    )

    aggregate_count = int(aggregate_mask.sum())
    df = df[~aggregate_mask].copy()

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

    before = len(df)

    df = df.drop_duplicates(
        subset=["지역명", "연도", "사업체수"]
    )

    duplicate_count = before - len(df)

    if df.duplicated(
        subset=["지역명", "연도"]
    ).any():
        raise ValueError(
            "사업체 수 데이터에 동일 지역·연도의 "
            "서로 다른 값이 있습니다."
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


# =====================================================
# 6. 직무별 종사자 수 전처리
# =====================================================

def clean_job_data(df):
    df = df.copy()

    original_count = len(df)

    required = ["직무별", "연도", "값"]

    for col in required:
        if col not in df.columns:
            raise ValueError(
                f"직무별 데이터에 {col} 열이 없습니다."
            )

    original_values = numeric_values(df["값"])

    header_mask = (
        df["직무별"].astype(str).str.strip() == "직무별"
    )

    header_count = int(header_mask.sum())
    df = df[~header_mask].copy()

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

    before = len(df)

    df = df.drop_duplicates(
        subset=["직무별", "연도", "종사자수"]
    )

    duplicate_count = before - len(df)

    if df.duplicated(
        subset=["직무별", "연도"]
    ).any():
        raise ValueError(
            "직무별 데이터에 동일 직무·연도의 "
            "서로 다른 값이 있습니다."
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


# =====================================================
# 7. 전처리 실행
# =====================================================

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
    st.error(f"데이터 전처리 오류: {e}")
    st.stop()


# =====================================================
# 상단 메뉴: 질문 / 데이터 분석 / 기준선 / 모델 평가
# =====================================================
tab_question, tab_analysis, tab_baseline, tab_evaluation = st.tabs(
    ["1. 질문", "2. 데이터 분석", "3. 기준선", "4. 모델 평가"]
)

with tab_question:
    st.header("1. 탐구 질문")
    st.subheader(
        "국내 애니메이션산업의 지역별 매출액과 종사자 수는 어떤 관계가 있으며, "
        "사업체 수는 이러한 관계와 어떻게 연결되는가?"
    )
    st.write(
        "지역별 종사자 수와 사업체 수, 연도를 이용해 매출액을 예측하고, "
        "회귀 모델이 단순 평균 예측 기준선보다 나은 성능을 보이는지 확인합니다."
    )
    st.markdown(
        "**분석 방법을 회귀로 선택한 이유:** 이번 탐구의 예측 대상인 매출액은 "
        "높음·낮음과 같은 범주가 아니라 연속적인 수치입니다. 따라서 수치형 결과를 "
        "예측하는 회귀 분석이 적합합니다. 분류는 범주를 구분하고, 군집은 유사한 "
        "자료끼리 묶는 방법이므로 이번 매출액 예측 질문과는 목적이 다릅니다."
    )
    st.markdown(
        "**분석 변수:** 종속변수는 예측 대상인 매출액이며, 독립변수는 종사자 수·"
        "사업체 수·연도입니다. 단순 선형회귀는 종사자 수만 사용하고, 나머지 다중 "
        "변수 기반 모델은 세 변수를 함께 사용합니다. 전국 단위 직무별 자료는 "
        "지역 단위가 일치하지 않아 회귀 입력 변수에서 제외했습니다."
    )
    st.caption(
        "분석 자료는 2020~2022년 지역별 통계입니다. 관찰 자료를 이용한 분석이므로 "
        "변수 사이의 인과관계를 증명하는 것은 아닙니다."
    )

with tab_analysis:
    # =====================================================
    # 8. 전처리 결과 요약
    # =====================================================

    st.header("1. 데이터 전처리 결과")

    logs = {
        "매출액": revenue_log,
        "종사자 수": employee_log,
        "사업체 수": company_log,
        "직무별 종사자 수": job_log
    }

    log_df = pd.DataFrame(logs).T.fillna(0)

    st.dataframe(
        log_df,
        use_container_width=True
    )

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

    st.plotly_chart(fig_count, use_container_width=True)

    st.write(
        "반복 제목 행과 집계 행 등을 구분하여 개별 지역 분석에 "
        "적합한 형태로 정리했습니다."
    )


    # =====================================================
    # 9. 데이터 속성 및 유형 안내
    # =====================================================

    st.divider()
    st.header("2. 데이터 속성별 유형 및 시각화 방법")

    type_df = pd.DataFrame([
        {
            "데이터": "지역별 매출액",
            "속성": "지역, 연도, 매출액",
            "유형": "범주형, 시계열형, 수치형",
            "시각화": "막대그래프, 선그래프, 산점도",
            "파악할 내용": "지역별 매출 규모와 연도별 변화"
        },
        {
            "데이터": "지역별 종사자 수",
            "속성": "지역, 연도, 종사자 수",
            "유형": "범주형, 시계열형, 이산형 수치",
            "시각화": "막대그래프, 선그래프, 상자그림",
            "파악할 내용": "지역별 고용 규모와 분포"
        },
        {
            "데이터": "지역별 사업체 수",
            "속성": "지역, 연도, 사업체 수",
            "유형": "범주형, 시계열형, 이산형 수치",
            "시각화": "막대그래프, 선그래프, 산점도",
            "파악할 내용": "사업체의 지역별 분포와 변화"
        },
        {
            "데이터": "직무별 종사자 수",
            "속성": "직무, 연도, 종사자 수",
            "유형": "범주형, 시계열형, 이산형 수치",
            "시각화": "막대그래프, 누적 막대그래프",
            "파악할 내용": "직무별 인력 구성과 변화"
        }
    ])

    st.dataframe(type_df, use_container_width=True)


    # =====================================================
    # 10. 원본과 전처리 데이터 통계 비교
    # =====================================================

    st.divider()
    st.header("3. 원본과 전처리 데이터 통계 비교")

    comparison_data = {
        "매출액": (revenue_before, revenue_after),
        "종사자 수": (employee_before, employee_after),
        "사업체 수": (company_before, company_after),
        "직무별 종사자 수": (job_before, job_after)
    }

    comparison_rows = []

    for name, (before, after) in comparison_data.items():
        for label, series in [
            ("원본", before),
            ("전처리 후", after)
        ]:
            comparison_rows.append({
                "데이터": name,
                "구분": label,
                "개수": len(series),
                "평균": series.mean() if len(series) else None,
                "중앙값": series.median() if len(series) else None,
                "표준편차": series.std() if len(series) else None,
                "최솟값": series.min() if len(series) else None,
                "최댓값": series.max() if len(series) else None
            })

    comparison_df = pd.DataFrame(comparison_rows)

    st.dataframe(
        comparison_df,
        use_container_width=True
    )

    selected_comparison = st.selectbox(
        "분포 비교 데이터 선택",
        list(comparison_data.keys())
    )

    before, after = comparison_data[selected_comparison]

    distribution_df = pd.concat([
        pd.DataFrame({
            "구분": "원본",
            "값": before
        }),
        pd.DataFrame({
            "구분": "전처리 후",
            "값": after
        })
    ], ignore_index=True)

    fig_comparison = px.box(
        distribution_df,
        x="구분",
        y="값",
        color="구분",
        points="all",
        title=f"{selected_comparison} 원본과 전처리 후 분포 비교"
    )

    st.plotly_chart(
        fig_comparison,
        use_container_width=True
    )

    st.caption(
        "원본에는 합계·소계 등이 포함될 수 있으므로 "
        "두 분포의 차이는 집계 행 제거 등의 영향을 받을 수 있습니다."
    )


    # =====================================================
    # 11. 지역별 데이터 통합
    # =====================================================

    st.divider()
    st.header("4. 지역별 통합 데이터")

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

    data = merged.dropna(
        subset=["매출액", "종사자수", "사업체수"]
    ).copy()

    unmatched = merged[
        merged[
            ["매출액", "종사자수", "사업체수"]
        ].isna().any(axis=1)
    ].copy()

    st.write(f"전체 지역·연도 조합: {len(merged)}개")
    st.write(f"세 자료가 모두 일치하는 조합: {len(data)}개")
    st.write(f"일부 자료가 누락된 조합: {len(unmatched)}개")

    if not unmatched.empty:
        with st.expander("일부 자료가 누락된 조합 확인"):
            st.dataframe(unmatched, use_container_width=True)

    data = data.sort_values(
        ["연도", "지역명"]
    ).reset_index(drop=True)

    if data.empty:
        st.error(
            "세 자료가 모두 일치하는 지역·연도 조합이 없습니다. "
            "지역명과 연도를 확인해 주세요."
        )
        st.stop()

    st.dataframe(data, use_container_width=True)


    # =====================================================
    # 12. 지역별 연도별 합계 및 변화 분석
    # =====================================================

    st.divider()
    st.header("5. 연도별 산업 규모 변화")

    # 개별 지역 자료의 합계
    yearly = (
        data.groupby("연도")[
            ["매출액", "종사자수", "사업체수"]
        ]
        .sum()
        .reset_index()
        .sort_values("연도")
    )

    st.write(
        "아래 합계는 통합된 개별 지역 자료의 합입니다. "
        "원본의 전국 합계 행과 집계 범위가 다를 수 있습니다."
    )

    st.dataframe(yearly, use_container_width=True)

    metric_choice = st.selectbox(
        "연도별 변화 확인",
        ["매출액", "종사자수", "사업체수"]
    )

    fig_yearly = px.line(
        yearly,
        x="연도",
        y=metric_choice,
        markers=True,
        text=metric_choice,
        title=f"연도별 {metric_choice} 합계 변화"
    )

    st.plotly_chart(fig_yearly, use_container_width=True)

    if len(yearly) >= 2:
        first = yearly.iloc[0]
        last = yearly.iloc[-1]

        start_value = first[metric_choice]
        end_value = last[metric_choice]

        if start_value != 0:
            change = (
                (end_value - start_value)
                / abs(start_value) * 100
            )
        else:
            change = None

        st.subheader("연도별 변화 해석")

        st.write(
            f"{int(first['연도'])}년부터 {int(last['연도'])}년까지 "
            f"{metric_choice}은 "
            f"{format_change(change)} 했습니다."
        )

        if end_value > start_value:
            st.success(
                f"분석 대상 지역의 {metric_choice} 합계가 "
                "증가하는 흐름을 보였습니다."
            )
        elif end_value < start_value:
            st.warning(
                f"분석 대상 지역의 {metric_choice} 합계가 "
                "감소하는 흐름을 보였습니다."
            )
        else:
            st.info(
                f"분석 대상 지역의 {metric_choice} 합계는 "
                "시작 연도와 마지막 연도에 같은 수준입니다."
            )


    # =====================================================
    # 13. 지역별 비교
    # =====================================================

    st.divider()
    st.header("6. 지역별 규모 비교")

    selected_year = st.selectbox(
        "비교할 연도",
        sorted(data["연도"].unique()),
        key="region_year"
    )

    selected_data = data[
        data["연도"] == selected_year
    ].copy()

    region_metric = st.selectbox(
        "비교할 항목",
        ["매출액", "종사자수", "사업체수"],
        key="region_metric"
    )

    region_sorted = selected_data.sort_values(
        region_metric,
        ascending=False
    )

    fig_region = px.bar(
        region_sorted,
        x="지역명",
        y=region_metric,
        color="지역명",
        text_auto=".2s",
        title=f"{int(selected_year)}년 지역별 {region_metric} 비교"
    )

    fig_region.update_layout(showlegend=False)

    st.plotly_chart(fig_region, use_container_width=True)

    if not region_sorted.empty:
        highest = region_sorted.iloc[0]
        lowest = region_sorted.iloc[-1]

        st.subheader("지역별 비교 해석")

        st.write(
            f"{int(selected_year)}년 {region_metric}이 가장 높은 지역은 "
            f"{highest['지역명']}이며, 값은 "
            f"{format_number(highest[region_metric])}입니다."
        )

        st.write(
            f"가장 낮은 지역은 {lowest['지역명']}이며, 값은 "
            f"{format_number(lowest[region_metric])}입니다."
        )

        st.write(
            "지역별 값의 차이를 통해 산업 규모가 지역에 따라 "
            "다르게 분포하는지 확인할 수 있습니다."
        )


    # =====================================================
    # 14. 지역별 분포 분석
    # =====================================================

    st.divider()
    st.header("7. 지역별 데이터 분포 분석")

    box_metric = st.selectbox(
        "분포를 확인할 항목",
        ["매출액", "종사자수", "사업체수"],
        key="box_metric"
    )

    fig_box = px.box(
        data,
        x="연도",
        y=box_metric,
        points="all",
        title=f"연도별 지역 {box_metric} 분포"
    )

    st.plotly_chart(fig_box, use_container_width=True)

    st.write(
        "상자그림에서 중앙값, 사분위 범위와 이상치 등을 확인하여 "
        "지역별 값의 차이와 분포 변화를 살펴볼 수 있습니다."
    )

    box_stats = (
        data.groupby("연도")[box_metric]
        .agg(["median", "min", "max"])
        .reset_index()
    )

    box_stats["범위"] = (
        box_stats["max"] - box_stats["min"]
    )

    st.dataframe(box_stats, use_container_width=True)

    if len(box_stats) >= 2:
        first_range = box_stats.iloc[0]["범위"]
        last_range = box_stats.iloc[-1]["범위"]

        if last_range > first_range:
            st.write(
                "첫 연도와 마지막 연도를 비교하면 지역별 최댓값과 "
                "최솟값의 차이가 커졌습니다."
            )
        elif last_range < first_range:
            st.write(
                "첫 연도와 마지막 연도를 비교하면 지역별 최댓값과 "
                "최솟값의 차이가 작아졌습니다."
            )
        else:
            st.write(
                "첫 연도와 마지막 연도의 지역별 최댓값과 "
                "최솟값의 차이가 같습니다."
            )


    # =====================================================
    # 15. 매출액과 종사자 수의 관계
    # =====================================================

    st.divider()
    st.header("8. 매출액과 종사자 수의 관계")

    fig_scatter = px.scatter(
        data,
        x="종사자수",
        y="매출액",
        color="연도",
        hover_name="지역명",
        size="사업체수",
        title="지역별 종사자 수와 매출액의 관계"
    )

    st.plotly_chart(fig_scatter, use_container_width=True)

    pearson = safe_corr(
        data["종사자수"],
        data["매출액"],
        "pearson"
    )

    spearman = safe_corr(
        data["종사자수"],
        data["매출액"],
        "spearman"
    )

    col1, col2 = st.columns(2)

    col1.metric(
        "피어슨 상관계수",
        format_number(pearson)
    )

    col2.metric(
        "스피어만 상관계수",
        format_number(spearman)
    )

    st.subheader("매출액과 종사자 수 해석")

    st.write(
        "피어슨 상관계수: "
        + corr_description(pearson)
    )

    st.write(
        "스피어만 상관계수: "
        + corr_description(spearman)
    )

    if pearson is not None:
        if pearson > 0:
            st.write(
                "종사자 수가 많은 지역일수록 매출액도 높은 "
                "경향이 나타났습니다."
            )
        elif pearson < 0:
            st.write(
                "종사자 수가 많은 지역일수록 매출액이 낮은 "
                "경향이 나타났습니다."
            )
        else:
            st.write(
                "두 변수 사이에 뚜렷한 선형 관계가 나타나지 않았습니다."
            )

    st.warning(
        "상관관계는 두 변수의 관련성을 나타내며, "
        "한 변수가 다른 변수의 원인이라는 것을 증명하지 않습니다."
    )


    # =====================================================
    # 16. 사업체 수와 매출액의 관계
    # =====================================================

    st.divider()
    st.header("9. 사업체 수와 매출액의 관계")

    fig_company = px.scatter(
        data,
        x="사업체수",
        y="매출액",
        color="연도",
        hover_name="지역명",
        title="지역별 사업체 수와 매출액의 관계"
    )

    st.plotly_chart(fig_company, use_container_width=True)

    company_pearson = safe_corr(
        data["사업체수"],
        data["매출액"],
        "pearson"
    )

    company_spearman = safe_corr(
        data["사업체수"],
        data["매출액"],
        "spearman"
    )

    col1, col2 = st.columns(2)

    col1.metric(
        "피어슨 상관계수",
        format_number(company_pearson)
    )

    col2.metric(
        "스피어만 상관계수",
        format_number(company_spearman)
    )

    st.write(
        "사업체 수와 매출액의 관계는 "
        + corr_description(company_pearson)
        + "로 나타났습니다."
    )

    if company_pearson is not None:
        if company_pearson > 0:
            st.write(
                "사업체 수가 많은 지역일수록 매출액도 높은 "
                "경향이 관찰됩니다."
            )
        elif company_pearson < 0:
            st.write(
                "사업체 수가 많은 지역일수록 매출액이 낮은 "
                "경향이 관찰됩니다."
            )
        else:
            st.write(
                "두 변수 사이에 뚜렷한 선형 관계가 나타나지 않았습니다."
            )


    # =====================================================
    # 17. 지역별 평균 매출 지표
    # =====================================================

    st.divider()
    st.header("10. 지역별 사업체당·종사자당 매출 지표")

    data["사업체당 매출액"] = (
        data["매출액"] / data["사업체수"]
    )

    data["종사자당 매출액"] = (
        data["매출액"] / data["종사자수"]
    )

    st.write(
        "아래 지표는 지역별 매출액을 사업체 수 또는 종사자 수로 "
        "나눈 단순 계산값입니다. 원자료의 평균매출액 통계와 "
        "정의 및 집계 방식이 동일하다고 확인된 것은 아닙니다."
    )

    ratio_choice = st.selectbox(
        "비교할 지표",
        ["사업체당 매출액", "종사자당 매출액"]
    )

    fig_ratio = px.bar(
        data,
        x="지역명",
        y=ratio_choice,
        color="연도",
        barmode="group",
        title=f"지역별 {ratio_choice} 비교"
    )

    st.plotly_chart(fig_ratio, use_container_width=True)


    # =====================================================
    # 18. 직무별 종사자 분석
    # =====================================================

    st.divider()
    st.header("11. 직무별 종사자 구성")

    job_years = sorted(job["연도"].unique())

    selected_job_year = st.selectbox(
        "직무 분석 연도",
        job_years
    )

    job_selected = job[
        (job["연도"] == selected_job_year)
        & (job["직무별"] != "합계")
    ].copy()

    fig_job = px.bar(
        job_selected.sort_values(
            "종사자수",
            ascending=False
        ),
        x="직무별",
        y="종사자수",
        color="직무별",
        text_auto=True,
        title=f"{int(selected_job_year)}년 직무별 종사자 수"
    )

    fig_job.update_layout(showlegend=False)

    st.plotly_chart(fig_job, use_container_width=True)

    job_total = job_selected["종사자수"].sum()

    if job_total > 0:
        job_selected["비중(%)"] = (
            job_selected["종사자수"] / job_total * 100
        )

        job_selected = job_selected.sort_values(
            "종사자수",
            ascending=False
        )

        st.dataframe(
            job_selected[
                ["직무별", "종사자수", "비중(%)"]
            ],
            use_container_width=True
        )

        top_job = job_selected.iloc[0]

        st.write(
            f"{int(selected_job_year)}년 가장 많은 종사자가 분포한 "
            f"직무는 {top_job['직무별']}이며, "
            f"직무별 합계에서 약 {top_job['비중(%)']:.1f}%를 "
            "차지합니다."
        )


    # =====================================================
    # 19. 보조 분석: 매출액 변화와 직무별 종사자 구성 비교
    # =====================================================

    st.divider()
    st.header("12. 보조 분석: 매출액 변화와 직무별 종사자 구성")

    st.markdown(
        """
        이 분석은 전국 매출액의 연도별 변화와 전국 직무별 종사자 수의
        변화를 같은 기간에 비교합니다. 지역별 매출액 분석을 대체하는 것이
        아니라, 산업 전체의 인력 구성이 어떻게 변화했는지 추가로 살펴보는
        보조 분석입니다.

        두 자료는 전국 단위의 연도별 자료이므로 지역별 매출액과 직무별
        인원을 직접 연결할 수 없습니다. 또한 관측 연도가 제한적이므로
        변화 방향을 비교하는 데 목적이 있으며, 인과관계를 증명하지 않습니다.
        """
    )

    # 원자료의 전국 매출액 합계 행 추출
    national_revenue_aux = revenue_raw.copy()

    if {"지역", "연도", "값"}.issubset(national_revenue_aux.columns):
        national_revenue_aux["지역"] = (
            national_revenue_aux["지역"].astype(str).str.strip()
        )
        national_revenue_aux["연도"] = pd.to_numeric(
            national_revenue_aux["연도"], errors="coerce"
        )
        national_revenue_aux["매출액"] = pd.to_numeric(
            national_revenue_aux["값"].astype(str).str.replace(",", "", regex=False),
            errors="coerce"
        )

        national_revenue_aux = national_revenue_aux[
            (national_revenue_aux["지역"] == "합계")
            & national_revenue_aux["연도"].between(2020, 2022)
        ][["연도", "매출액"]].dropna()

        national_revenue_aux = national_revenue_aux.drop_duplicates(
            subset=["연도"], keep="last"
        )

    else:
        national_revenue_aux = pd.DataFrame(
            columns=["연도", "매출액"]
        )

    # 원자료에서 전국 합계 행을 찾지 못한 경우에만
    # 전처리된 개별 지역 매출액의 합계를 대체값으로 사용
    revenue_source_note = "원자료의 전국 합계 행"

    if national_revenue_aux.empty:
        national_revenue_aux = (
            revenue.groupby("연도", as_index=False)["매출액"]
            .sum()
        )
        revenue_source_note = "전처리된 개별 지역 매출액의 합계(전국 공식 합계와 다를 수 있음)"

    job_detail_aux = job[
        job["직무별"].astype(str).str.strip() != "합계"
    ].copy()

    job_detail_aux = job_detail_aux[
        job_detail_aux["연도"].between(2020, 2022)
    ]

    job_pivot_aux = job_detail_aux.pivot_table(
        index="연도",
        columns="직무별",
        values="종사자수",
        aggfunc="sum"
    ).reset_index()

    job_revenue_comparison = national_revenue_aux.merge(
        job_pivot_aux,
        on="연도",
        how="inner"
    ).sort_values("연도")

    st.caption(
        f"매출액 자료 기준: {revenue_source_note}. "
        "매출액의 단위는 원자료의 단위를 따릅니다."
    )

    if job_revenue_comparison.empty:
        st.warning(
            "매출액과 직무별 종사자 자료에서 공통 연도를 찾지 못했습니다. "
            "각 CSV의 연도와 전국 합계 행을 확인해 주세요."
        )
    else:
        st.subheader("12-1. 전국 매출액과 직무별 종사자 수 비교")

        st.dataframe(
            job_revenue_comparison,
            use_container_width=True,
            hide_index=True
        )

        # 연도별 증감률 계산
        comparison_change_aux = job_revenue_comparison.copy()
        comparison_columns_aux = [
            col for col in comparison_change_aux.columns
            if col != "연도"
        ]

        for col in comparison_columns_aux:
            comparison_change_aux[f"{col}_변화율(%)"] = (
                comparison_change_aux[col].pct_change() * 100
            )

        change_columns_aux = ["연도"] + [
            f"{col}_변화율(%)"
            for col in comparison_columns_aux
        ]

        st.subheader("12-2. 매출액과 직무별 종사자 수의 연도별 변화율")

        st.dataframe(
            comparison_change_aux[change_columns_aux].round(2),
            use_container_width=True,
            hide_index=True
        )

        st.caption(
            "변화율은 직전 공통 연도 대비 증감률입니다. "
            "첫 번째 연도는 비교 대상이 없어 값이 표시되지 않습니다."
        )

        # 서로 다른 단위의 지표를 비교하기 위해 첫 공통 연도를 100으로 환산
        st.subheader("12-3. 기준 연도 대비 매출액·직무별 인원 변화")

        indexed_aux = job_revenue_comparison.copy()
        index_columns_aux = [
            col for col in indexed_aux.columns
            if col != "연도"
        ]

        base_values_aux = indexed_aux.iloc[0]

        for col in index_columns_aux:
            if pd.notna(base_values_aux[col]) and base_values_aux[col] != 0:
                indexed_aux[f"{col}_지수"] = (
                    indexed_aux[col] / base_values_aux[col] * 100
                )

        index_long_columns_aux = [
            f"{col}_지수"
            for col in index_columns_aux
            if f"{col}_지수" in indexed_aux.columns
        ]

        index_long_aux = indexed_aux[
            ["연도"] + index_long_columns_aux
        ].melt(
            id_vars="연도",
            var_name="항목",
            value_name="지수"
        )

        index_long_aux["항목"] = (
            index_long_aux["항목"]
            .str.replace("_지수", "", regex=False)
        )

        fig_index_aux = px.line(
            index_long_aux,
            x="연도",
            y="지수",
            color="항목",
            markers=True,
            title=(
                f"{int(indexed_aux.iloc[0]['연도'])}년을 100으로 설정한 "
                "매출액·직무별 종사자 수 변화"
            )
        )

        fig_index_aux.add_hline(
            y=100,
            line_dash="dash",
            annotation_text="기준 지수 100"
        )

        fig_index_aux.update_layout(
            xaxis_title="연도",
            yaxis_title="기준 연도 대비 지수"
        )

        st.plotly_chart(fig_index_aux, use_container_width=True)

        st.caption(
            "지수는 서로 단위가 다른 매출액과 종사자 수의 변화 방향을 "
            "비교하기 위한 값입니다. 기준 연도는 공통으로 확보된 첫 연도이며, "
            "지수 100보다 높으면 기준 연도보다 증가한 것입니다."
        )

        # 직무별 구성 비중의 연도별 변화
        st.subheader("12-4. 직무별 종사자 구성 비중 변화")

        job_share_aux = job_detail_aux.copy()
        job_share_aux["직무별 합계"] = (
            job_share_aux.groupby("연도")["종사자수"].transform("sum")
        )
        job_share_aux["비중(%)"] = (
            job_share_aux["종사자수"]
            / job_share_aux["직무별 합계"] * 100
        )

        fig_share_aux = px.bar(
            job_share_aux,
            x="연도",
            y="비중(%)",
            color="직무별",
            barmode="stack",
            title="연도별 직무 구성 비중"
        )

        fig_share_aux.update_layout(
            xaxis_title="연도",
            yaxis_title="직무별 비중 (%)"
        )

        st.plotly_chart(fig_share_aux, use_container_width=True)

        st.caption(
            "각 연도의 직무별 항목 합계를 100%로 환산했습니다. "
            "직무별 합계가 공식 전체 종사자 수와 일치하는지는 별도로 확인해야 합니다."
        )

        # 제작 직무에 초점을 맞춰 매출액과 나란히 비교
        st.subheader("12-5. 매출액과 제작 직무 종사자 수 비교")

        if "제작" in job_revenue_comparison.columns:
            production_compare_aux = job_revenue_comparison[
                ["연도", "매출액", "제작"]
            ].copy()

            production_long_aux = production_compare_aux.melt(
                id_vars="연도",
                var_name="항목",
                value_name="값"
            )

            fig_production_aux = px.line(
                production_long_aux,
                x="연도",
                y="값",
                color="항목",
                markers=True,
                title="전국 매출액과 제작 직무 종사자 수의 변화"
            )

            fig_production_aux.update_layout(
                xaxis_title="연도",
                yaxis_title="각 항목의 원자료 값"
            )

            st.plotly_chart(
                fig_production_aux,
                use_container_width=True
            )

            st.caption(
                "두 항목의 단위와 규모가 다르므로 이 그래프는 변화 추세를 "
                "함께 확인하는 용도입니다. 수치의 높이를 직접 비교하면 안 됩니다."
            )
        else:
            st.info(
                "직무별 데이터에서 '제작' 항목을 찾지 못해 "
                "제작 직무 비교 그래프를 표시하지 못했습니다."
            )

        # 보조 분석 자동 해석
        st.subheader("12-6. 보조 분석 해석")

        first_aux = job_revenue_comparison.iloc[0]
        last_aux = job_revenue_comparison.iloc[-1]

        revenue_change_aux = None
        if first_aux["매출액"] != 0:
            revenue_change_aux = (
                (last_aux["매출액"] - first_aux["매출액"])
                / abs(first_aux["매출액"]) * 100
            )

        if revenue_change_aux is not None:
            st.write(
                f"공통 분석 기간인 {int(first_aux['연도'])}년부터 "
                f"{int(last_aux['연도'])}년까지 매출액은 "
                f"{format_change(revenue_change_aux)} 했습니다."
            )

        if "제작" in job_revenue_comparison.columns:
            first_production_aux = first_aux["제작"]
            last_production_aux = last_aux["제작"]

            if pd.notna(first_production_aux) and first_production_aux != 0:
                production_change_aux = (
                    (last_production_aux - first_production_aux)
                    / abs(first_production_aux) * 100
                )

                st.write(
                    f"같은 기간 제작 직무 종사자 수는 "
                    f"{format_change(production_change_aux)} 했습니다."
                )

                if (
                    revenue_change_aux is not None
                    and revenue_change_aux > 0
                    and production_change_aux > 0
                ):
                    st.write(
                        "분석 기간에 매출액과 제작 직무 종사자 수가 "
                        "모두 증가하는 방향을 보였습니다."
                    )
                elif (
                    revenue_change_aux is not None
                    and revenue_change_aux < 0
                    and production_change_aux < 0
                ):
                    st.write(
                        "분석 기간에 매출액과 제작 직무 종사자 수가 "
                        "모두 감소하는 방향을 보였습니다."
                    )
                else:
                    st.write(
                        "분석 기간에 매출액과 제작 직무 종사자 수의 "
                        "변화 방향이 서로 다르게 나타났거나, 한쪽의 변화가 "
                        "뚜렷하지 않았습니다."
                    )

        st.info(
            "이 결과는 매출액과 직무별 인력 변화가 같은 시기에 어떻게 "
            "나타났는지 보여주는 탐색적 비교입니다. 연도별 관측값이 적고 "
            "직무별 자료가 전국 단위이므로, 직무별 인력 변화가 매출액 변화의 "
            "원인이라고 결론 내릴 수는 없습니다. 또한 제작 직무 전체를 "
            "CG·VFX 종사자 수로 해석해서는 안 됩니다."
        )




# =====================================================
# 기준선과 회귀 모델 학습 및 평가
# =====================================================

try:
    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LinearRegression
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.dummy import DummyRegressor
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    model_data = data[["지역명", "연도", "종사자수", "사업체수", "매출액"]].copy()
    for col in ["연도", "종사자수", "사업체수", "매출액"]:
        model_data[col] = pd.to_numeric(model_data[col], errors="coerce")
    model_data = model_data.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)

    if len(model_data) < 12:
        with tab_baseline:
            st.warning(f"모델 학습에 사용할 수 있는 데이터가 {len(model_data)}개입니다. 표본이 적어 성능 비교가 불안정할 수 있습니다.")
        with tab_evaluation:
            st.warning("평가할 데이터가 충분하지 않습니다.")
    else:
        Xs = model_data[["종사자수"]]
        Xm = model_data[["종사자수", "사업체수", "연도"]]
        y = model_data["매출액"]
        idx = np.arange(len(model_data))
        train_idx, test_idx = train_test_split(idx, test_size=0.25, random_state=42)
        Xs_train, Xs_test = Xs.iloc[train_idx], Xs.iloc[test_idx]
        Xm_train, Xm_test = Xm.iloc[train_idx], Xm.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        # 기준선: 학습 데이터 매출액 평균을 모든 평가 행에 예측
        baseline = DummyRegressor(strategy="mean")
        baseline.fit(Xm_train, y_train)
        baseline_pred = baseline.predict(Xm_test)
        baseline_mae = mean_absolute_error(y_test, baseline_pred)
        baseline_rmse = np.sqrt(mean_squared_error(y_test, baseline_pred))
        baseline_r2 = r2_score(y_test, baseline_pred)

        models = {
            "단순 선형회귀": (LinearRegression(), Xs_train, Xs_test),
            "다중 선형회귀": (LinearRegression(), Xm_train, Xm_test),
            "K-최근접 이웃 회귀": (make_pipeline(StandardScaler(), KNeighborsRegressor(n_neighbors=min(5, len(train_idx)))), Xm_train, Xm_test),
            "랜덤 포레스트 회귀": (RandomForestRegressor(n_estimators=300, random_state=42), Xm_train, Xm_test)
        }
        excluded_models = []
        try:
            from xgboost import XGBRegressor
            models["XGBoost 회귀"] = (XGBRegressor(n_estimators=300, max_depth=3, learning_rate=0.03, subsample=0.9, colsample_bytree=0.9, objective="reg:squarederror", random_state=42, n_jobs=1), Xm_train, Xm_test)
        except ImportError:
            excluded_models.append({
                "모델": "XGBoost 회귀",
                "제외 이유": "xgboost 라이브러리를 불러오지 못해 이번 실행의 수치 비교에서 제외함."
            })

        rows, preds, fitted = [], {}, {}
        for name, (estimator, xtr, xte) in models.items():
            estimator.fit(xtr, y_train)
            pred = estimator.predict(xte)
            fitted[name] = estimator
            preds[name] = pred
            rows.append({"모델": name, "MAE": mean_absolute_error(y_test, pred), "RMSE": np.sqrt(mean_squared_error(y_test, pred)), "R²": r2_score(y_test, pred)})
        model_scores = pd.DataFrame(rows)
        baseline_scores = pd.DataFrame([{"모델":"기준선(평균 예측)", "MAE":baseline_mae, "RMSE":baseline_rmse, "R²":baseline_r2}])
        all_scores = pd.concat([baseline_scores, model_scores], ignore_index=True)

        with tab_baseline:
            st.header("3. 기준선")
            st.write("기준선은 수업에서 배운 평균 모델을 적용한 것으로, 학습 데이터의 매출액 평균을 모든 평가 데이터의 예측값으로 사용하는 단순한 비교 대상입니다.")
            st.caption("즉, 지역별 종사자 수나 사업체 수의 차이를 반영하지 않고 모든 평가 사례에 동일한 평균 매출액을 예측합니다. 후보 회귀 모델이 이 단순한 기준보다 실제로 나은지 판단하기 위해 사용합니다.")
            c1,c2,c3=st.columns(3)
            c1.metric("전체 데이터", f"{len(model_data)}개")
            c2.metric("학습 데이터", f"{len(train_idx)}개")
            c3.metric("평가 데이터", f"{len(test_idx)}개")
            st.dataframe(baseline_scores.round(4), use_container_width=True, hide_index=True)
            st.caption("MAE와 RMSE는 낮을수록 좋고, R²는 높을수록 좋습니다. 기준선의 R²가 음수일 수도 있습니다.")
            base_plot=pd.DataFrame({"실제 매출액":y_test.to_numpy(), "기준선 예측 매출액":baseline_pred})
            st.plotly_chart(px.scatter(base_plot, x="실제 매출액", y="기준선 예측 매출액", title="기준선: 실제값과 평균 예측값"), use_container_width=True)

        with tab_evaluation:
            st.header("4. 모델 평가")
            st.write("모든 모델은 동일한 학습·평가 데이터 분할(학습 75%, 평가 25%, random_state=42)을 사용합니다. 다중 선형회귀·KNN·랜덤 포레스트·XGBoost는 종사자 수·사업체 수·연도라는 동일한 변수를 사용하므로 이 네 모델을 공정한 주 비교 대상으로 삼습니다. 단순 선형회귀는 종사자 수만 사용하는 추가 참고 모델입니다.")
            with st.expander("비교한 후보 모델과 작동 방식", expanded=True):
                st.markdown(
                    """
                    | 후보 모델 | 사용 변수 | 특징 |
                    |---|---|---|
                    | 단순 선형회귀 | 종사자 수 | 종사자 수와 매출액의 선형 관계를 직선으로 나타냄 |
                    | 다중 선형회귀 | 종사자 수·사업체 수·연도 | 여러 독립변수의 관계를 함께 고려함 |
                    | K-최근접 이웃 회귀(KNN) | 종사자 수·사업체 수·연도 | 특성이 가까운 학습 자료의 값을 활용해 예측함 |
                    | 랜덤 포레스트 회귀 | 종사자 수·사업체 수·연도 | 여러 결정 트리의 예측을 종합함 |
                    | XGBoost 회귀 | 종사자 수·사업체 수·연도 | 트리를 순차적으로 추가해 앞선 예측 오차를 보완함 |
                    """
                )
                st.caption("동일 변수 조건의 주 비교 모델은 다중 선형회귀·KNN·랜덤 포레스트·XGBoost입니다. 단순 선형회귀는 종사자 수만 사용하는 보조 비교이므로 주 비교 모델과 변수 구성이 다르다는 점을 구분해 해석합니다. 모든 모델은 같은 학습·평가 행을 사용합니다.")
            st.subheader("후보 모델 성능 비교")
            st.write("MAE와 RMSE는 낮을수록 좋고, R²는 높을수록 좋습니다. 아래 수치로 후보 모델 간 차이를 비교할 수 있습니다.")
            st.dataframe(all_scores.sort_values("RMSE").round(4), use_container_width=True, hide_index=True)
            improvement=[]
            for _,r in model_scores.iterrows():
                improvement.append({"모델":r["모델"], "MAE 개선율(%)":(baseline_mae-r["MAE"])/baseline_mae*100 if baseline_mae else np.nan, "RMSE 개선율(%)":(baseline_rmse-r["RMSE"])/baseline_rmse*100 if baseline_rmse else np.nan, "기준선보다 RMSE가 낮은가":r["RMSE"]<baseline_rmse})
            st.subheader("기준선 대비 성능 개선")
            improvement_df = pd.DataFrame(improvement)
            st.dataframe(improvement_df.round(2), use_container_width=True, hide_index=True)

            # 평가 지표의 의미와 실제 결과에 따른 해석
            st.subheader("평가 결과 해석")
            st.markdown(
                """
                - **MAE(평균 절대 오차)**: 예측값과 실제값의 차이를 절댓값으로 평균한 값입니다. 낮을수록 평균적인 예측 오차가 작습니다.
                - **RMSE(평균 제곱근 오차)**: 오차를 제곱한 뒤 평균을 내고 제곱근을 취한 값입니다. 큰 오차에 더 민감하며, 낮을수록 좋습니다.
                - **R²(결정계수)**: 모델이 실제 매출액의 변동을 얼마나 설명하는지 나타냅니다. 일반적으로 1에 가까울수록 좋으며, 0보다 작으면 단순히 실제값의 평균을 예측하는 기준보다도 성능이 낮을 수 있습니다.
                """
            )

            best_rmse = model_scores.loc[model_scores["RMSE"].idxmin()]
            best_mae = model_scores.loc[model_scores["MAE"].idxmin()]
            best_r2 = model_scores.loc[model_scores["R²"].idxmax()]
            st.write(
                f"RMSE 기준으로 가장 낮은 오차를 보인 모델은 **{best_rmse['모델']}**이며, "
                f"RMSE는 **{best_rmse['RMSE']:,.4f}**입니다. "
                f"MAE 기준으로는 **{best_mae['모델']}**이 가장 낮은 "
                f"**{best_mae['MAE']:,.4f}**를 기록했습니다. "
                f"R² 기준으로는 **{best_r2['모델']}**이 가장 높은 "
                f"**{best_r2['R²']:.4f}**를 기록했습니다."
            )

            st.markdown("**베이스라인과 비교한 해석**")
            for _, row in model_scores.iterrows():
                mae_change = (
                    (baseline_mae - row["MAE"]) / baseline_mae * 100
                    if baseline_mae != 0 else np.nan
                )
                rmse_change = (
                    (baseline_rmse - row["RMSE"]) / baseline_rmse * 100
                    if baseline_rmse != 0 else np.nan
                )
                if row["RMSE"] < baseline_rmse:
                    comparison = "베이스라인보다 RMSE가 낮아 평가 데이터에서 오차가 감소했습니다."
                elif row["RMSE"] > baseline_rmse:
                    comparison = "베이스라인보다 RMSE가 높아 평가 데이터에서 오차가 커졌습니다."
                else:
                    comparison = "베이스라인과 RMSE가 같습니다."
                mae_text = f"MAE는 {mae_change:+.2f}%" if pd.notna(mae_change) else "MAE 개선율은 계산할 수 없습니다"
                rmse_text = f"RMSE는 {rmse_change:+.2f}%" if pd.notna(rmse_change) else "RMSE 개선율은 계산할 수 없습니다"
                st.write(
                    f"- **{row['모델']}**: {comparison} "
                    f"베이스라인 대비 {mae_text}, {rmse_text}입니다."
                )

            if best_rmse["RMSE"] < baseline_rmse:
                st.success(
                    f"현재 평가 분할에서는 RMSE가 가장 낮은 **{best_rmse['모델']}**을 "
                    "우선 후보로 볼 수 있습니다. 다만 한 번의 데이터 분할 결과이므로 "
                    "최종적으로 일반화 성능이 가장 좋다고 단정할 수는 없습니다."
                )
            else:
                st.warning(
                    "이번 평가 분할에서는 어떤 후보 모델도 베이스라인보다 낮은 RMSE를 "
                    "기록하지 못했습니다. 따라서 복잡한 모델이 평균 예측 기준선보다 "
                    "더 낫다고 판단하기 어렵습니다."
                )

            st.info(
                "이 결과는 현재 확보한 2020~2022년 지역별 자료와 한 번의 무작위 학습·평가 "
                "분할을 기준으로 합니다. 관측치가 많지 않고 같은 지역의 서로 다른 연도가 "
                "학습·평가 데이터에 나뉘어 들어갈 수 있으므로, 결과를 미래 연도 예측이나 "
                "인과관계의 증거로 해석해서는 안 됩니다."
            )

            st.subheader("최종 모델 선정 및 비교 기록")
            # RMSE를 주된 선정 기준으로 삼되 MAE와 R²도 함께 기록
            fair_model_names = ["다중 선형회귀", "K-최근접 이웃 회귀", "랜덤 포레스트 회귀", "XGBoost 회귀"]
            fair_model_scores = model_scores[model_scores["모델"].isin(fair_model_names)]
            if fair_model_scores.empty:
                fair_model_scores = model_scores
            selected_row = fair_model_scores.sort_values("RMSE").iloc[0]
            st.write(
                f"**선정 후보: {selected_row['모델']}** — 이번 평가 자료에서 RMSE가 "
                f"가장 낮았기 때문에 우선 후보로 선정했습니다. 해당 모델의 MAE는 "
                f"{selected_row['MAE']:,.4f}, RMSE는 {selected_row['RMSE']:,.4f}, "
                f"R²는 {selected_row['R²']:.4f}입니다. 이 선정은 현재의 한 번의 "
                "분할 결과에 한정되며, 다른 자료에서도 항상 가장 좋다는 뜻은 아닙니다."
            )
            rejected_rows = fair_model_scores[fair_model_scores["모델"] != selected_row["모델"]]
            if not rejected_rows.empty:
                st.markdown("**이번 비교에서 최종 우선 후보로 선택하지 않은 모델과 이유**")
                for _, rejected in rejected_rows.iterrows():
                    st.write(
                        f"- **{rejected['모델']}**: RMSE가 {rejected['RMSE']:,.4f}로 "
                        f"선정 후보({selected_row['RMSE']:,.4f})보다 높아 우선 후보에서 제외했습니다. "
                        "이는 이번 평가 분할에서의 상대적 결과이며, 모델 자체가 항상 부적합하다는 뜻은 아닙니다."
                    )
            if excluded_models:
                st.markdown("**실행 단계에서 비교 대상에서 제외된 모델**")
                st.dataframe(pd.DataFrame(excluded_models), use_container_width=True, hide_index=True)
            else:
                st.caption("라이브러리 오류로 실행 단계에서 제외된 모델은 없습니다.")
            st.info(
                "기록용 요약: 회귀를 선택한 이유는 매출액이라는 연속형 수치를 예측하기 위해서이며, "
                f"동일한 세 변수를 사용한 주 비교 모델 {len(fair_model_scores)}개를 같은 학습·평가 분할에서 비교했습니다. "
                f"RMSE 기준 우선 후보는 {selected_row['모델']}입니다. 실제 GitHub 커밋은 "
                "저장소에서 직접 수행해야 하며, 아래 비교 기록을 로그로 저장해 활용할 수 있습니다."
            )
            model_log_lines = [
                "국내 애니메이션산업 매출액 예측 모델 비교 기록",
                "분석 방법: 회귀 (매출액은 연속형 수치)",
                "분할 조건: 학습 75%, 평가 25%, random_state=42",
                "동일 변수 주 비교 모델: 다중 선형회귀, KNN, 랜덤 포레스트, XGBoost(설치된 경우)",
                "단순 선형회귀는 종사자 수만 사용하는 보조 참고 모델",
                "베이스라인: 학습 데이터 매출액 평균을 모든 평가 자료에 예측",
                "선정 기준: RMSE 최저 모델을 우선 후보로 선정하고 MAE와 R²도 함께 확인",
                f"우선 후보: {selected_row['모델']}",
                f"MAE={selected_row['MAE']:.6f}, RMSE={selected_row['RMSE']:.6f}, R²={selected_row['R²']:.6f}",
                "비교 모델 및 결과:"
            ]
            for _, log_row in model_scores.iterrows():
                reason = "우선 후보" if log_row['모델'] == selected_row['모델'] else "RMSE가 우선 후보보다 높아 이번 평가에서는 최종 우선 후보로 선택하지 않음"
                model_log_lines.append(
                    f"- {log_row['모델']}: MAE={log_row['MAE']:.6f}, RMSE={log_row['RMSE']:.6f}, R²={log_row['R²']:.6f}; {reason}"
                )
            for excluded in excluded_models:
                model_log_lines.append(f"- 제외 모델: {excluded['모델']}; 사유: {excluded['제외 이유']}")
            model_log_lines.append("주의: 한 번의 무작위 분할 결과이며, 미래 연도 예측이나 인과관계를 보장하지 않음.")
            model_log_text = "\n".join(model_log_lines)
            st.download_button(
                "모델 선택·제외 기록 로그 다운로드",
                data=model_log_text,
                file_name="모델_비교_선정_제외_기록.txt",
                mime="text/plain"
            )

            metric=st.selectbox("비교할 평가 지표", ["RMSE","MAE","R²"], key="tab_model_metric")
            chart=px.bar(all_scores, x="모델", y=metric, color="모델", text_auto=".3f", title=f"기준선과 모델별 {metric} 비교")
            chart.update_layout(showlegend=False)
            st.plotly_chart(chart, use_container_width=True)
            chosen=st.selectbox("실제값과 예측값을 비교할 모델", list(preds.keys()), key="tab_pred_model")
            pred_df=pd.DataFrame({"실제 매출액":y_test.to_numpy(), "예측 매출액":preds[chosen], "지역명":model_data.iloc[test_idx]["지역명"].to_numpy(), "연도":model_data.iloc[test_idx]["연도"].to_numpy()})
            st.dataframe(pred_df.round(3), use_container_width=True, hide_index=True)
            st.plotly_chart(px.scatter(pred_df, x="실제 매출액", y="예측 매출액", hover_data=["지역명","연도"], title=f"{chosen}: 실제값과 예측값"), use_container_width=True)
            if chosen in ["랜덤 포레스트 회귀","XGBoost 회귀"]:
                imp=pd.DataFrame({"변수":Xm_train.columns,"중요도":fitted[chosen].feature_importances_}).sort_values("중요도",ascending=False)
                st.subheader(f"{chosen} 변수 중요도")
                st.plotly_chart(px.bar(imp,x="변수",y="중요도",text_auto=".3f"),use_container_width=True)
            best=model_scores.sort_values("RMSE").iloc[0]
            if best["RMSE"] < baseline_rmse:
                st.success(f"이번 평가 분할에서는 {best['모델']}이 기준선보다 낮은 RMSE({best['RMSE']:,.3f})를 기록했습니다.")
            else:
                st.warning("이번 평가 분할에서는 어떤 모델도 기준선보다 낮은 RMSE를 기록하지 못했습니다.")
            st.info("자료가 2020~2022년의 지역별 관측치로 제한적입니다. 무작위 분할은 동일 지역의 다른 연도를 학습·평가에 나눌 수 있으므로 새로운 연도에 대한 예측 성능을 보장하지 않습니다. 결과는 탐색적 비교이며 인과관계를 증명하지 않습니다.")
except ImportError as e:
    with tab_baseline:
        st.error(f"기준선·모델 학습 라이브러리를 불러오지 못했습니다: {e}. requirements.txt에 numpy와 scikit-learn을 추가하세요.")
    with tab_evaluation:
        st.error(f"모델 라이브러리 오류: {e}")
except Exception as e:
    with tab_baseline:
        st.error(f"기준선 계산 오류: {e}")
    with tab_evaluation:
        st.error(f"모델 학습 오류: {e}")


with tab_analysis:
    # =====================================================
    # 21. 탐구 주제에 따른 종합 해석
    # =====================================================

    st.divider()
    st.header("14. 탐구 주제에 따른 종합 해석")

    st.subheader(
        "탐구 주제: 국내 애니메이션산업의 지역별 매출액과 "
        "종사자 수는 어떤 관계가 있으며, 사업체 수는 "
        "이러한 관계와 어떻게 연결되는가?"
    )

    interpretations = []

    # 연도별 사업체 수 변화
    if len(yearly) >= 2:
        first = yearly.iloc[0]
        last = yearly.iloc[-1]

        if first["사업체수"] != 0:
            company_change = (
                (last["사업체수"] - first["사업체수"])
                / first["사업체수"] * 100
            )

            interpretations.append(
                f"① 사업체 수 변화: {int(first['연도'])}년부터 "
                f"{int(last['연도'])}년까지 분석 대상 지역의 "
                f"사업체 수 합계는 "
                f"{format_change(company_change)} 했습니다."
            )

    # 매출액과 종사자 수
    if pearson is not None:
        interpretations.append(
            f"② 매출액과 종사자 수: 피어슨 상관계수는 "
            f"{pearson:.3f}이며, "
            f"{corr_description(pearson)}가 관찰됩니다."
        )
    else:
        interpretations.append(
            "② 매출액과 종사자 수: 자료가 부족하거나 변동이 없어 "
            "상관관계를 계산할 수 없습니다."
        )

    # 사업체 수와 매출액
    if company_pearson is not None:
        interpretations.append(
            f"③ 사업체 수와 매출액: 피어슨 상관계수는 "
            f"{company_pearson:.3f}이며, "
            f"{corr_description(company_pearson)}가 관찰됩니다."
        )
    else:
        interpretations.append(
            "③ 사업체 수와 매출액: 상관관계를 계산할 수 없습니다."
        )

    # 매출액 최고 지역
    latest_year = data["연도"].max()

    latest_data = data[
        data["연도"] == latest_year
    ]

    if not latest_data.empty:
        highest_revenue = latest_data.loc[
            latest_data["매출액"].idxmax()
        ]

        highest_employee = latest_data.loc[
            latest_data["종사자수"].idxmax()
        ]

        highest_company = latest_data.loc[
            latest_data["사업체수"].idxmax()
        ]

        interpretations.append(
            f"④ 지역별 매출액: {int(latest_year)}년 "
            f"분석 대상 중 매출액이 가장 높은 지역은 "
            f"{highest_revenue['지역명']}입니다."
        )

        interpretations.append(
            f"⑤ 지역별 종사자 수: {int(latest_year)}년 "
            f"분석 대상 중 종사자 수가 가장 높은 지역은 "
            f"{highest_employee['지역명']}입니다."
        )

        interpretations.append(
            f"⑥ 지역별 사업체 수: {int(latest_year)}년 "
            f"분석 대상 중 사업체 수가 가장 높은 지역은 "
            f"{highest_company['지역명']}입니다."
        )

    for item in interpretations:
        st.write(item)

    st.subheader("종합 결론")

    if pearson is not None and company_pearson is not None:
        if pearson > 0 and company_pearson > 0:
            conclusion = (
                "분석 대상 지역에서는 종사자 수와 매출액, "
                "사업체 수와 매출액 사이에 모두 양의 상관관계가 "
                "관찰됩니다. 이는 고용 규모와 사업체 규모가 "
                "지역별 매출액과 함께 움직이는 경향이 있음을 "
                "보여줍니다."
            )
        elif pearson > 0:
            conclusion = (
                "종사자 수와 매출액 사이에는 양의 상관관계가 "
                "관찰되지만, 사업체 수와 매출액 사이에서는 "
                "같은 방향의 관계가 확인되지 않았습니다. "
                "따라서 사업체 수뿐 아니라 고용 규모도 함께 "
                "살펴볼 필요가 있습니다."
            )
        elif company_pearson > 0:
            conclusion = (
                "사업체 수와 매출액 사이에는 양의 상관관계가 "
                "관찰되지만, 종사자 수와 매출액 사이에서는 "
                "같은 방향의 관계가 확인되지 않았습니다."
            )
        else:
            conclusion = (
                "두 변수 관계에서 뚜렷한 양의 상관관계가 "
                "모두 확인되지는 않았습니다. 지역별 산업 규모를 "
                "해석할 때 다른 요인도 함께 살펴볼 필요가 있습니다."
            )

        st.write(conclusion)

    else:
        st.write(
            "현재 데이터만으로 매출액과 종사자 수, 사업체 수의 "
            "관계를 모두 확인하기 어렵습니다. 자료의 누락 여부와 "
            "지역·연도별 대응 관계를 추가로 확인해야 합니다."
        )

    st.warning(
        "위 결론은 관찰된 지역별 자료의 상관관계에 관한 것입니다. "
        "사업체 수나 종사자 수가 매출액을 직접 증가시켰다고 "
        "단정할 수 없습니다."
    )


    # =====================================================
    # 22. 보고서용 해석문 자동 생성
    # =====================================================

    st.divider()
    st.header("15. 보고서에 활용할 수 있는 분석 결과")

    report_lines = []

    report_lines.append(
        "이번 탐구에서는 2020년부터 2022년까지의 "
        "국내 애니메이션산업 자료를 활용하여 "
        "지역별 매출액, 종사자 수, 사업체 수의 관계를 분석하였다."
    )

    if "job_revenue_comparison" in locals() and not job_revenue_comparison.empty:
        report_lines.append(
            "보조 분석에서는 전국 매출액과 직무별 종사자 수의 연도별 변화를 "
            "비교하여 산업 내부의 인력 구성 변화도 살펴보았다. "
            "이는 두 지표의 변화 양상을 비교한 것이며, 직무별 인력 변화가 "
            "매출액 변화의 원인이라는 의미는 아니다."
        )

    if len(yearly) >= 2:
        first = yearly.iloc[0]
        last = yearly.iloc[-1]

        report_lines.append(
            f"사업체 수는 {int(first['연도'])}년 "
            f"{first['사업체수']:,.0f}개에서 "
            f"{int(last['연도'])}년 "
            f"{last['사업체수']:,.0f}개로 변화하였다."
        )

    if pearson is not None:
        report_lines.append(
            f"매출액과 종사자 수의 피어슨 상관계수는 "
            f"{pearson:.3f}으로, "
            f"{corr_description(pearson)}가 나타났다."
        )

    if company_pearson is not None:
        report_lines.append(
            f"사업체 수와 매출액의 피어슨 상관계수는 "
            f"{company_pearson:.3f}으로, "
            f"{corr_description(company_pearson)}가 나타났다."
        )

    report_lines.append(
        "지역별 막대그래프와 상자그림을 통해 지역 간 산업 규모와 "
        "분포의 차이를 비교하고, 산점도와 상관계수를 이용하여 "
        "매출액과 고용 및 사업체 규모 사이의 관계를 확인하였다."
    )

    report_lines.append(
        "이러한 결과는 지역별 애니메이션산업의 규모를 파악하는 데 "
        "도움이 되지만, 상관관계만으로 변수 사이의 인과관계를 "
        "확정할 수는 없다."
    )

    report_text = "\n\n".join(report_lines)

    st.text_area(
        "자동 생성된 보고서용 해석문",
        value=report_text,
        height=250
    )

    st.download_button(
        "보고서용 해석문 다운로드",
        data=report_text,
        file_name="애니메이션산업_분석결과.txt",
        mime="text/plain"
    )


    # =====================================================
    # 23. 데이터 다운로드
    # =====================================================

    st.divider()
    st.header("16. 전처리 데이터 다운로드")

    st.download_button(
        "지역별 통합 데이터 CSV 다운로드",
        data=data.to_csv(
            index=False,
            encoding="utf-8-sig"
        ),
        file_name="애니메이션산업_지역별_통합데이터.csv",
        mime="text/csv"
    )

    st.download_button(
        "직무별 전처리 데이터 CSV 다운로드",
        data=job.to_csv(
            index=False,
            encoding="utf-8-sig"
        ),
        file_name="애니메이션산업_직무별_전처리데이터.csv",
        mime="text/csv"
    )

    st.success(
        "데이터 전처리, 시각화, 통계분석 및 해석이 완료되었습니다."
    )
