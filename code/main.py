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
