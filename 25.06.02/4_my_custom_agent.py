# pip install streamlit crewai duckduckgo-search langchain-openai yfinance
import streamlit as st
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process
from crewai.tools import BaseTool
import yfinance as yf
from langchain_community.tools import DuckDuckGoSearchRun

load_dotenv()


### --- 툴 래퍼들 --- ###
class FinanceTool(BaseTool):
    name: str = "Finance Data Tool"
    description: str = "주식 가격과 재무 지표를 가져옵니다."

    def _run(self, ticker: str) -> dict:
        info = yf.Ticker(ticker).info
        return {
            "price": info.get("currentPrice"),
            "pe": info.get("trailingPE"),
            "eps": info.get("trailingEps"),
            "beta": info.get("beta"),
        }


class MyCustomDuckDuckGoTool(BaseTool):
    name: str = "DuckDuckGo Stock News Tool"
    description: str = "특정 주식 종목에 대한 최신 뉴스를 검색하는 도구입니다."

    def _run(self, query: str) -> str:
        duckduckgo_tool = DuckDuckGoSearchRun()
        return duckduckgo_tool.invoke(query)


finance_tool = FinanceTool()
news_tool = MyCustomDuckDuckGoTool()


### --- 서브 에이전트 정의 (coworker 후보) --- ###
news_agent = Agent(
    role="News 연구원",  # ← 이 문자열과 manager_agent에서 호출할 때 쓰는 "coworker"가 100% 일치해야 합니다.
    goal="{ticker} 관련 최신 뉴스를 DuckDuckGoStockNewsTool 도구를 사용해 3줄로 요약한다.",
    backstory="금융 뉴스 크롤링과 요약에 특화된 애널리스트야. 반드시 DuckDuckGoStockNewsTool을 사용해야 합니다.",
    tools=[news_tool],
    verbose=True,
)

fund_agent = Agent(
    role="펀더멘털 애널리스트",  # ← 이 문자열과 반드시 동일하게 사용
    goal="{ticker}의 재무 상태를 FinanceDataTool 도구를 사용해서 평가한다.",
    backstory="밸류에이션 모델링 5년 차 애널리스트야. 반드시 FinanceDataTool을 사용해야 합니다.",
    tools=[finance_tool],
    verbose=True,
)

risk_agent = Agent(
    role="리스크 매니저",  # ← 이 문자열과 반드시 동일하게 사용
    goal="{ticker}의 최근 1년 치 변동성과 β를 FinanceDataTool 도구로 조회하여 위험 등급을 산출한다.",
    backstory="시장 리스크 관리 경험이 풍부한 전문가야. 반드시 FinanceDataTool을 사용해야 합니다.",
    tools=[finance_tool],
    verbose=True,
)

port_agent = Agent(
    role="포트폴리오 메이커",  # ← 이 문자열과 반드시 동일하게 사용
    goal="{budget}원 예산과 {risk} 리스크 성향에 기반하여, 위임받은 정보를 토대로 최적의 자산 비중을 추천한다.",
    backstory="자산배분 전략 컨설턴트로 활동 중이야. 위에서 받은 정보(뉴스 요약, 재무 평가, 위험 등급)를 필수로 반영해야 합니다.",
    verbose=True,
)


### --- 관리자 에이전트 정의 (manager_agent) --- ###
manager_agent = Agent(
    role="투자 분석 관리자",
    goal=(
        "{ticker} 주식을 분석하기 위해 다음과 같이 순서대로 하위 에이전트에게 작업을 위임하세요:\n\n"
        "1. “News 연구원”에게 “{ticker} 관련 최신 뉴스를 3줄 요약하고 링크 제공” 작업을 요청하고, \n"
        "   Context로 “{ticker}은 현재 상승세이며, 곧 실적 발표가 있습니다.”를 전달합니다.\n\n"
        "2. “펀더멘털 애널리스트”에게 “{ticker}의 PER과 EPS를 조회하여 투자 매력도를 평가” 작업을 요청하고, \n"
        "   필요한 Context(예: “{ticker}은 최근 실적 발표를 앞두고 있습니다.”)를 함께 전달합니다.\n\n"
        "3. “리스크 매니저”에게 “{ticker}의 최근 1년 변동성과 β를 조회하여 위험 등급을 산출” 작업을 요청합니다.\n\n"
        "4. “포트폴리오 메이커”에게 “{budget}원 예산과 {risk} 리스크 성향을 반영하여 최적의 포트폴리오 비중을 표로 제시” 작업을 요청합니다.\n\n"
        "각 위임은 반드시 DelegateWorkTool을 사용해 아래 형식으로 실행하세요:\n"
        "이 과정을 차례로 진행한 후, 모든 결과를 모아 최종 투자 리포트를 작성합니다."
    ),
    backstory="여러 전문가(서브 에이전트)에게 업무를 위임하고 총괄하는 투자 전략 컨설턴트야.",
    allow_delegation=True,
    verbose=True,
)


manager_task = Task(
    description=(
        "지금부터 {ticker}에 대해 다음 과정을 진행해:  \n"
        "1. News 연구원이 최신 뉴스 3줄 요약 및 링크 제공  \n"
        "2. 펀더멘털 애널리스트가 PER, EPS 기반 투자 매력도 평가  \n"
        "3. 리스크 매니저가 최근 1년 변동성·β로 위험 등급 산출  \n"
        "4. 포트폴리오 메이커가 예산: {budget}, 리스크: {risk} 반영해 포트폴리오 표 생성  \n"
        "이 네 결과를 종합해서, 한글로 깔끔한 최종 투자 리포트를 작성해 주세요.\n"
        "※ 하위 에이전트에게 보내는 task와 context는 반드시 순수한 텍스트(문장) 형태로만 제공해야 합니다."
    ),
    expected_output=(
        "1) 뉴스 요약 3줄  \n"
        "2) 재무 평가 결과(투자 매력도)  \n"
        "3) 위험 등급(숫자 포함)  \n"
        "4) 표 형태 포트폴리오 비중 및 이유  \n"
        "5) 위 모든 세부 결과를 한 문서로 요약한 최종 리포트"
    ),
    agent=manager_agent,
)


### --- Streamlit UI 구성 --- ###
st.set_page_config(page_title="계층적 투자 분석 앱", page_icon="💹")
st.title("💹 계층적 투자 분석")
st.caption("CrewAI를 활용한 다중 에이전트 투자 분석")

# ★★ 1) Yahoo Finance 티커 검색 섹션 ★★
st.header("📈 회사명으로 티커 검색하기")
company_name = st.text_input("회사명을 입력하여 티커를 찾아주세요", value="Apple")
if st.button("티커 검색"):
    lookup_url = f"https://finance.yahoo.com/lookup?s={company_name}"
    st.markdown(f"[🔗 여기 클릭해서 '{company_name}' 티커 찾기]({lookup_url})")

# ★★ 2) 투자 분석 입력폼 ★★
ticker_input = st.text_input("🔎 분석할 종목명 (티커)", value="AAPL")
budget_input = st.number_input(
    "💰 예산 입력 (정수)",
    min_value=100000,
    max_value=1000000000,
    step=100000,
    value=10000000,
)
risk_input = st.selectbox(
    "🎯 리스크 성향 선택", options=["낮음", "중간", "높음"], index=1
)

if st.button("크루 실행"):
    with st.spinner("AI 분석 중..."):
        crew = Crew(
            agents=[
                news_agent,
                fund_agent,
                risk_agent,
                port_agent,
            ],  # ← manager_agent는 절대로 여기 넣지 않습니다!
            tasks=[manager_task],
            process=Process.hierarchical,
            manager_agent=manager_agent,
        )
        result = crew.kickoff(
            inputs={"ticker": ticker_input, "budget": budget_input, "risk": risk_input}
        )

    st.success("✅ 분석 완료!")
    st.subheader("📋 최종 투자 리포트")
    st.write(result.raw)

    st.subheader("🔍 세부 단계별 출력")
    try:
        for idx, out in enumerate(result.tasks_output, start=1):
            st.markdown(f"**Task {idx} 결과:**")
            st.write(out.raw)
    except:
        for step in result.full_output:
            agent_name = step.get("agent_name", "알 수 없음")
            response = step.get("response", "")
            st.markdown(f"**{agent_name} 응답:**")
            st.write(response)
