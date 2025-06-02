# pip install streamlit crewai duckduckgo-search langchain-openai
import streamlit as st
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process
from crewai.tools import BaseTool
from langchain_community.tools import DuckDuckGoSearchRun

# 환경 변수 로드
load_dotenv()


# DuckDuckGo 검색 툴 정의
class MyCustomDuckDuckGoTool(BaseTool):
    name: str = "DuckDuckGo Stock News Tool"
    description: str = "특정 주식 종목에 대한 최신 뉴스를 검색하는 도구입니다."

    def _run(self, query: str) -> str:
        duckduckgo_tool = DuckDuckGoSearchRun()
        return duckduckgo_tool.invoke(query)


def create_crew(ticker, perspective):
    search_tool = MyCustomDuckDuckGoTool()

    # 뉴스 분석 에이전트
    news_agent = Agent(
        role="증시 뉴스 분석가",
        goal=f"{ticker} 관련 최신 뉴스를 수집하고 요약합니다.",
        backstory="증권사 리서치 센터에서 활동 중인 금융 뉴스 전문가입니다.",
        tools=[search_tool],
        verbose=True,
    )

    # 투자 의견 에이전트
    opinion_agent = Agent(
        role="투자 조언 전문가",
        goal=f"{perspective}에서 {ticker} 주식에 대한 의견을 작성합니다.",
        backstory="초보 투자자에게도 이해하기 쉬운 설명을 제공하는 전문가입니다.",
        verbose=True,
    )

    news_task = Task(
        description=f"{ticker}에 대한 최신 뉴스 3~5개를 수집하고 요약해 주세요. 주요 사건 위주로 간결하게 정리해 주세요.",
        agent=news_agent,
        expected_output=f"{ticker} 관련 뉴스 요약 리스트",
    )

    opinion_task = Task(
        description=(
            f"위 뉴스들을 참고하여, 현재 {ticker} 주식을 {perspective}에서 분석했을 때 "
            "주의할 점이나 기대 요인을 3~4문장으로 정리해 주세요."
        ),
        agent=opinion_agent,
        context=[news_task],
        expected_output=f"{ticker}에 대한 {perspective} 기반의 요약 투자 의견",
    )

    crew = Crew(
        agents=[news_agent, opinion_agent],
        tasks=[news_task, opinion_task],
        process=Process.sequential,
        verbose=True,
    )

    return crew


# --- Streamlit UI 구성 ---
st.set_page_config(page_title="투자 뉴스 분석기", page_icon="📈")

st.title("📈 투자 뉴스 기반 요약 분석기")
st.caption("CrewAI를 활용한 종목 뉴스 요약 + 투자 의견 생성")

ticker_input = st.text_input(
    "🔎 분석할 종목명 또는 키워드 입력", value="삼성전자 반도체"
)
perspective_input = st.selectbox(
    "🎯 투자 관점 선택", options=["장기 투자자 관점", "단기 투자자 관점"], index=0
)

if st.button("크루 실행!"):
    with st.spinner("AI 분석 중..."):
        crew = create_crew(ticker_input, perspective_input)
        result = crew.kickoff(
            inputs={"ticker": ticker_input, "perspective": perspective_input}
        )

    st.success("✅ 분석 완료!")
    st.subheader("📰 뉴스 요약")
    st.write(result.tasks_output[0].raw)
    st.subheader("📈 투자 의견")
    st.write(result.tasks_output[1].raw)
