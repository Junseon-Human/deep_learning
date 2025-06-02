from crewai.tools import BaseTool
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.document_loaders import WebBaseLoader
from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
openai_client = OpenAI(api_key=openai_api_key)


# 1. 덕덕고 검색도구 정의
class DuckDuckGoSearchTool(BaseTool):
    name: str = "DuckDuckGo Search"
    description: str = "DuckDuckGo 에서 최신 정보를 검색합니다."

    def _run(self, query: str) -> str:
        duckduckgo_tool = DuckDuckGoSearchRun()
        response = duckduckgo_tool.invoke(query)
        return response


# 2. 웹 스크래퍼 도구 정의
class WebScraperTool(BaseTool):
    name: str = "웹 스크래퍼"
    description: str = "웹페이지 내용을 추출합니다."

    def _run(self, url: str) -> str:
        loader = WebBaseLoader(url)
        docs = loader.load()
        return docs[0].page_content if docs else "내용을 찾을 수 없습니다."


# 3. 계산기 도구 정의
class CalculatorTool(BaseTool):
    name: str = "계산기"
    description: str = "수학 계산을 수행합니다."

    def _run(self, expression: str) -> str:
        try:
            return str(eval(expression))
        except Exception as e:
            return f"계산 오류 : {e}"
