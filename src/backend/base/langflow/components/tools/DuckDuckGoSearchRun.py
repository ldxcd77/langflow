from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field
from langchain_community.tools import DuckDuckGoSearchRun
from langflow.base.langchain_utilities.model import LCToolComponent
from langflow.inputs import MessageTextInput, IntInput
from langflow.schema import Data
from langflow.field_typing import Tool
from langchain.tools import StructuredTool

class DuckDuckGoSearchComponent(LCToolComponent):
    display_name: str = "DuckDuckGo Search"
    description: str = "Perform web searches using the DuckDuckGo search engine with result limiting"
    name = "DuckDuckGoSearch"
    documentation: str = "https://python.langchain.com/docs/integrations/tools/ddg"
    icon: str = "DuckDuckGo"
    inputs = [
        MessageTextInput(
            name="input_value",
            display_name="Search Query",
            required=True,
        ),
        IntInput(name="max_results", display_name="Max Results", value=5, advanced=True),
        IntInput(name="max_snippet_length", display_name="Max Snippet Length", value=100, advanced=True),
    ]

    class DuckDuckGoSearchSchema(BaseModel):
        query: str = Field(..., description="The search query")
        max_results: int = Field(5, description="Maximum number of results to return")
        max_snippet_length: int = Field(100, description="Maximum length of each result snippet")

    def _build_wrapper(self):
        return DuckDuckGoSearchRun()

    def build_tool(self) -> Tool:
        wrapper = self._build_wrapper()

        def search_func(
            query: str, max_results: int = 5, max_snippet_length: int = 100
        ) -> List[Dict[str, Any]]:
            full_results = wrapper.run(f"{query} (site:*)")
            result_list = full_results.split('\n')[:max_results]
            limited_results = []
            for result in result_list:
                limited_result = {
                    "snippet": result[:max_snippet_length],
                }
                limited_results.append(limited_result)
            return limited_results

        tool = StructuredTool.from_function(
            name="duckduckgo_search",
            description="Search for recent results using DuckDuckGo with result limiting",
            func=search_func,
            args_schema=self.DuckDuckGoSearchSchema,
        )
        self.status = f"DuckDuckGo Search Tool created"
        return tool

    def run_model(self) -> List[Data]:
        tool = self.build_tool()
        results = tool.run(
            {
                "query": self.input_value,
                "max_results": self.max_results,
                "max_snippet_length": self.max_snippet_length,
            }
        )
        data_list = [Data(data=result, text=result.get("snippet", "")) for result in results]
        self.status = data_list
        return data_list
