import urllib
from typing import Optional

import requests
from langchain.pydantic_v1 import BaseModel, Field, create_model
from langchain_core.tools import StructuredTool

from langflow.base.langchain_utilities.model import LCToolComponent
from langflow.io import DictInput, IntInput, SecretStrInput, StrInput
from langflow.schema import Data


class AstraCQLToolComponent(LCToolComponent):
    display_name: str = "Astra DB CQL Tool"
    description: str = "Create a tool to get data from DataStax Astra DB CQL Table"
    documentation: str = "https://astra.datastax.com"
    icon: str = "AstraDB"

    inputs = [
        StrInput(name="tool_name", display_name="Tool Name", info="The name of the tool.", required=True),
        StrInput(
            name="tool_description",
            display_name="Tool Description",
            info="The tool description to be passed to the model.",
            required=True,
        ),
        StrInput(
            name="keyspace",
            display_name="Keyspace",
            value="default_keyspace",
            info="The keyspace name within Astra DB where the data is stored.",
            required=True,
        ),
        StrInput(
            name="table",
            display_name="Table Name",
            info="The name of the table within Astra DB where the data is stored.",
            required=True,
        ),
        SecretStrInput(
            name="token",
            display_name="Astra DB Application Token",
            info="Authentication token for accessing Astra DB.",
            value="ASTRA_DB_APPLICATION_TOKEN",
            required=True,
        ),
        StrInput(
            name="api_endpoint",
            display_name="API Endpoint",
            info="API endpoint URL for the Astra DB service.",
            value="ASTRA_DB_API_ENDPOINT",
            required=True,
        ),
        StrInput(
            name="projection",
            display_name="Projection fields",
            info="Attributes to return separated by comma.",
            required=True,
            value="*",
        ),
        DictInput(
            name="partition_keys",
            display_name="Partition Keys",
            is_list=True,
            info="Field name and description to the model",
            required=True,
        ),
        DictInput(
            name="clustering_keys",
            display_name="Clustering Keys",
            is_list=True,
            info="Field name and description to the model",
        ),
        DictInput(
            name="static_filters",
            display_name="Static Filters",
            is_list=True,
            info="Field name and value. When filled, it will not be generated by the LLM.",
        ),
        IntInput(name="limit", display_name="Limit", value=5),
    ]

    def astra_rest(self, args):
        headers = {"Accept": "application/json", "X-Cassandra-Token": f"{self.token}"}
        ASTRA_URL = f"{self.api_endpoint}/api/rest/v2/keyspaces/{self.keyspace}/{self.table}/"
        key = []
        # Partition keys are mandatory
        for k in self.partition_keys.keys():
            if k in args.keys():
                key.append(args[k])
            elif self.static_filters[k] is not None:
                key.append(self.static_filters[k])
            else:
                # TO-DO: Raise error - Missing information
                key.append("none")

        # Clustering keys are optional
        for k in self.clustering_keys.keys():
            if k in args.keys():
                key.append(args[k])
            elif self.static_filters[k] is not None:
                key.append(self.static_filters[k])

        url = f'{ASTRA_URL}{"/".join(key)}?page-size={self.limit}'

        if self.projection != "*":
            url += f'&fields={urllib.parse.quote(self.projection.replace(" ","")) }'

        print("=" * 10)
        print(url)

        res = requests.request("GET", url=url, headers=headers)

        if int(res.status_code) >= 400:
            return res.text

        try:
            res_data = res.json()
            return res_data["data"]
        except ValueError:
            res_data = res.status_code
            return res_data

    def create_args_schema(self) -> dict[str, BaseModel]:
        args = {}

        for key in self.partition_keys.keys():
            # Partition keys are mandatory is it doesn't have a static filter
            if key not in self.static_filters.keys():
                args[key] = (str, Field(description=self.partition_keys[key]))

        for key in self.clustering_keys.keys():
            # Partition keys are mandatory if has the exclamation mark and doesn't have a static filter
            if key not in self.static_filters.keys():
                if key.startswith("!"):  # Mandatory
                    args[key[1:]] = (str, Field(description=self.clustering_keys[key]))
                else:  # Optional
                    args[key] = (Optional[str], Field(description=self.clustering_keys[key], default=None))

        model = create_model("ToolInput", **args, __base__=BaseModel)
        return {"ToolInput": model}

    def build_tool(self) -> StructuredTool:
        """
        Builds a Astra DB CQL Table tool.

        Args:
            name (str, optional): The name of the tool.

        Returns:
            Tool: The built AstraDB tool.
        """
        schema_dict = self.create_args_schema()
        return StructuredTool.from_function(
            name=self.tool_name,
            args_schema=schema_dict["ToolInput"],
            description=self.tool_description,
            func=self.run_model,
            return_direct=False,
        )

    def projection_args(self, input_str: str) -> dict:
        elements = input_str.split(",")
        result = {}

        for element in elements:
            if element.startswith("!"):
                result[element[1:]] = False
            else:
                result[element] = True

        return result

    def run_model(self, **args) -> Data | list[Data]:
        results = self.astra_rest(args)
        self.status = results
        return results
