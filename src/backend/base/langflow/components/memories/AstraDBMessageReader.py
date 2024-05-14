from typing import Optional, cast

from langchain_community.chat_message_histories.astradb import AstraDBChatMessageHistory

from langflow.base.memory.memory import BaseMemoryComponent
from langflow.field_typing import Text
from langflow.schema.schema import Record


class AstraDBMessageReaderComponent(BaseMemoryComponent):
    display_name = "Astra DB Message Reader"
    description = "Retrieves stored chat messages from Astra DB."

    def build_config(self):
        return {
            "session_id": {
                "display_name": "Session ID",
                "info": "Session ID of the chat history.",
                "input_types": ["Text"],
            },
            "token": {
                "display_name": "Astra DB Application Token",
                "info": "Token for the Astra DB instance.",
                "password": True,
            },
            "api_endpoint": {
                "display_name": "Astra DB API Endpoint",
                "info": "API Endpoint for the Astra DB instance.",
                "password": True,
            },
            "query": {
                "display_name": "Query",
                "info": "Query to search for in the chat history.",
            },
            "metadata": {
                "display_name": "Metadata",
                "info": "Optional metadata to attach to the message.",
                "advanced": True,
            },
            "limit": {
                "display_name": "Limit",
                "info": "Limit of search results.",
                "advanced": True,
            },
        }

    def get_messages(self, **kwargs) -> list[Record]:
        """
        Retrieves messages from the AstraDBChatMessageHistory memory.

        If a query is provided, the search method is used to search for messages in the memory, otherwise all messages are returned.

        Args:
            memory (AstraDBChatMessageHistory): The AstraDBChatMessageHistory instance to retrieve messages from.

        Returns:
            list[Record]: A list of Record objects representing the search results.
        """
        memory: AstraDBChatMessageHistory = cast(AstraDBChatMessageHistory, kwargs.get("memory"))
        if not memory:
            raise ValueError("AstraDBChatMessageHistory instance is required.")

        # Get messages from the memory
        messages = memory.messages
        results = [Record.from_lc_message(message) for message in messages]

        return list(results)

    def build(
        self,
        session_id: Text,
        collection_name: str,
        token: str,
        api_endpoint: str,
        namespace: Optional[str] = None,
    ) -> list[Record]:
        try:
            import astrapy
        except ImportError:
            raise ImportError(
                "Could not import astrapy package. " "Please install it with `pip install astrapy`."
            )

        memory = AstraDBChatMessageHistory(
            session_id=session_id,
            collection_name=collection_name,
            token=token,
            api_endpoint=api_endpoint,
            namespace=namespace,
        )

        records = self.get_messages(memory=memory)
        self.status = records

        return records
