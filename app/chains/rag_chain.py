"""
Conversational RAG chain -- optimised for low latency with local Ollama models.
"""
from __future__ import annotations
from typing import Iterator, List, Tuple
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from app import config
from app.retrieval.vector_store import VectorStoreManager
from app.utils.logger import get_logger

logger = get_logger(__name__)

_QA_SYSTEM = """You are a concise knowledge assistant. Answer ONLY from the context below.
- If the answer is not in the context, say: I don't have enough information in the provided documents.
- Cite the source filename at the end: Source: <filename>
- Be brief and direct.

Context:
{context}"""


def _build_llm():
    if config.LLM_PROVIDER == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=config.OPENAI_CHAT_MODEL, temperature=config.LLM_TEMPERATURE, max_tokens=config.LLM_MAX_TOKENS, openai_api_key=config.OPENAI_API_KEY, streaming=True)
    elif config.LLM_PROVIDER == "azure_openai":
        from langchain_openai import AzureChatOpenAI
        return AzureChatOpenAI(azure_deployment=config.AZURE_CHAT_DEPLOYMENT, azure_endpoint=config.AZURE_OPENAI_ENDPOINT, api_version=config.AZURE_OPENAI_API_VERSION, openai_api_key=config.OPENAI_API_KEY, temperature=config.LLM_TEMPERATURE, max_tokens=config.LLM_MAX_TOKENS, streaming=True)
    elif config.LLM_PROVIDER == "ollama":
        from langchain_ollama import ChatOllama
        return ChatOllama(model=config.OLLAMA_CHAT_MODEL, base_url=config.OLLAMA_BASE_URL, temperature=config.LLM_TEMPERATURE, num_predict=config.LLM_MAX_TOKENS)
    raise ValueError(f"Unknown LLM_PROVIDER: {config.LLM_PROVIDER}")


def _format_docs(docs: List[Document]) -> str:
    return "\n\n".join(f"[{d.metadata.get('filename','unknown')}]\n{d.page_content}" for d in docs)


class ChatSession:
    def __init__(self, vs_manager: VectorStoreManager) -> None:
        self._retriever = vs_manager.as_retriever()
        self._llm = _build_llm()
        self._history: List[BaseMessage] = []
        self._qa_prompt = ChatPromptTemplate.from_messages([
            ("system", _QA_SYSTEM),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])
        self._stream_chain = self._qa_prompt | self._llm | StrOutputParser()
        logger.info("ChatSession ready provider=%s model=%s", config.LLM_PROVIDER, config.OLLAMA_CHAT_MODEL if config.LLM_PROVIDER == "ollama" else config.OPENAI_CHAT_MODEL)

    def _trim_history(self) -> List[BaseMessage]:
        max_msgs = config.MAX_CHAT_HISTORY * 2
        return self._history[-max_msgs:] if len(self._history) > max_msgs else self._history

    def _retrieve(self, question: str) -> Tuple[List[Document], str]:
        docs = self._retriever.invoke(question)
        return docs, _format_docs(docs)

    def stream_chat(self, question: str) -> Iterator[Tuple[str, List[Document]]]:
        docs, context_str = self._retrieve(question)
        full_answer: List[str] = []
        for chunk in self._stream_chain.stream({"input": question, "chat_history": self._trim_history(), "context": context_str}):
            full_answer.append(chunk)
            yield chunk, []
        answer = "".join(full_answer)
        self._history.append(HumanMessage(content=question))
        self._history.append(AIMessage(content=answer))
        yield "", docs

    def chat(self, question: str) -> Tuple[str, List[Document]]:
        docs, context_str = self._retrieve(question)
        answer = self._stream_chain.invoke({"input": question, "chat_history": self._trim_history(), "context": context_str})
        self._history.append(HumanMessage(content=question))
        self._history.append(AIMessage(content=answer))
        return answer, docs

    def reset_history(self) -> None:
        self._history.clear()

    @property
    def history(self) -> List[BaseMessage]:
        return list(self._history)