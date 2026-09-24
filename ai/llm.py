from langchain_groq import ChatGroq
import os
llm = ChatGroq(
    model = "openai/gpt-oss-120b",
    temperature=0,
    max_tokens=2048,
)
