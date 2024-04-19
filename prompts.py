standalone_question_prompt = """
Given the following conversation and a follow up question, rephrase the
follow up question to be a standalone question, in its original language.

Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:"""

standalone_question_answer_prompt = """
Answer the question based only on the following context:
{context}

Question: {question}
"""

question_answer_prompt = """
Given the following chat history, context, and question please
answer the question based only on the following context:

Chat History:
{chat_history}

Context:
{context}

Question: {question}
"""
