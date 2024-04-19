from typing import List, Tuple, Optional
from openai import OpenAI
import utils
from generator import Generator
from prompts import (standalone_question_prompt,
                     standalone_question_answer_prompt, question_answer_prompt)

MODELS = {
    "gpt3.5": "gpt-3.5-turbo-1106",
    "gpt4": "gpt-4-0125-preview"
}


class OpenAIGenerator(Generator):
    # will get from os.environ.get("OPENAI_API_KEY") by default
    api_key: Optional[str] = None

    def __init__(self, **data):
        super().__init__(**data)
        self.create_client()

    def create_client(self):
        self.client = OpenAI(api_key=self.api_key)

    def format_chat_history(self, chat_history: List[Tuple[str, str]]) -> str:
        buffer = ""
        for human_turn, ai_turn in chat_history:
            human = f"user: {human_turn}"
            ai = f"assistant: {ai_turn}"
            buffer += f"\n{human}\n{ai}"
        return buffer

    def call_openai(self, prompt: str, model: str = "gpt3.5") -> str:
        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": prompt})
        response = self.client.chat.completions.create(
            model=MODELS[model],
            messages=messages
        )

        try:
            answer = response.choices[0].message.content
            if answer is None:
                raise ValueError(f"Chat completion for: {prompt}\n returned None.")
        except KeyError:
            raise KeyError(f"Chat completion for: {prompt}\n had a key error when accessing the response object.")

        return answer

    def answer_standalone_question(self, query: str, chat_history: str) -> str:
        formatted_sq_prompt = standalone_question_prompt.format(chat_history=chat_history, question=query)
        standalone_query = self.call_openai(formatted_sq_prompt)
        relevant_docs = self.retriever.retrieve_similar_docs(standalone_query)
        context_str = utils.format_docs(relevant_docs)
        formatted_sqa_prompt = standalone_question_answer_prompt.format(context=context_str, question=standalone_query)
        answer = self.call_openai(formatted_sqa_prompt)
        return answer

    def answer_user_question(self, query: str, chat_history: List[Tuple[str, str]]) -> str:
        chat_history_str = self.format_chat_history(chat_history)

        try:
            if self.standalone:
                answer = self.answer_standalone_question(query, chat_history_str)
            else:
                relevant_docs = self.retriever.retrieve_similar_docs(query)
                context_str = utils.format_docs(relevant_docs)
                formatted_qa_prompt = question_answer_prompt.format(chat_history=chat_history_str,
                                                                    context=context_str, question=query)
                answer = self.call_openai(formatted_qa_prompt)
        except Exception as e:
            answer = f"Question failed due to:\n {e.args[0]}"

        return answer
