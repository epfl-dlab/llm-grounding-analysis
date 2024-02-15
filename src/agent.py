from abc import ABC

import colorama
from typing import List

from langchain import PromptTemplate
from langchain.callbacks import get_openai_callback
from langchain.chat_models import ChatOpenAI
from langchain.schema import get_buffer_string

from typing import Dict, Optional

from langchain.memory import ChatMessageHistory
from langchain.schema import SystemMessage
import torch

from transformers import pipeline

colorama.init()


class AgentMessageHistory(ChatMessageHistory):
    has_system_message: bool = False

    def __init__(self, system_message: str = None):
        super().__init__()
        if system_message is not None:
            self.set_system_message(system_message)

    def set_system_message(self, message: str) -> None:
        if message is not None:
            if self.has_system_message:
                self.messages[0] = SystemMessage(content=message)
            else:
                self.messages.insert(0, SystemMessage(content=message))
                self.has_system_message = True

    def __str__(self):
        text = get_buffer_string(self.messages)
        return text


class Agent(ABC):
    def __init__(
        self,
        name: str,
        generation_parameters: Dict,
        human_message_prompt_template: str,
        system_message: str = None,
        demonstrations: List[dict] = None,
        demonstrations_response_template: PromptTemplate = None,
        verbose: Optional[bool] = True,
    ):
        self.name = name

        # ~~~ Generation ~~~
        self.generation_parameters = generation_parameters

        # ~~~ Prompting ~~~
        self.system_message = system_message
        self.human_message_prompt_template = PromptTemplate.from_template(
            template=human_message_prompt_template, template_format="jinja2", validate_template=False
        )

        # ~~~ Demonstrations ~~~
        self.demonstrations = demonstrations
        self.demonstrations_response_template = demonstrations_response_template

        # ~~~ Debugging ~~~
        self.verbose = verbose

        # ~~~ Cost ~~~
        self.total_cost = 0
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0

    def _get_human_message_content(self, sample_data: Dict):
        return self.human_message_prompt_template.format(**sample_data)

    def _get_response_message_content(self, sample_data: Dict):
        return self.demonstrations_response_template.format(**sample_data)

    def _add_demonstrations(self, message_history: AgentMessageHistory):
        if self.demonstrations is not None:
            for example in self.demonstrations:
                query = self._get_human_message_content(example["query"])
                response = self._get_response_message_content(example["response"])

                message_history.add_user_message(message=query)

                message_history.add_ai_message(message=response)

    def _initialize_conversation(self, message_history: AgentMessageHistory, system_message: str = None):
        # ~~~ Sanity check ~~~
        conversation_messages = message_history.messages
        assert len(conversation_messages) == 0, f"Conversation already has {len(conversation_messages)} messages"

        # ~~~ Add the system message ~~~
        message_history.set_system_message(
            message=system_message,
        )

        # ~~~ Add the demonstration query-response tuples (if any) ~~~
        self._add_demonstrations(message_history)

    def _call(self, message_history: AgentMessageHistory):
        raise NotImplementedError

    def __call__(
        self,
        query_data: Dict,
        dry_run: bool = False,
    ):
        message_history = AgentMessageHistory()

        self._initialize_conversation(message_history, self.system_message)

        # ~~~ Construct the query from a query message ~~~
        query_message_content = self._get_human_message_content(query_data)
        message_history.add_user_message(message=query_message_content)

        if dry_run:
            print(
                f"\n{colorama.Fore.MAGENTA}~~~ Chat History ~~~\n" f"{colorama.Style.RESET_ALL}{str(message_history)}"
            )
            return {"chat_history": message_history, "response_message": None}

        # ~~~ Get response ~~~
        response_message, response_message_content = self._call(message_history=message_history)

        return {
            "query_message": query_message_content,
            "response_message": response_message,
            "response_content": response_message_content,
            "chat_history": message_history,
        }

    def get_name(self):
        return self.name


class OpenAIAgent(Agent):
    def __init__(
        self,
        api_key: str,
        model_name: str,
        generation_parameters: Dict,
        system_message: str,
        human_message_prompt_template: str,
        demonstrations: List[dict] = None,
        demonstrations_response_template: PromptTemplate = None,
        verbose: Optional[bool] = True,
    ):
        # ~~~ Model generation ~~~
        generation_parameters["max_tokens"] = generation_parameters.pop("max_new_tokens")
        super().__init__(
            model_name,
            generation_parameters,
            human_message_prompt_template,
            system_message,
            demonstrations,
            demonstrations_response_template,
            verbose,
        )
        self.api_key = api_key

    def _get_human_message_content(self, sample_data: Dict):
        return self.human_message_prompt_template.format(**sample_data)

    def _get_response_message_content(self, sample_data: Dict):
        return self.demonstrations_response_template.format(**sample_data)

    def _add_demonstrations(self, message_history: AgentMessageHistory):
        if self.demonstrations is not None:
            for example in self.demonstrations:
                query = self._get_human_message_content(example["query"])
                response = self._get_response_message_content(example["response"])

                message_history.add_user_message(message=query)

                message_history.add_ai_message(message=response)

    def _initialize_conversation(self, message_history: AgentMessageHistory, system_message: str = None):
        # ~~~ Sanity check ~~~
        conversation_messages = message_history.messages
        assert len(conversation_messages) == 0, f"Conversation already has {len(conversation_messages)} messages"

        # ~~~ Add the system message ~~~
        message_history.set_system_message(
            message=system_message,
        )

        # ~~~ Add the demonstration query-response tuples (if any) ~~~
        self._add_demonstrations(message_history)

    def _call(self, message_history: AgentMessageHistory):
        with get_openai_callback() as cb:
            backend = ChatOpenAI(model_name=self.name, openai_api_key=self.api_key, **self.generation_parameters)
            messages = message_history.messages
            response = backend(messages)
            response.content = response.content
            message_history.add_ai_message(message=response.content)
            self.total_cost += cb.total_cost
            self.total_prompt_tokens += cb.prompt_tokens
            self.total_completion_tokens += cb.completion_tokens
            if self.verbose:
                messages_str = str(message_history)
                print(f"\n{colorama.Fore.MAGENTA}~~~ Chat History ~~~\n" f"{colorama.Style.RESET_ALL}{messages_str}")
                print("Number of tokens in the generated response:", cb.total_tokens)
                print("Cumulative cost in dollars:", self.total_cost)
                print(f"Cumulative prompt tokens: {self.total_prompt_tokens}")
                print(f"Cumulative completion Tokens: {self.total_completion_tokens}")

        return response, response.content


class HFAgent(Agent):
    def __init__(
        self,
        model_path: str,
        tokenizer_path: str,
        generation_parameters: Dict,
        system_message: str,
        human_message_prompt_template: str,
        demonstrations: List[dict] = None,
        demonstrations_response_template: PromptTemplate = None,
        bfloat16=True,
        merge_system_message=False,
        verbose: Optional[bool] = True,
    ):
        # ~~~ Model generation ~~~
        # Extract the model name from the checkpoint directory
        model_name = model_path.split("/")[-1]

        if merge_system_message:
            human_message_prompt_template = system_message + "\n\n" + human_message_prompt_template
            system_message = None

        if "temperature" in generation_parameters and generation_parameters["temperature"] == 0.0:
            generation_parameters.pop("temperature")
            generation_parameters["do_sample"] = False

        super().__init__(
            model_name,
            generation_parameters,
            human_message_prompt_template,
            system_message,
            demonstrations,
            demonstrations_response_template,
            verbose,
        )

        self.model = pipeline(
            "conversational",
            model=model_path,
            tokenizer=tokenizer_path,
            device_map="auto",
            torch_dtype=torch.bfloat16 if bfloat16 else torch.float32,
        )

    def _call(self, message_history: AgentMessageHistory):
        dialog = [
            {"role": message.type if message.type != "human" else "user", "content": message.content}
            for message in message_history.messages
        ]

        results = self.model(dialog, **self.generation_parameters)[-1]
        response_content = results["content"]

        message_history.add_ai_message(message=response_content)
        if self.verbose:
            messages_str = str(message_history)
            print(f"\n{colorama.Fore.MAGENTA}~~~ Chat History ~~~\n" f"{colorama.Style.RESET_ALL}{messages_str}")

        return results, response_content
