# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.3
#   kernelspec:
#     display_name: pyrit-kernel
#     language: python
#     name: pyrit-kernel
# ---

# %%
from pyrit.prompt_target.prompt_chat_target.openai_chat_target import AzureOpenAITextChatAttackTarget
from pyrit.common import default_values
from pyrit.memory import DuckDBMemory
import uuid

system_prompt = "You are an assistant whose goal is to flatter me."

default_values.load_default_env()

target = AzureOpenAITextChatAttackTarget(use_aad_auth=False, memory=DuckDBMemory(), system_prompt=system_prompt)

# %%
conversation_id = uuid.uuid4()  # makes sure this is unique per conversation

while True:
    prompt = input("Enter your prompt: ")
    print(f"Request: {prompt}")
    response = await target.send_chat_prompt_async(prompt=prompt, conversation_id=conversation_id)
    print(f"Response: {response.request_pieces[0].converted_value}")
    if input("Continue conversation? (y/n) ").lower() != "y":
        break

# %%
