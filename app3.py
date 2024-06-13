from qdrant_client import QdrantClient
import autogen
from autogen.agentchat.contrib.qdrant_retrieve_user_proxy_agent import QdrantRetrieveUserProxyAgent
from autogen.agentchat.contrib.retrieve_assistant_agent import RetrieveAssistantAgent
from autogen.retrieve_utils import TEXT_FORMATS

config_list = autogen.config_list_from_json(
    env_or_file="OAI_CONFIG_LIST"
)

# 1. create an RetrieveAssistantAgent instance named "assistant"
assistant = RetrieveAssistantAgent(
    name="assistant",
    system_message="You are a helpful assistant.",
    llm_config={
        "timeout": 600,
        "cache_seed": 42,
        "config_list": config_list,
    },
)

# 2. create the QdrantRetrieveUserProxyAgent instance named "ragproxyagent"
ragproxyagent = QdrantRetrieveUserProxyAgent(
    name="ragproxyagent",
    human_input_mode="NEVER",
    max_consecutive_auto_reply=10,
    retrieve_config={
        "task": "code",
        "docs_path": "https://raw.githubusercontent.com/microsoft/autogen/main/README.md",  
        "chunk_token_size": 2000,
        "model": config_list[0]["model"],
        "client": QdrantClient(":memory:"),
        "embedding_model": "BAAI/bge-small-en-v1.5",
    },
)

assistant.reset()
qa_problem = "what is AutoGen?"
ragproxyagent.initiate_chat(assistant, problem=qa_problem)

assistant.reset()
qa_problem = "List all the benefits of AutoGen"
ragproxyagent.initiate_chat(assistant, problem=qa_problem)