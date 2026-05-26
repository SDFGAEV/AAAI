import json
import os

# Bypass macOS system proxy (Clash/Surge port 1082) for localhost connections
os.environ["NO_PROXY"] = "127.0.0.1,localhost"
os.environ["no_proxy"] = "127.0.0.1,localhost"

# Load API key from MindCraft keys.json
keys_path = os.path.join(os.path.dirname(__file__),
    "../MC复现/mindcraft-develop/keys.json")
with open(keys_path) as f:
    keys = json.load(f)

from voyager import Voyager

voyager = Voyager(
    mc_port=55916,                          # MindCraft server port
    openai_api_key=keys["OPENAI_API_KEY"],
    action_agent_model_name="gpt-4o",
    curriculum_agent_model_name="gpt-4o",
    curriculum_agent_qa_model_name="gpt-4o-mini",
    critic_agent_model_name="gpt-4o",
    skill_manager_model_name="gpt-4o-mini",
    max_iterations=5,                       # sanity run: 5轮即可
    openai_api_request_timeout=60,          # 单次 LLM 调用最多 60 秒
    ckpt_dir=os.path.join(os.path.dirname(__file__), "ckpt/baseline_run2"),
)

voyager.learn()
