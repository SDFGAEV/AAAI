"""
GPT-4o Planning Model for XENON — OpenAI-compatible API.
Replaces the commented-out import in agent.py.
"""
import json, os, time, urllib.request
import logging

logger = logging.getLogger(__name__)

# Default to the cloud API from experiment_main.py
DEFAULT_API_KEY = "sk-B6fd5mbqOslBVT1p75Cel2vMWaZfNLkUD3Vjl0By6fZIlmOW"
DEFAULT_BASE_URL = "https://api.vectorengine.ai/v1"
DEFAULT_MODEL = "gpt-4o"


class PlanningModel:
    """Drop-in GPT-4o planner matching XENON's PlanningModel interface."""

    def __init__(self, api_key=None, base_url=None, model=None):
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", DEFAULT_API_KEY)
        self.base_url = base_url or os.environ.get("OPENAI_BASE_URL", DEFAULT_BASE_URL)
        self.model = model or os.environ.get("OPENAI_MODEL", DEFAULT_MODEL)
        logger.info(f"GPT4PlanningModel: {self.model} @ {self.base_url}")

    def _call(self, system_prompt, user_prompt, max_tokens=1024, temperature=0.0, retries=3):
        """Call the OpenAI-compatible chat API."""
        for attempt in range(retries):
            try:
                data = json.dumps({
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                }).encode('utf-8')
                req = urllib.request.Request(
                    f"{self.base_url}/chat/completions", data=data,
                    headers={
                        "Content-Type": "application/json",
                        "Authorization": f"Bearer {self.api_key}"
                    }
                )
                resp = urllib.request.urlopen(req, timeout=120)
                return json.loads(resp.read())["choices"][0]["message"]["content"]
            except Exception as e:
                logger.warning(f"GPT call attempt {attempt+1}/{retries}: {e}")
                if attempt < retries - 1:
                    time.sleep(5)
        return "[API error]"

    def decomposed_plan(self, waypoint, rgb_obs, similar_wp_sg_dict=None, failed_sg_list_for_wp=None):
        """Generate a decomposed subgoal plan for a waypoint.
        Returns: (plan_json_str, prompt_str)"""
        context = ""
        if similar_wp_sg_dict:
            context += f"Previously succeeded subgoals for similar waypoints: {json.dumps(similar_wp_sg_dict)}\n"
        if failed_sg_list_for_wp:
            context += f"Previously failed subgoals: {failed_sg_list_for_wp}\n"

        system = (
            "You are a Minecraft planning agent. Output a JSON array of subgoal objects. "
            "Each subgoal: {\"task\": \"<action>\", \"goal\": [\"<item>\", <count>]}. "
            "Actions: craft, mine, smelt, equip, place. "
            "Return ONLY valid JSON, no explanation."
        )
        user = f"Generate a subgoal plan to achieve the waypoint: '{waypoint}'.\n{context}"

        plan = self._call(system, user, max_tokens=512)
        prompt = f"[GPT-4o plan for {waypoint}]"
        return plan, prompt

    def planning(self, task, rgb_obs, example=None, visual_info=None, graph=None):
        """Generate a plan for a task."""
        ctx = ""
        if example: ctx += f"Example: {example}\n"
        if graph: ctx += f"Recipe graph: {graph}\n"
        if visual_info: ctx += f"Visual info: {visual_info}\n"

        system = "You are a Minecraft planner. Output a step-by-step plan as a JSON array. Act only: craft, mine, smelt, equip, place."
        user = f"Plan to: {task}\n{ctx}"
        return self._call(system, user, max_tokens=512)

    def replan(self, task, rgb_obs, error_info=None, examples=None, graph=None):
        """Replan after failure."""
        return self.planning(f"(Replan after failure: {error_info}) {task}", rgb_obs, examples, graph=graph)

    def context_aware_reasoning(self, task, goal, rgb_obs):
        """Generate context-aware reasoning for a task."""
        system = "You are a Minecraft reasoning agent. Describe what you see and how to approach the task."
        user = f"Task: {task}\nGoal: {goal}"
        reasoning = self._call(system, user, max_tokens=256)
        return reasoning, f"[visual: {task}]"

    def fix_json_format(self, errorneous_planning, rgb_obs):
        """Fix malformed JSON plan."""
        system = "Fix this JSON to be valid. Output ONLY the corrected JSON."
        user = f"Fix: {errorneous_planning}"
        return self._call(system, user, max_tokens=256)
