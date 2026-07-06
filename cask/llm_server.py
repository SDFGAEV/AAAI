"""Minimal LLM server matching XENON ServerAPI protocol. Wraps GPT4PlanningModel."""
import sys, os, json, logging
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from flask import Flask, request, jsonify
app = Flask(__name__)
logger = logging.getLogger("LLMServer")
logging.basicConfig(level=logging.INFO)

from optimus1.models.gpt4_planning import PlanningModel
planner = PlanningModel()

@app.route("/reset", methods=["GET"])
def reset():
    return jsonify({"status": "ok"})

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    chat_type = data.get("type", "plan")
    waypoint = data.get("waypoint", "")
    task = data.get("task_or_instruction", "")
    goal = data.get("goal", "")
    error_info = data.get("error_info", "")
    example = data.get("example", "")
    graph = data.get("graph", "")

    logger.info(f"Chat request: type={chat_type} waypoint={waypoint[:60]} task={task[:60]}")

    try:
        if chat_type == "decomposed_plan":
            plan, prompt = planner.decomposed_plan(
                waypoint, "", data.get("similar_wp_sg_dict"),
                data.get("failed_sg_list_for_wp"))
            return jsonify({"response": plan, "message": prompt})

        elif chat_type == "context_aware_reasoning":
            reasoning, visual = planner.context_aware_reasoning(task, goal, "")
            return jsonify({"response": reasoning, "message": visual})

        elif chat_type == "plan":
            plan = planner.planning(task, "", example, None, graph)
            return jsonify({"response": plan, "message": ""})

        elif chat_type == "replan":
            plan = planner.replan(task, "", error_info, example, graph)
            return jsonify({"response": plan, "message": ""})

        else:
            plan = planner.planning(task if task else waypoint, "")
            return jsonify({"response": plan, "message": ""})

    except Exception as e:
        logger.error(f"Chat error: {e}")
        return jsonify({"response": f"[error: {e}]", "message": ""}), 200


@app.route("/shutdown", methods=["POST"])
def shutdown():
    return jsonify({"status": "shutting down"})


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 9000
    print(f"LLM Server starting on port {port}")
    app.run(host="0.0.0.0", port=port, debug=False)
