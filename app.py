import os
import sys
import signal
import time
import argparse
import base64
import binascii
import hmac
import re
import random
from pathlib import Path
import numpy as np
import torch
import transformers

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import uvicorn

from optimus1.server.agent import AgentFactory
from optimus1.server.api.request import MCRequest, MCResponse
from optimus1.server.api.utils import base64_to_image, base64lst2img_path

import time

app = FastAPI()
agent = None
_PROJECT_ROOT = Path(__file__).resolve().parent
_API_TOKEN = os.getenv("CACT_API_TOKEN", "")
_MAX_REQUEST_BYTES = int(os.getenv("CACT_MAX_REQUEST_BYTES", str(16 * 1024 * 1024)))
_MAX_RETRIES = max(1, int(os.getenv("CACT_MAX_RETRIES", "5")))


@app.middleware("http")
async def _request_guard(request: Request, call_next):
    content_length = request.headers.get("content-length")
    try:
        content_length_value = int(content_length) if content_length else 0
    except ValueError:
        return JSONResponse({"detail": "invalid content-length"}, status_code=400)
    if content_length_value > _MAX_REQUEST_BYTES:
        return JSONResponse({"detail": "request too large"}, status_code=413)
    if _API_TOKEN and request.url.path != "/health":
        provided = request.headers.get("X-CACT-Token", "")
        if not hmac.compare_digest(provided, _API_TOKEN):
            return JSONResponse({"detail": "unauthorized"}, status_code=401)
    return await call_next(request)


def _safe_image_root(hydra_path: str | None, run_uuid: str | None) -> str:
    root = Path(os.getenv("CACT_IMAGE_ROOT", str(_PROJECT_ROOT / "exp_results" / "api_images"))).resolve()
    if os.getenv("CACT_ALLOW_CLIENT_IMAGE_ROOT") == "1" and hydra_path:
        requested = Path(hydra_path).expanduser().resolve()
        allowed = [Path(x).expanduser().resolve() for x in os.getenv("CACT_ALLOWED_IMAGE_ROOTS", str(_PROJECT_ROOT)).split(os.pathsep) if x]
        if any(requested == item or item in requested.parents for item in allowed):
            root = requested
    safe_uuid = re.sub(r"[^A-Za-z0-9_.-]", "_", str(run_uuid or "anonymous"))[:128] or "anonymous"
    target = (root / safe_uuid / "imgs").resolve()
    if target != root and root not in target.parents:
        raise ValueError("image path escaped configured root")
    target.mkdir(parents=True, exist_ok=True)
    return str(target)


def _img2base64(img_path: str):
    with open(img_path, "rb") as f:
        img = base64.b64encode(f.read())
    return img.decode("utf-8")


def _filter_task_obs(task: str, image_root: str) -> str:
    """
    Filter the task observations based on the given task.

    Args:
        task (str): The task to filter the observations for.

    Returns:
        str: The path of the first image that matches the given task.

    """
    task = task.replace(" ", "_")
    task_imgs = [img for img in os.listdir(image_root) if ".jpg" in img and task in img]
    task_imgs.sort(key=lambda x: int(x.split("_")[-1].replace(".jpg", "")))
    return os.path.join(image_root, task_imgs[0])


def stop_server():
    print("Stopping server...")
    os.kill(os.getpid(), signal.SIGINT)  # Graceful shutdown using SIGINT


@app.post("/shutdown")
def shutdown():
    stop_server()
    return {"message": "Server is stopping..."}


@app.get("/health")
def health():
    """Light probe: return 200 once the server process is up.  Model loading
    may still be in progress; the multi-GPU launcher polls this endpoint."""
    return {"status": "ok"}

@app.get("/reset")
def reset() -> MCResponse:
    global agent
    if agent is None:
        agent = AgentFactory.reset()
        print("agent reset (first time)")
    return MCResponse(response="reset done")


@app.post("/chat")
def chat(req: MCRequest) -> MCResponse:
    global agent

    # print(f'req.type: {req.type}')

    if req.type is None:
        req.type = "plan"
    
    # Save current obs (bytes) to image file, and return the path
    # print(req)

    hydra_path = req.hydra_path
    run_uuid = req.run_uuid

    image_root = _safe_image_root(hydra_path, run_uuid)
    if agent is None:
        agent = AgentFactory.reset()
    rgb_obs = base64_to_image(
        req.rgb_images,
        image_root=image_root,
        task=req.task_or_instruction,
        step=req.current_step,
    )
    # print(f"HERE req.type: {req.type}")
    response = None
    match req.type:
        case "decomposed_plan":
            retry = 0
            while retry < _MAX_RETRIES:
                try:
                    plans, prompt = agent.decomposed_plan(
                        req.waypoint,
                        rgb_obs[-1],
                        req.similar_wp_sg_dict,
                        req.failed_sg_list_for_wp,
                    )
                    response = MCResponse(response=plans, message=prompt)
                    break
                except Exception as exc:
                    retry += 1
                    print("connection error, retry: ", retry)
        case "context_aware_reasoning":
            retry = 0
            while retry < _MAX_RETRIES:
                try:
                    reasoning, visual_description = agent.context_aware_reasoning(
                        req.task_or_instruction,
                        req.goal,
                        rgb_obs[-1],
                    )
                    response = MCResponse(response=reasoning, message=visual_description)
                    break
                except Exception as exc:
                    retry += 1
                    print("connection error, retry: ", retry)
        case "retrieval":
            retry = 0
            while retry < _MAX_RETRIES:
                try:
                    plans_retrieval = agent.retrieve(
                        req.task_or_instruction,
                        rgb_obs[-1],
                    )
                    response = MCResponse(response=plans_retrieval)
                    break
                except Exception as exc:
                    retry += 1
                    print("connection error while retrieval, retry: ", retry)
        case "plan":
            retry = 0
            while retry < _MAX_RETRIES:
                try:
                    plans = agent.plan(
                        req.task_or_instruction,
                        rgb_obs[-1],
                        req.example,
                        req.visual_info,
                        req.graph,
                    )
                    response = MCResponse(response=plans)
                    break
                except Exception as exc:
                    retry += 1
                    print("connection error, retry: ", retry)
        case "fixjson":
            retry = 0
            # print(f"HERE!!!@#!%$@%!")
            # print(f"HERE!!! req.errorneous_planning: {req.errorneous_planning}")
            # print(f"rgb_obs[-1]: {rgb_obs[-1]}")
            while retry < 10:
                try:
                    fixed_json = agent.fix_json_format(
                        req.errorneous_planning, rgb_obs[-1]
                    )
                    response = MCResponse(response=fixed_json)
                    break
                except Exception as exc:
                    retry += 1
                    print("connection error, retry: ", retry)

        case "action":

            start = time.perf_counter()
            minrl_action = agent.action(req.task_or_instruction, rgb_obs)
            end = time.perf_counter()
            # print(end - start, " s")  # 0.04s
            response = MCResponse(response=minrl_action)
            # print(response)
        case "reflection":
            # old_obs: path of the obs when the current task is given
            old_obs = _filter_task_obs(req.task_or_instruction, image_root)
            print(f"old_obs {old_obs} current step {req.current_step}")
            retry = 0

            done_imgs, cont_imgs, replan_imgs = (
                req.done_imgs,
                req.cont_imgs, # str data (bytes) of the images
                req.replan_imgs,
            )
            done, cont, replan = (
                base64lst2img_path(done_imgs, image_root), # save image data (bytes) to file and return path
                base64lst2img_path(cont_imgs, image_root),
                base64lst2img_path(replan_imgs, image_root),
            )
            while retry < 10:
                try:
                    # NOTE: Can VLM determine the progress only using 2 images (current obs, old obs)?
                    reflection = agent.reflection(
                        req.task_or_instruction,
                        old_obs, # obs when the current task is given
                        rgb_obs[-1], # current obs
                        done_img_path=done,
                        cont_img_path=cont,
                        replan_img_path=replan,
                    )
                    print(f"{old_obs} <-> {rgb_obs[-1]}: {reflection}")
                    response = MCResponse(
                        response=reflection, appendix=_img2base64(old_obs)
                    )
                    break
                except Exception as exc:
                    retry += 1
                    time.sleep(1)
                    print("connection error while reflection, retry: ", retry)
        case "replan":
            retry = 0
            while retry < 10:
                try:
                    replan = agent.replan(
                        req.task_or_instruction,
                        rgb_obs[-1],
                        req.error_info,
                        req.example,
                        req.graph,
                    )
                    response = MCResponse(response=replan)
                    print(replan)
                    break
                except Exception as e:
                    retry += 1
                    time.sleep(1)
                    print(f"connection error while replan {e}, retry: {retry}")
        case _:
            response = MCResponse(message=f"{req.type} not support...", status_code=400)
    if response is None:
        response = MCResponse(status_code=503, message="VLM inference failed after bounded retries")
    # Attach cumulative token stats from the plan model
    if hasattr(agent.plan_model, 'token_stats'):
        response.tokens = agent.plan_model.token_stats
    return response


@app.post("/batch_chat")
def batch_chat(reqs: list[MCRequest]) -> list[MCResponse]:
    """Batch VLM inference — N requests in one GPU forward pass.

    Accepts a list of MCRequest objects, extracts all prompts and images,
    runs one model.generate() call, and returns N MCResponse objects.

    This is 2-4x faster than N individual /chat calls when multiple
    workers submit requests concurrently.
    """
    global agent

    if not reqs:
        return []
    if agent is None:
        agent = AgentFactory.reset()

    instructions = []
    images_list = []

    for req in reqs:
        instruction = req.task_or_instruction or ""
        rgb_images = req.rgb_images
        if rgb_images and len(rgb_images) > 0:
            images_list.append(rgb_images[0] if isinstance(rgb_images, list) else rgb_images)
        else:
            images_list.append(None)
        instructions.append(instruction)

    responses = agent.batch_inference(instructions, images_list)

    return [
        MCResponse(response=r, message="batch")
        for r in responses
    ]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Start FastAPI server with custom AgentFactory configuration."
    )
    parser.add_argument("--plan_with_gpt", action="store_true", help="Use GPT for planning.")
    parser.add_argument("--plan_model", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct", help="Model for planning.")
    parser.add_argument("--in_model", type=str, default="checkpoints/vpt/2x.model", help="Model for input.")
    parser.add_argument("--in_weights", type=str, default="checkpoints/steve1/steve1.weights", help="Weights for input.")
    parser.add_argument("--prior_weights", type=str, default="checkpoints/steve1/steve1_prior.pt", help="Weights for prior.")
    parser.add_argument("--host", type=str, default=os.getenv("CACT_HOST", "127.0.0.1"), help="Bind address; loopback is the safe default.")
    parser.add_argument("--port", type=int, default=12345, help="Port to run the server on.")
    parser.add_argument("--api-token", type=str, default=os.getenv("CACT_API_TOKEN", ""), help="Optional token for non-loopback deployments.")
    parser.add_argument("--seed", type=int, default=0)

    args = parser.parse_args()
    _API_TOKEN = args.api_token
    if args.host not in {"127.0.0.1", "localhost", "::1"} and not _API_TOKEN:
        raise SystemExit("Refusing non-loopback bind without --api-token/CACT_API_TOKEN")
    print("Starting server...")
    print(f'args: {args}')

    seed = int(args.seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    transformers.set_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    AgentFactory.set_args(
        plan_with_gpt=args.plan_with_gpt,
        plan_model=args.plan_model,
        in_model=args.in_model,
        in_weights=args.in_weights,
        prior_weights=args.prior_weights,
    )

    uvicorn.run(app, host=args.host, port=args.port, timeout_keep_alive=600, limit_concurrency=int(os.getenv("CACT_HTTP_MAX_CONCURRENCY", "64")), limit_max_requests=int(os.getenv("CACT_HTTP_MAX_REQUESTS", "0")) or None)
