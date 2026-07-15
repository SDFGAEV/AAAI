import copy
import pickle
import warnings
from collections.abc import Mapping

import gym
from ..config import DEVICE, MINECLIP_CONFIG
from ..mineclip_code.load_mineclip import load
from ..MineRLConditionalAgent import MineRLConditionalAgent
from ..VPT.agent import ENV_KWARGS, POLICY_KWARGS, PI_HEAD_KWARGS


def load_model_parameters(path_to_model_file):
    """Load VPT architecture metadata across legacy and torch state-dict files.

    Older VPT releases stored a pickled training bundle with ``model.args``;
    the released ``2x.model`` checkpoint is a torch state dict instead.  The
    latter intentionally has no architecture metadata, so use the canonical
    policy constants shipped with this repository rather than attempting an
    unsafe raw ``pickle.load`` or guessing from tensor shapes.
    """
    try:
        with open(path_to_model_file, "rb") as handle:
            agent_parameters = pickle.load(handle)
    except (pickle.UnpicklingError, EOFError, ValueError, AttributeError) as exc:
        warnings.warn(
            f"{path_to_model_file} is a torch state-dict checkpoint; using the "
            f"repository VPT architecture defaults ({exc})",
            RuntimeWarning,
            stacklevel=2,
        )
        return copy.deepcopy(POLICY_KWARGS), copy.deepcopy(PI_HEAD_KWARGS)

    if not isinstance(agent_parameters, Mapping):
        raise ValueError(f"unsupported VPT metadata type: {type(agent_parameters).__name__}")
    try:
        policy_kwargs = copy.deepcopy(agent_parameters["model"]["args"]["net"]["args"])
        pi_head_kwargs = copy.deepcopy(agent_parameters["model"]["args"]["pi_head_opts"])
    except (KeyError, TypeError) as exc:
        raise ValueError(f"VPT metadata missing model architecture fields: {path_to_model_file}") from exc
    if "temperature" in pi_head_kwargs:
        pi_head_kwargs["temperature"] = float(pi_head_kwargs["temperature"])
    return policy_kwargs, pi_head_kwargs


def load_mineclip_wconfig():
    print("Loading MineClip...")
    return load(MINECLIP_CONFIG, device=DEVICE)


def make_env(seed):
    from minerl.herobraine.env_specs.human_survival_specs import HumanSurvival

    print("Loading MineRL...")
    env = HumanSurvival(**ENV_KWARGS).make()
    print("Starting new env...")
    env.reset()
    if seed is not None:
        print(f"Setting seed to {seed}...")
        env.seed(seed)
    return env


def make_agent(in_model, in_weights, cond_scale):
    print(f"Loading agent with cond_scale {cond_scale}...")
    agent_policy_kwargs, agent_pi_head_kwargs = load_model_parameters(in_model)
    env = gym.make("MineRLBasaltFindCave-v0")
    # Make conditional agent
    agent = MineRLConditionalAgent(
        env,
        device="cuda",
        policy_kwargs=agent_policy_kwargs,
        pi_head_kwargs=agent_pi_head_kwargs,
    )
    agent.load_weights(in_weights)
    agent.reset(cond_scale=cond_scale)
    env.close()
    return agent


def load_mineclip_agent_env(in_model, in_weights, seed, cond_scale):
    mineclip = load_mineclip_wconfig()
    agent = make_agent(in_model, in_weights, cond_scale=cond_scale)
    env = make_env(seed)
    return agent, mineclip, env
