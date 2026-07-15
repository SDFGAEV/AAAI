from typing import List, Optional

import torch
try:
    from transformers import Qwen2_5_VLForConditionalGeneration
except ImportError:
    Qwen2_5_VLForConditionalGeneration = None
try:
    from transformers import Qwen2VLForConditionalGeneration
except ImportError:
    Qwen2VLForConditionalGeneration = None
from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info

from ..util.prompt import language_action_to_subgoal

from .base_model import BasePlanningModel

prompt_decomposed_plan = """For an item name, you need to make a plan using examples.
"""

#################### Our context-aware reasoning prompt ####################
description_prompt = """Given a Minecraft game image, describe nearby Minecraft objects, like tree, grass, cobblestone, etc.

[Example]
"There is a large tree with dark green leaves surrounding the area."
"The image shows a dark, cave-like environment in Minecraft. The player is digging downwards. There are no visible trees or grass in this particular view."
"The image shows a dark, narrow tunnel made of stone blocks. The player is digging downwards."

[Your turn]
Describe the given image, simply and clearly like the examples."""

context_aware_reasoning_prompt = """
Given <task> and <visual_description>, determine if the player needs intervention to achieve the goal. If intervention is needed, suggest a task that the player should perform.
I will give you examples.

[Example]
<task>: chop tree
<visual_description>: There is a large tree with dark green leaves surrounding the area.
<goal>: logs
<reasoning>:
{{
    "need_intervention": false,
    "thoughts": "The player can see a tree and can chop it down to get logs.",
    "task": "",
}}

[Example]
<task>: chop tree
<visual_description>: The image shows a dirt block in Minecraft. There is a tree in the image, but it is too far from here.
<goal>: logs
<reasoning>:
{{
    "need_intervention": true,
    "thoughts": "The player is far from trees. The player needs to move to the trees.",
    "task": "explore to find trees",
}}

[Example]
<task>: dig down to mine iron_ore
<visual_description>: The image shows a dark, narrow tunnel made of stone blocks. The player is digging downwards.
<goal>: iron_ore
<reasoning>:
{{
    "need_intervention": false,
    "thoughts": "The player is already digging down and is likely to find iron ore.",
    "task": "",
}}

[Your turn]
Here is the <task>, <visual_description>, and <goal>.
You MUST output the <reasoning> in JSON format.
<task>: {task}
<visual_description>: {visual_description}
<goal>: {goal}
<reasoning>:
"""


def is_path(path):
    if len(path) == 2:
        return True
    else:
        return False


class PlanningModel(BasePlanningModel):

    def __init__(self, model_path: str = "Qwen/Qwen2.5-VL-7B-Instruct", device_id: int = 0,
                 system_prompt: Optional[str] = None) -> None:
        self.device = f"cuda:{device_id}"

        normalized_model = model_path.lower()
        if "qwen2.5" in normalized_model or "qwen2_5" in normalized_model:
            if Qwen2_5_VLForConditionalGeneration is None:
                raise ImportError(
                    "Qwen2.5-VL requires transformers>=4.49; install a compatible "
                    "version instead of falling back to the incompatible Qwen2-VL class")
            ModelClass = Qwen2_5_VLForConditionalGeneration
        elif "qwen2-vl" in normalized_model or "qwen2vl" in normalized_model:
            if Qwen2VLForConditionalGeneration is None:
                raise ImportError(
                    "Qwen2-VL support is unavailable in the installed transformers version")
            ModelClass = Qwen2VLForConditionalGeneration
        else:
            raise ValueError(
                f"Unsupported Qwen-VL model path {model_path!r}; include qwen2.5 or qwen2-vl "
                "in the model identifier so the architecture cannot be mis-selected")
        self.model = ModelClass.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map=self.device,
        )
        self.model.eval()
        self.processor = AutoProcessor.from_pretrained(model_path)
        self._cum_input_tokens = 0
        self._cum_output_tokens = 0
        self._cum_calls = 0

    @property
    def token_stats(self):
        return {"in": self._cum_input_tokens, "out": self._cum_output_tokens,
                "calls": self._cum_calls}

    def decomposed_plan(
        self,
        waypoint: str,
        images: str | List[str],
        similar_wp_sg_dict: dict | None = None,
        failed_sg_list_for_wp: List[str] | None = None,
    ):
        images = None
        prompt = prompt_decomposed_plan

        if similar_wp_sg_dict is not None and len(similar_wp_sg_dict) > 0:
            prompt += "I will give you examples of which plans are needed to achieve an item.\n"
            for similar_wp, sg_str in similar_wp_sg_dict.items():
                prompt += f"""[Example]
<item name>
{similar_wp}
<task planning>
{sg_str}

"""
        else:
            # similar waypoints are not available
            # That is, it does not use memory
            pass

        # The paper's formal high-level action space is exactly {mine, craft, smelt}.
        language_actions = ["mine", "craft", "smelt"]
        failed_sg_list_for_wp = failed_sg_list_for_wp or []
        for failed_sg_str in failed_sg_list_for_wp:
            failed = str(failed_sg_str).lower()
            if any(x in failed for x in ("mine", "chop", "punch", "gather")):
                if "mine" in language_actions:
                    language_actions.remove("mine")
            elif "craft" in failed and "craft" in language_actions:
                language_actions.remove("craft")
            elif "smelt" in failed and "smelt" in language_actions:
                language_actions.remove("smelt")
        language_action_options = [f"{action} {waypoint}" for action in language_actions]
        if not language_action_options:
            language_action_options = [f"mine {waypoint}", f"craft {waypoint}", f"smelt {waypoint}"]
        language_subgoal_options = []
        i = 1
        for action in language_action_options:
            _, subgoal = language_action_to_subgoal(action, waypoint)
            language_subgoal_options.append(f"{i}. {subgoal}")
            i += 1
        options_str = "\n".join(language_subgoal_options)

        prompt += f"""
[Your turn]
Here is <item name>, you MUST output <task planning> in JSON format.
You can make <task planning> by selecting an option from below:
{options_str}

<item name>
{waypoint}
<task planning>
"""

        print(f"====\n{prompt}\n====")
        return self._inference(prompt, None), prompt


    def context_aware_reasoning(
        self,
        task: str,
        goal: str,
        image_path: str,
    ):
        visual_description = self._inference(description_prompt, image_path)

        new_reasoning_prompt = context_aware_reasoning_prompt.format(
            task=task,
            visual_description=visual_description,
            goal=goal,
        )
        reasoning = self._inference(new_reasoning_prompt, None)
        return reasoning, visual_description
    


    # From Optimus-1

    # def retrieve(
    #     self,
    #     task: str,
    #     image_path: str,
    # ):
    #     return self._inference(retrieve_prompt.format(task=task), image_path)


#     def replan(
#         self,
#         task: str,
#         image_path: str,
#         error_info: str | None = None,
#         examples: str | None = None,
#         graph_summary: str | None = None,
#     ):
#         logic1 = ""
#         if examples is None or examples == "":
#             logic1 = """craft 1 crafting_table summary:
# 1. log: need 1
# 2. planks: need 4
# 3. crafting_table: need 1"""
#             examples = """<task>: craft wooden_pickaxe.
# <error>: missing material: {"crafting_table": 1}.
# <replan>: 
# {
#     "step 1": {"task": "chop tree", "goal": ["logs", 1]},
#     "step 2": {"task": "craft planks", "goal": ["planks", 4]},
#     "step 3": {"task": "craft crafting table", "goal": ["crafting_table", 1]
# }
# """

#         if logic1 == "":
#             logic1 = graph_summary

#         if graph_summary is None or graph_summary == "":
#             prompt = non_reflection_replan_prompt.format(
#                 task1=task,
#                 logic1=logic1,
#                 example=examples.strip(),
#                 error=error_info,
#             )
#         else:
#             prompt = replan_prompt.format(
#                 task1=task,
#                 logic1=logic1,
#                 logic=graph_summary.strip(),  # type: ignore
#                 example=examples.strip(),
#                 error=error_info,
#             )

#         return self._inference(prompt, image_path)


    # def planning(
    #     self,
    #     task: str,
    #     images: str | List[str],
    #     example: str | None = None,
    #     visual_info: str | None = None,
    #     graph: str | None = None,
    # ):
    #     if visual_info is None and graph is None:
    #         prompt = no_reflection_plan_prompt.format(task=task, example=example)
    #     else:
    #         prompt = plan_prompt.format(
    #             task=task,
    #             example=example,
    #             visual=visual_info,
    #             graph=graph,
    #         )
    #     print(f"====\n{prompt}\n====")
    #     return self._inference(prompt, images)

    # def reflection(
    #     self,
    #     task: str,
    #     done_path: List[str],
    #     continue_path: List[str],
    #     replan_path: List[str],
    #     image_path: List[str],
    # ):
    #     is_done, is_continue, is_replan = (
    #         is_path(done_path),
    #         is_path(continue_path),
    #         is_path(replan_path),
    #     )
    #     prompt = reflection_systerm.format(task=task)
    #     imgs = []
    #     if is_done or is_continue or is_replan:
    #         prompt += "\n" + reflection_examples
    #         if is_done:
    #             prompt += f"\n<done>:\n{self.IMAGE_TAG} {self.IMAGE_TAG}"
    #             imgs += done_path

    #         if is_continue:
    #             prompt += f"\n<continue>:\n{self.IMAGE_TAG} {self.IMAGE_TAG}"
    #             imgs += continue_path

    #         if is_replan:
    #             prompt += f"\n<replan>:\n{self.IMAGE_TAG} {self.IMAGE_TAG}"
    #             imgs += replan_path

    #     imgs += image_path
    #     prompt += reflection_prompt

    #     return self._inference(prompt, imgs)


    def _inference(self, instruction: str, images: str | List[str] = None) -> str:
        """Single inference call. For batch, use _inference_batch()."""
        return self._inference_batch([(instruction, images)])[0]

    def _inference_batch(self, requests: list) -> list:
        """Batch VLM inference — N prompts in one GPU forward pass.

        Args:
            requests: List of (instruction, images) tuples.
                      images can be None (text-only), a path string, or a list of paths.

        Returns:
            List of response strings, one per request.
        """
        if len(requests) == 0:
            return []

        # Build N messages, one per request
        messages_list = []
        all_have_images = True
        for instruction, images in requests:
            if images is None:
                all_have_images = False
                msg_content = [{"type": "text", "text": instruction}]
            else:
                msg_content = [
                    {"type": "text", "text": instruction},
                    {"type": "image", "image": images},
                ]
            messages_list.append([{"role": "user", "content": msg_content}])

        # Apply chat template to each message independently
        texts = [
            self.processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            for msgs in messages_list
        ]

        if not all_have_images:
            # All text-only — simple batch
            inputs = self.processor(
                text=texts,
                padding=True,
                return_tensors="pt",
            )
        else:
            # Mixed text+image — process vision info for all messages
            all_image_inputs = []
            all_video_inputs = []
            for msgs in messages_list:
                imgs, vids = process_vision_info(msgs)
                all_image_inputs.append(imgs)
                all_video_inputs.append(vids)
            # Flatten image/video lists for the processor
            flat_images = [img for imgs in all_image_inputs for img in (imgs or [])]
            flat_videos = [vid for vids in all_video_inputs for vid in (vids or [])]
            inputs = self.processor(
                text=texts,
                images=flat_images if flat_images else None,
                videos=flat_videos if flat_videos else None,
                padding=True,
                return_tensors="pt",
            )
        inputs = inputs.to(self.model.device)

        generated_ids = self.model.generate(**inputs, max_new_tokens=512)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        responses = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        # Track token usage across batch
        self._cum_input_tokens += int(inputs.input_ids.shape[1]) * len(requests)
        self._cum_output_tokens += sum(
            int(ids.shape[0]) for ids in generated_ids_trimmed
        ) if generated_ids_trimmed else 0
        self._cum_calls += 1  # Count batch as 1 GPU call

        return responses
