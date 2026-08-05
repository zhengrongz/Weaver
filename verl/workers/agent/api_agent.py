import re
import logging
import ast
import json
import base64
import requests
import numpy as np
from copy import deepcopy
from io import BytesIO
from PIL import Image
from typing import List, Dict, Any, Tuple
from omegaconf import DictConfig
from .qwen_tools import fetch_tools
import time

logger = logging.getLogger(__name__)

class APIAgent:
    """
    Agent class for handling multi-turn action/observation interactions, using OpenAI API for conversation generation.
    Handles interaction between LLM and environment, supports multimodal observation.
    """
    def __init__(
        self,
        config: DictConfig,
    ):
        """
        Initialize Agent.
        
        Args:
            config: Configuration information
        """
        self.config = config
        self.tools = fetch_tools(placeholder='<image>')
        
        # Define regex for action detection
        self.tool_pattern = config.get("tool_pattern", r"<tool_call>(.*?)</tool_call>")
        self.image_placeholder = config.get("image_placeholder", "<image>")
        
        # Configure max turns and max tokens per turn
        self.max_turns = config.get("max_turns", 4)
        self.max_tokens_per_turn = config.get("max_tokens_per_turn", 700)
        self.max_results = config.get("max_results", 3)
        
        # Whether to save intermediate responses
        self.save_intermediate_responses = config.get("save_intermediate_responses", False)
        
        # OpenAI API config
        self.openai_api_key = config.get("openai_api_key")
        self.openai_api_base = config.get("openai_api_base", "https://api.openai.com/v1/chat/completions")
        self.openai_model = config.get("openai_model", "gpt-4.1")
        self.openai_temperature = config.get("openai_temperature", 0.7)
        self.openai_top_p = config.get("openai_top_p", 0.95)

    def process_action_and_execute(self, text: str, env_state: Dict[str, Any] = None) -> Tuple[bool, bool, Dict[str, Any]]:
        """
        Extracts action list from text and executes them one by one, returns merged result.
        Result text is wrapped with <observation> tag.

        Args:
            text: Generated text
            env_state: Environment state

        Returns:
            (has_action, all_actions_successful, result_dict):
            - has_action: Boolean, whether tool_call block found.
            - all_actions_successful: Boolean, whether all actions executed successfully.
            - result_dict: Dict containing merged observation result.
        """
        obs_template = "{image}\n{text}\nNow think first again, then decide to call tools or answer."

        tool_match = re.search(self.tool_pattern, text, re.DOTALL)
        if not tool_match:
            tool_match = re.search(r'"tool_calls": (\[.*\])', text, re.DOTALL)
        if not tool_match:
            no_action_msg = obs_template.format(text="No action found.",image='')
            return False, False, {"text": no_action_msg, "image": None}

        try:
            tool_content_str = tool_match.group(1).strip()
            tool_calls_list = ast.literal_eval(tool_content_str)

            if not isinstance(tool_calls_list, list):
                if isinstance(tool_calls_list, dict):
                    tool_calls_list = [tool_calls_list]
                else:
                    raise ValueError("Tool content inside <tool_call> is not a valid list or dictionary of tool calls.")
            if len(tool_calls_list) == 0:
                return False, False, {"text": obs_template.format(text="No tool calls found in the list.",image=''), "image": None}

        except Exception as e:
            error_msg = obs_template.format(text=f"Error parsing tool calls list: {str(e)}",image='')
            return True, False, {"text": error_msg, "image": None}

        if not tool_calls_list:
            empty_list_msg = obs_template.format(text="Tool call list is empty.",image='')
            return True, False, {"text": empty_list_msg, "image": None}

        aggregated_obs_texts = []
        aggregated_images = []
        all_actions_successful = True

        for i, tool_content in enumerate(tool_calls_list[:self.max_results]):
            if not isinstance(tool_content, dict):
                error_msg = f"Invalid item in tool call list: Expected a dictionary, got {type(tool_content)} ({str(tool_content)[:100]})."
                aggregated_obs_texts.append(error_msg)
                all_actions_successful = False
                continue

            try:
                tool_name = tool_content.get('name')
                args = tool_content.get('arguments')

                if not tool_name:
                    error_msg = f"Missing 'name' in tool content: {str(tool_content)[:100]}"
                    aggregated_obs_texts.append(error_msg)
                    all_actions_successful = False
                    continue
                
                if args is None:
                    args = {}
                elif not isinstance(args, dict):
                    error_msg = f"Invalid 'arguments' for tool '{tool_name}': Expected a dictionary, got {type(args)}."
                    aggregated_obs_texts.append(error_msg)
                    all_actions_successful = False
                    continue

            except Exception as e:
                error_msg = f"Error parsing individual tool call '{str(tool_content)[:100]}': {str(e)}"
                aggregated_obs_texts.append(error_msg)
                all_actions_successful = False
                continue

            if tool_name not in self.tools:
                error_msg = f"Error: There is no tool named '{tool_name}'."
                aggregated_obs_texts.append(error_msg)
                all_actions_successful = False
                continue

            try:
                result = self.tools[tool_name].call(args, env_state)

                tool_obs_text = result.get('text', '')
                if tool_obs_text:
                    aggregated_obs_texts.append(f"Image {i+len(env_state['image'])}:" + str(tool_obs_text))

                tool_obs_image = result.get("image", None)
                if tool_obs_image is not None:
                    if isinstance(tool_obs_image, list):
                        aggregated_images.extend(tool_obs_image)
                    else:
                        aggregated_images.append(tool_obs_image)
            
            except Exception as e:
                error_msg = f"Failed to call tool '{tool_name}' with args '{str(args)[:100]}' due to error: {str(e)}"
                aggregated_obs_texts.append(error_msg)
                all_actions_successful = False

        final_obs_content = "\n".join(aggregated_obs_texts)
        if not final_obs_content and tool_calls_list:
            if all_actions_successful:
                final_obs_content = "All actions processed successfully with no textual output."
            else:
                final_obs_content = "Actions processed with errors, but no specific textual error messages were generated."
        elif not tool_calls_list:
             final_obs_content = "No actions to process."

        formatted_final_obs_text = obs_template.format(text=final_obs_content,image=self.image_placeholder*len(aggregated_images))
        
        final_images = aggregated_images if aggregated_images else None

        return True, all_actions_successful, {"text": formatted_final_obs_text, "image": final_images}

    def is_base64_string(self, s):
        """Check if string is valid base64 encoding"""
        if not isinstance(s, str):
            return False
        try:
            if "base64," in s:
                s = s.split("base64,")[1]
            base64.b64decode(s)
            return True
        except Exception:
            return False

    def _encode_image_to_base64(self, image):
        """Convert PIL image, numpy array, or base64 string to base64 encoding"""
        if isinstance(image, str) and self.is_base64_string(image):
            if "base64," in image:
                return image.split("base64,")[1]
            return image
            
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        if isinstance(image, Image.Image):
            buffer = BytesIO()
            image.save(buffer, format="PNG")
            return base64.b64encode(buffer.getvalue()).decode("utf-8")
            
        raise ValueError(f"Unsupported image type: {type(image)}")

    def _prepare_message_with_images(self, text, images=None):
        """
        Prepare mixed content of text and images.
        Handles <image> placeholders in text, inserts corresponding images.
        """
        content = []
        
        if "<image>" not in text or not images:
            if images and "<image>" not in text:
                for image in images:
                    if isinstance(image, str) and image.startswith("http"):
                        content.append({
                            "type": "image_url",
                            "image_url": {"url": image}
                        })
                    else:
                        base64_image = self._encode_image_to_base64(image)
                        content.append({
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{base64_image}"}
                        })
            content.append({"type": "text", "text": text})
            return content
        parts = text.split("<image>")
        image_count = len(parts) - 1
        
        if parts[0]:
            content.append({"type": "text", "text": parts[0]})
        
        for i in range(image_count):
            if i < len(images):
                image = images[i]
                if isinstance(image, str) and image.startswith("http"):
                    content.append({
                        "type": "image_url",
                        "image_url": {"url": image}
                    })
                else:
                    base64_image = self._encode_image_to_base64(image)
                    content.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{base64_image}"}
                    })
            
            if i+1 < len(parts) and parts[i+1]:
                content.append({"type": "text", "text": parts[i+1]})
        
        for i in range(image_count, len(images)):
            image = images[i]
            if isinstance(image, str) and image.startswith("http"):
                content.append({
                    "type": "image_url",
                    "image_url": {"url": image}
                })
            else:
                base64_image = self._encode_image_to_base64(image)
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{base64_image}"}
                })
                
        return content

    def _call_openai_api(self, messages, max_tokens=None, temperature=None, stop=None):
        """Call OpenAI API for generation"""
        if not self.openai_api_key:
            raise ValueError("OpenAI API key not provided")
        
        url = f"{self.openai_api_base}"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.openai_api_key}"
        }
        
        payload = {
            "model": self.openai_model,
            "messages": messages,
            "temperature": temperature or self.openai_temperature,
            "top_p": self.openai_top_p,
        }
        
        if max_tokens:
            payload["max_tokens"] = max_tokens
            
        if stop:
            payload["stop"] = stop
        
        max_retries = 3
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                response = requests.post(url, headers=headers, json=payload)
                response.raise_for_status()
                return response.json()
            except Exception as e:
                retry_count += 1
                logger.error(f"Error calling OpenAI API (attempt {retry_count}/{max_retries}): {str(e)}")
                
                if retry_count >= max_retries:
                    logger.error(f"Failed after {max_retries} attempts")
                    raise
                
                time.sleep(1)
                
    def generate_conversation(self, system_prompt, initial_prompt, images=None):
        """
        Generate multi-turn conversation, handle tool calls and observations
        
        Args:
            system_prompt: System prompt text
            initial_prompt: Initial prompt text
            images: Initial image list (optional)
            env_state: Environment state (optional)
            
        Returns:
            conversation_history: Conversation history
        """
        conversation_history = []
        
        if system_prompt:
            conversation_history.append({
                "role": "system",
                "content": system_prompt
            })
        
        initial_content = self._prepare_message_with_images(initial_prompt, images)
        
        conversation_history.append({
            "role": "user",
            "content": initial_content
        })
        
        current_mm_data = {"image": images} if images else None
        
        for turn in range(self.max_turns):
            print(f"Start turn {turn+1}")
            
            try:
                response = self._call_openai_api(
                    messages=conversation_history,
                    max_tokens=self.max_tokens_per_turn,
                    stop=["</tool_call>"]
                )
                
                generated_text = response['choices'][0]['message']['content']
                
                if "<tool_call>" in generated_text and "</tool_call>" not in generated_text:
                    generated_text += "</tool_call>"
                
                if self.save_intermediate_responses:
                    print(f"Turn {turn+1} assistant reply: {generated_text}")

                conversation_history.append({
                    "role": "assistant", 
                    "content": generated_text
                })
                    
                has_action, all_actions_successful, observation = self.process_action_and_execute(
                    generated_text, 
                    current_mm_data
                )
                
                if not has_action or turn == self.max_turns - 1:
                    break
                
                obs_text = observation.get("text", "")
                obs_images = observation.get("image", None)
                
                if self.save_intermediate_responses:
                    print(f"Turn {turn+1} observation: {obs_text}")
                    if not all_actions_successful:
                        print(f"Warning: Turn {turn+1} tool call failed")
                
                obs_message_content = self._prepare_message_with_images(obs_text, obs_images)
                
                conversation_history.append({
                    "role": "user",
                    "content": obs_message_content
                })
                
                if obs_images is not None:
                    if current_mm_data is None:
                        current_mm_data = {"image": obs_images}
                    elif "image" not in current_mm_data:
                        current_mm_data["image"] = obs_images
                    else:
                        current_mm_data["image"].extend(obs_images)
                
            except Exception as e:
                logger.error(f"Exception in turn {turn+1}: {str(e)}")
                break
            
            print(f"Turn {turn+1} finished")
        
        return conversation_history

    def chat_with_tools(self, prompt, images=None, system_prompt=None):
        """
        Simplified API entry for multi-turn tool interaction conversation
        
        Args:
            prompt: User question
            images: Optional image list
            system_prompt: Optional system prompt
            env_state: Optional environment state
            
        Returns:
            final_response: Final response text
            conversation: Full conversation history
        """
        
        images_copy = deepcopy(images) if images else None
        
        conversation = self.generate_conversation(system_prompt, prompt, images_copy)
        
        final_response = None
        for message in reversed(conversation):
            if message["role"] == "assistant":
                final_response = message["content"]
                break
                
        return final_response, conversation