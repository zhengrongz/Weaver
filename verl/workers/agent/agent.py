import os
import re
import logging
import torch
import numpy as np
import ast
import torch.distributed as dist
from typing import List, Dict, Any, Optional, Tuple, Union
from omegaconf import DictConfig
from copy import deepcopy
from .qwen_tools import fetch_tools
from verl.utils.torch_functional import pad_2d_list_to_length
from verl.models.transformers.qwen2_vl import get_rope_index

logger = logging.getLogger(__name__)

def transform(a, b):
    return 1 if a == 1 and b in (0, 1) else 2

class Agent:
    """
    Agent class for handling multi-turn action/observation interactions.
    Manages the interaction between the LLM and the environment, supporting multi-modal observations.
    """
    def __init__(
        self,
        config: DictConfig,
        tokenizer,
        processor,
    ):
        """
        Initialize the Agent.
        
        Args:
            config: Configuration information
            tokenizer: Tokenizer for processing text
            tools: Optional dictionary of tools for executing actions
        """
        self.config = config
        self.tokenizer = tokenizer
        self.processor = processor

        # Tool service URLs, configurable via agent config
        # Priority: explicit url config > tool_host + default port > hardcoded localhost
        tool_urls = {}
        default_host = config.get("tool_host", "localhost") or "localhost"
        tool_urls['TemporalGroundingTool'] = (
            config.get("temporal_grounding_url") or f"http://{default_host}:9995/temporal"
        )
        tool_urls['TrackingTool'] = (
            config.get("tracking_url") or f"http://{default_host}:9999/tracking"
        )
        tool_urls['SpatialGroundingTool'] = (
            config.get("spatial_grounding_url") or f"http://{default_host}:9998/grounding"
        )
        tool_urls['TemporalCountTool'] = (
            config.get("temporal_count_url") or f"http://{default_host}:9997/count"
        )
        tool_urls['FrameSelectionTool_frame'] = (
            config.get("frame_selection_frame_url") or f"http://{default_host}:9997/frame"
        )
        tool_urls['FrameSelectionTool_temporal'] = (
            config.get("frame_selection_temporal_url") or f"http://{default_host}:9995/temporal"
        )
        self.tools = fetch_tools(tool_urls=tool_urls)
        
        # Regular expression for action detection
        self.tool_pattern = config.get("tool_pattern", r"<tool_call>(.*?)</tool_call>")
        self.image_placeholder = config.get("image_placeholder", "<|vision_start|><|image_pad|><|vision_end|>")
        self.video_placeholder = config.get("video_placeholder", "<|vision_start|><|video_pad|><|vision_end|>")
        
        # Configuration for the maximum number of turns and maximum tokens per turn
        self.max_turns = config.get("max_turns", 5)
        self.max_tokens_per_turn = config.get("max_tokens_per_turn", 512)
        self.response_length = config.get("response_length", 2048)
        self.inference_batch_size = config.get("inference_batch_size", 8)
        self.max_results = config.get("max_results", 2)
        self.max_prompt_length = config.get("max_prompt_length", 8192)
        self.max_response_length = config.get("max_response_length", 20480)

        
        # eos Token ID for marking the end of a sequence
        self.eos_token_id = self.tokenizer.eos_token_id
        self.pad_token_id = self.tokenizer.pad_token_id if hasattr(self.tokenizer, "pad_token_id") else self.eos_token_id
        self.eot_token_id = self.tokenizer.encode('</tool_call>')
        
        print(self)
        
    def __repr__(self):
        return f"Agent(tools={self.tools}, max_turns={self.max_turns}, max_tokens_per_turn={self.max_tokens_per_turn}, response_length={self.response_length})"

    def process_action_and_execute(self, text: str, env_state: Dict[str, Any] = None, extra_info: Dict[str, Any] = None) -> Tuple[bool, bool, Dict[str, Any]]:
        """
        Extract action list from text and execute them one by one, returning the merged result.
        Result text is wrapped in <observation> tags.

        Args:
            text: Generated text
            env_state: Environment state

        Returns:
            (has_action, all_actions_successful, result_dict):
            - has_action: A boolean indicating whether a tool_call block was found.
            - all_actions_successful: A boolean indicating whether all extracted actions were successfully executed.
            - result_dict: A dictionary containing the merged observation result.
        """
        # Define the observation template as it's used in the original function
        # obs_template = "<|im_end|>\n<|im_start|>user\n{image}\n{text}\nNow think first again, then decide to call tools or answer.<|im_end|>\n<|im_start|>assistant\n"
        success_obs_template = "<|im_end|>\n<|im_start|>user\n{vision}\n{text}.<|im_end|>\n<|im_start|>assistant\n"
        obs_template = "<|im_end|>\n<|im_start|>user\n{vision}\n{text}\nNow think first again, then decide to call tools or answer.<|im_end|>\n<|im_start|>assistant\n"
        video_path = extra_info.get("video_path", None)
        duration = extra_info.get("duration", None)
        if "video" in env_state:
            placeholder = self.video_placeholder
        else:
            placeholder = self.image_placeholder

        # Extract the content within the tool_pattern
        tool_match = re.search(self.tool_pattern, text, re.DOTALL)
        if not tool_match:
            no_action_msg = obs_template.format(text="No action found.",vision='')
            return False, False, {"text": no_action_msg, "image": None}

        try:
            # tool_content_str is the string representation of the list of tool calls
            tool_content_str = tool_match.group(1).strip()
            # ast.literal_eval safely parses this string into a Python list of dicts
            tool_calls_list = ast.literal_eval(tool_content_str)

            # Ensure it's a list. If it's a single dict (old format), wrap it in a list.
            if not isinstance(tool_calls_list, list):
                if isinstance(tool_calls_list, dict): # For backward compatibility or if a single tool call is not in a list
                    tool_calls_list = [tool_calls_list]
                else:
                    # If it's neither a list nor a dict after eval, it's an unexpected format.
                    raise ValueError("Tool content inside <tool_call> is not a valid list or dictionary of tool calls.")

        except Exception as e:
            error_msg = obs_template.format(text=f"Error parsing tool calls list: {str(e)}", vision='')
            # Return True for has_action because a <tool_call> block was found, but parsing failed.
            return True, False, {"text": error_msg, "image": None}

        if not tool_calls_list:  # Handles the case of an empty list like "<tool_call>[]</tool_call>"
            # Found a tool_call block, but the list of tools is empty.
            # This is considered a successful processing of an empty action list.
            empty_list_msg = obs_template.format(text="Tool call list is empty.\n", vision='')
            return True, False, {"text": empty_list_msg, "image": None}

        aggregated_obs_texts = []
        aggregated_images = []  # List to hold all images from all tool calls
        aggregated_videos = []
        all_actions_successful = True # Assume success until a failure occurs

        for i, tool_content in enumerate(tool_calls_list[:self.max_results]):
            if not isinstance(tool_content, dict):
                error_msg = f"Invalid item in tool call list: Expected a dictionary, got {type(tool_content)} ({str(tool_content)[:100]})."
                aggregated_obs_texts.append(error_msg)
                all_actions_successful = False
                continue  # Skip to the next tool call in the list

            try:
                tool_name = tool_content.get('tool_name')
                # args = tool_content.get('arguments')
                args = {}
                temporal_query = tool_content.get("query", None)
                spatial_objects = tool_content.get("objects", None)
                temporal_start = tool_content.get("start", None)
                temporal_end = tool_content.get("end", None)
                
                if temporal_query:
                    args['query'] = temporal_query
                if spatial_objects:
                    args['objects'] = spatial_objects
                if duration:
                    args['duration'] = duration
                if video_path:
                    args['video_path'] = video_path
                if temporal_start:
                    args['start'] = temporal_start
                if temporal_end:
                    args['end'] = temporal_end


                if not tool_name:
                    error_msg = f"Missing 'name' in tool content: {str(tool_content)[:100]}"
                    aggregated_obs_texts.append(error_msg)
                    all_actions_successful = False
                    continue
                
                # Ensure args is a dictionary if the tool expects it, even if 'arguments' is missing or None
                if args is None:
                    args = {}
                elif not isinstance(args, dict):
                    error_msg = f"Invalid 'arguments' for tool '{tool_name}': Expected a dictionary, got {type(args)}."
                    aggregated_obs_texts.append(error_msg)
                    all_actions_successful = False
                    continue
                
                if video_path is not None:
                    args['video_path'] = video_path


            except Exception as e:  # Catch errors during extraction of name/args from an individual tool_content
                error_msg = f"Error parsing individual tool call '{str(tool_content)[:100]}': {str(e)}"
                aggregated_obs_texts.append(error_msg)
                all_actions_successful = False
                continue  # Skip to the next tool call

            # Check if tool_name is a valid string before checking if it's in self.tools
            if not isinstance(tool_name, str):
                error_msg = f"Error: Tool name must be a string, got {type(tool_name)}."
                aggregated_obs_texts.append(error_msg)
                all_actions_successful = False
                continue
                
            if tool_name not in self.tools:
                error_msg = f"Error: There is no tool named '{tool_name}'."
                aggregated_obs_texts.append(error_msg)
                all_actions_successful = False
                continue  # Tool name is not available in the tools dictionary

            try:
                # Execute the tool call
                # import ipdb; ipdb.set_trace()
                result = self.tools[tool_name].call(args, env_state) # Assuming tool.call() interface

                # Aggregate text observation
                tool_obs_text = result.get('text', '') # Default to empty string if no text
                if tool_obs_text: # Append if there is text
                    aggregated_obs_texts.append(tool_obs_text)
                #     aggregated_obs_texts.append(f"Image {i+len(env_state['image'])}: {str(tool_obs_text)}") #这句话其实没太看懂

                # Aggregate image observation(s)
                tool_obs_image = result.get("image", None)
                tool_obs_video = result.get("video", None)
                if tool_obs_image is not None:
                    if isinstance(tool_obs_image, list):
                        aggregated_images.extend(tool_obs_image)
                    else:
                        aggregated_images.append(tool_obs_image)
                if tool_obs_video is not None:
                    if isinstance(tool_obs_video, list):
                        aggregated_videos.extend(tool_obs_video)
                    else:
                        aggregated_videos.append(tool_obs_video)

            
            except Exception as e:
                error_msg = f"Failed to call tool '{tool_name}' with args '{str(args)[:100]}' due to error: {str(e)}"
                aggregated_obs_texts.append(error_msg)
                all_actions_successful = False
                # Continue processing other tools in the list even if one fails

        # Consolidate all observations
        # Join individual text observations with a newline.
        # If no text observations were generated but actions were processed, provide a generic message.
        final_obs_content = "\n".join(aggregated_obs_texts)
        # print(tool_calls_list)
        # if not all_actions_successful:
        #     print(error_msg)
        if not final_obs_content and tool_calls_list: # Actions were processed, but no text output
            if all_actions_successful:
                final_obs_content = "All actions processed successfully with no textual output."
                formatted_final_obs_text = obs_template.format(text=final_obs_content, vision=placeholder*(len(aggregated_images)+len(aggregated_videos)))
            else:
                final_obs_content = "Actions processed with errors, but no specific textual error messages were generated."
                formatted_final_obs_text = obs_template.format(text=final_obs_content, vision=placeholder*(len(aggregated_images)+len(aggregated_videos)))
        elif not tool_calls_list: # Should have been caught by "Tool call list is empty"
             final_obs_content = "No actions to process."
             formatted_final_obs_text = obs_template.format(text=final_obs_content, vision=placeholder*(len(aggregated_images)+len(aggregated_videos)))
        
        if final_obs_content and tool_calls_list:
            if all_actions_successful:
                # final_obs_content = "New visual information. Continue Analyzing."
                formatted_final_obs_text = success_obs_template.format(text=final_obs_content, vision=placeholder*(len(aggregated_images)+len(aggregated_videos)))
            else:
                final_obs_content = "Tools use unsuccessful."
                formatted_final_obs_text = obs_template.format(text=final_obs_content, vision=placeholder*(len(aggregated_images)+len(aggregated_videos)))
        # formatted_final_obs_text = obs_template.format(text=final_obs_content, vision=placeholder*(len(aggregated_images)+len(aggregated_videos)))
        
        # If aggregated_images is empty, final_images should be None, otherwise it's the list of images.
        final_images = aggregated_images if aggregated_images else None
        final_videos = aggregated_videos if aggregated_videos else None

        # Return True for has_action as the <tool_call> block was found and processed.
        # all_actions_successful reflects if all tools in the list were processed without error.
        # import ipdb; ipdb.set_trace()
        return True, all_actions_successful, {"text": formatted_final_obs_text, "image": final_images, "video": final_videos}


    def multi_turn_generate(
        self,
        engine,
        vllm_inputs: List[Dict],
        running_states,
        total_attention_masks,
        extra_info,
        sampling_params,
        multi_modal_inputs=None,
    ) -> Tuple[List[List[int]], List[float], List[Dict]]:
        """
        Execute multi-turn generation, handling tool calls and observations
        
        Args:
            engine: vLLM inference engine
            vllm_inputs: List of vLLM inputs
            sampling_params: Sampling parameters
            
        Returns:
            (final_tokens, response_masks, final_mm_data): Final token sequences, response masks, and multi-modal data
        """
        sampling_params_ = deepcopy(sampling_params)
        sampling_params_.n = 1
        sampling_params_.stop_token_ids = self.eot_token_id
        sampling_params_.max_tokens = self.max_tokens_per_turn
        sampling_params_.repetition_penalty = 1.05
        max_total_length = self.max_prompt_length + self.max_response_length

        extra_info = deepcopy(extra_info)
            
        batch_size = len(vllm_inputs)
        # print("batch_size: ",batch_size)
                
        active_indices = [i for i in range(batch_size)]
        response_tool_use = [0] * batch_size
        
        response_tokens = [[] for _ in range(batch_size)]
        loss_mask = [[] for _ in range(batch_size)]
        current_pixel_data = [[deepcopy(item)] for item in multi_modal_inputs] 
        # Handle multi-turn interaction for each sample
        for turn in range(self.max_turns):
            # print(f"Starting round {turn+1} interaction")
            
            # Prepare vLLM inputs for the current turn
            # current_vllm_inputs = []
            
            # for b_idx in active_indices:
            #     current_vllm_inputs.append(vllm_inputs[b_idx])
            
            # print(f"Current active samples: {len(active_indices)}")
            # If there are no active samples, exit the loop
            if not active_indices:
                break
                
            outputs = []
            # print(turn)
            # Process samples in batches based on inference_batch_size
            for batch_start in range(0, len(active_indices), self.inference_batch_size):
                batch_end = min(batch_start + self.inference_batch_size, len(active_indices))
                batch_indices = active_indices[batch_start:batch_end]
                batch_vllm_inputs = [vllm_inputs[idx] for idx in batch_indices]

                
                # Generate outputs for current batch
                batch_outputs = engine.generate(
                    prompts=batch_vllm_inputs,
                    sampling_params=sampling_params_,
                    use_tqdm=False
                )
                
                # Add results to outputs list in correct order
                for i, output in enumerate(batch_outputs):
                    outputs.append(output)
            
            removed_indices = []
            # Process generation results, extract actions and execute
            for i, b_idx in enumerate(active_indices):
                output = outputs[i]
                
                # Get newly generated tokens (excluding the existing prompt part)
                new_tokens = []
                for sample_id in range(len(output.outputs)):
                    new_tokens = output.outputs[sample_id].token_ids
                    # Filter out token IDs greater than 151664
                    new_tokens = [token_id for token_id in new_tokens if token_id <= 151664]
                    break  # We only process the first output (n=1)
                
                # If no content is generated, skip
                if len(new_tokens) == 0:
                    removed_indices.append(b_idx)
                    continue
                
                # Append generated tokens to the current token sequence (including tool call part)
                vllm_inputs[b_idx]['prompt_token_ids'].extend(new_tokens)
                response_tokens[b_idx].extend(new_tokens)
                loss_mask[b_idx].extend([1] * len(new_tokens))

                response_token_ids = torch.tensor(new_tokens, dtype=torch.int64, device=running_states[b_idx].device)
                running_states[b_idx] = torch.cat([running_states[b_idx], response_token_ids])
                action_mask = torch.ones_like(response_token_ids, dtype=torch.int64, device=running_states[b_idx].device)
                total_attention_masks[b_idx] = torch.cat([total_attention_masks[b_idx], action_mask])
                
                # No response in the last round
                if turn == self.max_turns - 1:
                    continue
                    
                # Decode the generated text
                generated_text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
                
                has_action, action_success, observation = self.process_action_and_execute(generated_text, vllm_inputs[b_idx]['multi_modal_data'], extra_info[b_idx])

                # If no action is processed or EOS is included, complete the sample
                if not has_action or self.eos_token_id in new_tokens:
                    removed_indices.append(b_idx)
                    continue
                
                response_tool_use[b_idx] = transform(action_success, response_tool_use[b_idx])
                    
                # Get observation results
                obs_text = observation.get("text", "")
                obs_images = observation.get("image", None)
                obs_videos = observation.get("video", None)

                inputs = self.processor(
                    text=[obs_text],
                    images=obs_images,
                    videos=obs_videos,
                    return_tensors="pt",
                )
                
                obs_inputs = inputs['input_ids'].squeeze(0).tolist()
                if len(obs_inputs) + len(response_tokens[b_idx]) > self.response_length:
                    removed_indices.append(b_idx)
                    continue
                
                # Encode observation text and add to sequence
                if obs_text:
                    obs_tokens = self.tokenizer.encode(obs_text, add_special_tokens=False)
                    vllm_inputs[b_idx]['prompt_token_ids'].extend(obs_tokens)
                    loss_mask[b_idx].extend([1])


                # Update multi-modal data
                if obs_images is not None:
                    vllm_inputs[b_idx]['multi_modal_data']['image'].extend(obs_images)
                if obs_videos is not None:
                    vllm_inputs[b_idx]['multi_modal_data']['video'].extend(obs_videos)
                    
                response_tokens[b_idx].extend(inputs['input_ids'].squeeze(0).tolist())
                loss_mask[b_idx].extend([0] * (len(inputs['input_ids'].squeeze(0).tolist())-1))

                obs_token_ids = torch.tensor(inputs['input_ids'].squeeze(0).tolist(), dtype=torch.int64, device=running_states[b_idx].device)
                running_states[b_idx] = torch.cat([running_states[b_idx], obs_token_ids])
                action_mask = torch.ones_like(obs_token_ids, dtype=torch.int64, device=running_states[b_idx].device)
                total_attention_masks[b_idx] = torch.cat([total_attention_masks[b_idx], action_mask])
                
                if obs_images is not None:
                    current_pixel_data[b_idx].append({
                        "pixel_values": inputs['pixel_values'],
                        "image_grid_thw": inputs['image_grid_thw'],
                    })
                if obs_videos is not None:
                    current_pixel_data[b_idx].append(
                        {
                            "pixel_values_videos": inputs['pixel_values_videos'],
                            "video_grid_thw": inputs['video_grid_thw']
                        }
                    )
                    
            for b_idx in range(batch_size):
                if len(response_tokens[b_idx]) >= self.response_length:
                    # print(f"Overlength: Sample {b_idx} token sequence exceeded the maximum length.")
                    # If exceeds length limit, truncate and ensure no token is broken
                    max_length = self.response_length
                    response_tokens[b_idx] = response_tokens[b_idx][:max_length]
                    loss_mask[b_idx] = loss_mask[b_idx][:max_length]
                    removed_indices.append(b_idx)
            
            # Update active sample indices
            active_indices = [i for i in active_indices if i not in removed_indices]
            if not active_indices:
                break
        
        all_text = [self.tokenizer.decode(response_tokens[i], skip_special_tokens=False) for i in range(batch_size)]
        # self.save_response(all_text)
        target_device = running_states[0].device
        running_states = [state[: max_total_length] for state in running_states]
        state_tensor = pad_2d_list_to_length(running_states, self.tokenizer.pad_token_id, max_length=max_total_length).to(target_device)

        total_attention_masks = [mask[: max_total_length] for mask in total_attention_masks]
        attn_mask_tensor = pad_2d_list_to_length(total_attention_masks, 0, max_length=max_total_length).to(target_device)

        
        response_multi_modal_inputs = []
        if 'pixel_values' in multi_modal_inputs[0]:
            p_dtype = multi_modal_inputs[0]['pixel_values'].dtype
            i_dtype = multi_modal_inputs[0]['image_grid_thw'].dtype
            for i in range(batch_size):
                response_multi_modal_inputs.append({
                    "pixel_values": torch.cat([item['pixel_values'] for item in current_pixel_data[i]], dim=0).to(p_dtype) \
                        if current_pixel_data[i] else torch.tensor([]),
                    "image_grid_thw": torch.cat([item['image_grid_thw'] for item in current_pixel_data[i]], dim=0).to(i_dtype) \
                        if current_pixel_data[i] else torch.tensor([]),
                })
        else:
            p_dtype = multi_modal_inputs[0]['pixel_values_videos'].dtype
            i_dtype = multi_modal_inputs[0]['video_grid_thw'].dtype
            for i in range(batch_size):
                response_multi_modal_inputs.append({
                    "pixel_values_videos": torch.cat([item['pixel_values_videos'] for item in current_pixel_data[i]], dim=0).to(p_dtype) \
                        if current_pixel_data[i] else torch.tensor([]),
                    "video_grid_thw": torch.cat([item['video_grid_thw'] for item in current_pixel_data[i]], dim=0).to(i_dtype) \
                        if current_pixel_data[i] else torch.tensor([]),
                })
        if self.processor is not None and self.processor.image_processor.__class__.__name__ == "Qwen2VLImageProcessor":
            # For Qwen-VL: (n*bs, 3, seq_len)
            position_ids_list = [
                get_rope_index(
                    self.processor,
                    input_ids=state_tensor[i, :],
                    image_grid_thw=response_multi_modal_inputs[i].get("image_grid_thw", None),
                    video_grid_thw=response_multi_modal_inputs[i].get("video_grid_thw", None),
                    second_per_grid_ts=response_multi_modal_inputs[i].get("second_per_grid_ts", None),
                    attention_mask=attn_mask_tensor[i, :],
                ) for i in range(batch_size)
            ]
            position_ids_tensor = torch.stack(position_ids_list, dim=0)
        else:
            # For LM: (n*bs, seq_len)
            position_ids_tensor = None
        
        
        return response_tokens, loss_mask, response_multi_modal_inputs, position_ids_tensor, response_tool_use
    



