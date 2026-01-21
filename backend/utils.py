"""
Utility functions for text processing and chat templates
"""
from typing import List, Dict


def sanitize_utf8(text: str) -> str:
    """Remove incomplete UTF-8 characters from the end of text"""
    if not text:
        return text
    
    # Encode to bytes and check for incomplete sequences at the end
    try:
        encoded = text.encode('utf-8')
    except UnicodeEncodeError:
        # If encoding fails, try to recover
        return text.encode('utf-8', errors='ignore').decode('utf-8')
    
    # Check last few bytes for incomplete multi-byte sequences
    valid_length = len(encoded)
    
    for i in range(min(4, len(encoded))):
        idx = len(encoded) - 1 - i
        byte = encoded[idx]
        
        # 4-byte sequence start (11110xxx)
        if (byte & 0xF8) == 0xF0:
            if i < 3:  # Not enough bytes for complete sequence
                valid_length = idx
            break
        # 3-byte sequence start (1110xxxx)
        elif (byte & 0xF0) == 0xE0:
            if i < 2:
                valid_length = idx
            break
        # 2-byte sequence start (110xxxxx)
        elif (byte & 0xE0) == 0xC0:
            if i < 1:
                valid_length = idx
            break
        # Continuation byte (10xxxxxx) - keep looking
        elif (byte & 0xC0) == 0x80:
            continue
        # ASCII or valid start - we're good
        else:
            break
    
    if valid_length < len(encoded):
        return encoded[:valid_length].decode('utf-8', errors='ignore')
    
    return text


def apply_chat_template(prompt: str, model_path: str) -> str:
    """Apply appropriate chat template based on model type"""
    model_name_lower = model_path.lower()
    
    # Gemma format
    if "gemma" in model_name_lower:
        return f"<start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n"
    
    # Qwen format (if instruct)
    elif "qwen" in model_name_lower and "instruct" in model_name_lower:
        return f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
    
    # Swallow/TinySwallow format (alpaca-style)
    elif "swallow" in model_name_lower and "instruct" in model_name_lower:
        return f"### 指示:\n{prompt}\n\n### 応答:\n"
    
    # Default: no template (for base models)
    else:
        return prompt


def build_vllm_prompt_with_history(messages: List[Dict], model_path: str) -> str:
    """Build a prompt with conversation history for vLLM"""
    model_name_lower = model_path.lower()
    
    # Gemma format
    if "gemma" in model_name_lower:
        prompt_parts = []
        for msg in messages:
            if msg["role"] == "user":
                prompt_parts.append(f"<start_of_turn>user\n{msg['content']}<end_of_turn>")
            elif msg["role"] == "assistant":
                prompt_parts.append(f"<start_of_turn>model\n{msg['content']}<end_of_turn>")
        prompt_parts.append("<start_of_turn>model\n")
        return "\n".join(prompt_parts)
    
    # Qwen format (ChatML)
    elif "qwen" in model_name_lower and "instruct" in model_name_lower:
        prompt_parts = ["<|im_start|>system\nYou are a helpful assistant.<|im_end|>"]
        for msg in messages:
            if msg["role"] == "user":
                prompt_parts.append(f"<|im_start|>user\n{msg['content']}<|im_end|>")
            elif msg["role"] == "assistant":
                prompt_parts.append(f"<|im_start|>assistant\n{msg['content']}<|im_end|>")
        prompt_parts.append("<|im_start|>assistant\n")
        return "\n".join(prompt_parts)
    
    # Swallow/TinySwallow format (alpaca-style) - doesn't support multi-turn well
    elif "swallow" in model_name_lower and "instruct" in model_name_lower:
        # For Swallow, we'll concatenate the conversation
        conversation = ""
        for msg in messages:
            if msg["role"] == "user":
                conversation += f"ユーザー: {msg['content']}\n"
            elif msg["role"] == "assistant":
                conversation += f"アシスタント: {msg['content']}\n"
        return f"### 指示:\n以下の会話を続けてください。\n\n{conversation}\n### 応答:\n"
    
    # Default: simple concatenation
    else:
        prompt_parts = []
        for msg in messages:
            if msg["role"] == "user":
                prompt_parts.append(f"User: {msg['content']}")
            elif msg["role"] == "assistant":
                prompt_parts.append(f"Assistant: {msg['content']}")
        prompt_parts.append("Assistant:")
        return "\n".join(prompt_parts)

