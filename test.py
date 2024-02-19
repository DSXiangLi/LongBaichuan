# -*-coding:utf-8 -*-
import re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.utils import GenerationConfig

if __name__ == '__main__':
    model_dir = '/data/llm_models/models/Baichuan2-13B-Chat-Modify'
    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=False, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_dir, device_map="auto", torch_dtype=torch.float16,
                                                 trust_remote_code=True)
    model.generation_config = GenerationConfig.from_pretrained(model_dir)
    model.eval()

    device = 'cuda'
    context = """测试长内容"""
    text = '<reserved_106>{}<reserved_105>'.format(context)
    input_ids = tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
    generate_input = {
        "input_ids": input_ids,
        "pad_token_id": 0,
        "bos_token_id": 1,
        "eos_token_id": 2,
        "max_new_tokens": 4096,
        "temperature": 0.3,
        "top_k": 5,
        "top_p": 0.85,
        "repetition_penalty": 1.1,
        "do_sample": True
    }

    with torch.no_grad():
        outputs = model.generate(**generate_input)
        response = tokenizer.decode(outputs[0])
        pattern = re.compile(r'<reserved_105>(.*?)</s>', flags=re.S)
        content = pattern.search(response).group(1)
        print(content)
