# -*-coding:utf-8 -*-
import re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.utils import GenerationConfig
from streaming.inference import streaming_inference
from streaming.kv_cache import StartRecentKVCache


if __name__ == '__main__':
    # 这里不测试多轮streaming，只模拟测试单轮
    model_dir = '/data/llm_models/models/Baichuan2-13B-Chat-Modify'
    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=False, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_dir, device_map="auto", torch_dtype=torch.float16,
                                                 trust_remote_code=True)
    model.generation_config = GenerationConfig.from_pretrained(model_dir)
    model.eval()
    device = 'cuda'
    context = """测试长内容"""
    text = '<reserved_106>{}<reserved_105>'.format(context)

    kv_cache = StartRecentKVCache(
        start_size=4,
        recent_size=4000,
        k_seq_dim=2,
        v_seq_dim=2, # baichuan的seq dim
    )

    streaming_inference(model, tokenizer, [text],
                        kv_cache=kv_cache, max_gen_len=1000, greedy=False)
