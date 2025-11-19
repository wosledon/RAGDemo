#!/usr/bin/env python3
import argparse
import json
import os
import sys

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-dir', required=False)
    parser.add_argument('--text-file', required=True)
    parser.add_argument('--jieba', action='store_true', help='Use jieba for Chinese tokenization')
    args = parser.parse_args()
    model_dir = args.model_dir if hasattr(args, 'model_dir') else None
    tf = args.text_file
    # 支持中文分词（jieba）
    if getattr(args, 'jieba', False):
        try:
            import jieba
            text = open(tf, 'r', encoding='utf-8').read()
            tokens = list(jieba.cut(text, cut_all=False))
            print(' '.join(tokens))
            return
        except Exception as e:
            print(json.dumps({'error': 'jieba not available', 'detail': str(e)}))
            sys.exit(4)
    if not os.path.exists(tf):
        print(json.dumps({'error':'text file not found'}))
        sys.exit(2)
    text = open(tf, 'r', encoding='utf-8').read()
    # Try fast Tokenizers first
    try:
        from tokenizers import Tokenizer
        tok_path = os.path.join(model_dir, 'tokenizer.json')
        if os.path.exists(tok_path):
            tokenizer = Tokenizer.from_file(tok_path)
            enc = tokenizer.encode(text)
            ids = enc.ids
            # tokenizers library doesn't provide token_type_ids; we set defaults
            attention_mask = [1] * len(ids)
            token_type_ids = [0] * len(ids)
            print(json.dumps({'input_ids': ids, 'attention_mask': attention_mask, 'token_type_ids': token_type_ids}))
            return
    except Exception as e:
        # fallthrough to transformers
        pass

    # Fallback to transformers AutoTokenizer
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
        enc = tokenizer(text, return_tensors='pt')
        ids = enc['input_ids'][0].tolist()
        attention_mask = enc['attention_mask'][0].tolist() if 'attention_mask' in enc else [1]*len(ids)
        token_type_ids = enc['token_type_ids'][0].tolist() if 'token_type_ids' in enc else [0]*len(ids)
        print(json.dumps({'input_ids': ids, 'attention_mask': attention_mask, 'token_type_ids': token_type_ids}))
        return
    except Exception as e:
        print(json.dumps({'error': 'tokenizer not available', 'detail': str(e)}))
        sys.exit(3)

if __name__ == '__main__':
    main()
