import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache, infer_device

device = f"{infer_device()}:0"

model_id = "meta-llama/Llama-2-7b-chat-hf"
model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16, device_map=device)
tokenizer = AutoTokenizer.from_pretrained(model_id)

past_key_values = DynamicCache()
messages = [{"role": "user", "content": "Hello, what's your name."}]
inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt", return_dict=True).to(
    model.device
)

generated_ids = inputs.input_ids
cache_position = torch.arange(inputs.input_ids.shape[1], dtype=torch.int64, device=model.device)
max_new_tokens = 10

for _ in range(max_new_tokens):
    outputs = model(**inputs, cache_position=cache_position, past_key_values=past_key_values, use_cache=True)
    # Greedily sample one next token
    next_token_ids = outputs.logits[:, -1:].argmax(-1)
    generated_ids = torch.cat([generated_ids, next_token_ids], dim=-1)
    # Prepare inputs for the next generation step by leaving unprocessed tokens, in our case we have only one new token
    # and expanding attn mask for the new token, as explained above
    attention_mask = inputs["attention_mask"]
    attention_mask = torch.cat([attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))], dim=-1)
    inputs = {"input_ids": next_token_ids, "attention_mask": attention_mask}
    cache_position = cache_position[-1:] + 1  # add one more position for the next token

print(tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0])
"[INST] Hello, what's your name. [/INST]  Hello! My name is LLaMA,"
