from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("/root/autodl-tmp/tensorrt_llm/examples/llama/llama-7b")
model = AutoModelForCausalLM.from_pretrained("/root/autodl-tmp/tensorrt_llm/examples/llama/llama-7b")
model.half()
model.cuda()

inp = '谁是美国的第一个总统？'
input_ids = tokenizer.encode(inp, return_tensors="pt").cuda()
max_token = 1024
out = model.generate(input_ids, max_length=max_token, num_beams=1)
print(tokenizer.decode(out[0]))
