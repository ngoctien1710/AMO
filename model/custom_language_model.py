# ...existing code...

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

import warnings
warnings.filterwarnings("ignore")

from transformers.utils import logging
logging.set_verbosity_error()

from utils.seed import set_seed



class LlamaLLM:
	"""
	Core utility class for loading and running meta-llama/Llama-3.1-8B-Instruct with 4-bit quantization.
	Designed for extensibility and efficient use of GPU memory.
	"""

	# ===== ADD: GLOBAL CACHE =====
	_shared_model = None
	_shared_tokenizer = None
	_global_call_count = 0
	# =============================
	# model = "meta-llama/Llama-3.2-3B-Instruct"
	# model = "meta-llama/Meta-Llama-3.1-8B-Instruct"
	# model = "Qwen/Qwen2.5-7B-Instruct"
	# model = "Qwen/Qwen3-4B-Instruct-2507"
	# model = "Salesforce/xgen-small-4B-instruct-r"
	# model = "google/gemma-2-2b"
	# model = Qwen/Qwen2.5-0.5B-Instruct
	def __init__(
		self,
		model_name: str = "meta-llama/Llama-3.2-3B-Instruct",
		quantization: bool = True,
		device: str = None,
		**hf_kwargs
	):
		"""
		Initialize the LlamaLLM core module.
		Args:
			model_name (str): Hugging Face model name.
			quantization (bool): Whether to use 4-bit quantization (bitsandbytes).
			device (str): 'cuda', 'cpu', or None for auto-detect.
			hf_kwargs: Additional kwargs for model loading.
		"""
		
		self.model_name = model_name
		self.quantization = quantization
		self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
		self.hf_kwargs = hf_kwargs
		self.seed = 42
		self.model = None
		self.tokenizer = None

		# ===== ADD: REUSE MODEL IF ALREADY LOADED =====
		if LlamaLLM._shared_model is not None:

			self.model = LlamaLLM._shared_model
			self.tokenizer = LlamaLLM._shared_tokenizer

			print("Reusing existing LLM from GPU cache")

		else:

			self._load_model_and_tokenizer()

			LlamaLLM._shared_model = self.model
			LlamaLLM._shared_tokenizer = self.tokenizer
		# ==============================================



	def _load_model_and_tokenizer(self):
		"""
		Loads the model and tokenizer with optional 4-bit quantization.
		"""

		print("Loading LLM model to GPU...")

		quant_config = None

		if self.quantization:
			quant_config = BitsAndBytesConfig(
				load_in_4bit=True,
				bnb_4bit_use_double_quant=True,
				bnb_4bit_quant_type="nf4",
				bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
				# ===== fix: cho phép offload CPU khi thiếu VRAM =====
        		#load_in_4bit_fp32_cpu_offload=True
			)

		self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
		self.model = AutoModelForCausalLM.from_pretrained(
			self.model_name,
			device_map="cuda:0" if self.device == "cuda" else None,
			#max_memory=max_memory if self.device == "cuda" else None,
			quantization_config=quant_config,
			torch_dtype=torch.bfloat16 if self.device == "cuda" else torch.float32,
			**self.hf_kwargs
		)

		self.model.eval()

	def get_call_count(self):
		return LlamaLLM._global_call_count

	def reset_call_count(self):
		LlamaLLM._global_call_count = 0

	def generate(
		self,
		prompt: str,
		temperature: float = 0.1,
		max_new_tokens: int = 512,
		top_p: float = 0.9,
		do_sample: bool = False,
		seed: int = 42,
		**gen_kwargs
	) -> str:
		self.seed = seed
		set_seed(self.seed)
		messages = [{"role": "user", "content": prompt}]

		inputs = self.tokenizer.apply_chat_template(
			messages,
			return_tensors="pt",
			add_generation_prompt=True
		).to(self.model.device)

		LlamaLLM._global_call_count += 1
		
		with torch.no_grad():
			output = self.model.generate(
				inputs,
				temperature=temperature,
				max_new_tokens=max_new_tokens,
				top_p=top_p,
				do_sample=do_sample,
				eos_token_id=self.tokenizer.eos_token_id,
				pad_token_id=self.tokenizer.eos_token_id,
				use_cache=True,
				**gen_kwargs
			)

		response = self.tokenizer.decode(output[0], skip_special_tokens=True)

		return response.split("assistant")[-1].strip()


	def get_model(self):
		"""Returns the underlying model object."""
		return self.model


	def get_tokenizer(self):
		"""Returns the underlying tokenizer object."""
		return self.tokenizer


# Example usage (remove or comment out in production):
if __name__ == "__main__":
	llm = LlamaLLM()
	prompt = "What is the capital of France?"
	print("Prompt:", prompt)
	print("Response:", llm.generate(prompt))