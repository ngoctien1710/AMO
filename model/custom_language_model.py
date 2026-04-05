import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import warnings
warnings.filterwarnings("ignore")
from transformers.utils import logging
logging.set_verbosity_error()
from utils.seed import set_seed

class LlamaLLM:
    _shared_model = None
    _shared_tokenizer = None
    _global_call_count = 0

    def __init__(
        self,
        model_name: str = "meta-llama/Llama-3.2-3B-Instruct",
        quantization: bool = True,
        device: str = None,
        **hf_kwargs
    ):
        self.model_name = model_name
        self.quantization = quantization
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.hf_kwargs = hf_kwargs
        self.seed = 42

        if LlamaLLM._shared_model is not None:
            self.model = LlamaLLM._shared_model
            self.tokenizer = LlamaLLM._shared_tokenizer
            print("Reusing existing LLM from GPU cache")
        else:
            self._load_model_and_tokenizer()
            LlamaLLM._shared_model = self.model
            LlamaLLM._shared_tokenizer = self.tokenizer

    def _load_model_and_tokenizer(self):
        print(f"Loading {self.model_name} to GPU...")
        quant_config = None
        if self.quantization:
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            )

<<<<<<< HEAD
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            device_map="auto" if self.device == "cuda" else None,
            quantization_config=quant_config,
            torch_dtype=torch.bfloat16 if self.device == "cuda" else torch.float32,
=======
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            device_map="cuda:1" if self.device == "cuda" else None,
            quantization_config=quant_config,
            torch_dtype=torch.bfloat16 if self.device == "cuda" else torch.float32,
            trust_remote_code=True,
>>>>>>> a540a6f (EM 30.8, F1 44.13)
            **self.hf_kwargs
        )
        self.model.eval()

    def generate(
        self,
        prompt: str,
        temperature: float = 0.1,
        max_new_tokens: int = 512,
        top_p: float = 0.9,
        do_sample: bool = False,
        seed: int = 42,
        use_logprob: bool = False,
        use_entropy: bool = False,
        **gen_kwargs
    ) -> dict:
        self.seed = seed
        set_seed(self.seed)
        messages = [{"role": "user", "content": prompt}]

        inputs = self.tokenizer.apply_chat_template(
            messages,
            return_tensors="pt",
<<<<<<< HEAD
            add_generation_prompt=True
        ).to(self.model.device)

=======
            add_generation_prompt=True,
            padding=True, 
            truncation=True
        ).to(self.model.device)
        attention_mask = (inputs != self.tokenizer.pad_token_id).long()
>>>>>>> a540a6f (EM 30.8, F1 44.13)
        LlamaLLM._global_call_count += 1
        
        # Chỉ bật output_scores nếu cần tính logprob hoặc entropy
        output_scores = use_logprob or use_entropy

        with torch.no_grad():
            output = self.model.generate(
                inputs,
<<<<<<< HEAD
=======
                attention_mask=attention_mask,
>>>>>>> a540a6f (EM 30.8, F1 44.13)
                temperature=temperature,
                max_new_tokens=max_new_tokens,
                top_p=top_p,
                do_sample=do_sample,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.eos_token_id,
                use_cache=True,
                return_dict_in_generate=True,
                output_scores=output_scores,
                **gen_kwargs
            )

        # 1. Decode văn bản
        generated_sequences = output.sequences[0]
        new_tokens = generated_sequences[inputs.shape[-1]:]
        response_text = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
<<<<<<< HEAD
    
=======
>>>>>>> a540a6f (EM 30.8, F1 44.13)
        # Khởi tạo result với các mảng rỗng thay vì score tính sẵn
        result = {
            "response": response_text,
            "thought_logprobs": [],
            "thought_entropy": [],
            "action_logprobs": [],
            "action_entropy": []
        }

        # 2. Thu hoạch Logprobs và Entropy
<<<<<<< HEAD
=======
        # 2. Thu hoạch Logprobs và Entropy
        # 2. Thu hoạch Logprobs và Entropy
>>>>>>> a540a6f (EM 30.8, F1 44.13)
        if output_scores and hasattr(output, "scores"):
            logits = torch.stack(output.scores, dim=0).squeeze(1) 
            
            all_logprobs = []
            all_entropy = []

            if use_logprob:
                log_probs_dist = torch.log_softmax(logits, dim=-1)
<<<<<<< HEAD
                all_logprobs = log_probs_dist.gather(dim=-1, index=new_tokens.unsqueeze(-1)).squeeze(-1).cpu().tolist()

            if use_entropy:
                probs = torch.softmax(logits, dim=-1)
                all_entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1).cpu().tolist()

            # 3. Phân tách dựa trên marker "Action:" mà không tính toán score
            marker = "Action:"
            if marker in response_text:
                # Tìm điểm cắt dựa trên số lượng token của phần Thought
                thought_part = response_text.split(marker)[0]
                split_idx = len(self.tokenizer.encode(thought_part, add_special_tokens=False))

                # Gán mảng thô về cho các phần
=======
                current_new_tokens = new_tokens[:logits.size(0)]
                all_logprobs = log_probs_dist.gather(dim=-1, index=current_new_tokens.unsqueeze(-1)).squeeze(-1).cpu().tolist()

            if use_entropy:
                probs = torch.softmax(logits, dim=-1)
                entropy_tensor = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1)
                all_entropy = entropy_tensor.cpu().tolist()

            marker = "Action:"
            if marker in response_text:
                marker_start_pos = response_text.find(marker)
                thought_part = response_text[:marker_start_pos]
                
                # Encode chuẩn phần text trước marker
                tokenized_thought = self.tokenizer.encode(thought_part, add_special_tokens=False)
                split_idx = len(tokenized_thought)

>>>>>>> a540a6f (EM 30.8, F1 44.13)
                if use_logprob:
                    result["thought_logprobs"] = all_logprobs[:split_idx]
                    result["action_logprobs"] = all_logprobs[split_idx:]
                if use_entropy:
                    result["thought_entropy"] = all_entropy[:split_idx]
                    result["action_entropy"] = all_entropy[split_idx:]
            else:
<<<<<<< HEAD
                # Nếu không có Action, coi như toàn bộ là Thought
=======
>>>>>>> a540a6f (EM 30.8, F1 44.13)
                if use_logprob: result["thought_logprobs"] = all_logprobs
                if use_entropy: result["thought_entropy"] = all_entropy

        return result
<<<<<<< HEAD
    
=======

>>>>>>> a540a6f (EM 30.8, F1 44.13)
    def get_call_count(self):
        return LlamaLLM._global_call_count

    def reset_call_count(self):
        LlamaLLM._global_call_count = 0

if __name__ == "__main__":
<<<<<<< HEAD
    llm = LlamaLLM()
    prompt = "Who is the author of 'The Changing Scottish Landscape'?"
    # Chạy thử với cả 2 tham số
    res = llm.generate(prompt, use_logprob=True, use_entropy=True)
    print(f"Response: {res['response']}")
    if res['entropy']:
        print(f"Avg Entropy: {sum(res['entropy'])/len(res['entropy']):.4f}")
    if res['logprobs']:
        print(f"Avg Logprob: {sum(res['logprobs'])/len(res['logprobs']):.4f}")
=======
    llm = LlamaLLM() # Model mặc định hoặc truyền Qwen vào đây
    prompt = "Who is the author of 'The Changing Scottish Landscape'?"
    
    # Chạy thử với cả 2 tham số
    res = llm.generate(prompt, use_logprob=True, use_entropy=True)
    
    print(f"Response: {res['response']}")
    
    # Check và in kết quả Thought/Action thay vì logprobs chung
    if res['thought_logprobs']:
        print(f"Thought Tokens: {len(res['thought_logprobs'])}")
        avg_lp = sum(res['thought_logprobs'])/len(res['thought_logprobs'])
        print(f"Avg Thought Logprob: {avg_lp:.4f}")
        
    if res['action_logprobs']:
        print(f"Action Tokens: {len(res['action_logprobs'])}")
>>>>>>> a540a6f (EM 30.8, F1 44.13)
