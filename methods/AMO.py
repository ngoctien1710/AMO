from model.custom_language_model import LlamaLLM
from prompt.Prompts import build_amo_prompt, build_amo_summary_prompt, build_amo_final_prompt
from typing import List, Dict
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import re
from methods.Tools import Tools
class AMO:
    def __init__(self, model):
        self.model = model
        self.name = "AMO"
        self.max_steps = 5
        self.tools = Tools()
        self.embed_model = SentenceTransformer('all-MiniLM-L6-v2')

    def filter_duplicate_queries_internal_first(self, old_queries, new_queries, threshold=0.85):
        if not new_queries: return []
        # Batch encode để tiết kiệm thời gian
        new_embeds = self.embed_model.encode(new_queries)
        unique_indices = []
        for i in range(len(new_queries)):
            is_dup = False
            for j in unique_indices:
                if cosine_similarity([new_embeds[i]], [new_embeds[j]])[0][0] >= threshold:
                    is_dup = True; break
            if not is_dup: unique_indices.append(i)
        unique_new_queries = [new_queries[i] for i in unique_indices]
        if not old_queries: return unique_new_queries
        # So sánh với query cũ bằng matrix multiplication
        old_embeds = self.embed_model.encode(old_queries)
        sim_matrix = cosine_similarity(new_embeds[unique_indices], old_embeds)
        return [unique_new_queries[i] for i in range(len(unique_new_queries)) if np.max(sim_matrix[i]) < threshold]
    
    def summarize_tool_results(self, queries: List[str], observations: List[str]) -> List[str]:
        """
        Tóm tắt kết quả tool theo từng query.

        Args:
            queries: list các query mà LLM đã generate
            observations: list kết quả tương ứng từ tool (cùng index với queries)

        Returns:
            summaries: list tóm tắt cho mỗi query
        """
        summaries = []

        # check length match
        if len(queries) != len(observations):
            raise ValueError("Length of queries and observations must match!")

        for query, obs in zip(queries, observations):
            if not obs or "no relevant information" in obs.lower() or len(obs.strip()) < 10:
                summaries.append("No relevant information found in the knowledge base.")
            else:
                prompt = build_amo_summary_prompt(query, obs)
                summary = self.model.generate(prompt, temperature=0.1)
                summaries.append(summary)
        return summaries

    def parse_llm_output_with_thought(self, text: str) -> Dict:
        """
        Parse LLM output để lấy Thought + Action ([QUERY] hoặc [FINAL]).
        Bổ sung: Lấy cả Reason để đưa vào History giúp Agent bớt 'ngáo'.
        """
        text = text.strip()

        # --- 1. Lấy Thought ---
        thought_match = re.search(r'Thought:(.*?)(Action:|$)', text, re.DOTALL)
        thought_text = thought_match.group(1).strip() if thought_match else ""

        # --- 2. Kiểm tra [FINAL] ---
        final_match = re.search(r'\[FINAL\](.*?)(\[|$)', text, re.DOTALL)
        if final_match:
            final_text = final_match.group(1).strip()
            return {"thought": thought_text, "type": "FINAL", "items": [final_text]}

        # --- 3. Parse [QUERY] (Lấy cả Question và Reason) ---
        # Sửa regex để linh hoạt hơn với khoảng trắng và dấu ngoặc
        query_matches = re.findall(
            r'\[QUERY\]\s*Question:\s*["\']?(.*?)["\']?\s*Reason:\s*["\']?(.*?)(?=\s*\[|$)',
            text,
            re.IGNORECASE | re.DOTALL
        )

        if query_matches:
            # items: vẫn trả về list questions để ông chạy hàm filter_duplicate
            questions = [q for q, r in query_matches]
            # reasons: list các lý do tương ứng
            reasons = [r for q, r in query_matches]
            
            return {
                "thought": thought_text, 
                "type": "QUERY", 
                "items": questions,
                "reasons": reasons # <--- Thêm cái này vào đây
            }

        # --- 4. Không tìm thấy gì ---
        return {"thought": thought_text, "type": "UNKNOWN", "items": [], "reasons": []}

    # def tool_calls_to_observations(self, queries: List[str]) -> List[str]:
    #     """
    #     Gọi tool tương ứng với từng query và trả về list observation.

    #     Args:
    #         queries: list câu hỏi để gọi tool

    #     Returns:
    #         observations: list kết quả trả về từ tool, cùng index với queries
    #     """
    #     observations = []
    #     for query in queries:
    #         obs = self.tools.search_sentences(query)
    #         observations.append(obs)
    #     return observations
    def tool_calls_to_observations(self, queries: List[str]) -> List[str]:
        """
        Gọi tool tương ứng với từng query và trả về list observation.
        Với mỗi obs, lấy 2 mục đầu tiên, dùng title để gọi get_page() rồi ghép vào obs.
        """
        observations = []
        for query in queries:
            obs = self.tools.search_sentences(query)  # obs là string
            obs_lines = obs.split("\n")  # giả sử mỗi result 1 line
            first_two_titles = []

            # Lấy 2 line đầu để extract title
            for line in obs_lines[:2]:
                # parse title từ format "i. [Title page: {title}] {sentence}"
                match = re.search(r'\[Title page: (.*?)\]', line)
                if match:
                    first_two_titles.append(match.group(1))

            # gọi get_page với 2 title đó và ghép vào obs
            page_results = []
            for title in first_two_titles:
                page_results.append(self.tools.get_page(title))  # giả sử trả về string

            # ghép obs + page_results
            combined_obs = obs + "\n" + "\n".join(page_results)
            observations.append(combined_obs.strip())

        return observations

    def inference(self, question: str, qid: str) -> Dict:
        feedback = None
        self.tools.set_qid(qid)
        current_step = 0
        run_history = {}
        
        redundancy_streak = 0 
        SATURATED = False
        
        while current_step < self.max_steps:
            # --- 1. CHỌN PROMPT DỰA TRÊN TRẠNG THÁI ---
            # Nếu đã bão hòa hoặc đến bước cuối, dùng Final Prompt để ép chốt
            is_final_attempt = SATURATED or current_step >= self.max_steps - 1
            
            if is_final_attempt:
                prompt = build_amo_final_prompt(question, history=run_history)
                response = self.model.generate(prompt, temperature=0.0) # Ép tính ổn định
            else:
                prompt = build_amo_prompt(question, history=run_history, feedback=feedback)
                response = self.model.generate(prompt, temperature=0.3)
            
            #print(f"LLM Response at step {current_step}:\n{response}\n")
            parsed = self.parse_llm_output_with_thought(response)
            
            # Khởi tạo step history (Lưu Thought để báo cáo cho đẹp)
            run_history[current_step] = {
                "thought": parsed["thought"] if parsed["thought"] else "Synthesizing final answer...",
                "queries": [],
                "action": []
            }

            # --- 2. LOGIC THOÁT (EXIT STRATEGY) ---
            # Nếu model tự chốt [FINAL] HOẶC chúng ta đang ở bước ép chốt
            if parsed["type"] == "FINAL" or is_final_attempt:
                # Nếu parse không ra (do dùng Final Prompt format khác), lấy nguyên response
                final_ans = parsed["items"][0] if parsed["items"] else response.strip()
                run_history[current_step]["final"] = final_ans
                #print(f"--- Process Terminated at Step {current_step} ---")
                return run_history

            # --- 3. LOGIC TRUY XUẤT (QUERY) ---
            elif parsed["type"] == "QUERY":
                past_queries = []
                for s_idx in range(current_step):
                    past_queries.extend(run_history[s_idx].get("queries", []))
                
                new_queries = self.filter_duplicate_queries_internal_first(
                    old_queries=past_queries,
                    new_queries=parsed["items"]
                )
                
                run_history[current_step]["queries"] = new_queries

                if not new_queries:
                    redundancy_streak += 1
                    feedback = f"Duplicate Detected. You've searched for {parsed['items']} before. Pivot or conclude."
                    if redundancy_streak >= 2: SATURATED = True
                    
                else:
                    # Reset streak nếu tìm được cái mới
                    redundancy_streak = 0 
                    
                    raw_observations = self.tool_calls_to_observations(new_queries)
                    summaries = self.summarize_tool_results(new_queries, raw_observations)
                    
                    for i, q in enumerate(new_queries):
                        idx_in_parsed = parsed["items"].index(q)
                        reason = parsed["reasons"][idx_in_parsed]
                        
                        run_history[current_step]["action"].append({
                            "Question": q,
                            "Reason": reason,
                            "Raw Observation": raw_observations[i],
                            "Summary": summaries[i]
                        })
                    
                    feedback = None
            else:
                feedback = "Format Error: Use [QUERY] or [FINAL] tags strictly."
                #redundancy_streak += 1
                #if redundancy_streak >= 2: SATURATED = True
            current_step += 1
            #print(f"--- End Step {current_step-1} ---")
            
        return run_history


