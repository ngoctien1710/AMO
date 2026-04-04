from unittest import result
from urllib import response
from collections import defaultdict
from model.custom_language_model import LlamaLLM
from prompt.Prompts import build_amo_prompt, build_amo_summary_prompt, build_amo_final_prompt
from typing import List, Dict
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer, util
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
    
    def summarize_tool_results(self, queries: List[str], observations: List[str], k=2) -> List[dict]:
        """
        Xử lý hàng loạt List[Query] và List[Obs] để trích xuất Graph Facts.
        """
        if len(queries) != len(observations):
            raise ValueError("Length mismatch!")

        # Bước 1: Thu thập tất cả raw_output từ LLM (vẫn phải chạy vòng lặp vì LLM gọi từng query)
        raw_outputs = []
        for query, obs in zip(queries, observations):
            if not obs or len(obs.strip()) < 10:
                raw_outputs.append(None)
                continue
            
            prompt = build_amo_summary_prompt(query, obs)
            # Giả sử self.model.generate trả về dict có key "response"
            resp = self.model.generate(prompt, temperature=0.1)["response"]
            raw_outputs.append(resp)

        # Bước 2: Xử lý hậu kỳ (Parse + Deduplicate + k-hop) cho từng kết quả
        final_results = []
        for i, raw_output in enumerate(raw_outputs):
            if raw_output is None:
                final_results.append({"graph_facts": []})
                continue

            # Gọi hàm parse anh em mình đã chốt (nên tích hợp batch embed bên trong hàm này nếu được)
            all_triplets = self.parse_and_deduplicate(raw_output, self.embed_model)
            
            # Lấy seed dựa trên query tương ứng
            seeds = self.extract_seeds_from_query(queries[i], all_triplets, self.embed_model)
            
            # Duyệt k-hop
            sub_graph = self.expand_graph_k_hop(seeds, all_triplets, k=k)
            
            if not sub_graph and all_triplets:
                sub_graph = all_triplets[:3] 
            # -----------------------------------------------

            final_results.append({
                "graph_facts": sub_graph
            })

        return final_results

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

    def parse_and_deduplicate(self, llm_output, embed_model, threshold=0.92):
        raw_triplets = []
        lines = llm_output.strip().split('\n')
        
        # --- Bước 1: Parse thô (V3) ---
        for line in lines:
            if '|' not in line: continue
            line = re.sub(r"^[(\s]+|[)\s]+$", "", line.strip())
            parts = [p.strip().lower() for p in line.split('|')]
            if len(parts) == 3 and all(len(p) > 1 for p in parts):
                if not any(x in parts[0] or x in parts[2] for x in ["entity a", "entity b", "relationship"]):
                    raw_triplets.append(tuple(parts))
        
        if not raw_triplets: return []

        # --- Bước 2: Lọc trùng bằng Sklearn Cosine Similarity ---
        triplet_texts = [f"{s} {r} {o}" for s, r, o in raw_triplets]
        embeddings = embed_model.encode(triplet_texts) # Đầu ra là numpy array
        
        unique_indices = []
        for i in range(len(embeddings)):
            is_duplicate = False
            for j in unique_indices:
                # Reshape về 2D: (1, n_features)
                vec_i = embeddings[i].reshape(1, -1)
                vec_j = embeddings[j].reshape(1, -1)
                
                sim = cosine_similarity(vec_i, vec_j)[0][0]
                if sim > threshold:
                    is_duplicate = True
                    break
            if not is_duplicate:
                unique_indices.append(i)
                
        return [raw_triplets[i] for i in unique_indices]
    
    def extract_seeds_from_query(self, query, triplets, embed_model, threshold=0.85):
        all_nodes = set()
        for s, r, o in triplets:
            all_nodes.add(s); all_nodes.add(o)
            
        query_lower = query.lower()
        seeds = set()

        # --- Bước 1: Exact Match ---
        for node in all_nodes:
            if node in query_lower:
                seeds.add(node)
                
        # --- Bước 2: Semantic Seed với Sklearn ---
        if not seeds and all_nodes:
            node_list = list(all_nodes)
            query_emb = embed_model.encode([query_lower]) # Encode list để ra 2D ngay
            node_embs = embed_model.encode(node_list)     # Matrix 2D
            
            # Tính similarity giữa 1 query và tất cả nodes cùng lúc (Batch processing)
            similarities = cosine_similarity(query_emb, node_embs)[0]
            
            for idx, sim in enumerate(similarities):
                if sim > threshold:
                    seeds.add(node_list[idx])
                        
        return list(seeds)
    

    def expand_graph_k_hop(self, seeds, all_triplets, k=2):
        """
        Input: 
            - seeds: Danh sách các thực thể gốc (từ Query).
            - all_triplets: Toàn bộ kho Triplet đã parse và lọc trùng.
            - k: Số bước nhảy tối đa.
        Output:
            - result_triplets: List các bộ ba (S, R, O) nằm trong phạm vi k-hop.
        """
        if not seeds or k <= 0:
            return []

        # --- Bước 1: Xây dựng đồ thị (Adjacency List) để truy xuất nhanh ---
        # Ta xây dựng đồ thị vô hướng (bi-directional) để có thể duyệt từ S sang O và ngược lại
        graph = defaultdict(list)
        for triplet in all_triplets:
            s, r, o = triplet
            graph[s].append(triplet)
            graph[o].append(triplet)

        visited_nodes = set(seeds)
        visited_triplets = set()
        current_layer = set(seeds)

        # --- Bước 2: Duyệt theo chiều rộng (BFS) đến k-hop ---
        for _ in range(k):
            next_layer = set()
            for node in current_layer:
                # Lấy tất cả các cạnh liên quan đến node hiện tại
                for triplet in graph[node]:
                    if triplet not in visited_triplets:
                        visited_triplets.add(triplet)
                        
                        # Xác định node tiếp theo để nhảy (neighbor)
                        s, r, o = triplet
                        neighbor = s if node == o else o
                        
                        if neighbor not in visited_nodes:
                            visited_nodes.add(neighbor)
                            next_layer.add(neighbor)
            
            # Cập nhật lớp hiện tại cho hop tiếp theo
            current_layer = next_layer
            if not current_layer: # Hết đường để đi
                break

        return list(visited_triplets)

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

    def compute_confidence_score(self, logprobs: list, entropy: list, theta: float = 1e-8) -> float:
        if not logprobs or not entropy:
            return 0.0
    
        avg_logprob = sum(logprobs) / len(logprobs)
        avg_entropy = sum(entropy) / len(entropy)
        
        # Cách 1: Dùng xác suất thực (Dễ đặt ngưỡng từ 0 đến +vô cùng) 
        # confidence = torch.exp(torch.tensor(avg_logprob)).item()
        # return confidence / (avg_entropy + theta)

        # Cách 2: Dùng hiệu số (Đơn giản, tính toán nhanh)
        return avg_logprob - avg_entropy

    def inference(self, question: str, qid: str) -> Dict:
        feedback = None
        self.tools.set_qid(qid)
        current_step = 0
        run_history = {}
        thougt_logprobs = []    
        thought_entropy = []
        redundancy_streak = 0 
        SATURATED = False
        should_summarize_now = False # Cờ này để ép LLM tóm tắt lại history nếu nó bị lạc hướng hoặc đi vào vòng lặp
        while current_step < self.max_steps:
            # --- 1. CHỌN PROMPT DỰA TRÊN TRẠNG THÁI ---
            # Nếu đã bão hòa hoặc đến bước cuối, dùng Final Prompt để ép chốt
            
            is_final_attempt = SATURATED or current_step >= self.max_steps - 1 
            if is_final_attempt:
                prompt = build_amo_final_prompt(question, history=run_history, FORCE_SUMMARIZE=True)
                response = self.model.generate(prompt, temperature=0.0) # Ép tính ổn định
                response = response["response"]
            else:
                prompt = build_amo_prompt(question, history=run_history, feedback=feedback, FORCE_SUMMARIZE=should_summarize_now)
                output = self.model.generate(prompt, temperature=0.3)
                response = output["response"]
                thougt_logprobs = output.get("thought_logprobs", [])
                thought_entropy = output.get("thought_entropy", [])
            should_summarize_now = False # Reset cờ sau khi dùng prompt có ép tóm tắt, cho phép LLM tự do hơn ở bước tiếp theo nếu không bị bão hòa
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

                if not new_queries :
                    if self.compute_confidence_score(logprobs=thougt_logprobs, entropy=thought_entropy) <= 1 and current_step > 0:
                        SATURATED = True
                    should_summarize_now = True # Nếu không có query mới nào được generate, ép LLM tóm tắt lại history để tìm manh mối mới
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
                            "Graph Facts": summaries[i]["graph_facts"]
                        })
                    
                    feedback = None
            else:
                feedback = "Format Error: Use [QUERY] or [FINAL] tags strictly."
                redundancy_streak += 1
                if redundancy_streak >= 2: SATURATED = True
                should_summarize_now = True # Ép tóm tắt lại history để LLM có cái nhìn tổng quan và hy vọng thoát khỏi format lỗi
            current_step += 1
            #print(f"--- End Step {current_step-1} ---")
            
        return run_history


