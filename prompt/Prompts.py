def history_to_text(history, FORCE_SUMMARIZE=False):
    history_text = ""
    steps = sorted(history.keys())
    if not steps:
        return "No history yet."
    last_step = steps[-1] # Xác định bước gần nhất vừa thực hiện
    for step in steps:
        data = history[step]
        history_text += f"Step {step}:\n"
        history_text += f"Thought: {data.get('thought', '')}\n"
        actions = data.get('action', [])
        for act in actions:
            history_text += f"Action:\n"
            history_text += f"[QUERY]\n"
            # Lưu ý: Key phải khớp với lúc ông lưu (Question/question)
            history_text += f"Question: {act.get('Question')}\n"
            history_text += f"Reason: {act.get('Reason')}\n"
            # --- LOGIC HYBRID TẠI ĐÂY ---
            if step == last_step and not FORCE_SUMMARIZE:
                # Bước vừa xong: Đưa bản thô (Raw) để tìm manh mối cho bước tiếp theo
                obs = act.get('Raw Observation', 'No data found.')
                history_text += f"Observation (Detailed): {obs}\n"
            else:
                # Các bước cũ: Chỉ đưa bản tóm tắt (Summary) để tránh nhiễu
                graph_facts = act.get('Graph Facts', []) # Mặc định là list rỗng thay vì string
                if graph_facts:
                    facts_str = "\n".join([f"- ({s} | {r} | {o})" for s, r, o in graph_facts])
                    history_text += f"Graph Facts:\n{facts_str}\n"
                else:
                    history_text += "Graph Facts: (No information extracted)\n"

        history_text += "-" * 20 + "\n"
    return history_text

def build_amo_prompt(question, history=None, feedback=None, FORCE_SUMMARIZE=False):
    history_text = history_to_text(history, FORCE_SUMMARIZE) if history else "No history yet."
    thought = """Use this space to summarize what you currently know from the observations, what information is missing, and what you plan to retrieve next. Keep this short (1–2 sentences)
""" if history else """ you need to identify the core entities in the question to start the search """
    prompt = f"""
You are an intelligent language model. Use the following iterative process:

Thought: {thought}
Action: Decide what to do next. **You have exactly two options for this Action step**:
1. Generate a batch of ONE OR MORE [QUERY] blocks to retrieve missing information. **All queries in this step must be [QUERY]; do not include [FINAL] here.**
2. Provide [FINAL] if you have all information and are ready to answer the question. **Do not include any [QUERY] in this step.**

[QUERY] format:
[QUERY]
Question: <your retrieval question>
Reason: <why you need this question>

[FINAL] format:
[FINAL] 
<your concise answer here>

Example 1 - Using [QUERY]:

Thought:
I know the capital cities of some countries, but I do not know the capital of France. I need this information to answer the question.

Action:
[QUERY]
Question: What is the capital of France?
Reason: Retrieve the capital city to answer the question about France.

Example 2 - Using [FINAL]:

Thought:
Observation states that the capital of France is known.

Action:
[FINAL] 
Paris

Important rules:
- Treat all information in the observations as factual and fully reliable.
- Do **not guess** or invent anything.
- When writing Thought, **only summarize facts explicitly present in the observations**.
- Do not add any inferred or assumed information to Thought.
- If some information is missing, clearly state: "Information missing" in Thought.
- If the observations contain the answer, respond immediately with [FINAL].
- Keep the answer concise (2–3 words) and directly relevant to the original question.
- Never include [QUERY] if providing [FINAL].


Question: {question}
History chat: {history_text}
"""
    if feedback:
        prompt += f"""
ATTENTION (Self-Correction):
{feedback}
Please reconsider your strategy and generate NEW, DISTINCT queries that target missing information.
"""
    return prompt

def build_amo_summary_prompt(query, observation):
    """
    Xây dựng prompt để LLM trích xuất Knowledge Graph (Triplets) từ kết quả raw.
    Mục tiêu: Chống tóm tắt lụi, tập trung vào thực thể và quan hệ liên quan đến Query.
    """
    prompt = f"""
You are a Knowledge Graph Extractor. Your task is to transform raw text into a structured set of triplets (Subject, Relation, Object).

Query: "{query}"

Raw Information: "{observation}"

Instructions:
1. ONLY extract information that directly helps in answering the Query. Ignore all distractors.
2. Focus on "Nodes" (Entities: Names, Dates, Locations, Events) and "Edges" (Relationships between them).
3. Format each finding as a triplet: (Subject | Relationship | Object).
4. DO NOT write full sentences. DO NOT answer the query.
5. If the information is conflicting across sources, include both but mark them clearly.


Output format MUST be:
[SUMMARY]:
(Entity A | relationship | Entity B)
(Entity B | relationship | Entity C)
...

[SUMMARY]:
"""
    return prompt


def build_amo_final_prompt(question, history, FORCE_SUMMARIZE=False):
    history_text = history_to_text(history, FORCE_SUMMARIZE) if history else "No history yet."
    prompt = f"""
Question: {question}
History chat: {history_text}
You have gathered all the necessary information from the observations. Now, provide a concise final answer to the original question based on the facts from the observations. Do NOT include any new information that was not
explicitly stated in the observations. Your answer should be 2-3 words at most and directly address the original question.

Example - 1 - Using [FINAL]:

Thought:
Observation states that the capital of France is known.

Action:
[FINAL] Paris

Example - 2 - Using [FINAL]:
Thought:
Observation states that the largest mammal is known.

Action:
[FINAL] Blue whale
"""
    return prompt