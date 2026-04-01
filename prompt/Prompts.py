def history_to_text(history):
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
            history_text += f"[Action]\n"
            # Lưu ý: Key phải khớp với lúc ông lưu (Question/question)
            history_text += f"Question: {act.get('Question')}\n"
            history_text += f"Reason: {act.get('Reason')}\n"
            # --- LOGIC HYBRID TẠI ĐÂY ---
            if step == last_step:
                # Bước vừa xong: Đưa bản thô (Raw) để tìm manh mối cho bước tiếp theo
                obs = act.get('Raw Observation', 'No data found.')
                history_text += f"Observation (Detailed): {obs}\n"
            else:
                # Các bước cũ: Chỉ đưa bản tóm tắt (Summary) để tránh nhiễu
                summary = act.get('Summary', 'Information processed.')
                history_text += f"Observation (Summary): {summary}\n"
        history_text += "-" * 20 + "\n"
    return history_text

def build_amo_prompt(question, history=None, feedback=None):
    history_text = history_to_text(history) if history else "No history yet."
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

Example 1 - Using [QUERY]:

Thought:
I know the capital cities of some countries, but I do not know the capital of France. I need this information to answer the question.

Action:
[QUERY]
Question: "What is the capital of France?"
Reason: "Retrieve the capital city to answer the question about France."

Example 2 - Using [FINAL]:

Thought:
Observation states that the capital of France is known.

Action:
[FINAL] Paris

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
    Xây dựng prompt để LLM tóm tắt kết quả tool (observation) theo query.

    Args:
        query: str, câu hỏi hoặc query mà agent đã tạo
        observation: str, kết quả raw từ tool

    Returns:
        prompt: str, prompt hoàn chỉnh để gọi LLM
    """
    prompt = f"""
You are a summarization assistant. You are NOT allowed to answer the query. 
Your ONLY task is to summarize the raw results provided.

Query: "{query}"

Raw results: "{observation}"

Instructions:
1. DO NOT answer the query.
2. Summarize the information in 1-2 sentences only.
3. Focus ONLY on the information relevant to the query.
4. Make it concise and easy to understand.
5. Avoid repeating information or adding your own opinions.
6. Output format MUST be exactly:
   [SUMMARY]: <your summary here>

[SUMMARY]:
"""
    return prompt

def build_amo_final_prompt(question, history):
    history_text = history_to_text(history) if history else "No history yet."
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