from data.hotpot_loader import load_hotpotqa_qa
from rank_bm25 import BM25Okapi

class Tools:
    def __init__(self):
        raw_data = load_hotpotqa_qa()
        # build index
        self.data = {}
        self.bm25_index = {}
        self.title_list = {}
        for qid, item in raw_data.items():
            pages = {}
            for title, sentences in item["context"]:
                pages[title] = sentences
            self.data[qid] = pages
            # build BM25 index for page titles
            titles = list(pages.keys())
            tokenized_titles = [title.lower().split() for title in titles]
            self.bm25_index[qid] = BM25Okapi(tokenized_titles)
            self.title_list[qid] = titles
        self.current_qid = None
        self.toolConfig = self._tools()

    def set_qid(self, qid):
        """Set current question id"""
        self.current_qid = qid

    def list_tools(self):
        return [func for func in dir(self) if not func.startswith("_")]

    def _tools(self):
        return [
            {
                "name": "search_sentences",
                "description": "Find sentences in the current context that contain the given sentences.",
                "args": ["sentences"]
            },
            {
                "name": "get_page",
                "description": "Retrieves the entire content of that page. Use this only if you know the exact page name.",
                "args": ["title"]
            }
        ]

    def get_page(self, title):
        title = title.strip('"')  # Remove quotes if present
        title = title.strip("'")  # Remove single quotes if present
        pages = self.data[self.current_qid]
        if title in pages:
            results = pages[title]
        else:
            return f"No relevant information found."
        text = "\n".join(results)
        return f"""Page: {title}
{text}
"""

    def search_sentences(self, entity, max_results=5): 
        pages = self.data[self.current_qid]
        all_sentences = []
        sentence_metadata = []  
        for title, sentences in pages.items():
            for sent in sentences:
                all_sentences.append(sent)
                sentence_metadata.append(title)
        if not all_sentences:
            return f"No relevant information found."
        tokenized_sentences = [sent.lower().split() for sent in all_sentences]
        bm25 = BM25Okapi(tokenized_sentences)
        query_tokens = entity.lower().split()
        scores = bm25.get_scores(query_tokens)
        ranked_results = []
        for i, (sent, title, score) in enumerate(zip(all_sentences, sentence_metadata, scores)):
            if score > 0:
                ranked_results.append({
                    "score": score,
                    "title": title,
                    "sentence": sent,
                    "index": i
                })
        ranked_results.sort(key=lambda x: x["score"], reverse=True)
        top_results = ranked_results[:max_results]
        if len(top_results) == 0:
            return f"No relevant information found."
        text_lines = []
        for i, result in enumerate(top_results, 1):
            text_lines.append(f"{i}. [Title page: {result['title']}] {result['sentence']}")
        text = "\n".join(text_lines)
        return f"""{text}
"""