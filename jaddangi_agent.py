"""
JADDANGI AGENTIC INTERCEPTOR v1.0
Designed by: Kurumalla Venkataramana | Jaddangi IT & AI Consultancy
Description: A native ReAct (Reason + Act) interceptor loop. 
Capabilities: Live Web Search, Deterministic Math, Private Vector RAG.
"""

import torch
import sympy
from ddgs import DDGS
from sentence_transformers import SentenceTransformer, util

# ====================================================================
# 1. THE TOOLBOX (Sensors & Actuators)
# ====================================================================
class JaddangiTools:
    def __init__(self):
        print("🧰 Initializing Jaddangi Tool Belt...")
        self.search_engine = DDGS()
        # Lightweight embedding model for offline/local RAG
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2') 
        self.knowledge_base_text = []
        self.knowledge_base_embeddings = None

    def load_rag_documents(self, documents):
        """Encodes private text into 3D mathematical vectors for instant retrieval."""
        print(f"📚 Embedding {len(documents)} RAG documents into Vector Vault...")
        self.knowledge_base_text = documents
        self.knowledge_base_embeddings = self.embedder.encode(documents, convert_to_tensor=True)

    def execute_tool(self, tool_name, query):
        """Routing hub for tool execution."""
        if tool_name == "SEARCH":
            return self._tool_search(query)
        elif tool_name == "MATH":
            return self._tool_math(query)
        elif tool_name == "RAG":
            return self._tool_rag(query)
        else:
            return f"Error: Unknown Tool '{tool_name}'."

    def _tool_search(self, query):
        print(f"   🌐 [ACT] Searching the live internet for: {query}")
        try:
            results = self.search_engine.text(query, max_results=3)
            return "\n".join([f"- {res['title']}: {res['body']}" for res in results])
        except Exception as e:
            return f"Search failed: {e}"

    def _tool_math(self, equation):
        print(f"   🧮 [ACT] Calculating exact deterministic math: {equation}")
        try:
            # SymPy completely eliminates AI math hallucination
            result = sympy.sympify(equation).evalf()
            return str(result)
        except Exception as e:
            return f"Math Error: {e}"

    def _tool_rag(self, query):
        print(f"   📂 [ACT] Scanning Private Vector Vault for: {query}")
        if self.knowledge_base_embeddings is None:
            return "RAG Vault is empty."
        
        query_emb = self.embedder.encode(query, convert_to_tensor=True)
        scores = util.cos_sim(query_emb, self.knowledge_base_embeddings)[0]
        best_match_idx = torch.argmax(scores).item()
        
        # 0.3 is the baseline confidence threshold for cosine similarity
        if scores[best_match_idx] > 0.3: 
            return self.knowledge_base_text[best_match_idx]
        return "No highly relevant information found in private documents."

# ====================================================================
# 2. THE AGENTIC LOOP (ReAct Interceptor)
# ====================================================================
class JaddangiAgent:
    def __init__(self, model, tokenizer, tools):
        self.model = model
        self.tokenizer = tokenizer
        self.tools = tools
        self.device = next(model.parameters()).device
        
        # 🔬 Few-Shot Prompting: Hardcoding the exact bracket syntax we want it to use
        self.system_prompt = """You are Jaddangi-Alfa, a precise AI Agent.
You CANNOT guess. You MUST use tools to answer questions.
Tools available: [SEARCH: query], [MATH: equation], [RAG: topic]

Example 1:
User: Calculate 50 * 22
Jaddangi: [MATH: 50 * 22]

Example 2:
User: What is the secret code?
Jaddangi: [RAG: secret code]

Example 3:
User: Who won the 2022 World Cup?
Jaddangi: [SEARCH: 2022 World Cup winner]

Now answer the following User query using exactly one tool:

User: """

    def run(self, user_prompt):
        print("\n" + "="*60)
        print(f"👤 USER: {user_prompt}")
        print("="*60)
        
        current_context = self.system_prompt + user_prompt + "\nJaddangi: "
        
        # Allow the agent to think and act up to 3 times before timing out
        for step in range(3):
            input_ids = self.tokenizer.encode(current_context, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                # Generate a short burst to see if it triggers a tool
                output_ids = self.model.generate(input_ids, max_new_tokens=25, temperature=0.0)
            
            response = self.tokenizer.decode(output_ids[0][input_ids.shape[1]:], skip_special_tokens=True)
            
            # 🛑 THE INTERCEPTOR: Halt generation if brackets are detected
            if "[" in response and "]" in response:
                start_idx = response.find("[")
                end_idx = response.find("]")
                command = response[start_idx+1 : end_idx]
                
                try:
                    tool_name, tool_query = command.split(":", 1)
                    tool_name = tool_name.strip()
                    tool_query = tool_query.strip()
                    
                    print(f"🧠 [REASON] Jaddangi decided to use {tool_name}")
                    
                    # Execute the tool in Python
                    tool_result = self.tools.execute_tool(tool_name, tool_query)
                    
                    # Inject the real-world answer back into the AI's context
                    current_context += response[:end_idx+1] + f"\n[TOOL RESULT: {tool_result}]\nJaddangi Final Answer:"
                except ValueError:
                    print(f"⚠️ Syntax Error in Tool Trigger: {command}")
                    break
            
            else:
                # No tool triggered, the AI is giving its final conversational response
                print(f"🤖 JADDANGI: {response.strip()}")
                return

        print("🤖 JADDANGI: Task timeout or loop limit reached.")

# ====================================================================
# 3. STANDALONE TEST
# ====================================================================
if __name__ == "__main__":
    print("Testing Agent Framework (Requires Engine and Tokenizer)...")
    # Note: To run this standalone, import JaddangiAlfaEngine from jaddangi_engine
    # load the tokenizer, and instantiate the Agent.
