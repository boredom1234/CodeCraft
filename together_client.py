# together_client.py
import aiohttp
from typing import List, Dict
import os
from dotenv import load_dotenv
import asyncio

class TogetherAIClient:
    def __init__(self, api_key: str = None):
        load_dotenv()  # Load environment variables from .env file
        self.api_key = api_key or os.getenv("TOGETHER_API_KEY")
        if not self.api_key:
            raise ValueError("Together AI API key is required. Set it in .env file or pass directly.")
        
        print(f"API Key length: {len(self.api_key)}")  # Debug line
        
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        self.base_url = "https://api.together.xyz/v1"
        self.max_retries = 3
        self.retry_delay = 1  # seconds

    async def _make_request(self, method: str, endpoint: str, **kwargs) -> dict:
        """Make API request with retry logic"""
        url = f"{self.base_url}/{endpoint}"
        
        for attempt in range(self.max_retries):
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.request(
                        method, 
                        url,
                        headers=self.headers,
                        **kwargs
                    ) as response:
                        if response.status == 429:  # Rate limit
                            retry_after = int(response.headers.get('Retry-After', self.retry_delay))
                            await asyncio.sleep(retry_after)
                            continue
                            
                        if response.status != 200:
                            text = await response.text()
                            raise Exception(f"API error ({response.status}): {text}")
                            
                        return await response.json()
                        
            except aiohttp.ClientError as e:
                if attempt == self.max_retries - 1:
                    raise Exception(f"Network error after {self.max_retries} attempts: {str(e)}")
                await asyncio.sleep(self.retry_delay * (attempt + 1))
                
        raise Exception("Max retries exceeded")

    async def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings using Together AI's embedding model"""
        if not texts:
            return []
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url}/embeddings",
                    headers=self.headers,
                    json={
                        "model": "togethercomputer/m2-bert-80M-32k-retrieval",
                        "input": texts
                    }
                ) as response:
                    if response.status != 200:
                        text = await response.text()
                        raise Exception(f"API error ({response.status}): {text}")
                    
                    data = await response.json()
                    print(f"Embedding response structure: {type(data)}")
                    
                    # Adjusted to handle different response structures
                    if isinstance(data, dict) and "data" in data:
                        return [item["embedding"] for item in data["data"]]
                    elif isinstance(data, list):
                        return [item["embedding"] for item in data]
                    else:
                        print(f"Unexpected response structure: {data}")
                        raise Exception("Unexpected API response structure")
        except Exception as e:
            raise Exception(f"Failed to generate embeddings: {str(e)}")

    async def get_completion(self, 
                           prompt: str, 
                           context: str = None,
                           history: str = None,
                           temperature: float = 0.7,
                           max_tokens: str = None) -> str:
        """Get completion from Together AI's LLM"""
        try:
            # Better system prompt with explicit context maintenance instructions
            system_prompt = (
                "You are a highly skilled AI consultant that helps developers understand, debug, and improve codebases. "
                "Your primary role is to provide precise, well-structured explanations, insights, and actionable recommendations based on the provided code context. Always include relevant line numbers when discussing specific code segments. "

                "Professional Communication Guidelines:\n"
                "1. **Structure & Clarity:** Organize your responses with clear sections and headers. Use concise language and maintain a professional, authoritative tone throughout your responses.\n"
                "2. **Technical Precision:** Use technically accurate terminology and maintain consistent naming conventions. Avoid colloquial language and imprecise descriptions.\n"
                "3. **Evidence-Based Analysis:** Reference specific parts of the code to support your observations and recommendations. Always cite line numbers and file names.\n"

                "Key Areas of Expertise:\n"
                "1. **Context Awareness & Continuity:** "
                "   - Maintain a professional thread across conversation exchanges, referencing prior discussions with precision. "
                "   - Track essential elements such as function names, architectural patterns, and dependencies systematically. "

                "2. **Code Analysis & Performance Optimization:** "
                "   - Provide multi-level analysis appropriate to the technical context (e.g., architectural, implementation, algorithm complexity). "
                "   - Identify potential defects, performance bottlenecks, and security vulnerabilities with actionable remediation steps. "

                "3. **Professional Code Review & Industry Standards:** "
                "   - Assess code quality dimensions: readability, maintainability, performance, security, and scalability. "
                "   - Present alternative implementation approaches with clear technical trade-off analysis. "
                "   - Reference relevant industry best practices, design patterns, and standards where applicable. "

                "4. **Testing & Quality Assurance Guidance:** "
                "   - Provide detailed guidance on test coverage strategies and effective debugging methodologies. "
                "   - Recommend appropriate testing frameworks and quality assurance tools with justification. "

                "5. **Developer Productivity Enhancement:** "
                "   - Deliver concise, high-value summaries of complex code sections while preserving critical details. "
                "   - Clarify module relationships, dependencies, and system architecture to aid in navigation. "
                "   - Provide references to relevant documentation, specifications, and learning resources when appropriate. "

                "6. **Professional Presentation:** "
                "   - Present code snippets with proper syntax highlighting and consistent formatting. "
                "   - Use appropriate technical diagrams or structured explanations for complex concepts. "
                "   - When clarification is needed, formulate precise technical questions. "

                "IMPORTANT: Limit your analysis strictly to the provided codebase context. Do not address unrelated inquiries. "
                "For any code review, maintain a balanced perspective highlighting both strengths and areas for improvement."

                "When receiving feedback, acknowledge it professionally and adjust your subsequent responses accordingly to better serve the technical requirements."
            )

            
            messages = [{"role": "system", "content": system_prompt}]
            
            # Add conversation history as chat messages for better context
            if history and history.strip():
                # Convert history to actual conversation format
                conversation_so_far = []
                history_lines = history.strip().split('\n\n')
                
                for i in range(0, len(history_lines), 2):
                    if i+1 < len(history_lines):
                        q_part = history_lines[i]
                        a_part = history_lines[i+1]
                        
                        if q_part.startswith('Question'):
                            q_text = q_part.split('\n', 1)[1] if '\n' in q_part else q_part
                            messages.append({"role": "user", "content": q_text})
                        
                        if a_part.startswith('Answer'):
                            a_text = a_part.split('\n', 1)[1] if '\n' in a_part else a_part
                            messages.append({"role": "assistant", "content": a_text})
                        
                        # Add to readable conversation summary
                        conversation_so_far.append(f"User: {q_text}")
                        conversation_so_far.append(f"Assistant: {a_text}")
                
                # Add a system message with conversation summary to help the LLM understand
                if conversation_so_far:
                    summary = "\n\n".join(conversation_so_far)
                    messages.append({
                        "role": "system", 
                        "content": f"Here's a summary of the conversation so far, which you MUST consider when answering:\n\n{summary}"
                    })
            
            # Add the current context and question
            if context:
                messages.append({
                    "role": "user",
                    "content": f"Context from the codebase:\n{context}\n\nQuestion: {prompt}"
                })
            else:
                messages.append({
                    "role": "user",
                    "content": prompt
                })

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url}/chat/completions",
                    headers=self.headers,
                    json={
                        "model": "meta-llama/Llama-3.3-70B-Instruct-Turbo",
                        "messages": messages,
                        "max_tokens": max_tokens,
                        "temperature": temperature,
                        "top_p": 0.7,
                        "top_k": 50
                    }
                ) as response:
                    if response.status != 200:
                        text = await response.text()
                        raise Exception(f"API error ({response.status}): {text}")
                    
                    data = await response.json()
                    print(f"Completion response structure: {type(data)}")
                    
                    # Handle response
                    if isinstance(data, dict) and "choices" in data:
                        return data["choices"][0]["message"]["content"]
                    else:
                        print(f"Unexpected response structure: {data}")
                        raise Exception("Unexpected API response structure")
        except Exception as e:
            raise Exception(f"Failed to get completion: {str(e)}")