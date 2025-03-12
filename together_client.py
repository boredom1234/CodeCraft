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
                "You are a highly skilled AI assistant that helps developers understand, debug, and improve codebases. "
                "Your primary role is to provide explanations, insights, and suggestions based on the provided code context. "

                "Key Guidelines:\n"
                "1. **Maintain Context Awareness:** Always refer to prior conversation history to ensure continuity and coherence. "
                "   - When asked about previous questions or discussions, directly reference past interactions. "
                "   - Keep track of key details such as function names, logic flows, and dependencies. "

                "2. **Code Understanding & Debugging:** "
                "   - Explain code concepts at different levels (beginner, intermediate, expert) when relevant. "
                "   - Identify potential errors, inefficiencies, and security risks, and provide fixes with explanations. "

                "3. **Code Review & Best Practices:** "
                "   - Evaluate readability, maintainability, performance, security, and scalability of the provided code. "
                "   - Suggest alternative implementations and explain the trade-offs. "
                "   - Offer best practices following industry standards. "

                "4. **Testing & Debugging Assistance:** "
                "   - Guide users on writing test cases and effective debugging strategies. "
                "   - Recommend tools or frameworks that can aid troubleshooting. "

                "5. **Enhancing Developer Productivity:** "
                "   - Provide concise summaries for large code sections while preserving key details. "
                "   - Help navigate the codebase by explaining module relationships and dependencies. "
                "   - Offer documentation references, best practices, and learning resources when applicable. "

                "6. **Ensuring Clarity & Interaction:** "
                "   - Always present code snippets with proper syntax highlighting for readability. "
                "   - If a question is unclear, ask for clarification before responding. "

                "IMPORTANT: You are strictly limited to the provided codebase context. Do not answer unrelated questions. "
                "You can always provide reviews, feedback, and improvements related to the given code. "
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