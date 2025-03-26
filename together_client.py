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
                           max_tokens: int = 4000) -> str:
        """Get completion from Together AI's LLM"""
        try:
            # Enhanced system prompt for more detailed responses
            system_prompt = (
                "You are a highly knowledgeable AI code assistant. Your responses should be comprehensive and detailed. "
                "When analyzing code:"
                "\n1. Always include relevant code snippets in your explanations"
                "\n2. Explain what each piece of code does and how it works"
                "\n3. Reference specific file paths and line numbers"
                "\n4. Highlight important patterns and considerations"
                "\n5. Provide context about dependencies and related components"
                "\n6. Use markdown formatting for clarity"
                "\n\nWhen answering questions about specific files:"
                "\n- Always show the ACTUAL content of the file in your response"
                "\n- DO NOT claim the file is not in the context if it's mentioned in the 'Relevant code context' section"
                "\n- Start with an overview of the file's purpose and structure"
                "\n- Show the most important parts of the file with code blocks"
                "\n- Explain each function, class, or code section's purpose"
                "\n\nWhen discussing code in general:"
                "\n- Show the actual implementation details"
                "\n- Explain the purpose and functionality"
                "\n- Point out any important patterns or practices"
                "\n- Include relevant error handling and edge cases"
                "\n\nMaintain a technical and precise tone while being thorough in your explanations."
                "\nFor any code review, maintain a balanced perspective highlighting both strengths and areas for improvement."
                "\nWhen receiving feedback, acknowledge it professionally and adjust your subsequent responses accordingly."
                "\n\nIMPORTANT: Your responses should be very detailed and comprehensive. Short answers are insufficient."
                "\nTake the time to provide thorough explanations with examples, code snippets, and detailed analysis."
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
                        console_msg = f"API error ({response.status}): {text}"
                        print(console_msg)  # Print directly for debugging
                        raise Exception(console_msg)
                    
                    data = await response.json()
                    print(f"Completion response structure: {type(data)}")
                    
                    # Handle response
                    if isinstance(data, dict) and "choices" in data:
                        response_content = data["choices"][0]["message"]["content"]
                        # Check if response is suspiciously short
                        if len(response_content.split()) < 100:
                            print(f"Warning: Generated response is unusually short ({len(response_content.split())} words)")
                        return response_content
                    else:
                        print(f"Unexpected response structure: {data}")
                        raise Exception("Unexpected API response structure")
        except Exception as e:
            error_msg = f"Failed to get completion: {str(e)}"
            print(error_msg)  # Print directly for debugging
            raise Exception(error_msg)