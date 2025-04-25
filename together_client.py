# together_client.py
import aiohttp
from typing import List, Dict
import os
from dotenv import load_dotenv
import asyncio
import yaml
from pathlib import Path
import numpy as np
import gc

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

    async def get_embeddings(self, texts: List[str], batch_size: int = 64) -> List[List[float]]:
        """Get embeddings for a list of texts using the Together AI API.
        
        Args:
            texts: List of texts to get embeddings for
            batch_size: Number of texts to process in a single API call
            
        Returns:
            List of embeddings, one for each input text
        """
        try:
            all_embeddings = []
            
            # Process in batches to avoid memory issues
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i+batch_size]
                
                # Log batch progress
                print(f"Processing embedding batch {i//batch_size + 1}/{(len(texts) + batch_size - 1)//batch_size}")
                
                response = await self._make_request('post', 'embeddings', json={
                    "model": "togethercomputer/m2-bert-80M-32k-retrieval",
                    "input": batch_texts
                })
                
                # Get embeddings and convert to float16 for memory efficiency
                raw_embeddings = [np.array(data['embedding'], dtype=np.float16) for data in response['data']]
                all_embeddings.extend(raw_embeddings)
                
                # Clear batch from memory
                del batch_texts
                gc.collect()
                
            return all_embeddings
        except Exception as e:
            raise Exception(f"Failed to get embeddings: {str(e)}")

    async def get_completion(self, 
                           prompt: str, 
                           context: str = None,
                           history: str = None,
                           temperature: float = 0.7,
                           max_tokens: int = 4000,
                           concise: bool = None,
                           model: str = None) -> str:
        """Get a completion from the Together AI API.
        
        Args:
            prompt: The input prompt
            context: Additional context to provide
            history: Conversation history
            temperature: Temperature for generation (higher = more creative)
            max_tokens: Maximum tokens to generate
            concise: Override to force concise responses
            model: Specific model to use (overrides config)
        """
        try:
            # Load config to check for concise setting
            import yaml
            import os
            from pathlib import Path
            
            # Default to verbose responses unless specifically overridden
            use_concise = False
            
            # Check config.yml for concise mode setting
            config_path = Path('config.yml')
            config_model = None
            if config_path.exists():
                with open(config_path, 'r') as f:
                    config = yaml.safe_load(f)
                    # Check for concise mode in multiple possible locations
                    use_concise = (
                        config.get('response_mode') == 'concise' or
                        config.get('model', {}).get('concise_responses', False) or
                        config.get('model', {}).get('verbosity') == 'low'
                    )
                    
                    # Get model from config
                    config_model = config.get('model', {}).get('default', 'meta-llama/Llama-3.3-70B-Instruct-Turbo')
            
            # Parameter overrides config if provided
            if concise is not None:
                use_concise = concise
                
            # Get max token setting from config
            config_max_tokens = None
            if config_path.exists():
                with open(config_path, 'r') as f:
                    config = yaml.safe_load(f)
                    config_max_tokens = config.get('model', {}).get('max_tokens')
            
            # Use config max tokens if available and not explicitly overridden
            if config_max_tokens is not None and max_tokens == 4000:  # 4000 is the default
                max_tokens = config_max_tokens
            
            # Select system prompt based on verbosity setting
            if use_concise:
                system_prompt = (
                    "You are a concise AI code assistant. Your responses should be brief and to the point. "
                    "When analyzing code:"
                    "\n1. Keep explanations very brief, 1-2 sentences maximum"
                    "\n2. Only include the most critical information"
                    "\n3. Focus only on answering the direct question without elaboration"
                    "\n4. Avoid showing code snippets unless specifically requested"
                    "\n5. Never add background information or context"
                    "\n6. No formatting, markdown, or explanatory text"
                    "\n\nKeep your total response under 150 words. Be direct and avoid any explanation beyond what's needed."
                )
            else:
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
            
            # Better handling of conversation history - parse the formatted history into proper chat format
            if history and history.strip():
                # Try to parse User/Assistant format
                history_lines = history.split('\n\n')
                
                for entry in history_lines:
                    if entry.startswith('User:'):
                        user_content = entry.replace('User:', '', 1).strip()
                        messages.append({"role": "user", "content": user_content})
                    elif entry.startswith('Assistant:'):
                        assistant_content = entry.replace('Assistant:', '', 1).strip()
                        messages.append({"role": "assistant", "content": assistant_content})
            
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

            # Determine which model to use (priority: function param > config file > default)
            model_to_use = model if model else config_model if config_model else "meta-llama/Llama-3.3-70B-Instruct-Turbo"

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url}/chat/completions",
                    headers=self.headers,
                    json={
                        "model": model_to_use,
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