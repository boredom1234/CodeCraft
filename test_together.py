import asyncio
import os
from dotenv import load_dotenv

async def test_together():
    load_dotenv()
    api_key = os.getenv("TOGETHER_API_KEY")
    print(f"API Key length: {len(api_key)}")
    
    # Test direct HTTP request using aiohttp
    import aiohttp
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    async with aiohttp.ClientSession() as session:
        async with session.get(
            "https://api.together.xyz/v1/models",
            headers=headers
        ) as response:
            if response.status == 200:
                print("API key works!")
                data = await response.json()
                print(f"Response structure: {type(data)}")
                # Print the first few keys to understand the structure
                if isinstance(data, dict):
                    print(f"Keys: {list(data.keys())}")
                elif isinstance(data, list):
                    print(f"Available models: {len(data)}")
                    # Print first model if available
                    if data:
                        print(f"First model: {data[0]}")
            else:
                print(f"API error: {response.status}")
                print(await response.text())

if __name__ == "__main__":
    asyncio.run(test_together()) 