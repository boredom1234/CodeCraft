import asyncio
import os
from dotenv import load_dotenv
import pytest
from unittest.mock import Mock, patch

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

@pytest.fixture
async def mock_client():
    with patch('together_client.TogetherAIClient') as mock:
        client = Mock()
        mock.return_value = client
        yield client

@pytest.mark.asyncio
async def test_embeddings(mock_client):
    texts = ["test text"]
    mock_client.get_embeddings.return_value = [[0.1, 0.2, 0.3]]
    
    result = await mock_client.get_embeddings(texts)
    assert len(result) == 1
    assert len(result[0]) == 3

@pytest.mark.asyncio
async def test_completion(mock_client):
    mock_client.get_completion.return_value = "Test response"
    
    result = await mock_client.get_completion("test prompt")
    assert isinstance(result, str)
    assert len(result) > 0

if __name__ == "__main__":
    asyncio.run(test_together()) 