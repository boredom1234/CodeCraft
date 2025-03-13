import asyncio
from analyzer import CodebaseAnalyzer
import os
from dotenv import load_dotenv

async def main():
    # Load environment variables
    load_dotenv()
    api_key = os.getenv("TOGETHER_API_KEY")
    
    if not api_key:
        print("Error: TOGETHER_API_KEY not found in environment variables")
        return
    
    print(f"API Key length: {len(api_key)}")
    
    # Create a small test directory with a few files
    os.makedirs("test_files", exist_ok=True)
    
    # Create test files
    with open("test_files/file1.py", "w") as f:
        f.write("""
# This is a test file about data processing
def process_data(data):
    \"\"\"Process the input data and return results\"\"\"
    results = []
    for item in data:
        results.append(item * 2)
    return results

def filter_data(data, condition):
    \"\"\"Filter data based on a condition\"\"\"
    return [item for item in data if condition(item)]
""")
    
    with open("test_files/file2.py", "w") as f:
        f.write("""
# This is a test file about user authentication
def authenticate_user(username, password):
    \"\"\"Authenticate a user with username and password\"\"\"
    if username == "admin" and password == "password":
        return True
    return False

def create_user(username, password, email):
    \"\"\"Create a new user\"\"\"
    # In a real system, this would save to a database
    print(f"Created user {username} with email {email}")
    return {"username": username, "email": email}
""")
    
    with open("test_files/file3.py", "w") as f:
        f.write("""
# This is a test file about data visualization
def create_bar_chart(data, labels):
    \"\"\"Create a bar chart from data and labels\"\"\"
    print("Creating bar chart...")
    for i, value in enumerate(data):
        print(f"{labels[i]}: {'#' * value}")

def create_line_graph(x_values, y_values):
    \"\"\"Create a line graph from x and y values\"\"\"
    print("Creating line graph...")
    # This would use a plotting library in a real application
    return {"x": x_values, "y": y_values}
""")
    
    # Initialize the analyzer with the test directory
    analyzer = CodebaseAnalyzer("test_files", api_key, "test_project")
    
    # Index the test files
    await analyzer.index()
    
    # Test queries with different topics to show similarity scores
    queries = [
        "How does the data processing work?",
        "How does user authentication work?",
        "How can I create visualizations?",
        "What functions are available for filtering data?"
    ]
    
    for query in queries:
        print("\n" + "="*50)
        print(f"QUERY: {query}")
        print("="*50)
        
        # Get response (this will show similarity scores with our changes)
        response = await analyzer.query(query)
        
        # Just print a separator after each query
        print("\n" + "-"*50)

if __name__ == "__main__":
    asyncio.run(main()) 