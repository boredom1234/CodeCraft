#!/usr/bin/env python
"""
Test script for demonstrating the improved code-aware retrieval in CodeCraft.

This script shows how the new retrieval system can find exact code lines and provide
better context-aware search results for code-specific queries.
"""

import argparse
import sys
import os
import yaml
from pathlib import Path
import subprocess
import time

def test_improved_retrieval():
    """Test and demonstrate the improved code-aware retrieval functionality."""
    # Check if CodeCraft is properly installed
    try:
        result = subprocess.run(
            ["python", "cli.py", "--help"],
            capture_output=True,
            text=True,
            check=True
        )
        print("CodeCraft installation verified")
    except subprocess.CalledProcessError as e:
        print(f"ERROR: CodeCraft CLI not working properly: {e}")
        print(e.stdout)
        print(e.stderr)
        sys.exit(1)
    
    # Define test questions that would benefit from code-aware retrieval
    test_questions = [
        # Exact line location queries
        "Where is this line located: const { copied, trigger } = useCopiedDelay();",
        "Find the definition of function useCopiedDelay",
        "In which file is the function copyText defined?",
        
        # Implementation queries
        "How is the useCopiedDelay hook implemented?",
        "Show me the implementation of copyText function",
        
        # Usage queries
        "How is the useCopiedDelay hook used in the codebase?",
        "Show me examples of where copyText is called"
    ]
    
    # Run tests with each question
    for i, question in enumerate(test_questions):
        print("\n" + "="*80)
        print(f"TEST QUERY {i+1}: {question}")
        print("="*80 + "\n")
        
        # Run the query and capture results
        print(f"Processing query: {question}")
        print("Running CLI command...")
        
        try:
            # Use the concise flag to keep responses focused
            result = subprocess.run(
                ["python", "cli.py", "ask", "--concise", question],
                capture_output=True,
                text=True,
                check=True,
                timeout=60  # Add timeout to prevent hanging
            )
            print("RESPONSE:")
            print("-"*80)
            print(result.stdout)
            print("-"*80)
        except subprocess.CalledProcessError as e:
            print(f"ERROR running query: {e}")
            print(e.stdout)
            print(e.stderr)
        except subprocess.TimeoutExpired:
            print("ERROR: Command timed out after 60 seconds")
        
        # Add a pause between queries
        time.sleep(2)
    
    print("\nAll tests completed.")

def main():
    parser = argparse.ArgumentParser(description="Test CodeCraft's improved code-aware retrieval")
    parser.add_argument('--run', action='store_true', help='Run the retrieval tests')
    
    args = parser.parse_args()
    
    if args.run:
        test_improved_retrieval()
    else:
        print("To run the retrieval tests, use: python test_retrieval.py --run")
        print("This will demonstrate the improved code-aware retrieval capabilities.")
        print("Queries will test exact line location, function definitions, and usage patterns.")

if __name__ == "__main__":
    main() 