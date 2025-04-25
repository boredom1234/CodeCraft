#!/usr/bin/env python
"""
Test script for demonstrating concise mode in CodeCraft.

This script shows the difference between verbose and concise responses
from the CodeCraft assistant.
"""

import argparse
import sys
import os
import yaml
from pathlib import Path
import subprocess
import time

def test_concise_mode():
    """Test and demonstrate concise mode functionality."""
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
    
    # Define test questions
    test_questions = [
        "What does the CodebaseAnalyzer class do?",
        "How are code chunks processed in the system?",
        "What is the purpose of the get_completion method?"
    ]
    
    # First run in verbose mode
    print("\n" + "="*80)
    print("TESTING VERBOSE MODE (DEFAULT)")
    print("="*80 + "\n")
    
    # Temporarily make sure config is in verbose mode
    config_path = Path('config.yml')
    if config_path.exists():
        # Backup existing config
        with open(config_path, 'r') as f:
            original_config = yaml.safe_load(f)
        
        # Create verbose config
        verbose_config = original_config.copy()
        if 'response_mode' in verbose_config:
            verbose_config.pop('response_mode')
        if 'model' in verbose_config:
            if 'concise_responses' in verbose_config['model']:
                verbose_config['model']['concise_responses'] = False
            if 'verbosity' in verbose_config['model']:
                verbose_config['model']['verbosity'] = 'high'
        
        # Write verbose config
        with open(config_path, 'w') as f:
            yaml.dump(verbose_config, f)
    
    # Run test with first question in verbose mode
    question = test_questions[0]
    print(f"Question: {question}")
    print("\nGenerating verbose response...\n")
    
    try:
        result = subprocess.run(
            ["python", "cli.py", "ask", question],
            capture_output=True,
            text=True,
            check=True
        )
        print("VERBOSE RESPONSE:")
        print("-"*80)
        print(result.stdout)
        print("-"*80)
    except subprocess.CalledProcessError as e:
        print(f"ERROR running verbose mode: {e}")
        print(e.stdout)
        print(e.stderr)
    
    # Now test concise mode
    print("\n" + "="*80)
    print("TESTING CONCISE MODE")
    print("="*80 + "\n")
    
    # Update config for concise mode
    if config_path.exists():
        concise_config = original_config.copy()
        concise_config['response_mode'] = 'concise'
        if 'model' not in concise_config:
            concise_config['model'] = {}
        concise_config['model']['concise_responses'] = True
        concise_config['model']['verbosity'] = 'low'
        concise_config['model']['max_tokens'] = 150
        
        # Write concise config
        with open(config_path, 'w') as f:
            yaml.dump(concise_config, f)
    
    # Run test with same question in concise mode
    print(f"Question: {question}")
    print("\nGenerating concise response...\n")
    
    try:
        result = subprocess.run(
            ["python", "cli.py", "ask", "--concise", question],
            capture_output=True,
            text=True,
            check=True
        )
        print("CONCISE RESPONSE:")
        print("-"*80)
        print(result.stdout)
        print("-"*80)
    except subprocess.CalledProcessError as e:
        print(f"ERROR running concise mode: {e}")
        print(e.stdout)
        print(e.stderr)
    
    # Restore original config
    if config_path.exists():
        with open(config_path, 'w') as f:
            yaml.dump(original_config, f)
    
    print("\nTest completed. Original configuration restored.")

def main():
    parser = argparse.ArgumentParser(description="Test CodeCraft's concise mode functionality")
    parser.add_argument('--run', action='store_true', help='Run the concise mode test')
    
    args = parser.parse_args()
    
    if args.run:
        test_concise_mode()
    else:
        print("To run the concise mode test, use: python test_concise.py --run")
        print("This will demonstrate the difference between verbose and concise responses.")

if __name__ == "__main__":
    main() 