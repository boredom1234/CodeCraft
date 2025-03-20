#!/bin/bash

# If TOGETHER_API_KEY is provided as an environment variable, save it to .env
if [ ! -z "$TOGETHER_API_KEY" ]; then
    echo "TOGETHER_API_KEY=$TOGETHER_API_KEY" > .env
fi

# Execute the CLI tool with all arguments passed to the container
exec python cli.py "$@" 