# Docker Usage Guide

This guide explains how to use the CodeAI tool with Docker.

## Prerequisites

- Docker installed on your system
- Docker Compose (optional, but recommended)
- Together AI API key

## Quick Start

### Using Docker Compose (Recommended)

1. Set your Together AI API key as an environment variable:
```bash
export TOGETHER_API_KEY=your_api_key_here
```

2. Build and run the container:
```bash
docker-compose build
docker-compose run --rm codeai [command]
```

Example commands:
```bash
# Show help
docker-compose run --rm codeai --help

# Initialize a project
docker-compose run --rm codeai create-project my-project
docker-compose run --rm codeai init /workspace/my-code --project my-project

# Ask questions about the code
docker-compose run --rm codeai ask --interactive --project my-project
```

### Using Docker Directly

1. Build the image:
```bash
docker build -t codeai .
```

2. Run the container:
```bash
docker run -it --rm \
  -e TOGETHER_API_KEY=your_api_key_here \
  -v $(pwd):/workspace \
  -v codeai_data:/root/.codeai \
  -w /workspace \
  codeai [command]
```

Example commands:
```bash
# Show help
docker run -it --rm codeai --help

# Initialize a project
docker run -it --rm -e TOGETHER_API_KEY=your_key -v $(pwd):/workspace -v codeai_data:/root/.codeai codeai create-project my-project
docker run -it --rm -e TOGETHER_API_KEY=your_key -v $(pwd):/workspace -v codeai_data:/root/.codeai codeai init /workspace/my-code --project my-project

# Ask questions about the code
docker run -it --rm -e TOGETHER_API_KEY=your_key -v $(pwd):/workspace -v codeai_data:/root/.codeai codeai ask --interactive --project my-project
```

## Understanding the Docker Setup

### Volumes

The Docker setup uses two volumes:
1. `.:/workspace` - Mounts your current directory to /workspace in the container
2. `codeai_data:/root/.codeai` - Persistent volume for storing project data

### Environment Variables

- `TOGETHER_API_KEY` - Your Together AI API key

### Working Directory

The container's working directory is set to `/workspace`, which maps to your current directory. This means you can reference local files and directories relative to your current path.

## Best Practices

1. Always use the `--rm` flag to remove the container after it exits
2. Use Docker Compose for simpler command execution
3. Keep your API key secure by using environment variables
4. Use the persistent volume to maintain project data between runs

## Troubleshooting

### Common Issues

1. **Permission Issues**:
   - Ensure the mounted directories have proper permissions
   - Try running Docker with your user ID: `docker run -u $(id -u):$(id -g) ...`

2. **Missing API Key**:
   - Verify that the TOGETHER_API_KEY environment variable is set
   - Check if the .env file is properly created in the container

3. **Volume Mounting**:
   - Ensure paths are absolute when using Docker run
   - Check if the volumes are properly created: `docker volume ls`

### Getting Help

If you encounter issues:
1. Check the container logs: `docker logs [container_id]`
2. Run the container with bash: `docker run -it --rm codeai bash`
3. Verify volume mounts: `docker inspect [container_id]` 