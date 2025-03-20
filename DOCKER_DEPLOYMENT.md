# CodeAI Docker Deployment Guide

This guide explains how to share, deploy, and use the CodeAI Docker container. The container provides a portable way to run the CodeAI tool without worrying about dependencies or setup.

## Table of Contents
- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
- [Sharing Options](#sharing-options)
- [Deployment Options](#deployment-options)
- [Environment Variables](#environment-variables)
- [Volume Management](#volume-management)
- [Troubleshooting](#troubleshooting)

## Prerequisites

- Docker installed on the target machine
- Together AI API key ([Get one here](https://www.together.ai/))
- Git (only for building from source)

## Quick Start

### Option 1: Pull from Docker Hub (Recommended)

```bash
# Pull the image
docker pull boredom1234/codeai:latest

# Run with your API key
docker run -it --rm \
  -e TOGETHER_API_KEY="your_api_key_here" \
  -v $(pwd):/workspace \
  -v codeai_data:/root/.codeai \
  boredom1234/codeai:latest --help
```

### Option 2: Build from Source

```bash
# Clone the repository
git clone https://github.com/boredom1234/CodeWhisperer.git
cd CodeWhisperer

# Build the image
docker build -t codeai .

# Run with your API key
docker run -it --rm \
  -e TOGETHER_API_KEY="your_api_key_here" \
  -v $(pwd):/workspace \
  -v codeai_data:/root/.codeai \
  codeai --help
```

## Sharing Options

### 1. Share via Docker Hub

```bash
# Tag your image
docker tag codeai boredom1234/codeai:latest

# Push to Docker Hub
docker push boredom1234/codeai:latest
```

Others can then pull and use your image:
```bash
docker pull boredom1234/codeai:latest
```

### 2. Share via Docker Save/Load

```bash
# Save the image to a file
docker save codeai > codeai.tar

# Share the file (e.g., via file transfer)

# On recipient's machine
docker load < codeai.tar
```

### 3. Share via GitHub Package Registry

```bash
# Tag your image for GitHub
docker tag codeai ghcr.io/boredom1234/codeai:latest

# Push to GitHub
docker push ghcr.io/boredom1234/codeai:latest
```

## Deployment Options

### 1. Local Development

Use Docker Compose for the easiest setup:

```yaml
# docker-compose.yml
services:
  codeai:
    image: boredom1234/codeai:latest
    environment:
      - TOGETHER_API_KEY=${TOGETHER_API_KEY}
    volumes:
      - .:/workspace
      - codeai_data:/root/.codeai
    working_dir: /workspace

volumes:
  codeai_data:
```

Run with:
```bash
docker-compose run --rm codeai [command]
```

### 2. Server Deployment

For server deployment, you can use systemd service:

```ini
# /etc/systemd/system/codeai.service
[Unit]
Description=CodeAI Service
After=docker.service
Requires=docker.service

[Service]
Environment=TOGETHER_API_KEY=your_api_key_here
ExecStart=/usr/bin/docker run --rm \
  -e TOGETHER_API_KEY=${TOGETHER_API_KEY} \
  -v /path/to/workspace:/workspace \
  -v codeai_data:/root/.codeai \
  boredom1234/codeai:latest
Restart=always

[Install]
WantedBy=multi-user.target
```

Enable and start the service:
```bash
sudo systemctl enable codeai
sudo systemctl start codeai
```

### 3. Cloud Deployment

#### AWS ECS Example

```json
{
  "family": "codeai",
  "containerDefinitions": [
    {
      "name": "codeai",
      "image": "boredom1234/codeai:latest",
      "environment": [
        {
          "name": "TOGETHER_API_KEY",
          "value": "your_api_key_here"
        }
      ],
      "mountPoints": [
        {
          "sourceVolume": "workspace",
          "containerPath": "/workspace",
          "readOnly": false
        },
        {
          "sourceVolume": "codeai_data",
          "containerPath": "/root/.codeai",
          "readOnly": false
        }
      ]
    }
  ],
  "volumes": [
    {
      "name": "workspace",
      "host": {
        "sourcePath": "/path/to/workspace"
      }
    },
    {
      "name": "codeai_data",
      "dockerVolumeConfiguration": {
        "scope": "shared",
        "autoprovision": true,
        "driver": "local"
      }
    }
  ]
}
```

## Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `TOGETHER_API_KEY` | Your Together AI API key | Yes |
| `GIT_PYTHON_REFRESH` | Git Python refresh mode (default: quiet) | No |

## Volume Management

The container uses two main volumes:

1. Workspace Volume (`/workspace`):
   - Maps to your local code directory
   - Used for analyzing codebases
   - Should be mounted as read-write

2. Data Volume (`/root/.codeai`):
   - Stores persistent data like project configurations
   - Maintains conversation history
   - Should be preserved between container runs

### Managing Volumes

```bash
# List volumes
docker volume ls

# Create a named volume
docker volume create codeai_data

# Backup a volume
docker run --rm -v codeai_data:/source -v $(pwd):/backup alpine tar czf /backup/codeai_data.tar.gz /source

# Restore a volume
docker run --rm -v codeai_data:/source -v $(pwd):/backup alpine tar xzf /backup/codeai_data.tar.gz -C /source
```

## Troubleshooting

### Common Issues

1. **Permission Issues**
```bash
# Run container with your user ID
docker run -u $(id -u):$(id -g) ...
```

2. **Volume Mount Issues**
```bash
# Check volume mounts
docker inspect container_name | grep Mounts -A 20
```

3. **API Key Issues**
```bash
# Verify environment variable
docker exec container_name env | grep TOGETHER_API_KEY
```

4. **Git Issues**
```bash
# Check git installation
docker exec container_name which git
```

### Getting Help

1. Check container logs:
```bash
docker logs container_name
```

2. Access container shell:
```bash
docker exec -it container_name bash
```

3. View resource usage:
```bash
docker stats container_name
```

## Best Practices

1. Always use the `--rm` flag for one-off commands
2. Use named volumes for persistent data
3. Keep your API key secure using environment variables
4. Use Docker Compose for development
5. Tag your images with meaningful versions
6. Document any custom modifications

## Security Considerations

1. Never build the image with API keys included
2. Use secrets management in production
3. Regularly update the base image
4. Follow the principle of least privilege
5. Scan images for vulnerabilities:
```bash
docker scan boredom1234/codeai:latest
``` 