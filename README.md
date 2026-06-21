# Claude MLX Bridge

A protocol bridge that lets **Claude Code (`cc switch`)** use **MLX local models** on Apple Silicon.

## Problem

- **Claude Code** internally uses the **Anthropic API** (`POST /v1/messages`) to talk to all local models
- **MLX** (`mlx-lm`) only provides an **OpenAI-compatible API** (`POST /v1/chat/completions`)
- These two protocols are incompatible — Claude Code cannot talk to MLX directly

## Solution

This bridge translates **Anthropic API** requests from Claude Code into **OpenAI API** requests that MLX understands:

```
Claude Code (cc switch)
    ↓ POST /v1/messages        Anthropic API
claude-mlx-bridge  :11435
    ↓ translate to
    ↓ POST /v1/chat/completions  OpenAI API
mlx_lm.server      :18080
    ↓ MLX inference
Your local model (e.g. Qwen3-8B-4bit)
```

**Port 11435** is used to avoid conflict with Ollama (default 11434).

## Features

- ✅ **Anthropic → OpenAI translation** (`/v1/messages` → `/v1/chat/completions`)
- ✅ OpenAI API proxy (`/v1/models`, `/v1/chat/completions`)
- ✅ Ollama-compatible model discovery (`/api/tags`, `/api/version`) — for `cc switch` listing
- ✅ Streaming and non-streaming responses
- ✅ Tool call support
- ✅ No API key required

## Prerequisites

- Go 1.21+
- Python 3.9+ with `mlx-lm` installed
- A local MLX model downloaded

## Installation

```bash
git clone git@github.com:hellowBai/claude-mlx-bridge.git
cd claude-mlx-bridge
go build -o claude-mlx-bridge
```

## Usage

### 1. Start MLX Server

```bash
mlx_lm.server --model mlx-community/Qwen3-8B-4bit --port 18080
```

### 2. Start Bridge

```bash
./claude-mlx-bridge
```

Listens on `:11435` by default. Set `PORT` env var to change.

### 3. Configure Claude Code

Tell `cc switch` to connect to port 11435 instead of Ollama's default 11434:

```bash
export OLLAMA_HOST=localhost:11435
cc switch
# Select your MLX model
```

Or add `OLLAMA_HOST=localhost:11435` to your shell profile.

## One-Command Startup

```bash
./start-mlx-bridge.sh [model-name]
```

## Architecture

### Protocol Translation

| Endpoint | Incoming Protocol | Action |
|----------|-------------------|--------|
| `POST /v1/messages` | **Anthropic** | **Translate to OpenAI, forward to MLX** |
| `POST /v1/chat/completions` | OpenAI | Proxy to MLX |
| `GET /v1/models` | OpenAI | Proxy to MLX |
| `GET /api/tags` | Ollama | Model list for `cc switch` discovery |

### Why Ollama API?

`cc switch` uses `/api/tags` to discover what local models are available. This bridge emulates just enough Ollama API for discovery — the actual chat goes through the Anthropic → OpenAI translation layer.

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `PORT` | `11435` | Bridge listen port (avoids Ollama conflict) |
| `OPENAI_BASE_URL` | `http://127.0.0.1:18080/v1/` | MLX server endpoint |
| `OPENAI_API_KEY` | (none) | Optional |

## Why Not Use the Original?

The original [debackerl/ollama-proxy](https://github.com/debackerl/ollama-proxy) only does **Ollama API → OpenAI API** for cloud providers (OpenRouter). It does not support the **Anthropic API** that Claude Code uses.

| Feature | Original | This Project |
|---------|----------|-------------|
| **API Key** | Required | Optional |
| **Default Backend** | OpenRouter cloud | `localhost:18080` |
| **`/v1/messages`** | ❌ Not implemented | ✅ **Anthropic → OpenAI translation** |
| **`/v1/*` proxy** | ❌ Not implemented | ✅ Proxied to MLX |
| **`/api/version`** | ❌ Not implemented | ✅ Added |
| **`/api/ps`** | ❌ Not implemented | ✅ Added |
| **Model name** | Strips vendor prefix | Keeps full name |

## Docker

```bash
docker build -t claude-mlx-bridge .
docker run -p 11435:11435 -e OPENAI_BASE_URL=http://host.docker.internal:18080/v1/ claude-mlx-bridge
```

## License

MIT
