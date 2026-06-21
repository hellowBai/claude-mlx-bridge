#!/bin/bash
# Start MLX server + Ollama proxy bridge
# Usage: ./start-mlx-bridge.sh [model-name]

set -e

MODEL="${1:-mlx-community/Qwen3-8B-4bit}"
MLX_PORT=18080
PROXY_PORT=11434

echo "🚀 Starting MLX Server (model: $MODEL, port: $MLX_PORT)..."
mlx_lm.server --model "$MODEL" --port "$MLX_PORT" &
MLX_PID=$!

# Wait for MLX server
echo "⏳ Waiting for MLX server..."
for i in {1..30}; do
    if curl -s http://127.0.0.1:$MLX_PORT/v1/models > /dev/null 2>&1; then
        echo "✅ MLX server ready"
        break
    fi
    sleep 1
done

echo "🚀 Starting Ollama Proxy (port: $PROXY_PORT) → MLX (port: $MLX_PORT)..."
./ollama-proxy-mlx &
PROXY_PID=$!

# Wait for proxy
echo "⏳ Waiting for proxy..."
for i in {1..30}; do
    if curl -s http://127.0.0.1:$PROXY_PORT/ > /dev/null 2>&1; then
        echo "✅ Proxy ready"
        break
    fi
    sleep 1
done

echo ""
echo "═══════════════════════════════════════════════"
echo "  🎉 Bridge is running!"
echo ""
echo "  Ollama API:  http://127.0.0.1:$PROXY_PORT"
echo "  MLX Backend: http://127.0.0.1:$MLX_PORT"
echo "  Model: $MODEL"
echo ""
echo "  Now run: cc switch"
echo "═══════════════════════════════════════════════"
echo ""

trap "echo ''; echo '🛑 Stopping...'; kill $PROXY_PID $MLX_PID 2>/dev/null; exit 0" INT TERM
wait
