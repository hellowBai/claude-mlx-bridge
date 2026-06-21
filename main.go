package main

import (
	"bufio"
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"os"
	"strings"
	"time"

	"gopkg.in/yaml.v3"
	"github.com/gin-gonic/gin"
	openai "github.com/sashabaranov/go-openai"
)

type OllamaMessage struct {
	Role string `json:"role"`
	Content string `json:"content"`
	Thinking string `json:"thinking,omitempty"`
	ToolCalls []OllamaToolCall `json:"tool_calls,omitempty"`
}

func (c *OllamaMessage) ToOpenAi() openai.ChatCompletionMessage {
	var toolCalls []openai.ToolCall
	for _, toolCall := range c.ToolCalls {
		toolCalls = append(toolCalls, toolCall.ToOpenAi())
	}

	return openai.ChatCompletionMessage{
		Role: c.Role,
		Content: c.Content,
		ToolCalls: toolCalls,
		ToolCallID: "", // TODO: should be set when role=tool, must match the "id" returned as part of a Tool Call from the LLM
		ReasoningContent: c.Thinking,
	}
}

type OllamaToolCall struct {
	Function OllamaFunctionCall `json:"function"`
}

func (c *OllamaToolCall) ToOpenAi() openai.ToolCall {
	return openai.ToolCall{
		Type: "function",
		Function: c.Function.ToOpenAi(),
	}
}

type OllamaFunctionCall struct {
	Name string `json:"name"`
	Arguments json.RawMessage `json:"arguments"`
}

func (c *OllamaFunctionCall) ToOpenAi() openai.FunctionCall {
	return openai.FunctionCall{
		Name: c.Name,
		Arguments: string(c.Arguments),
	}
}

var modelFilter map[string]struct{}

func loadModelFilter(path string) (map[string]struct{}, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	scanner := bufio.NewScanner(file)
	filter := make(map[string]struct{})

	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line != "" {
			filter[line] = struct{}{}
		}
	}

	if err := scanner.Err(); err != nil {
		return nil, err
	}

	return filter, nil
}

func parseChoices(choices []openai.ChatCompletionChoice) (string, string, []map[string]interface{}) {
	content := ""
	thinking := ""
	var parsedToolCalls []map[string]interface{}

	if len(choices) > 0 {
		msg := choices[0].Message

		toolCalls := msg.ToolCalls
		if toolCalls != nil && len(toolCalls) > 0 {
			for _, tc := range toolCalls {
				// Parse arguments using YAML to be more foregiving with improper JSON
				var argsMap map[string]interface{}
				if err := yaml.Unmarshal([]byte(tc.Function.Arguments), &argsMap); err == nil {
					parsedToolCall := map[string]interface{}{
						"function": map[string]interface{}{
							"name": tc.Function.Name,
							"arguments": argsMap,
						},
					}
					parsedToolCalls = append(parsedToolCalls, parsedToolCall)
					} else {
					slog.Error("Failed to parse arguments for tool call", "Error", err)
				}
			}
		}

		content = msg.Content
		thinking = msg.ReasoningContent
	}

	return content, thinking, parsedToolCalls
}

func main() {
	r := gin.Default()
	// Load config from environment variables or command-line arguments.
	// For local MLX server, no API key is needed.
	apiKey := os.Getenv("OPENAI_API_KEY")
	if apiKey == "" && len(os.Args) > 1 {
		apiKey = os.Args[len(os.Args)-1]
	}

	// OPENAI_BASE_URL points to the local MLX server by default
	baseUrl := os.Getenv("OPENAI_BASE_URL")
	if baseUrl == "" {
		if len(os.Args) > 2 {
			baseUrl = os.Args[1]
		} else {
			baseUrl = "http://127.0.0.1:18080/v1/"
		}
	}

	provider := NewOpenrouterProvider(baseUrl, apiKey)

	filter, err := loadModelFilter("models-filter")
	if err != nil {
		if os.IsNotExist(err) {
			slog.Info("models-filter file not found. Skipping model filtering.")
			modelFilter = make(map[string]struct{})
		} else {
			slog.Error("Error loading models filter", "Error", err)
			return
		}
	} else {
		modelFilter = filter
		slog.Info("Loaded models from filter:")
		for model := range modelFilter {
			slog.Info(" - " + model)
		}
	}

	r.GET("/", func(c *gin.Context) {
		c.String(http.StatusOK, "Ollama is running")
	})
	r.HEAD("/", func(c *gin.Context) {
		c.String(http.StatusOK, "")
	})

	r.GET("/api/version", func(c *gin.Context) {
		c.JSON(http.StatusOK, gin.H{"version": "0.5.7"})
	})

	r.GET("/api/ps", func(c *gin.Context) {
		c.JSON(http.StatusOK, gin.H{"models": []gin.H{}})
	})

	// Anthropic API compatibility layer for cc switch
	// Claude Code uses Anthropic's /v1/messages API, but MLX server only supports OpenAI API
	// We translate Anthropic format <-> OpenAI format
	r.POST("/v1/messages", func(c *gin.Context) {
		bodyBytes, _ := c.GetRawData()

		// Parse Anthropic request - content can be string or array
		var anthropicReq struct {
			Model     string `json:"model"`
			Messages  []struct {
				Role    string          `json:"role"`
				Content json.RawMessage `json:"content"`
			} `json:"messages"`
			System    json.RawMessage `json:"system"`
			MaxTokens int             `json:"max_tokens"`
			Stream    bool            `json:"stream"`
			Tools     interface{}     `json:"tools"`
		}
		if err := json.Unmarshal(bodyBytes, &anthropicReq); err != nil {
			c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
			return
		}

		// Helper to extract text from Anthropic content (string or array of blocks)
		extractContent := func(raw json.RawMessage) string {
			var str string
			if err := json.Unmarshal(raw, &str); err == nil {
				return str
			}
			var blocks []struct {
				Type string `json:"type"`
				Text string `json:"text"`
			}
			if err := json.Unmarshal(raw, &blocks); err == nil {
				var text string
				for _, b := range blocks {
					if b.Type == "text" {
						text += b.Text
					}
				}
				return text
			}
			return ""
		}

		// Convert to OpenAI format
		openAiMessages := []map[string]string{}
		if len(anthropicReq.System) > 0 {
			systemText := extractContent(anthropicReq.System)
			if systemText != "" {
				openAiMessages = append(openAiMessages, map[string]string{
					"role":    "system",
					"content": systemText,
				})
			}
		}
		for _, m := range anthropicReq.Messages {
			role := m.Role
			if role == "human" {
				role = "user"
			}
			content := extractContent(m.Content)
			openAiMessages = append(openAiMessages, map[string]string{
				"role":    role,
				"content": content,
			})
		}

		openAiReq := map[string]interface{}{
			"model":    anthropicReq.Model,
			"messages": openAiMessages,
			"stream":   anthropicReq.Stream,
		}
		if anthropicReq.MaxTokens > 0 {
			openAiReq["max_tokens"] = anthropicReq.MaxTokens
		}

		openAiBody, _ := json.Marshal(openAiReq)

		// Forward to MLX server (OpenAI API)
		backendUrl := strings.TrimSuffix(baseUrl, "/") + "/chat/completions"
		req, err := http.NewRequest("POST", backendUrl, bytes.NewReader(openAiBody))
		if err != nil {
			c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
			return
		}
		req.Header.Set("Content-Type", "application/json")

		client := &http.Client{Timeout: 300 * time.Second}
		resp, err := client.Do(req)
		if err != nil {
			c.JSON(http.StatusBadGateway, gin.H{"error": err.Error()})
			return
		}
		defer resp.Body.Close()

		if anthropicReq.Stream {
			// Stream response: convert OpenAI SSE to Anthropic SSE
			c.Writer.Header().Set("Content-Type", "text/event-stream")
			c.Writer.Header().Set("Cache-Control", "no-cache")
			c.Writer.Header().Set("Connection", "keep-alive")
			c.Writer.WriteHeader(http.StatusOK)

			flusher, ok := c.Writer.(http.Flusher)
			if !ok {
				return
			}

			// Send message_start event
			msgStart, _ := json.Marshal(map[string]interface{}{
				"type": "message_start",
				"message": map[string]interface{}{
					"id": "msg_" + fmt.Sprintf("%d", time.Now().Unix()),
					"type": "message",
					"role": "assistant",
					"model": anthropicReq.Model,
					"content": []interface{}{},
					"stop_reason": nil,
					"stop_sequence": nil,
					"usage": map[string]interface{}{
						"input_tokens":  0,
						"output_tokens": 0,
					},
				},
			})
			fmt.Fprintf(c.Writer, "event: message_start\ndata: %s\n\n", string(msgStart))

			// Send content_block_start
			blockStart, _ := json.Marshal(map[string]interface{}{
				"type": "content_block_start",
				"index": 0,
				"content_block": map[string]interface{}{
					"type": "text",
					"text": "",
				},
			})
			fmt.Fprintf(c.Writer, "event: content_block_start\ndata: %s\n\n", string(blockStart))

			reader := bufio.NewReader(resp.Body)
			var fullContent string
			for {
				line, err := reader.ReadString('\n')
				if err != nil {
					break
				}
				line = strings.TrimSpace(line)
				if !strings.HasPrefix(line, "data: ") {
					continue
				}
				data := strings.TrimPrefix(line, "data: ")
				if data == "[DONE]" {
					break
				}

				var chunk struct {
					Choices []struct {
						Delta struct {
							Content string `json:"content"`
						} `json:"delta"`
					} `json:"choices"`
				}
				if err := json.Unmarshal([]byte(data), &chunk); err != nil {
					continue
				}
				if len(chunk.Choices) == 0 {
					continue
				}
				content := chunk.Choices[0].Delta.Content
				if content == "" {
					continue
				}
				fullContent += content

				deltaEvent, _ := json.Marshal(map[string]interface{}{
					"type": "content_block_delta",
					"index": 0,
					"delta": map[string]interface{}{
						"type": "text_delta",
						"text": content,
					},
				})
				fmt.Fprintf(c.Writer, "event: content_block_delta\ndata: %s\n\n", string(deltaEvent))
				flusher.Flush()
			}

			// Send content_block_stop
			blockStop, _ := json.Marshal(map[string]interface{}{
				"type": "content_block_stop",
				"index": 0,
			})
			fmt.Fprintf(c.Writer, "event: content_block_stop\ndata: %s\n\n", string(blockStop))

			// Send message_stop
			msgStop, _ := json.Marshal(map[string]interface{}{
				"type": "message_stop",
			})
			fmt.Fprintf(c.Writer, "event: message_stop\ndata: %s\n\n", string(msgStop))
			flusher.Flush()
		} else {
			// Non-stream response: convert OpenAI response to Anthropic format
			var openAiResp struct {
				Choices []struct {
					Message struct {
						Role    string `json:"role"`
						Content string `json:"content"`
					} `json:"message"`
				} `json:"choices"`
				Usage struct {
					PromptTokens     int `json:"prompt_tokens"`
					CompletionTokens int `json:"completion_tokens"`
				} `json:"usage"`
			}
			if err := json.NewDecoder(resp.Body).Decode(&openAiResp); err != nil {
				c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
				return
			}

			content := ""
			if len(openAiResp.Choices) > 0 {
				content = openAiResp.Choices[0].Message.Content
			}

			anthropicResp := map[string]interface{}{
				"id": "msg_" + fmt.Sprintf("%d", time.Now().Unix()),
				"type": "message",
				"role": "assistant",
				"model": anthropicReq.Model,
				"content": []map[string]interface{}{
					{
						"type": "text",
						"text": content,
					},
				},
				"stop_reason": "end_turn",
				"stop_sequence": nil,
				"usage": map[string]interface{}{
					"input_tokens":  openAiResp.Usage.PromptTokens,
					"output_tokens": openAiResp.Usage.CompletionTokens,
				},
			}
			c.JSON(http.StatusOK, anthropicResp)
		}
	})

	// OpenAI-compatible API endpoints (used by cc switch)
	// Handle all /v1/* requests by proxying to the backend MLX server
	r.NoRoute(func(c *gin.Context) {
		path := c.Request.URL.Path
		if !strings.HasPrefix(path, "/v1/") {
			c.JSON(http.StatusNotFound, gin.H{"error": "not found"})
			return
		}

		backendUrl := strings.TrimSuffix(baseUrl, "/") + path[len("/v1"):]
		if c.Request.URL.RawQuery != "" {
			backendUrl += "?" + c.Request.URL.RawQuery
		}

		bodyBytes, _ := c.GetRawData()

		req, err := http.NewRequest(c.Request.Method, backendUrl, bytes.NewReader(bodyBytes))
		if err != nil {
			c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
			return
		}

		for key, values := range c.Request.Header {
			for _, value := range values {
				req.Header.Add(key, value)
			}
		}

		client := &http.Client{Timeout: 300 * time.Second}
		resp, err := client.Do(req)
		if err != nil {
			c.JSON(http.StatusBadGateway, gin.H{"error": err.Error()})
			return
		}
		defer resp.Body.Close()

		for key, values := range resp.Header {
			for _, value := range values {
				c.Writer.Header().Add(key, value)
			}
		}
		c.Writer.WriteHeader(resp.StatusCode)
		io.Copy(c.Writer, resp.Body)
	})

	r.GET("/api/tags", func(c *gin.Context) {
		models, err := provider.GetModels(c)
		if err != nil {
			slog.Error("Error getting models", "Error", err)
			c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
			return
		}
		filter := modelFilter
		// Construct a new array of model objects with extra fields
		newModels := make([]map[string]interface{}, 0, len(models))
		for _, m := range models {
			// If the filter is empty, skip the check and include all models
			if len(filter) > 0 {
				if _, ok := filter[m.Model]; !ok {
					continue
				}
			}
			newModels = append(newModels, map[string]interface{}{
				"name":        m.Name,
				"model":       m.Model,
				"modified_at": m.ModifiedAt,
				"size":        270898672,
				"digest":      "9077fe9d2ae1a4a41a868836b56b8163731a8fe16621397028c2c76f838c6907",
				"details":     m.Details,
			})
		}

		c.JSON(http.StatusOK, gin.H{"models": newModels})
	})

	r.POST("/api/show", func(c *gin.Context) {
		var request map[string]string
		if err := c.BindJSON(&request); err != nil {
			c.JSON(http.StatusBadRequest, gin.H{"error": "Invalid JSON payload"})
			return
		}

		modelName := request["name"]
		if modelName == "" {
			c.JSON(http.StatusBadRequest, gin.H{"error": "Model name is required"})
			return
		}

		details, err := provider.GetModelDetails(modelName)
		if err != nil {
			c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
			return
		}

		c.JSON(http.StatusOK, details)
	})

	r.POST("/api/chat", func(c *gin.Context) {
		var request struct {
			Model     string          `json:"model"`
			Messages  []OllamaMessage `json:"messages"`
			Tools     []openai.Tool   `json:"tools"`
			Stream    *bool           `json:"stream"`
			Think     *bool           `json:"think"`
			KeepAlive string          `json:"keep_alive"` // ex: 30.0s
			Options   map[string]interface{} `json:"options"` // ex: {"num_ctx": 4096.0}
		}

		// Parse the JSON request
		bodyBytes, _ := c.GetRawData()

		//slog.Info("Request", "Request", string(bodyBytes))

		if err := json.Unmarshal(bodyBytes, &request); err != nil {
		//if err := c.ShouldBindJSON(&request); err != nil {
			// Read the raw request body as a string for logging
			slog.Error("Invalid JSON payload", "Error", err, "RequestBody", string(bodyBytes))

			c.JSON(http.StatusBadRequest, gin.H{"error": "Invalid JSON payload"})
			return
		}

		var openAiMessages []openai.ChatCompletionMessage
		for _, message := range request.Messages {
			openAiMessages = append(openAiMessages, message.ToOpenAi())
		}

		//slog.Info("Requested model", "model", request.Model)
		fullModelName, err := provider.GetFullModelName(c, request.Model)
		if err != nil {
			slog.Error("Error getting full model name", "Error", err, "model", request.Model)
			// Ollama returns 404 for an incorrect model name
			c.JSON(http.StatusNotFound, gin.H{"error": err.Error()})
			return
		}

		// Determine if streaming is needed (defaults to true if not specified for /api/chat)
		// IMPORTANT: Open WebUI may NOT send "stream": true for /api/chat, implying it.
		// Need to check what request the Open WebUI sends. If it doesn't send it, default to true.
		streamRequested := true
		if request.Stream != nil {
			streamRequested = *request.Stream
		}

		// If streaming is not requested, separate logic is required to gather the full response and send it as one JSON.
		// For now, only streaming is implemented.
		if !streamRequested {
			// Handle non-streaming response

			req := openai.ChatCompletionRequest{
				Model:    request.Model,
				Messages: openAiMessages,
				Tools:    request.Tools,
				Stream:   false,
			}

			// Call Chat to get the complete response
			response, err := provider.Chat(c, req)
			if err != nil {
				slog.Error("Failed to get chat response", "Error", err)
				c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
				return
			}

			// Format the response according to Ollama's format
			if len(response.Choices) == 0 {
				c.JSON(http.StatusInternalServerError, gin.H{"error": "No response from model"})
				return
			}

			// Extract the content and tool calls from the response
			content, thinking, parsedToolCalls := parseChoices(response.Choices)

			// Get finish reason, default to "stop" if not provided
			finishReason := "stop"
			if response.Choices[0].FinishReason != "" {
				finishReason = string(response.Choices[0].FinishReason)
			}

			// Create Ollama-compatible response
			ollamaResponse := map[string]interface{}{
				"model":      fullModelName,
				"created_at": time.Now().Format(time.RFC3339),
				"message":    map[string]interface{}{
					"role":       "assistant",
					"content":    content,
					"thinking":   thinking,
					"tool_calls": parsedToolCalls,
				},
				"done":              true,
				"finish_reason":     finishReason,
				"total_duration":    0,
				"load_duration":     0,
				"prompt_eval_count": 0,
				"eval_count":        0,
				"eval_duration":     0,
			}

			c.JSON(http.StatusOK, ollamaResponse)
		} else {
			req := openai.ChatCompletionRequest{
				Model:    request.Model,
				Messages: openAiMessages,
				Tools:    request.Tools, // the doc (https://ollama.readthedocs.io/en/api/) says that streaming is not supported with tools, but HASS does it anyway
				Stream:   true,
			}

			//reqJson, _ := json.Marshal(req)
			//slog.Info("Request", "Request", string(reqJson))

			// Call ChatStream to get the stream
			stream, err := provider.ChatStream(c, req)
			if err != nil {
				slog.Error("Failed to create stream", "Error", err)
				c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
				return
			}
			defer stream.Close() // Ensure stream closure

			// Set headers correctly for Newline Delimited JSON
			c.Writer.Header().Set("Content-Type", "application/x-ndjson")
			c.Writer.Header().Set("Cache-Control", "no-cache")
			c.Writer.Header().Set("Connection", "keep-alive")
			// Transfer-Encoding: chunked is set automatically by Gin

			w := c.Writer // Get the ResponseWriter
			flusher, ok := w.(http.Flusher)
			if !ok {
				slog.Error("Expected http.ResponseWriter to be an http.Flusher")
				// Sending an error to the client is difficult as headers may have already been sent
				return
			}

			var lastFinishReason string
			var toolName string
			var argsBuffer bytes.Buffer // arguments for tool calls are streamed

			flushToolCall := func() {
				if toolName == "" {
					return
				}

				var parsedToolCalls []map[string]interface{}

				// Parse arguments using YAML to be more foregiving with improper JSON
				var argsMap map[string]interface{}
				if err := yaml.Unmarshal(argsBuffer.Bytes(), &argsMap); err == nil {
					parsedToolCall := map[string]interface{}{
						"function": map[string]interface{}{
							"name": toolName,
							"arguments": argsMap,
						},
					}
					parsedToolCalls = append(parsedToolCalls, parsedToolCall)
					} else {
					slog.Error("Failed to parse arguments for tool call", "Error", err)
				}

				toolName = ""
				argsBuffer.Reset()

				if len(parsedToolCalls) > 0 {
					// Build JSON response structure for intermediate chunks (Ollama chat format)
					responseJSON := map[string]interface{}{
						"model":      fullModelName,
						"created_at": time.Now().Format(time.RFC3339),
						"message":    map[string]interface{}{
							"role":       "assistant",
							"tool_calls": parsedToolCalls,
						},
						"done":       false, // Always false for intermediate chunks
					}

					// Marshal JSON
					jsonData, err := json.Marshal(responseJSON)
					if err != nil {
						slog.Error("Error marshaling intermediate response JSON", "Error", err)
						return // Return, as we cannot send data
					}
					//slog.Info("Response Chunk", "Data:", jsonData)

					// Send JSON object followed by a newline
					fmt.Fprintf(w, "%s\n", string(jsonData))
				}
			}

			// Stream responses back to the client
			for {
				response, err := stream.Recv()
				if errors.Is(err, io.EOF) {
					// End of stream from the backend provider
					break
				}
				if err != nil {
					slog.Error("Backend stream error", "Error", err)
					// Attempt to send an error in NDJSON format
					// Ollama usually just drops the connection or sends a 500 error before that
					errorMsg := map[string]string{"error": "Stream error: " + err.Error()}
					errorJson, _ := json.Marshal(errorMsg)
					fmt.Fprintf(w, "%s\n", string(errorJson)) // Send the error + \n
					flusher.Flush()
					return
				}

				if len(response.Choices) == 0 {
					continue
				}

				//slog.Info("Response", "Choices", response.Choices)

				// Extract the content and tool calls from the response
				content := ""
				thinking := ""

				if len(response.Choices) > 0 {
					delta := response.Choices[0].Delta

					toolCalls := delta.ToolCalls
					if toolCalls != nil && len(toolCalls) > 0 {
						for _, tc := range toolCalls {
							//slog.Info("Tool Call", "Name", tc.Function.Name, "Arguments", tc.Function.Arguments)

							if tc.Function.Name != "" {
								flushToolCall()

								// only given in the first chunk
								toolName = tc.Function.Name
							}

							argsBuffer.WriteString(tc.Function.Arguments)
						}
					}

					content = delta.Content
					thinking = delta.ReasoningContent
					if content != "" || thinking != "" {
						flushToolCall()
					}
				}

				// Save the stop reason if present in the chunk
				if response.Choices[0].FinishReason != "" {
					lastFinishReason = string(response.Choices[0].FinishReason)
				}

				if content != "" || thinking != "" {
					// Build JSON response structure for intermediate chunks (Ollama chat format)
					responseJSON := map[string]interface{}{
						"model":      fullModelName,
						"created_at": time.Now().Format(time.RFC3339),
						"message":    map[string]interface{}{
							"role":       "assistant",
							"content":    content,
							"thinking":   thinking,
						},
						"done":       false, // Always false for intermediate chunks
					}

					// Marshal JSON
					jsonData, err := json.Marshal(responseJSON)
					if err != nil {
						slog.Error("Error marshaling intermediate response JSON", "Error", err)
						return // Return, as we cannot send data
					}
					//slog.Info("Response Chunk", "Data:", jsonData)

					// Send JSON object followed by a newline
					fmt.Fprintf(w, "%s\n", string(jsonData))
				}

				// Flush data to send it immediately
				flusher.Flush()
			}

			// --- Sending final message (done: true) in Ollama style ---

			// Determine the stop reason (if the backend did not provide one, use 'stop')
			// Ollama uses 'stop', 'length', 'content_filter', 'tool_calls'
			if lastFinishReason == "" {
				lastFinishReason = "stop"
			}

			flushToolCall()

			// IMPORTANT: Replace nil with 0 for numeric stats fields
			finalResponse := map[string]interface{}{
				"model":      fullModelName,
				"created_at": time.Now().Format(time.RFC3339),
				"message": map[string]string{ // required by ollama-python
					"role":    "assistant",
					"content": "",
				},
				"done":              true,
				"finish_reason":     lastFinishReason, // Not required for /api/chat Ollama, but does no harm
				"total_duration":    0,
				"load_duration":     0,
				"prompt_eval_count": 0,
				"eval_count":        0,
				"eval_duration":     0,
			}

			finalJsonData, err := json.Marshal(finalResponse)
			if err != nil {
				slog.Error("Error marshaling final response JSON", "Error", err)
				return
			}

			// Send the final JSON object + newline
			fmt.Fprintf(w, "%s\n", string(finalJsonData))
			flusher.Flush()

			// IMPORTANT: For NDJSON there is NO 'data: [DONE]' marker.
			// The client detects the end of the stream by receiving an object with "done": true
			// and/or by the server closing the connection (Gin will close it automatically after exiting the handler).
		}
	})

	port := os.Getenv("PORT")
	if port == "" {
		port = "11435"
	}
	r.Run(":" + port)
}
