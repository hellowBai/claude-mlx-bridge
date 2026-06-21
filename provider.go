package main

import (
	"context"
	"fmt"
	"strings"
	"time"

	"github.com/sashabaranov/go-openai"
)

type OpenrouterProvider struct {
	client     *openai.Client
	modelNames []string // Shared storage for model names
}

func NewOpenrouterProvider(baseUrl string, apiKey string) *OpenrouterProvider {
	config := openai.DefaultConfig(apiKey)
	config.BaseURL = baseUrl
	return &OpenrouterProvider{
		client:     openai.NewClientWithConfig(config),
		modelNames: []string{},
	}
}

func (o *OpenrouterProvider) Chat(ctx context.Context, req openai.ChatCompletionRequest) (openai.ChatCompletionResponse, error) {
	// Call the OpenAI API to get a complete response
	resp, err := o.client.CreateChatCompletion(ctx, req)
	if err != nil {
		return openai.ChatCompletionResponse{}, err
	}

	// Return the complete response
	return resp, nil
}

func (o *OpenrouterProvider) ChatStream(ctx context.Context, req openai.ChatCompletionRequest) (*openai.ChatCompletionStream, error) {
	// Call the OpenAI API to get a streaming response
	stream, err := o.client.CreateChatCompletionStream(ctx, req)
	if err != nil {
		return nil, err
	}

	// Return the stream for further processing
	return stream, nil
}

type ModelDetails struct {
	ParentModel       string   `json:"parent_model"`
	Format            string   `json:"format"`
	Family            string   `json:"family"`
	Families          []string `json:"families"`
	ParameterSize     string   `json:"parameter_size"`
	QuantizationLevel string   `json:"quantization_level"`
}

type Model struct {
	Name       string       `json:"name"`
	Model      string       `json:"model,omitempty"`
	ModifiedAt string       `json:"modified_at,omitempty"`
	Size       int64        `json:"size,omitempty"`
	Digest     string       `json:"digest,omitempty"`
	Details    ModelDetails `json:"details,omitempty"`
}

func (o *OpenrouterProvider) GetModels(ctx context.Context) ([]Model, error) {
	currentTime := time.Now().Format(time.RFC3339)

	// Fetch models from the OpenAI API
	modelsResponse, err := o.client.ListModels(ctx)
	if err != nil {
		return nil, err
	}

	// Clear shared model storage
	o.modelNames = []string{}

	var models []Model
	for _, apiModel := range modelsResponse.Models {
		name := apiModel.ID

		// Store name in shared storage
		o.modelNames = append(o.modelNames, name)

		// Create model struct
		model := Model{
			Name:       name,
			Model:      name,
			ModifiedAt: currentTime,
			Size:       0, // Stubbed size
			Digest:     name,
			Details: ModelDetails{
				ParentModel:       "",
				Format:            "mlx",
				Family:            "mlx",
				Families:          []string{"mlx"},
				ParameterSize:     "8B",
				QuantizationLevel: "4bit",
			},
		}
		models = append(models, model)
	}

	return models, nil
}

func (o *OpenrouterProvider) GetModelDetails(modelName string) (map[string]interface{}, error) {
	// Stub response; replace with actual model details if available
	currentTime := time.Now().Format(time.RFC3339)
	return map[string]interface{}{
		"license":    "STUB License",
		"system":     "STUB SYSTEM",
		"modifiedAt": currentTime,
		"details": map[string]interface{}{
			"format":             "gguf",
			"parameter_size":     "200B",
			"quantization_level": "Q4_K_M",
		},
		"model_info": map[string]interface{}{
			"architecture":    "STUB",
			"context_length":  200000,
			"parameter_count": 200_000_000_000,
		},
	}, nil
}

func (o *OpenrouterProvider) GetFullModelName(ctx context.Context, alias string) (string, error) {
	// If modelNames is empty or not populated yet, try to get models first
	if len(o.modelNames) == 0 {
		_, err := o.GetModels(ctx)
		if err != nil {
			return "", fmt.Errorf("failed to get models: %w", err)
		}
	}

	// First try exact match
	for _, fullName := range o.modelNames {
		if fullName == alias {
			return fullName, nil
		}
	}

	// Then try suffix match
	for _, fullName := range o.modelNames {
		if strings.HasSuffix(fullName, alias) {
			return fullName, nil
		}
	}

	// If no match found, just use the alias as is
	// This allows direct use of model names that might not be in the list
	return alias, nil
}
