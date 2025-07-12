# OpenRouter Free Models Documentation

This document provides a comprehensive overview of free models available on OpenRouter, including their IDs, key specifications, and a guide for API integration using TypeScript. All information is gathered from the official OpenRouter website and the provided PDF documentation.

## Model Details by Provider




### DeepSeek

| Model Name | Model ID | Context (tokens) | Input ($/1M tokens) | Output ($/1M tokens) | Description |
|---|---|---|---|---|---|
| DeepSeek V3 0324 (free) | `deepseek/deepseek-chat-v3-0324:free` | 163,840 | $0 | $0 | DeepSeek V3, a 685B-parameter, mixture-of-experts model, is the latest iteration of the flagship chat model family from the DeepSeek team. It succeeds the DeepSeek V3 model and... |
| R1 0528 (free) | `deepseek/deepseek-r1-0528:free` | 163,840 | $0 | $0 | May 28th update to the original DeepSeek R1 Performance on par with OpenAI o1, but open-sourced and with fully open reasoning tokens. It's 671B parameters in size, with 37B active in ... |
| R1 (free) | `deepseek/deepseek-r1:free` | 163,840 | $0 | $0 | DeepSeek R1 is here: Performance on par with OpenAI o1, but open-sourced and with fully open reasoning tokens. It's 671B parameters in size, with 37B active in an inference pass. Full... |
| DeepSeek V3 (free) | `deepseek/deepseek-chat:free` | 163,840 | $0 | $0 | DeepSeek-V3 is the latest model from the DeepSeek team, building upon the instruction following and coding abilities of the previous versions. Pre-trained on nearly 15 trillion tokens,... |
| Deepseek R1 0528 Qwen3 8B (free) | `deepseek/deepseek-r1-0528-qwen3-8b:free` | 131,072 | $0 | $0 | DeepSeek-R1-0528 is a lightly upgraded release of DeepSeek R1 that taps more compute and smarter post-training tricks, pushing its reasoning and inference to the brink of flagship model... |
| DeepSeek V3 Base (free) | `deepseek/deepseek-v3-base:free` | 163,840 | $0 | $0 | Note that this is a base model mostly meant for testing, you need to provide detailed prompts for the model to return useful responses. DeepSeek-V3 Base is a 671B parameter open Mixture... |
| R1 Distill Llama 70B (free) | `deepseek/deepseek-r1-distill-llama-70b:free` | 8,192 | $0 | $0 | DeepSeek R1 Distill Llama 70B is a distilled large language model based on Llama-3.3-70B-Instruct using outputs from DeepSeek R1. The model combines advanced distillation ... |



### Google

| Model Name | Model ID | Context (tokens) | Input ($/1M tokens) | Output ($/1M tokens) | Description |
|---|---|---|---|---|---|
| Gemini 2.0 Flash Experimental (free) | `google/gemini-2.0-flash-exp:free` | 1,048,576 | $0 | $0 | Gemini Flash 2.0 offers a significantly faster time to first token (TTFT) compared to Gemini Flash 1.5, while maintaining quality on par with larger models like Gemini Pro 1.5. It introduces notab... |
| Gemma 3 27B (free) | `google/gemma-3-27b:free` | 131,072 | $0 | $0 | Gemma 3 introduces multimodality, supporting vision-language input and text outputs. It handles context windows up to 128k tokens, understands over 140 languages, and offers ... |
| Gemma 3n 4B (free) | `google/gemma-3n-4b:free` | 8,192 | $0 | $0 | Gemma 3n E4B-IT is optimized for efficient execution on mobile and low-resource devices, such as phones, laptops, and tablets. It supports multimodal inputs—including text, visual data... |
| Gemma 2 9B (free) | `google/gemma-2-9b:free` | 8,192 | $0 | $0 | Gemma 2 9B by Google is an advanced, open-source language model that sets a new standard for efficiency and performance in its size class. Designed for a wide variety of tasks, it... |
| Gemma 3 4B (free) | `google/gemma-3-4b:free` | 131,072 | $0 | $0 | Gemma 3 introduces multimodality, supporting vision-language input and text outputs. It handles context windows up to 128k tokens, understands over 140 languages, and offers ... |
| Gemma 3n 2B (free) | `google/gemma-3n-2b:free` | 8,192 | $0 | $0 | Gemma 3n E2B IT is a multimodal, instruction-tuned model developed by Google DeepMind, designed to operate efficiently at an effective parameter size of 2B while leveraging a 6B ... |



### Qwen

| Model Name | Model ID | Context (tokens) | Input ($/1M tokens) | Output ($/1M tokens) | Description |
|---|---|---|---|---|---|
| QwQ 32B (free) | `qwen/qwq-32b:free` | 40,960 | $0 | $0 | QwQ is the reasoning model of the Qwen series. Compared with conventional instruction-tuned models, QwQ, which is capable of thinking and reasoning, can achieve significantly enhanced performance in downstream tasks, especially hard problems. QwQ-32B is the medium-sized reasoning model, which is capable of achieving competitive performance against state-of-the-art reasoning models, e.g., DeepSeek-R1, o1-mini. |
| Qwen3 14B (free) | `qwen/qwen3-14b:free` | 40,960 | $0 | $0 | Qwen3-14B is a dense 14.8B parameter causal language model from the Qwen3 series, designed for both complex reasoning and efficient dialogue. It supports seamless switching ... |
| Qwen3 32B (free) | `qwen/qwen3-32b:free` | 40,960 | $0 | $0 | Qwen3-32B is a dense 32.8B parameter causal language model from the Qwen3 series, optimized for both complex reasoning and efficient dialogue. It supports seamless switching ... |
| Qwen3 235B A22B (free) | `qwen/qwen3-235b-a22b:free` | 131,072 | $0 | $0 | Qwen3-235B-A22B is a 235B parameter mixture-of-experts (MoE) model developed by Qwen, activating 22B parameters per forward pass. It supports seamless switching between a... |
| Qwen2.5 VL 72B Instruct (free) | `qwen/qwen2.5-vl-72b-instruct:free` | 32,768 | $0 | $0 | Qwen2.5-VL is proficient in recognizing common objects such as flowers, birds, fish, and insects. It is also highly capable of analyzing texts, charts, icons, graphics, and layouts within ... |
| Qwen3 30B A3B (free) | `qwen/qwen3-30b-a3b:free` | 40,960 | $0 | $0 | Qwen3, the latest generation in the Qwen large language model series, features both dense and mixture-of-experts (MoE) architectures to excel in reasoning, multilingual support, and ... |
| Qwen2.5 Coder 32B Instruct (free) | `qwen/qwen2.5-coder-32b-instruct:free` | 32,768 | $0 | $0 | Qwen2.5-Coder is the latest series of Code-Specific Qwen large language models (formerly known as CodeQwen). Qwen2.5-Coder brings the following improvements upon ... |
| Qwen2.5 72B Instruct (free) | `qwen/qwen2.5-72b-instruct:free` | 32,768 | $0 | $0 | Qwen2.5 72B is the latest series of Qwen large language models. Qwen2.5 brings the following improvements upon Qwen2: - Significantly more knowledge and has greatly improved ... |
| Qwen2.5 VL 32B Instruct (free) | `qwen/qwen2.5-vl-32b-instruct:free` | 8,192 | $0 | $0 | Qwen2.5-VL-32B is a multimodal vision-language model fine-tuned through reinforcement learning for enhanced mathematical reasoning, structured outputs, and visual problem-solvin... |
| Qwen3 4B (free) | `qwen/qwen3-4b:free` | 40,960 | $0 | $0 | Qwen3-4B is a 4 billion parameter dense language model from the Qwen3 series, designed to support both general-purpose and reasoning-intensive tasks. It introduces a dual-mode ... |



### Mistral

| Model Name | Model ID | Context (tokens) | Input ($/1M tokens) | Output ($/1M tokens) | Description |
|---|---|---|---|---|---|
| Mistral Nemo (free) | `mistralai/mistral-nemo:free` | 131,072 | $0 | $0 | A 12B parameter model with a 128k token context length built by Mistral in collaboration with NVIDIA. The model is multilingual, supporting English, French, German, Spanish, Italian, ... |
| Mistral Small 3.2 24B (free) | `mistralai/mistral-small-3.2-24b:free` | 131,072 | $0 | $0 | Mistral-Small-3.2-24B-Instruct-2506 is an updated 24B parameter model from Mistral optimized for instruction following, repetition reduction, and improved function calling. ... |
| Mistral Small 3.1 24B (free) | `mistralai/mistral-small-3.1-24b:free` | 131,072 | $0 | $0 | Mistral Small 3.1 24B Instruct is an upgraded variant of Mistral Small 3 (2501), featuring 24 billion parameters with advanced multimodal capabilities. It provides state-of-the-art ... |
| Devstral Small 2505 (free) | `mistralai/devstral-small-2505:free` | 32,768 | $0 | $0 | Devstral-Small-2505 is a 24B parameter agentic LLM fine-tuned from Mistral-Small-3.1, jointly developed by Mistral AI and All Hands AI for advanced software engineering tasks ... |
| Mistral 7B Instruct (free) | `mistralai/mistral-7b-instruct:free` | 32,768 | $0 | $0 | A high-performing, industry-standard 7.3B parameter model, with optimizations for speed and ... |
| Mistral Small 3 (free) | `mistralai/mistral-small-3:free` | 32,768 | $0 | $0 | Mistral Small 3 is a 24B-parameter language model optimized for low-latency performance across common AI tasks. Released under the Apache 2.0 license, it features both pre-trained ... |



### TNG

| Model Name | Model ID | Context (tokens) | Input ($/1M tokens) | Output ($/1M tokens) | Description |
|---|---|---|---|---|---|
| DeepSeek R1T Chimera (free) | `tngtech/deepseek-r1t-chimera:free` | 163,840 | $0 | $0 | DeepSeek-R1T-Chimera is created by merging DeepSeek-R1 and DeepSeek-V3 (0324), combining the reasoning capabilities of R1 with the token efficiency improvements of V3. ... |
| DeepSeek R1T2 Chimera (free) | `tngtech/deepseek-r1t2-chimera:free` | 163,840 | $0 | $0 | DeepSeek-TNG-R1T2-Chimera is the second-generation Chimera model from TNG Tech. It is a 671B-parameter mixture-of-experts text-generation model assembled from DeepSeek-A... |



### Other Providers

| Model Name | Model ID | Context (tokens) | Input ($/1M tokens) | Output ($/1M tokens) | Description |
|---|---|---|---|---|---|
| Cypher Alpha (free) | `openrouter/cypher-alpha:free` | 1,048,576 | $0 | $0 | This is a cloaked model provided to the community to gather feedback. It's an all-purpose model supporting real-world, long-context tasks including code generation. Note: All prompt... |
| Meta: Llama 4 Maverick (free) | `meta-llama/llama-4-maverick:free` | 131,072 | $0 | $0 | Llama 4 Maverick 17B Instruct (128E) is a high-capacity multimodal language model from Meta, built on a a mixture-of-experts (MoE) architecture with 128 experts and 17 billion active ... |
| THUDM: GLM Z1 32B (free) | `thudm/glm-z1-32b:free` | 32,768 | $0 | $0 | GLM-Z1-32B-0414 is an enhanced reasoning variant of GLM-4-32B, built for deep mathematical, logical, and code-oriented problem solving. It applies extended reinforcement ... |
| Meta: Llama 3.3 70B Instruct (free) | `meta-llama/llama-3.3-70b-instruct:free` | 65,536 | $0 | $0 | Llama 3.3 70B Instruct is a fine-tuned generative model in 70B (text in/text out). The Llama 3.3 instruction tuned text only mod... |
| Shisa AI: Shisa V2 Llama 3.3 70B (free) | `shisa-ai/shisa-v2-llama-3.3-70b:free` | 32,768 | $0 | $0 | Shisa V2 Llama 3.3 70B is a bilingual Japanese-English chat model fine-tuned by Shisa.AI on Meta's Llama-3.3-70B-Instruct base. It prioritizes Japanese language performance while ... |
| Dolphin3.0 Mistral 24B (free) | `cognitivecomputations/dolphin3.0-mistral-24b:free` | 32,768 | $0 | $0 | Dolphin 3.0 is the next generation of the Dolphin series of instruct-tuned models. Designed to be the ultimate general purpose local model, enabling coding, math, agentic, function c... |
| THUDM: GLM 4 32B (free) | `thudm/glm-4-32b:free` | 32,768 | $0 | $0 | GLM-4-32B-0414 is a 32B bilingual (Chinese-English) open-weight language model optimized for code generation, function calling, and agent-style tasks. Pretrained on 15T of ... |
| Nous: DeepHermes 3 Llama 3 8B Preview (free) | `nousresearch/deephermes-3-llama-3-8b-preview:free` | 131,072 | $0 | $0 | DeepHermes 3 Preview is the latest version of our flagship Hermes series of LLMs by Nous Research, and one of the first models in the world to unify Reasoning (long chains of thought ... |
| Dolphin3.0 R1 Mistral 24B (free) | `cognitivecomputations/dolphin3.0-r1-mistral-24b:free` | 32,768 | $0 | $0 | Dolphin 3.0 R1 is the next generation of the Dolphin series of instruct-tuned models. Designed to be the ultimate general purpose local model, enabling coding, math, agentic, function c... |
| ArliAI: QwQ 32B RpR v1 (free) | `arliai/qwq-32b-rpr-v1:free` | 32,768 | $0 | $0 | QwQ-32B-ArliAI-RpR-v1 is a 32B parameter model fine-tuned from Qwen/QwQ-32B using a curated creative writing and roleplay dataset originally developed for the RPMax series. It is ... |
| Reka: Flash 3 (free) | `rekaai/flash-3:free` | 32,768 | $0 | $0 | Reka Flash 3 is a general-purpose, instruction-tuned large language model with 21 billion parameters, developed by Reka. It excels at general chat, coding tasks, instruction-following, ... |
| Moonshot AI: Kimi VL A3B Thinking (free) | `moonshotai/kimi-vl-a3b-thinking:free` | 131,072 | $0 | $0 | Kimi-VL is a lightweight Mixture-of-Experts vision-language model that activates only 2.8B parameters per step while delivering strong performance on multimodal reasoning and long-... |
| Sarvam AI: Sarvam-M (free) | `sarvamai/sarvam-m:free` | 32,768 | $0 | $0 | Sarvam-M is a 24 B-parameter, instruction-tuned derivative of Mistral-Small-3.1-24B-Base-2503, post-trained on English plus eleven major Indic languages (bn, hi, kn, gu, mr, ml, or, pa, ... |
| Meta: Llama 3.1 405B Instruct (free) | `meta-llama/llama-3.1-405b-instruct:free` | 65,536 | $0 | $0 | The highly anticipated 400B class of Llama3 is here! Clocking in at 128k context with impressive eval scores, the Meta AI team continues to push the frontier of open-source LLMs.... |
| Qwerky 72B (free) | `featherless/qwerky-72b:free` | 32,768 | $0 | $0 | Qwerky-72B is a linear-attention RWKV variant of the Qwen 2.5 72B model, optimized to significantly reduce computational cost at scale. Leveraging linear attention, it achieves ... |
| Meta: Llama 3.2 3B Instruct (free) | `meta-llama/llama-3.2-3b-instruct:free` | 131,072 | $0 | $0 | Llama 3.2 3B is a 3-billion-parameter multilingual large language model, optimized for advanced natural language processing tasks like dialogue generation, reasoning, and ... |
| NVIDIA: Llama 3.1 Nemotron Ultra 253B v1 (free) | `nvidia/llama-3.1-nemotron-ultra-253b-v1:free` | 131,072 | $0 | $0 | Llama 3.1 Nemotron Ultra 253B v1 is a large language model (LLM) optimized for ... |




## API Configuration in TypeScript

OpenRouter's API is designed to be similar to the OpenAI Chat API, making integration straightforward. Below is a TypeScript example demonstrating how to configure API requests, including common parameters like `temperature`, `max_tokens`, and `top_p`.

```typescript
interface ChatCompletionRequestMessage {
  role: 'user' | 'assistant' | 'system';
  content: string;
}

interface ChatCompletionRequest {
  model: string; // e.g., 'deepseek/deepseek-chat-v3-0324:free' - The ID of the model to use.
  messages?: ChatCompletionRequestMessage[]; // A list of messages comprising the conversation so far.
  prompt?: string; // Either 'messages' or 'prompt' is required. A string prompt for completion.
  max_tokens?: number; // Range: [1, context_length] - The maximum number of tokens to generate in the completion.
  temperature?: number; // Range: [0, 2] - Controls randomness. Higher values mean more random outputs. Lower values make the model more deterministic.
  top_p?: number; // Range: [0, 1] - Controls diversity. Only tokens whose cumulative probability exceeds top_p are considered. Helps to avoid less likely tokens.
  top_k?: number; // Range: [0, infinity] - Limits the sampling pool to the top_k most likely next tokens. Useful for controlling the breadth of generated text.
  frequency_penalty?: number; // Range: [-2, 2] - Penalizes new tokens based on their existing frequency in the text so far. Encourages the model to use new topics.
  presence_penalty?: number; // Range: [-2, 2] - Penalizes new tokens based on whether they appear in the text so far. Encourages the model to talk about new things.
  repetition_penalty?: number; // Range: (0, 2] - Penalizes tokens that have already appeared in the text. Higher values reduce repetition.
  logit_bias?: { [key: number]: number }; // Modifies the likelihood of specified tokens appearing in the completion. Accepts a JSON object that maps tokens (specified by their token ID in the tokenizer) to an associated bias value.
  seed?: number; // Integer only. A random seed to make the generation deterministic.
  stop?: string | string[]; // Up to 4 sequences where the API will stop generating further tokens. The generated text will not contain the stop sequence.
  stream?: boolean; // If true, partial message deltas will be sent. Tokens are sent as they are generated.
  response_format?: { type: 'json_object' }; // Only supported by certain models. Forces the model to produce a JSON object.
  min_p?: number; // Range: [0, 1] - Minimum probability for a token to be considered.
  top_a?: number; // Range: [0, 1] - Similar to top_p, but for attention scores.
  prediction?: { type: 'content'; content: string }; // Reduce latency by providing the model with a predicted output.
  transforms?: string[]; // OpenRouter-only parameter. See "Prompt Transforms" section in OpenRouter documentation.
  models?: string[]; // OpenRouter-only parameter. See "Model Routing" section in OpenRouter documentation. Allows specifying a list of models to try in order.
  route?: 'fallback'; // OpenRouter-only parameter. Specifies routing behavior, e.g., 'fallback' to try other providers if the primary fails.
  provider?: any; // OpenRouter-only parameter. See "Provider Routing" section in OpenRouter documentation. Allows specifying provider preferences.
  user?: string; // A stable identifier for your end-users. Used to help detect and prevent abuse.
  // ... any other parameters as per OpenRouter API documentation
}

async function getChatCompletion(request: ChatCompletionRequest) {
  const API_KEY = process.env.OPENROUTER_API_KEY; // Ensure your API key is set as an environment variable

  if (!API_KEY) {
    throw new Error('OPENROUTER_API_KEY environment variable is not set.');
  }

  const response = await fetch('https://openrouter.ai/api/v1/chat/completions', {
    method: 'POST',
    headers: {
      'Authorization': `Bearer ${API_KEY}`,
      'Content-Type': 'application/json',
      'HTTP-Referer': 'YOUR_APP_URL', // Replace with your app's URL
      'X-Title': 'YOUR_APP_NAME', // Replace with your app's name
    },
    body: JSON.stringify(request),
  });

  if (!response.ok) {
    const errorData = await response.json();
    throw new Error(`API error: ${response.status} ${response.statusText} - ${errorData.message || JSON.stringify(errorData)}`);
  }

  return response.json();
}

// Example Usageasync function main() {
  try {
    // Example chat completion request with various configuration parameters
    const chatRequest: ChatCompletionRequest = {
      model: 'google/gemini-2.0-flash-exp:free', // Specify the model ID
      messages: [
        { role: 'user', content: 'What is the capital of France?' }
      ],
      temperature: 0.7, // Controls randomness (0.0 - 2.0)
      max_tokens: 150, // Maximum number of tokens to generate
      top_p: 0.9, // Nucleus sampling: only consider tokens with cumulative probability above 0.9
      top_k: 50, // Top-k sampling: consider only the top 50 most likely next tokens
      frequency_penalty: 0.5, // Penalize new tokens based on their frequency in the text so far
      presence_penalty: 0.2, // Penalize new tokens based on whether they appear in the text so far
      repetition_penalty: 1.1, // Penalize tokens that have already appeared
      seed: 123, // For deterministic results
      stop: ['\n\n'], // Stop generation at these sequences
      stream: false, // Set to true for streaming responses
      // logit_bias: { 123: 1.5 }, // Example: Boost the likelihood of token ID 123
      // min_p: 0.1, // Example: Minimum probability for a token to be considered
      // top_a: 0.5, // Example: Top-A sampling
      // prediction: { type: 'content', content: 'The capital of France is' }, // Predicted output to reduce latency
      // transforms: ['some_transform'], // OpenRouter-specific prompt transforms
      // models: ['model/id-1', 'model/id-2'], // List of models to try in order
      // route: 'fallback', // Routing behavior
      // provider: { /* provider preferences */ }, // Provider-specific preferences
      user: 'user-123', // Stable identifier for the end-user
    };

    console.log('Sending chat completion request...');
    const completion = await getChatCompletion(chatRequest);
    console.log('Completion:', completion.choices[0].message.content);

    // Example JSON object generation request
    const jsonRequest: ChatCompletionRequest = {
      model: 'deepseek/deepseek-chat-v3-0324:free', // Example free model ID that supports JSON
      messages: [
        { role: 'user', content: 'Generate a JSON object with a name and age.\n```json\n{\n  "name": "John Doe",\n  "age": 30\n}\n```' }
      ],
      response_format: { type: 'json_object' }, // Force JSON output
    };

    console.log('Sending JSON generation request...');
    const jsonCompletion = await getChatCompletion(jsonRequest);
    console.log('JSON Completion:', jsonCompletion.choices[0].message.content);

  } catch (error) {
    console.error('Error:', error);
  }
}

main();
```

**Note:**
*   Replace `YOUR_APP_URL` and `YOUR_APP_NAME` with your actual application details.
*   Ensure your OpenRouter API key is securely stored and accessed, for example, via environment variables.
*   The `response_format: { type: 'json_object' }` parameter is only supported by certain models. Refer to the OpenRouter model page for compatibility.



