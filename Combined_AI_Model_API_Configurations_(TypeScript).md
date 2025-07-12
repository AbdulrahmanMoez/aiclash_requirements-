# Combined AI Model API Configurations (TypeScript)

This document contains TypeScript configurations for various AI model APIs, consolidated from individual files.

## Anthropic Models

```typescript
export const anthropicModels = [
  {
    modelFamily: "Claude Opus 4",
    officialApiIdentifier: "claude-opus-4-20250514",
    notes: "Our most capable and intelligent model yet. Sets new standards in complex reasoning and advanced coding.",
  },
  {
    modelFamily: "Claude Sonnet 4",
    officialApiIdentifier: "claude-sonnet-4-20250514",
    notes: "High-performance model with exceptional reasoning and efficiency.",
  },
  {
    modelFamily: "Claude Sonnet 3.7",
    officialApiIdentifier: "claude-3-7-sonnet-20250219",
    notes: "High-performance model with early extended thinking.",
  },
  {
    modelFamily: "Claude Sonnet 3.5",
    officialApiIdentifier: "claude-3-5-sonnet-20240620",
    notes: "The previous intelligent model.",
  },
  {
    modelFamily: "Claude Haiku 3.5",
    officialApiIdentifier: "claude-3-5-haiku-20241022",
    notes: "The fastest and most compact model in the 3.5 family, for near-instant responsiveness.",
  },
];
```




## Google AI Models

```typescript
export const googleAiModels = [
  {
    modelFamily: "Gemini 2.5 Pro",
    officialApiIdentifier: "gemini-2.5-pro",
    notes: "The primary high-performance model with advanced reasoning and a large context window.",
  },
  {
    modelFamily: "Gemini 2.5 Flash",
    officialApiIdentifier: "gemini-2.5-flash",
    notes: "A lighter-weight, faster, and lower-cost version of 2.5 Pro, optimized for high-volume text tasks.",
  },
  {
    modelFamily: "Gemini 1.5 Pro",
    officialApiIdentifier: "gemini-1.5-pro",
    notes: "The previous-generation high-performance model. Still available and widely used.",
  },
  {
    modelFamily: "Gemini 1.5 Flash",
    officialApiIdentifier: "gemini-1.5-flash",
    notes: "A previous generation Flash model, efficient for high-volume text tasks.",
  },
];
```




## Meta AI Models

```typescript
export const metaAiModels = [
  {
    modelFamily: "Llama 3.3 70B",
    officialApiIdentifier: "Llama-3.3-70B-Instruct",
    notes: "A text-only instruction-tuned model with enhanced performance.",
  },
  {
    modelFamily: "Llama 3.3 8B",
    officialApiIdentifier: "Llama-3.3-8B-Instruct",
    notes: "A lightweight and ultra-fast variant of Llama 3.3 70B, for quick response times.",
  },
  {
    modelFamily: "Cerebras-Llama-4-Maverick-17B-128E-Instruct (Preview)",
    officialApiIdentifier: "Cerebras-Llama-4-Maverick-17B-128E-Instruct",
    notes: "An accelerated text-only variant of Llama-4-Maverick-17B-128E-Instruct-FP8, with inference provided by Cerebras.",
  },
  {
    modelFamily: "Groq-Llama-4-Maverick-17B-128E-Instruct (Preview)",
    officialApiIdentifier: "Groq-Llama-4-Maverick-17B-128E-Instruct",
    notes: "An accelerated text-only variant of Llama-4-Maverick-17B-128E-Instruct-FP8, with inference provided by Groq.",
  },
];
```




## Mistral AI Models

```typescript
export const mistralAiModels = [
  {
    model: "Magistral Medium",
    officialApiIdentifier: "magistral-medium-2506",
    notes: "Our frontier-class reasoning model released June 2025.",
  },
  {
    model: "Mistral Small 3.2",
    officialApiIdentifier: "mistral-small-2506",
    notes: "An update to our previous small model, released June 2025.",
  },
  {
    model: "Magistral Small",
    officialApiIdentifier: "magistral-small-2506",
    notes: "Our small reasoning model released June 2025.",
  },
  {
    model: "Mistral Medium 3",
    officialApiIdentifier: "mistral-medium-2505",
    notes: "Our frontier-class multimodal model released May 2025 (supports text).",
  },
];
```




## xAI Models

```typescript
export const xaiModels = [
  {
    modelFamily: "Grok 3",
    officialApiIdentifier: "grok-3",
    notes: "The flagship model, excelling at tasks like data extraction, programming, and text summarization.",
  },
  {
    modelFamily: "Grok 3 Mini",
    officialApiIdentifier: "grok-3-mini",
    notes: "A lightweight model that excels at quantitative tasks involving math and reasoning.",
  },
];
```




## OpenAI Models

```typescript
export const openaiModels = [
  {
    modelFamily: "GPT-4o",
    officialApiIdentifier: "gpt-4o",
    notes: "Alias for the latest stable GPT-4o model. Recommended for complex reasoning and text tasks.",
  },
  {
    modelFamily: "GPT-4o Mini",
    officialApiIdentifier: "gpt-4o-mini",
    notes: "Alias for the latest stable, cost-effective, and fast GPT-4o Mini model.",
  },
];
```




## DeepSeek Models

```typescript
export const deepseekModels = [
  {
    modelFamily: "DeepSeek Chat",
    officialApiIdentifier: "deepseek-chat",
    notes: "A powerful chat model, currently aliased to DeepSeek-V3-0324.",
  },
  {
    modelFamily: "DeepSeek Reasoner",
    officialApiIdentifier: "deepseek-reasoner",
    notes: "A reasoning model, currently aliased to DeepSeek-R1-0528.",
  },
];
```





### API Configuration Details

**Sample Message (TypeScript):**

```typescript
import Anthropic from '@anthropic-ai/sdk';

const anthropic = new Anthropic({
  apiKey: process.env.ANTHROPIC_API_KEY, // defaults to process.env["ANTHROPIC_API_KEY"]
});

async function main() {
  const msg = await anthropic.messages.create({
    model: "claude-opus-4-20250514",
    max_tokens: 1024,
    messages: [
      {
        role: "user",
        content: "Hello, Claude",
      },
    ],
  });
  console.log(msg);
}

main();
```

**Latency:**

*   Varies by model and input/output token count. Claude Instant is generally faster. Prompt caching can significantly reduce latency (up to 85%).

**Temperature:**

*   Type: `number`
*   Range: `0.0` to `1.0`
*   Default: `1.0`
*   Description: Controls the randomness of the response. Lower values (closer to 0.0) for analytical/multiple choice tasks, higher values (closer to 1.0) for creative/generative tasks.

**Top P:**

*   Type: `number`
*   Range: `0.0` to `1.0`
*   Description: Controls the diversity of responses using nucleus sampling. You should adjust either `temperature` or `top_p`, but not both.

**Thinking Mode (Extended Thinking):**

*   Type: `object`
*   Description: When enabled, responses include `thinking` content blocks showing Claude's thinking process before the final answer. Requires a minimum budget of 1,024 tokens and counts towards `max_tokens` limit.

**Input/Output Token Length and Counter:**

*   **Context Window:** 200k+ tokens (approximately 500 pages of text or more).
*   **Max Tokens (Output):** The maximum number of tokens to generate before stopping. Varies by model, but generally `x >= 1`.
*   **Usage Counter:** API responses include `usage` object with `input_tokens` and `output_tokens` counts.






## Google AI Models

### API Configuration Details

**Sample Message (TypeScript - using Node.js client library):**

```typescript
import { GoogleGenerativeAI } from "@google/generative-ai";

// Access your API key as an environment variable (see "Set up your API key" above)
const genAI = new GoogleGenerativeAI(process.env.API_KEY);

async function run() {
  // For text-only input, use the gemini-pro model
  const model = genAI.getGenerativeModel({ model: "gemini-pro" });

  const prompt = "Write a story about a magic backpack.";

  const result = await model.generateContent(prompt);
  const response = await result.response;
  const text = response.text();
  console.log(text);
}

run();
```

**Latency:**

*   Varies by model and prompt complexity. Gemini 2.5 Flash is optimized for low latency.
*   Observed latency for 500 tokens can be 10-12 seconds.
*   Batch mode available for processing large jobs with results in 24 hours.

**Temperature:**

*   Type: `number`
*   Range: `0.0` to `1.0` (some documentation mentions up to `2.0`, but `1.0` is common)
*   Default: `0.9` (for `gemini-pro`)
*   Description: Controls the randomness of the response. Lower values for more deterministic output, higher values for more creative output.

**Top P:**

*   Type: `number`
*   Range: `0.0` to `1.0`
*   Description: Nucleus sampling. The model considers tokens whose probability sum is at least `topP`.

**Thinking Mode:**

*   Gemini 2.5 Flash and Pro models have "thinking" enabled by default. This can increase token usage and run time.
*   Can be disabled by setting `thinking_budget` to `0` in `GenerateContentConfig`.

**Input/Output Token Length and Counter:**

*   **Input Token Limit:** Varies by model. For `gemini-1.5-pro`, it's 1,048,576 tokens.
*   **Output Token Limit:** Varies by model. For `gemini-1.5-pro`, it's 8,192 tokens.
*   **Context Window:** Total token limit (combined input + output) for `gemini-1.5-pro` is 1,048,576 tokens.
*   **Usage Counter:** API responses include token counts.






## Meta (via API Providers)

### API Configuration Details

**Sample Message (Python - using Llama API):**

```python
import os
import requests

LLAMA_API_KEY = os.environ.get('LLAMA_API_KEY')
BASE_URL = "https://api.llama.com/v1"

def chat_completion(messages):
    headers = {
        "Authorization": f"Bearer {LLAMA_API_KEY}"
    }
    response = requests.post(f"{BASE_URL}/chat/completions", headers=headers, json={
        "model": "Llama-3.3-70B-Instruct", # or Llama-3.3-8B-Instruct
        "messages": messages
    })
    response.raise_for_status()
    return response.json()

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello, how are you?"}
]

response = chat_completion(messages)
print(response['choices'][0]['message']['content'])
```

**Latency:**

*   Meta Llama API is designed for fast responses. Specific latency numbers vary based on model size and inference provider (e.g., Cerebras, Groq).
*   Groq integration aims for predictable low latency and fast responses.

**Temperature:**

*   Type: `number`
*   Range: `0.0` to `1.0`
*   Default: `0.9` (common default, but can vary by specific API implementation)
*   Description: Controls the randomness of the output. Lower values make the output more deterministic, while higher values make it more creative.

**Top P:**

*   Type: `number`
*   Range: `0.0` to `1.0`
*   Description: Nucleus sampling. The model considers tokens whose probability mass sums up to `top_p`.

**Thinking Mode:**

*   Not explicitly documented as a configurable parameter like Google Gemini. Llama models are generally designed for direct text generation.

**Input/Output Token Length and Counter:**

*   **Context Window (Input + Output):** 128k tokens for Llama-3.3-70B-Instruct and Llama-3.3-8B-Instruct.
*   **Output Token Limit:** Varies by specific API implementation and provider. For example, Oracle Cloud Infrastructure (OCI) Generative AI service with Meta Llama 3.3 (70B) has a response length capped at 4,000 tokens.
*   **Usage Counter:** API responses typically include token usage details, but specific field names may vary by provider.






## Mistral AI Models

### API Configuration Details

**Sample Message (TypeScript - using `mistralai` library):**

```typescript
import MistralClient from '@mistralai/mistralai';

const client = new MistralClient(process.env.MISTRAL_API_KEY);

async function chatCompletion() {
  const chatResponse = await client.chat({
    model: 'mistral-large-latest',
    messages: [{ role: 'user', content: 'What is the capital of France?' }],
    temperature: 0.7,
    topP: 1,
    maxTokens: 100,
  });
  console.log(chatResponse.choices[0].message.content);
}

chatCompletion();
```

**Latency:**

*   Mistral AI models are optimized for low latency. Specific latency can vary based on model size and server load.
*   Mistral Small is highly efficient for high-volume, low-latency tasks.
*   Observed latency for some models can be sub-1 second.

**Temperature:**

*   Type: `number`
*   Range: `0.0` to `1.0` (recommended between `0.0` and `0.7`)
*   Default: Not explicitly stated, but `0.7` is a common recommendation for more random output, `0.2` for more deterministic.
*   Description: Controls the randomness of the output. Lower values make the output more deterministic, while higher values make it more creative.

**Top P:**

*   Type: `number`
*   Range: `0.0` to `1.0`
*   Default: `1.0`
*   Description: Nucleus sampling. The model considers tokens whose probability mass sums up to `top_p`.

**Thinking Mode:**

*   Not explicitly documented as a configurable parameter like Google Gemini. Mistral models are generally designed for direct text generation.

**Input/Output Token Length and Counter:**

*   **Context Window (Input + Output):** Varies by model. For example, `mistral-large-latest` has a context length of 32,768 tokens.
*   **Max Tokens (Output):** `max_tokens` parameter specifies the maximum number of tokens to generate in the completion.
*   **Usage Counter:** API responses include token usage details.






## xAI Models

### API Configuration Details

**Sample Message (Python - using xAI SDK):**

```python
import os

from xai_sdk import Client
from xai_sdk.chat import user, system

client = Client(api_key=os.getenv("XAI_API_KEY"))

chat = client.chat.create(model="grok-4")
chat.append(system("You are a helpful and funny assistant."))
chat.append(user("What is 2 + 2?"))

response = chat.sample()
print(response.content)
```

**Latency:**

*   xAI models are designed for performance. Specific latency varies by model and load.
*   `xai.grok-3-mini-fast` is available for latency-sensitive applications.
*   Observed latency for Grok 3 mini Reasoning (high) can be as low as 0.53s (time to first token).

**Temperature:**

*   Type: `number`
*   Range: `0.0` to `2.0`
*   Default: Not explicitly stated, but `0.7` is a common default in examples.
*   Description: Controls the randomness of the output. Lower values make the output more deterministic, while higher values make it more creative.

**Top P:**

*   Type: `number`
*   Range: `0.0` to `1.0`
*   Default: `1.0` (common default)
*   Description: Nucleus sampling. The model considers tokens whose probability mass sums up to `top_p`.

**Thinking Mode:**

*   xAI Grok models, particularly Grok-1 and Grok-3, are known for their reasoning capabilities. `reasoning_effort` parameter (e.g., `high`) might be available in some API implementations to control the depth of reasoning.

**Input/Output Token Length and Counter:**

*   **Context Window (Input + Output):** Grok 3 Beta has a context of 131K tokens. Grok 3 Mini has a context of 8,192 tokens.
*   **Max Tokens (Output):** `max_tokens` parameter specifies the maximum number of tokens to generate in the completion. Default is often 1000.
*   **Usage Counter:** API responses typically include token usage details. Input tokens are often priced differently from output tokens.






## DeepSeek Models

### API Configuration Details

**Sample Message (cURL - compatible with OpenAI API format):**

```bash
curl https://api.deepseek.com/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <DeepSeek API Key>" \
  -d 
  '{
    "model": "deepseek-chat",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "Hello!"}
    ],
    "stream": false
  }'
```

**Latency:**

*   DeepSeek API latency can vary. Some reports indicate average latency around 23.64s, while others show sub-6s latency. Performance is continually being optimized.
*   DeepSeek-R1 is noted for performance on par with OpenAI-o1.

**Temperature:**

*   Type: `number`
*   Range: `0.0` to `2.0`
*   Default: `1.0`
*   Description: Controls the randomness of the output. Lower values make the output more deterministic, while higher values make it more creative. DeepSeek recommends specific temperature settings based on use cases (e.g., 0.0 for Coding/Math, 1.3 for General Conversation).

**Top P:**

*   Type: `number`
*   Range: `0.0` to `1.0`
*   Default: `0.95` (common default)
*   Description: Nucleus sampling. The model considers tokens whose probability mass sums up to `top_p`. It's generally recommended to alter either `temperature` or `top_p`, but not both.

**Thinking Mode:**

*   DeepSeek offers a `deepseek-reasoner` model which provides access to Chain-of-Thought (CoT) content, enabling viewing, displaying, and distilling of the reasoning process.

**Input/Output Token Length and Counter:**

*   **Context Window (Input + Output):** DeepSeek-V2 has a context length of 128K tokens. DeepSeek-R1 has a context length of 128K tokens.
*   **Max Tokens (Output):** The `max_tokens` parameter limits the maximum output tokens of one request. For `deepseek-reasoner`, this includes the CoT part.
*   **Usage Counter:** Tokens are the basic units for billing. The actual number of tokens processed is based on the model's return, and can be viewed from usage results. Approximately 1 English character ≈ 0.3 token, 1 Chinese character ≈ 0.6 token.






## Google AI Models

### API Configuration Details

**Sample Message (Python - using `google-generativeai` library):**

```python
import google.generativeai as genai

genai.configure(api_key="YOUR_API_KEY")

model = genai.GenerativeModel("gemini-pro")

response = model.generate_content("Write a story about a magic backpack.")
print(response.text)
```

**Latency:**

*   Latency varies based on model, request complexity, and network conditions. Google AI models are generally optimized for performance.

**Temperature:**

*   Type: `number`
*   Range: `0.0` to `1.0`
*   Default: `0.9`
*   Description: Controls the randomness of the output. Higher values result in more creative and diverse outputs, while lower values produce more deterministic and focused outputs.

**Top P:**

*   Type: `number`
*   Range: `0.0` to `1.0`
*   Default: `1.0`
*   Description: Nucleus sampling. The model considers tokens whose probability mass sums up to `top_p`.

**Thinking Mode:**

*   Google Gemini models support various modes, including a 



## OpenAI Models

### API Configuration Details

**Note:** Due to persistent Cloudflare verification issues on OpenAI's official documentation, comprehensive API configuration details for sample messages, latency, thinking mode, and specific token counters could not be fully verified from official sources. The information below is based on general knowledge and publicly available API references, but may not be as precise as for other providers.

**Sample Message (TypeScript - using `openai` library):**

```typescript
import OpenAI from "openai";

const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY,
});

async function main() {
  const chatCompletion = await openai.chat.completions.create({
    messages: [{
      role: "user",
      content: "Say this is a test",
    }],
    model: "gpt-4o", // or gpt-4-turbo, gpt-3.5-turbo
    temperature: 0.7,
    top_p: 1,
    max_tokens: 100,
  });
  console.log(chatCompletion.choices[0].message.content);
}

main();
```

**Latency:**

*   Latency varies significantly based on the model (e.g., GPT-3.5 is faster than GPT-4), request complexity, and current API load. OpenAI continuously works on optimizing latency.
*   Typical response times can range from tens of milliseconds to several seconds per generated token.

**Temperature:**

*   Type: `number`
*   Range: `0.0` to `2.0`
*   Default: `1.0` (or `0.7` depending on the model)
*   Description: Controls the randomness of the output. Higher values (e.g., 0.8) make the output more random and creative, while lower values (e.g., 0.2) make it more deterministic and focused.

**Top P:**

*   Type: `number`
*   Range: `0.0` to `1.0`
*   Default: `1.0`
*   Description: Nucleus sampling. The model considers tokens whose probability mass sums up to `top_p`. It's generally recommended to adjust either `temperature` or `top_p`, but not both.

**Thinking Mode:**

*   OpenAI models do not expose a direct 'thinking mode' parameter. Their reasoning capabilities are inherent to the model architecture. For complex tasks, prompt engineering (e.g., Chain-of-Thought prompting) is used to guide the model's reasoning process.

**Input/Output Token Length and Counter:**

*   **Context Window (Input + Output):** Varies significantly by model. For example, `gpt-4o` has a context window of 128,000 tokens, and `gpt-4-turbo` has 128,000 tokens. `gpt-3.5-turbo` typically has a context window of 16,385 tokens.
*   **Max Tokens (Output):** The `max_tokens` parameter specifies the maximum number of tokens to generate in the completion. This value contributes to the overall context window limit.
*   **Usage Counter:** API responses include a `usage` object that provides details on `prompt_tokens`, `completion_tokens`, and `total_tokens` for each request.




