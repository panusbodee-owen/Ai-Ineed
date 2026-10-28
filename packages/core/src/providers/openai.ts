/**
 * OpenAI Provider Adapter
 * Implements LanguageModelV4 interface for OpenAI Chat Completions API (v4) and Responses API (v5)
 * Based on openai SDK patterns
 */

import OpenAI from 'openai';
import {
  type LanguageModelV4,
  type LanguageModelV4CallOptions,
  type LanguageModelV4GenerateResult,
  type LanguageModelV4StreamResult,
  type LanguageModelV4StreamPart,
  type LanguageModelV4Content,
  type LanguageModelV4FinishReason,
  type LanguageModelV4Usage,
  type LanguageModelV4Tool,
  type LanguageModelV4ToolChoice,
  LanguageModelError,
  LanguageModelRateLimitError,
  LanguageModelAuthenticationError,
  LanguageModelContextLengthError,
} from '../types/index.js';

// Type aliases for OpenAI SDK types
type ChatCompletionMessageParam = OpenAI.Chat.ChatCompletionMessageParam;
type ChatCompletionTool = OpenAI.Chat.ChatCompletionTool;
type ChatCompletion = OpenAI.Chat.ChatCompletion;
type ChatCompletionChunk = OpenAI.Chat.ChatCompletionChunk;
type ChatCompletionCreateParams = OpenAI.Chat.ChatCompletionCreateParams;

export interface OpenAIClientOptions {
  apiKey: string;
  baseURL?: string;
  organization?: string;
  project?: string;
  timeout?: number;
  maxRetries?: number;
  defaultHeaders?: Record<string, string>;
}

export class OpenAIChatLanguageModel implements LanguageModelV4 {
  readonly specificationVersion = 'v4' as const;
  readonly provider = 'openai.chat';
  readonly modelId: string;
  readonly supportedUrls = {};

  private client: OpenAI;

  constructor(modelId: string, options: OpenAIClientOptions) {
    this.modelId = modelId;
    this.client = new OpenAI({
      apiKey: options.apiKey,
      baseURL: options.baseURL,
      organization: options.organization,
      project: options.project,
      timeout: options.timeout,
      maxRetries: options.maxRetries,
      defaultHeaders: options.defaultHeaders,
    });
  }

  async doGenerate(options: LanguageModelV4CallOptions): Promise<LanguageModelV4GenerateResult> {
    const openaiOptions = this.mapCallOptions(options);

    try {
      const response = await this.client.chat.completions.create(openaiOptions);
      // Handle both ChatCompletion and Stream response
      if (Symbol.asyncIterator in response) {
        // It's a stream, collect it
        const chunks: ChatCompletionChunk[] = [];
        for await (const chunk of response as AsyncIterable<ChatCompletionChunk>) {
          chunks.push(chunk);
        }
        // Reconstruct the final message from chunks
        const finalResponse = this.reconstructFromChunks(chunks);
        return this.mapGenerateResult(finalResponse);
      }
      return this.mapGenerateResult(response);
    } catch (error) {
      throw this.mapError(error);
    }
  }

  async doStream(options: LanguageModelV4CallOptions): Promise<LanguageModelV4StreamResult> {
    const openaiOptions = this.mapCallOptions(options);
    openaiOptions.stream = true;

    try {
      const stream = await this.client.chat.completions.create(openaiOptions);
      const transformedStream = this.transformStream(stream as AsyncIterable<ChatCompletionChunk>);
      return { stream: transformedStream };
    } catch (error) {
      throw this.mapError(error);
    }
  }

  private reconstructFromChunks(chunks: ChatCompletionChunk[]): ChatCompletion {
    // Simple reconstruction - in practice would need more sophisticated handling
    const lastChunk = chunks[chunks.length - 1];
    const firstChunk = chunks[0];
    return {
      id: firstChunk.id,
      object: 'chat.completion',
      created: firstChunk.created,
      model: firstChunk.model,
      choices: [{
        index: 0,
        message: {
          role: 'assistant',
          content: chunks.map(c => c.choices[0]?.delta?.content).filter(Boolean).join(''),
          tool_calls: lastChunk.choices[0]?.delta?.tool_calls ? this.mergeToolCalls(chunks) : undefined,
        },
        finish_reason: lastChunk.choices[0]?.finish_reason ?? 'stop',
      }],
      usage: lastChunk.usage,
    } as ChatCompletion;
  }

  private mergeToolCalls(chunks: ChatCompletionChunk[]): OpenAI.Chat.ChatCompletionMessageToolCall[] {
    const toolCallsMap = new Map<number, { id: string; name: string; args: string }>();
    for (const chunk of chunks) {
      for (const tc of chunk.choices[0]?.delta?.tool_calls ?? []) {
        const index = tc.index ?? 0;
        if (!toolCallsMap.has(index)) {
          toolCallsMap.set(index, { id: tc.id ?? '', name: tc.function?.name ?? '', args: '' });
        }
        const toolCall = toolCallsMap.get(index)!;
        if (tc.id) toolCall.id = tc.id;
        if (tc.function?.name) toolCall.name = tc.function.name;
        if (tc.function?.arguments) toolCall.args += tc.function.arguments;
      }
    }
    return Array.from(toolCallsMap.entries()).map(([index, tc]) => ({
      index,
      id: tc.id,
      type: 'function' as const,
      function: { name: tc.name, arguments: tc.args },
    }));
  }

  private mapCallOptions(options: LanguageModelV4CallOptions): ChatCompletionCreateParams {
    const messages = this.mapPrompt(options.prompt);
    const tools = options.tools?.map(t => this.mapTool(t)) ?? [];
    const toolChoice = this.mapToolChoice(options.toolChoice);
    const responseFormat = this.mapResponseFormat(options.responseFormat);

    return {
      model: this.modelId,
      messages,
      max_tokens: options.maxOutputTokens,
      max_completion_tokens: options.maxOutputTokens,
      temperature: options.temperature,
      top_p: options.topP,
      stop: options.stopSequences,
      tools: tools.length > 0 ? tools : undefined,
      tool_choice: toolChoice,
      response_format: responseFormat,
      seed: options.seed,
      presence_penalty: options.presencePenalty,
      frequency_penalty: options.frequencyPenalty,
      stream_options: options.includeRawChunks ? { include_usage: true } : undefined,
      user: options.headers?.['x-user-id'],
    };
  }

  private mapPrompt(prompt: LanguageModelV4CallOptions['prompt']): ChatCompletionMessageParam[] {
    if (typeof prompt === 'string') {
      return [{ role: 'user', content: prompt }];
    }

    return prompt.map((msg): ChatCompletionMessageParam => {
      switch (msg.role) {
        case 'system':
          return { role: 'system', content: msg.content };
        case 'user':
          return { role: 'user', content: msg.content };
        case 'assistant':
          return { role: 'assistant', content: msg.content };
        case 'tool':
          return {
            role: 'tool',
            content: msg.content,
            tool_call_id: msg.toolCallId,
          } as ChatCompletionMessageParam;
        default: {
          // Exhaustiveness check - should never reach here
          const _exhaustive: never = msg;
          void _exhaustive;
          return { role: 'user', content: '' } as ChatCompletionMessageParam;
        }
      }
    });
  }

  private mapTool(tool: LanguageModelV4Tool): ChatCompletionTool {
    if (tool.type === 'function') {
      return {
        type: 'function',
        function: {
          name: tool.name,
          description: tool.description,
          parameters: tool.inputSchema,
          strict: true,
        },
      };
    }
    // Provider tools not directly supported in Chat Completions
    return {
      type: 'function',
      function: {
        name: tool.name,
        description: `Provider tool: ${tool.id}`,
        parameters: tool.inputSchema,
        strict: true,
      },
    };
  }

  private mapToolChoice(choice?: LanguageModelV4ToolChoice): OpenAI.Chat.ChatCompletionToolChoiceOption {
    if (!choice) return 'auto';
    if (choice === 'auto') return 'auto';
    if (choice === 'none') return 'none';
    if (choice === 'required') return 'required';
    if (choice.type === 'function') return { type: 'function', function: { name: choice.function.name } };
    if (choice.type === 'provider') return 'auto'; // Fallback
    return 'auto';
  }

  private mapResponseFormat(format?: LanguageModelV4CallOptions['responseFormat']): ChatCompletionCreateParams['response_format'] {
    if (!format) return undefined;
    if (format.type === 'text') return { type: 'text' };
    if (format.type === 'json' && format.schema) {
      return {
        type: 'json_schema',
        json_schema: {
          name: format.name ?? 'structured_output',
          schema: format.schema,
          strict: true,
        },
      };
    }
    return undefined;
  }

  private mapGenerateResult(response: ChatCompletion): LanguageModelV4GenerateResult {
    const choice = response.choices[0];
    const content = this.mapContent(choice.message);
    const finishReason = this.mapFinishReason(choice.finish_reason);
    const usage = this.mapUsage(response.usage);

    return {
      content,
      finishReason,
      usage,
      providerMetadata: {
        openai: {
          id: response.id,
          model: response.model,
          system_fingerprint: response.system_fingerprint,
        },
      },
      request: {},
      response: { headers: {}, body: response },
      warnings: [],
    };
  }

  private mapContent(message: OpenAI.Chat.ChatCompletionMessage): LanguageModelV4Content[] {
    const content: LanguageModelV4Content[] = [];

    if (message.content) {
      content.push({ type: 'text', text: message.content });
    }

    if (message.tool_calls) {
      for (const tc of message.tool_calls) {
        // tc is ChatCompletionMessageToolCall | ChatCompletionMessageCustomToolCall
        // Use type guard to access function property
        if ('function' in tc && tc.function) {
          content.push({
            type: 'tool-call',
            toolCallId: tc.id,
            toolName: tc.function.name,
            input: JSON.parse(tc.function.arguments),
            providerExecuted: false,
            dynamic: false,
          });
        }
      }
    }

    if (message.refusal) {
      content.push({ type: 'text', text: `Refusal: ${message.refusal}` });
    }

    return content.length > 0 ? content : [{ type: 'text', text: '' }];
  }

  private mapFinishReason(reason: string | null): LanguageModelV4FinishReason {
    const mapping: Record<string, LanguageModelV4FinishReason['unified']> = {
      stop: 'stop',
      length: 'length',
      tool_calls: 'tool-calls',
      content_filter: 'content-filter',
      function_call: 'tool-calls',
    };
    return { unified: mapping[reason ?? 'stop'] ?? 'other', raw: reason ?? undefined };
  }

  private mapUsage(usage?: OpenAI.CompletionUsage): LanguageModelV4Usage {
    if (!usage) {
      return { inputTokens: 0, outputTokens: 0, totalTokens: 0 };
    }
    return {
      inputTokens: usage.prompt_tokens,
      outputTokens: usage.completion_tokens,
      totalTokens: usage.total_tokens,
      inputTokensDetails: {
        cacheRead: usage.prompt_tokens_details?.cached_tokens,
        text: usage.prompt_tokens_details?.audio_tokens,
      },
      outputTokensDetails: {
        reasoning: usage.completion_tokens_details?.reasoning_tokens,
        text: usage.completion_tokens_details?.audio_tokens,
      },
    };
  }

  private transformStream(stream: AsyncIterable<ChatCompletionChunk>): ReadableStream<LanguageModelV4StreamPart> {
    let currentTextId = 'text-0';
    let toolCalls: Map<number, { id: string; name: string; args: string }> = new Map();
    let textStarted = false;

    const self = this; // Capture this for use in callback

    return new ReadableStream<LanguageModelV4StreamPart>({
      async start(controller) {
        controller.enqueue({ type: 'stream-start' });

        for await (const chunk of stream) {
          const choice = chunk.choices[0];
          if (!choice) continue;

          // Handle text content
          if (choice.delta.content) {
            if (!textStarted) {
              controller.enqueue({ type: 'text-start', id: currentTextId });
              textStarted = true;
            }
            controller.enqueue({ type: 'text-delta', id: currentTextId, delta: choice.delta.content });
          }

          // Handle tool calls
          if (choice.delta.tool_calls) {
            for (const tc of choice.delta.tool_calls) {
              const index = tc.index ?? 0;
              if (!toolCalls.has(index)) {
                toolCalls.set(index, { id: tc.id ?? '', name: tc.function?.name ?? '', args: '' });
              }
              const toolCall = toolCalls.get(index)!;
              if (tc.id) toolCall.id = tc.id;
              if (tc.function?.name) toolCall.name = tc.function.name;
              if (tc.function?.arguments) toolCall.args += tc.function.arguments;

              // Emit tool input delta
              if (tc.function?.arguments) {
                controller.enqueue({
                  type: 'tool-input-delta',
                  id: `tool-${index}`,
                  toolCallId: toolCall.id,
                  inputTextDelta: tc.function.arguments,
                });
              }
            }
          }

          // Handle finish
          if (choice.finish_reason) {
            if (textStarted) {
              controller.enqueue({ type: 'text-end', id: currentTextId });
              textStarted = false;
            }

            // Emit tool calls if any
            for (const [index, toolCall] of toolCalls) {
              if (toolCall.name) {
                controller.enqueue({
                  type: 'tool-input-end',
                  id: `tool-${index}`,
                  toolCallId: toolCall.id,
                  toolName: toolCall.name,
                  input: JSON.parse(toolCall.args || '{}'),
                });
                controller.enqueue({
                  type: 'tool-call',
                  id: `tool-${index}-call`,
                  toolCallId: toolCall.id,
                  toolName: toolCall.name,
                  input: JSON.parse(toolCall.args || '{}'),
                  providerExecuted: false,
                  dynamic: false,
                });
              }
            }
            toolCalls.clear();

            const finishReason = self.mapFinishReason(choice.finish_reason);
            controller.enqueue({
              type: 'finish',
              finishReason,
              usage: self.mapUsage(chunk.usage ?? undefined),
            });
          }
        }

        // Final finish if not already sent
        if (textStarted) {
          controller.enqueue({ type: 'text-end', id: currentTextId });
        }
        controller.close();
      },
    });
  }

  private mapError(error: unknown): Error {
    if (error instanceof OpenAI.RateLimitError) {
      const retryAfter = error.headers?.get('retry-after');
      return new LanguageModelRateLimitError(this.provider, this.modelId, retryAfter ? parseInt(retryAfter) * 1000 : undefined);
    }
    if (error instanceof OpenAI.AuthenticationError) {
      return new LanguageModelAuthenticationError(this.provider, this.modelId);
    }
    if (error instanceof OpenAI.APIError) {
      if (error.status === 400 && error.message.includes('context length')) {
        return new LanguageModelContextLengthError(this.provider, this.modelId);
      }
      return new LanguageModelError(error.message, this.provider, this.modelId, error.status, undefined);
    }
    return error instanceof Error ? error : new Error(String(error));
  }
}

/**
 * OpenAI Responses API Adapter (v5+)
 * For new features: built-in tools, previous_response_id, etc.
 */
export class OpenAIResponsesLanguageModel implements LanguageModelV4 {
  readonly specificationVersion = 'v4' as const;
  readonly provider = 'openai.responses';
  readonly modelId: string;
  readonly supportedUrls = {};

  private client: OpenAI;

  constructor(modelId: string, options: OpenAIClientOptions) {
    this.modelId = modelId;
    this.client = new OpenAI({
      apiKey: options.apiKey,
      baseURL: options.baseURL,
      organization: options.organization,
      project: options.project,
      timeout: options.timeout,
      maxRetries: options.maxRetries,
      defaultHeaders: options.defaultHeaders,
    });
  }

  async doGenerate(options: LanguageModelV4CallOptions): Promise<LanguageModelV4GenerateResult> {
    try {
      // Use Responses API for new features
      const response = await (this.client as any).responses.create({
        model: this.modelId,
        input: this.mapPromptToResponses(options.prompt),
        tools: options.tools?.map(t => this.mapToolToResponses(t)) ?? [],
        tool_choice: this.mapToolChoice(options.toolChoice),
        max_output_tokens: options.maxOutputTokens,
        temperature: options.temperature,
        top_p: options.topP,
        reasoning: options.reasoning ? { effort: this.mapEffort(options.reasoning) } : undefined,
        text: options.responseFormat?.type === 'json' ? { format: { type: 'json_schema', name: 'output', schema: options.responseFormat.schema, strict: true } } : undefined,
        stream: false,
      });

      return this.mapResponsesResult(response);
    } catch (error) {
      throw this.mapError(error);
    }
  }

  async doStream(options: LanguageModelV4CallOptions): Promise<LanguageModelV4StreamResult> {
    try {
      const stream = await (this.client as any).responses.create({
        model: this.modelId,
        input: this.mapPromptToResponses(options.prompt),
        tools: options.tools?.map(t => this.mapToolToResponses(t)) ?? [],
        tool_choice: this.mapToolChoice(options.toolChoice),
        max_output_tokens: options.maxOutputTokens,
        temperature: options.temperature,
        top_p: options.topP,
        reasoning: options.reasoning ? { effort: this.mapEffort(options.reasoning) } : undefined,
        text: options.responseFormat?.type === 'json' ? { format: { type: 'json_schema', name: 'output', schema: options.responseFormat.schema, strict: true } } : undefined,
        stream: true,
      });

      return { stream: this.transformResponsesStream(stream) };
    } catch (error) {
      throw this.mapError(error);
    }
  }

  private mapPromptToResponses(prompt: LanguageModelV4CallOptions['prompt']): any[] {
    if (typeof prompt === 'string') return [{ role: 'user', content: prompt }];
    return prompt.map(msg => ({
      role: msg.role === 'system' ? 'system' : msg.role === 'assistant' ? 'assistant' : 'user',
      content: msg.content,
    }));
  }

  private mapToolToResponses(tool: LanguageModelV4Tool): any {
    if (tool.type === 'function') {
      return {
        type: 'function',
        name: tool.name,
        description: tool.description,
        parameters: tool.inputSchema,
        strict: true,
      };
    }
    return { type: 'provider', id: tool.id, name: tool.name };
  }

  private mapToolChoice(choice?: LanguageModelV4ToolChoice): any {
    if (!choice) return 'auto';
    if (choice === 'auto') return 'auto';
    if (choice === 'none') return 'none';
    if (choice === 'required') return 'required';
    if (choice.type === 'function') return { type: 'function', function: { name: choice.function.name } };
    return 'auto';
  }

  private mapEffort(reasoning: string): 'low' | 'medium' | 'high' {
    if (reasoning === 'minimal' || reasoning === 'low') return 'low';
    if (reasoning === 'medium') return 'medium';
    return 'high';
  }

  private mapResponsesResult(response: any): LanguageModelV4GenerateResult {
    // Simplified - would need full implementation
    return {
      content: [{ type: 'text', text: response.output_text || '' }],
      finishReason: { unified: 'stop', raw: 'stop' },
      usage: { inputTokens: 0, outputTokens: 0, totalTokens: 0 },
      providerMetadata: { openai: { id: response.id } },
      request: {},
      response: { headers: {}, body: response },
      warnings: [],
    };
  }

  private transformResponsesStream(stream: any): ReadableStream<LanguageModelV4StreamPart> {
    // Simplified - would need full implementation
    return new ReadableStream({
      start(controller) {
        controller.enqueue({ type: 'stream-start' });
        controller.enqueue({ type: 'finish', finishReason: { unified: 'stop', raw: 'stop' }, usage: { inputTokens: 0, outputTokens: 0, totalTokens: 0 } });
        controller.close();
      },
    });
  }

  private mapError(error: unknown): Error {
    // Same as OpenAIChatLanguageModel
    if (error instanceof OpenAI.RateLimitError) {
      const retryAfter = error.headers?.get('retry-after');
      return new LanguageModelRateLimitError(this.provider, this.modelId, retryAfter ? parseInt(retryAfter) * 1000 : undefined);
    }
    if (error instanceof OpenAI.AuthenticationError) {
      return new LanguageModelAuthenticationError(this.provider, this.modelId);
    }
    if (error instanceof OpenAI.APIError) {
      if (error.status === 400 && error.message.includes('context length')) {
        return new LanguageModelContextLengthError(this.provider, this.modelId);
      }
      return new LanguageModelError(error.message, this.provider, this.modelId, error.status, undefined);
    }
    return error instanceof Error ? error : new Error(String(error));
  }
}

/**
 * Factory function to create OpenAI provider
 */
export function createOpenAIProvider(apiKey: string, options?: Partial<OpenAIClientOptions>) {
  return {
    specificationVersion: 'v4' as const,
    languageModel: (modelId: string) => new OpenAIChatLanguageModel(modelId, { apiKey, ...options }),
    responsesModel: (modelId: string) => new OpenAIResponsesLanguageModel(modelId, { apiKey, ...options }),
  };
}