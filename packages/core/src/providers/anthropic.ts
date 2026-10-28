/**
 * Anthropic Provider Adapter
 * Implements LanguageModelV4 interface for Anthropic Messages API
 * Based on @anthropic-ai/sdk patterns
 */

import Anthropic from '@anthropic-ai/sdk';
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

export interface AnthropicClientOptions {
  apiKey: string;
  baseURL?: string;
  timeout?: number;
  maxRetries?: number;
  defaultHeaders?: Record<string, string>;
}

export class AnthropicLanguageModel implements LanguageModelV4 {
  readonly specificationVersion = 'v4' as const;
  readonly provider = 'anthropic.messages';
  readonly modelId: string;
  readonly supportedUrls = {};

  private client: Anthropic;

  constructor(modelId: string, options: AnthropicClientOptions) {
    this.modelId = modelId;
    this.client = new Anthropic({
      apiKey: options.apiKey,
      baseURL: options.baseURL,
      timeout: options.timeout,
      maxRetries: options.maxRetries,
      defaultHeaders: options.defaultHeaders,
    });
  }

  async doGenerate(options: LanguageModelV4CallOptions): Promise<LanguageModelV4GenerateResult> {
    const anthropicOptions = this.mapCallOptions(options);

    try {
      const response = await this.client.messages.create(anthropicOptions);
      // The SDK returns either a Message or a Stream depending on stream option
      // Since we don't pass stream: true, it should be a Message
      return this.mapGenerateResult(response as Anthropic.Message);
    } catch (error) {
      throw this.mapError(error);
    }
  }

  async doStream(options: LanguageModelV4CallOptions): Promise<LanguageModelV4StreamResult> {
    const anthropicOptions = this.mapCallOptions(options);

    try {
      const stream = this.client.messages.stream(anthropicOptions);
      const transformedStream = this.transformStream(stream);
      return { stream: transformedStream };
    } catch (error) {
      throw this.mapError(error);
    }
  }

  private mapCallOptions(options: LanguageModelV4CallOptions): Anthropic.MessageCreateParams {
    const messages = this.mapPrompt(options.prompt);
    const system = this.extractSystemPrompt(messages);
    const filteredMessages = messages.filter(m => m.role !== 'system');

    const tools = options.tools?.map(t => this.mapTool(t)) ?? [];
    const toolChoice = this.mapToolChoice(options.toolChoice);

    // Map reasoning/effort
    let thinking: Anthropic.ThinkingConfigParam | undefined;
    if (options.reasoning && options.reasoning !== 'provider-default' && options.reasoning !== 'none') {
      thinking = { type: 'adaptive', display: 'summarized' };
    } else if (options.reasoning === 'none') {
      thinking = { type: 'disabled' };
    }

    const outputConfig = this.mapOutputConfig(options.responseFormat, options.reasoning);

    return {
      model: this.modelId,
      max_tokens: options.maxOutputTokens ?? 4096,
      messages: filteredMessages,
      system: system,
      thinking,
      tool_choice: toolChoice,
      tools: tools.length > 0 ? tools : undefined,
      output_config: outputConfig,
      stop_sequences: options.stopSequences,
      // temperature, top_p, top_k deprecated on Opus 4.7+
      metadata: options.headers ? { user_id: 'uaip' } : undefined,
    };
  }

  private mapPrompt(prompt: LanguageModelV4CallOptions['prompt']): Anthropic.MessageParam[] {
    if (typeof prompt === 'string') {
      return [{ role: 'user', content: prompt }];
    }

    return prompt.map((msg): Anthropic.MessageParam => {
      switch (msg.role) {
        case 'system':
          return { role: 'system', content: msg.content };
        case 'user':
          return { role: 'user', content: msg.content };
        case 'assistant':
          return { role: 'assistant', content: msg.content };
        case 'tool':
          return {
            role: 'user',
            content: [
              {
                type: 'tool_result',
                tool_use_id: msg.toolCallId,
                content: msg.content,
              },
            ],
          } as Anthropic.MessageParam;
        default: {
          // Exhaustiveness check - should never reach here
          const _exhaustive: never = msg;
          void _exhaustive;
          return { role: 'user', content: '' } as Anthropic.MessageParam;
        }
      }
    });
  }

  private extractSystemPrompt(messages: Anthropic.MessageParam[]): string | Anthropic.TextBlockParam[] | undefined {
    const systemMessages = messages.filter(m => m.role === 'system');
    if (systemMessages.length === 0) return undefined;
    if (systemMessages.length === 1) {
      const content = systemMessages[0].content;
      return typeof content === 'string' ? content : content;
    }
    // Multiple system messages - combine with cache_control
    return systemMessages.map(m => ({
      type: 'text' as const,
      text: typeof m.content === 'string' ? m.content : '',
      cache_control: { type: 'ephemeral' as const },
    }));
  }

  private mapTool(tool: LanguageModelV4Tool): Anthropic.Tool {
    if (tool.type === 'function') {
      return {
        name: tool.name,
        description: tool.description,
        input_schema: {
          type: 'object',
          ...tool.inputSchema,
        } as Anthropic.Tool.InputSchema,
        strict: true,
      };
    }
    // Provider tools - pass through as-is
    return tool as unknown as Anthropic.Tool;
  }

  private mapToolChoice(choice?: LanguageModelV4ToolChoice): Anthropic.ToolChoice {
    if (!choice) return { type: 'auto' };
    if (choice === 'auto') return { type: 'auto' };
    if (choice === 'none') return { type: 'none' };
    if (choice === 'required') return { type: 'any' };
    if (choice.type === 'function') return { type: 'tool', name: choice.function.name };
    if (choice.type === 'provider') return { type: 'tool', name: choice.id };
    return { type: 'auto' };
  }

  private mapOutputConfig(
    responseFormat?: LanguageModelV4CallOptions['responseFormat'],
    reasoning?: LanguageModelV4CallOptions['reasoning']
  ): Anthropic.OutputConfig | undefined {
    if (responseFormat?.type === 'json' && responseFormat.schema) {
      // Anthropic's output config expects schema directly on format with type 'json_schema'
      return {
        format: {
          type: 'json_schema' as const,
          schema: responseFormat.schema,
        },
      };
    }
    if (reasoning && reasoning !== 'provider-default' && reasoning !== 'none') {
      return { effort: this.mapEffort(reasoning) };
    }
    return undefined;
  }

  private mapEffort(reasoning: string): 'low' | 'medium' | 'high' | 'xhigh' | 'max' {
    switch (reasoning) {
      case 'minimal': return 'low';
      case 'low': return 'low';
      case 'medium': return 'medium';
      case 'high': return 'high';
      case 'xhigh': return 'xhigh';
      case 'max': return 'max';
      default: return 'high';
    }
  }

  private mapGenerateResult(response: Anthropic.Message): LanguageModelV4GenerateResult {
    const content = this.mapContent(response.content);
    const finishReason = this.mapFinishReason(response.stop_reason);
    const usage = this.mapUsage(response.usage);

    return {
      content,
      finishReason,
      usage,
      providerMetadata: {
        anthropic: {
          id: response.id,
          model: response.model,
          stop_sequence: response.stop_sequence,
        },
      },
      request: {},
      response: { headers: {}, body: response },
      warnings: [],
    };
  }

  private mapContent(blocks: Anthropic.ContentBlock[]): LanguageModelV4Content[] {
    return blocks.map(block => {
      switch (block.type) {
        case 'text':
          return { type: 'text', text: block.text };
        case 'thinking':
          return { type: 'reasoning', reasoning: block.thinking, details: block.signature };
        case 'redacted_thinking':
          return { type: 'reasoning', reasoning: '[redacted]', details: block.data };
        case 'tool_use':
          return {
            type: 'tool-call',
            toolCallId: block.id,
            toolName: block.name,
            input: block.input,
            providerExecuted: false,
            dynamic: false,
          };
        case 'server_tool_use':
          return {
            type: 'tool-call',
            toolCallId: block.id,
            toolName: block.name,
            input: block.input,
            providerExecuted: true,
            dynamic: false,
          };
        case 'web_search_tool_result':
        case 'web_fetch_tool_result':
        case 'code_execution_tool_result':
        case 'bash_code_execution_tool_result':
        case 'text_editor_code_execution_tool_result':
        case 'tool_search_tool_result':
        case 'container_upload':
          // These tool result blocks have different properties in the SDK
          return {
            type: 'tool-result',
            toolCallId: ('id' in block && typeof block.id === 'string') ? block.id : `call-${Date.now()}`,
            toolName: ('name' in block && typeof block.name === 'string') ? block.name : 'unknown',
            result: 'content' in block ? block.content : null,
            isError: ('is_error' in block && typeof block.is_error === 'boolean') ? block.is_error : false,
            preliminary: false,
            dynamic: false,
          };
        default:
          return { type: 'text', text: JSON.stringify(block) };
      }
    });
  }

  private mapFinishReason(reason: Anthropic.StopReason | null): LanguageModelV4FinishReason {
    const mapping: Record<string, LanguageModelV4FinishReason['unified']> = {
      end_turn: 'stop',
      max_tokens: 'length',
      tool_use: 'tool-calls',
      pause_turn: 'interrupted',
      refusal: 'content-filter',
      stop_sequence: 'stop',
    };
    return { unified: mapping[reason ?? 'stop'] ?? 'other', raw: reason ?? undefined };
  }

  private mapUsage(usage: Anthropic.Usage): LanguageModelV4Usage {
    return {
      inputTokens: usage.input_tokens,
      outputTokens: usage.output_tokens,
      totalTokens: usage.input_tokens + usage.output_tokens,
      inputTokensDetails: {
        cacheRead: usage.cache_read_input_tokens ?? undefined,
        cacheWrite: usage.cache_creation_input_tokens ?? undefined,
      },
    };
  }

  private transformStream(stream: ReturnType<Anthropic['messages']['stream']>): ReadableStream<LanguageModelV4StreamPart> {
    // Track the current content block info for use in delta/stop events
    let currentTextBlockId: string | null = null;
    let currentToolBlockId: string | null = null;
    let currentThinkingBlockId: string | null = null;
    let toolInputBuffer = '';
    // Store the current tool block's ID and name from start event
    let currentToolCallId: string | null = null;
    let currentToolName: string | null = null;

    const self = this; // Capture this for use in callback

    return new ReadableStream<LanguageModelV4StreamPart>({
      async start(controller) {
        controller.enqueue({ type: 'stream-start' });

        for await (const event of stream) {
          switch (event.type) {
            case 'content_block_start': {
              if (event.content_block.type === 'text') {
                currentTextBlockId = `text-${event.index}`;
                controller.enqueue({ type: 'text-start', id: currentTextBlockId });
              } else if (event.content_block.type === 'tool_use') {
                currentToolBlockId = `tool-${event.index}`;
                currentToolCallId = event.content_block.id;
                currentToolName = event.content_block.name;
                toolInputBuffer = '';
                controller.enqueue({
                  type: 'tool-input-start',
                  id: currentToolBlockId,
                  toolCallId: event.content_block.id,
                  toolName: event.content_block.name,
                });
              } else if (event.content_block.type === 'thinking') {
                currentThinkingBlockId = `thinking-${event.index}`;
                controller.enqueue({ type: 'reasoning-start', id: currentThinkingBlockId });
              }
              break;
            }

            case 'content_block_delta': {
              if (event.delta.type === 'text_delta' && currentTextBlockId) {
                controller.enqueue({ type: 'text-delta', id: currentTextBlockId, delta: event.delta.text });
              } else if (event.delta.type === 'input_json_delta' && currentToolBlockId) {
                toolInputBuffer += event.delta.partial_json;
                controller.enqueue({
                  type: 'tool-input-delta',
                  id: currentToolBlockId,
                  toolCallId: currentToolCallId ?? '',
                  inputTextDelta: event.delta.partial_json,
                });
              } else if (event.delta.type === 'thinking_delta' && currentThinkingBlockId) {
                controller.enqueue({ type: 'reasoning-delta', id: currentThinkingBlockId, delta: event.delta.thinking });
              }
              break;
            }

            case 'content_block_stop': {
              if (currentTextBlockId) {
                controller.enqueue({ type: 'text-end', id: currentTextBlockId });
                currentTextBlockId = null;
              }
              if (currentToolBlockId) {
                controller.enqueue({
                  type: 'tool-input-end',
                  id: currentToolBlockId,
                  toolCallId: currentToolCallId ?? '',
                  toolName: currentToolName ?? '',
                  input: JSON.parse(toolInputBuffer || '{}'),
                });
                currentToolBlockId = null;
                currentToolCallId = null;
                currentToolName = null;
                toolInputBuffer = '';
              }
              if (currentThinkingBlockId) {
                controller.enqueue({ type: 'reasoning-end', id: currentThinkingBlockId });
                currentThinkingBlockId = null;
              }
              break;
            }

            case 'message_delta': {
              if (event.delta.stop_reason) {
                const finishReason = self.mapFinishReason(event.delta.stop_reason);
                // Map MessageDeltaUsage to Usage by providing defaults for missing fields
                const usage = self.mapUsage({
                  input_tokens: event.usage.input_tokens,
                  output_tokens: event.usage.output_tokens,
                  cache_creation_input_tokens: event.usage.cache_creation_input_tokens,
                  cache_read_input_tokens: event.usage.cache_read_input_tokens,
                  // These fields might not exist in MessageDeltaUsage, provide defaults
                } as Anthropic.Usage);
                controller.enqueue({
                  type: 'finish',
                  finishReason,
                  usage,
                });
              }
              break;
            }
          }
        }

        // Ensure we have a finish event - use the stream's finalMessage method
        try {
          const finalMessage = await (stream as any).finalMessage();
          if (finalMessage) {
            const finishReason = self.mapFinishReason(finalMessage.stop_reason);
            controller.enqueue({
              type: 'finish',
              finishReason,
              usage: self.mapUsage(finalMessage.usage),
            });
          }
        } catch {
          // finalMessage might not be available, ignore
        }

        controller.close();
      },
    });
  }

  private mapError(error: unknown): Error {
    if (error instanceof Anthropic.RateLimitError) {
      // retryAfterMs may not exist, use a default
      return new LanguageModelRateLimitError(this.provider, this.modelId);
    }
    if (error instanceof Anthropic.AuthenticationError) {
      return new LanguageModelAuthenticationError(this.provider, this.modelId);
    }
    if (error instanceof Anthropic.APIError) {
      if (error.status === 400 && error.message.includes('context length')) {
        return new LanguageModelContextLengthError(this.provider, this.modelId);
      }
      return new LanguageModelError(
        error.message,
        this.provider,
        this.modelId,
        error.status,
        undefined // error.body not available in newer SDK
      );
    }
    return error instanceof Error ? error : new Error(String(error));
  }
}

/**
 * Factory function to create Anthropic provider
 */
export function createAnthropicProvider(apiKey: string, options?: Partial<AnthropicClientOptions>) {
  return {
    specificationVersion: 'v4' as const,
    languageModel: (modelId: string) => new AnthropicLanguageModel(modelId, { apiKey, ...options }),
  };
}