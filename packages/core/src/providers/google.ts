/**
 * Google GenAI Provider Adapter (NEW SDK: @google/genai)
 * Implements LanguageModelV4 interface for Google Gemini API
 * Based on @google/genai SDK (NOT the deprecated @google/generative-ai)
 */

import { GoogleGenAI, type GenerateContentConfig, type Content, type FunctionCallingConfigMode } from '@google/genai';
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

export interface GoogleClientOptions {
  apiKey: string;
  baseURL?: string;
  timeout?: number;
  maxRetries?: number;
  defaultHeaders?: Record<string, string>;
}

export class GoogleLanguageModel implements LanguageModelV4 {
  readonly specificationVersion = 'v4' as const;
  readonly provider = 'google.generative';
  readonly modelId: string;
  readonly supportedUrls = {};

  private client: GoogleGenAI;

  constructor(modelId: string, options: GoogleClientOptions) {
    this.modelId = modelId;
    this.client = new GoogleGenAI({
      apiKey: options.apiKey,
      // baseURL not directly supported in new SDK, use vertexAI for custom endpoints
    });
  }

  async doGenerate(options: LanguageModelV4CallOptions): Promise<LanguageModelV4GenerateResult> {
    const config = this.mapCallOptions(options);

    try {
      const response = await this.client.models.generateContent({
        model: this.modelId,
        contents: config.contents,
        config: config.config,
      });
      return this.mapGenerateResult(response);
    } catch (error) {
      throw this.mapError(error);
    }
  }

  async doStream(options: LanguageModelV4CallOptions): Promise<LanguageModelV4StreamResult> {
    const config = this.mapCallOptions(options);

    try {
      const stream = await this.client.models.generateContentStream({
        model: this.modelId,
        contents: config.contents,
        config: config.config,
      });
      const transformedStream = this.transformStream(stream);
      return { stream: transformedStream };
    } catch (error) {
      throw this.mapError(error);
    }
  }

  private mapCallOptions(options: LanguageModelV4CallOptions): {
    contents: Content[];
    config: GenerateContentConfig;
  } {
    const contents = this.mapPrompt(options.prompt);
    const tools = options.tools?.map(t => this.mapTool(t)) ?? [];
    const toolConfig = this.mapToolConfig(options.toolChoice);

    return {
      contents,
      config: {
        tools: tools.length > 0 ? tools : undefined,
        toolConfig,
        systemInstruction: this.extractSystemPrompt(contents),
        maxOutputTokens: options.maxOutputTokens,
        temperature: options.temperature,
        topP: options.topP,
        topK: options.topK,
        stopSequences: options.stopSequences,
        responseMimeType: options.responseFormat?.type === 'json' ? 'application/json' : undefined,
        responseSchema: options.responseFormat?.type === 'json' ? options.responseFormat.schema : undefined,
        candidateCount: 1,
      },
    };
  }

  private mapPrompt(prompt: LanguageModelV4CallOptions['prompt']): Content[] {
    if (typeof prompt === 'string') {
      return [{ role: 'user', parts: [{ text: prompt }] }];
    }

    return prompt.map((msg): Content => {
      switch (msg.role) {
        case 'system':
          // System prompts handled separately via systemInstruction
          return { role: 'user', parts: [{ text: `System: ${msg.content}` }] };
        case 'user':
          return { role: 'user', parts: [{ text: msg.content }] };
        case 'assistant':
          return { role: 'model', parts: [{ text: msg.content }] };
        case 'tool':
          return {
            role: 'function',
            parts: [{ functionResponse: { name: msg.toolName, response: { content: msg.content } } }],
          };
        default: {
          // Exhaustiveness check - should never reach here
          const _exhaustive: never = msg;
          void _exhaustive;
          return { role: 'user', parts: [{ text: '' }] };
        }
      }
    });
  }

  private extractSystemPrompt(contents: Content[]): string | undefined {
    const systemParts = contents
      .filter(c => c.role === 'user' && c.parts?.[0]?.text?.startsWith('System: '))
      .map(c => c.parts?.[0]?.text?.replace('System: ', '') || '')
      .filter(Boolean);
    return systemParts.length > 0 ? systemParts.join('\n\n') : undefined;
  }

  private mapTool(tool: LanguageModelV4Tool): any {
    if (tool.type === 'function') {
      return {
        functionDeclarations: [{
          name: tool.name,
          description: tool.description,
          parameters: tool.inputSchema,
        }],
      };
    }
    // Provider tools (code execution, google search)
    return {
      [tool.id]: {},
    };
  }

  private mapToolConfig(choice?: LanguageModelV4ToolChoice): GenerateContentConfig['toolConfig'] {
    if (!choice) return { functionCallingConfig: { mode: 'AUTO' as FunctionCallingConfigMode } };
    if (choice === 'auto') return { functionCallingConfig: { mode: 'AUTO' as FunctionCallingConfigMode } };
    if (choice === 'none') return { functionCallingConfig: { mode: 'NONE' as FunctionCallingConfigMode } };
    if (choice === 'required') return { functionCallingConfig: { mode: 'ANY' as FunctionCallingConfigMode } };
    if (choice.type === 'function') return { functionCallingConfig: { mode: 'ANY' as FunctionCallingConfigMode, allowedFunctionNames: [choice.function.name] } };
    return { functionCallingConfig: { mode: 'AUTO' as FunctionCallingConfigMode } };
  }

  private mapGenerateResult(response: any): LanguageModelV4GenerateResult {
    const candidate = response.candidates?.[0];
    if (!candidate) {
      return {
        content: [],
        finishReason: { unified: 'error', raw: 'no_candidate' },
        usage: { inputTokens: 0, outputTokens: 0, totalTokens: 0 },
        providerMetadata: {},
        request: {},
        response: { headers: {}, body: response },
        warnings: [],
      };
    }

    const content = this.mapContent(candidate.content?.parts || []);
    const finishReason = this.mapFinishReason(candidate.finishReason);
    const usage = this.mapUsage(response.usageMetadata);

    return {
      content,
      finishReason,
      usage,
      providerMetadata: {
        google: {
          modelVersion: response.modelVersion,
        },
      },
      request: {},
      response: { headers: {}, body: response },
      warnings: [],
    };
  }

  private mapContent(parts: any[]): LanguageModelV4Content[] {
    return parts.map(part => {
      if (part.text) return { type: 'text', text: part.text };
      if (part.functionCall) {
        return {
          type: 'tool-call',
          toolCallId: part.functionCall.id || `call-${Date.now()}`,
          toolName: part.functionCall.name,
          input: part.functionCall.args,
          providerExecuted: false,
          dynamic: false,
        };
      }
      if (part.functionResponse) {
        return {
          type: 'tool-result',
          toolCallId: part.functionResponse.id || '',
          toolName: part.functionResponse.name,
          result: part.functionResponse.response,
          isError: false,
          preliminary: false,
          dynamic: false,
        };
      }
      if (part.inlineData) {
        return { type: 'text', text: '[Image]' };
      }
      return { type: 'text', text: JSON.stringify(part) };
    });
  }

  private mapFinishReason(reason?: string): LanguageModelV4FinishReason {
    const mapping: Record<string, LanguageModelV4FinishReason['unified']> = {
      STOP: 'stop',
      MAX_TOKENS: 'length',
      SAFETY: 'content-filter',
      RECITATION: 'content-filter',
      FINISH_REASON_UNSPECIFIED: 'other',
    };
    return { unified: mapping[reason ?? 'STOP'] ?? 'other', raw: reason };
  }

  private mapUsage(metadata?: any): LanguageModelV4Usage {
    if (!metadata) {
      return { inputTokens: 0, outputTokens: 0, totalTokens: 0 };
    }
    return {
      inputTokens: metadata.promptTokenCount || 0,
      outputTokens: metadata.candidatesTokenCount || 0,
      totalTokens: metadata.totalTokenCount || 0,
      inputTokensDetails: {
        cacheRead: metadata.cachedContentTokenCount,
      },
    };
  }

  private transformStream(stream: AsyncIterable<any>): ReadableStream<LanguageModelV4StreamPart> {
    let textId = 'text-0';
    let textStarted = false;
    let toolCalls: Map<string, { name: string; args: string }> = new Map();

    return new ReadableStream<LanguageModelV4StreamPart>({
      async start(controller) {
        controller.enqueue({ type: 'stream-start' });

        for await (const chunk of stream) {
          const parts = chunk.candidates?.[0]?.content?.parts || [];
          for (const part of parts) {
            if (part.text) {
              if (!textStarted) {
                controller.enqueue({ type: 'text-start', id: textId });
                textStarted = true;
              }
              controller.enqueue({ type: 'text-delta', id: textId, delta: part.text });
            }
            if (part.functionCall) {
              const callId = part.functionCall.id || `call-${Date.now()}`;
              const existing = toolCalls.get(callId) || { name: part.functionCall.name, args: '' };
              existing.args += JSON.stringify(part.functionCall.args);
              toolCalls.set(callId, existing);

              controller.enqueue({
                type: 'tool-input-delta',
                id: `tool-${callId}`,
                toolCallId: callId,
                inputTextDelta: JSON.stringify(part.functionCall.args),
              });
            }
          }
        }

        if (textStarted) {
          controller.enqueue({ type: 'text-end', id: textId });
        }

        for (const [callId, toolCall] of toolCalls) {
          controller.enqueue({
            type: 'tool-input-end',
            id: `tool-${callId}`,
            toolCallId: callId,
            toolName: toolCall.name,
            input: JSON.parse(toolCall.args || '{}'),
          });
          controller.enqueue({
            type: 'tool-call',
            id: `tool-${callId}-call`,
            toolCallId: callId,
            toolName: toolCall.name,
            input: JSON.parse(toolCall.args || '{}'),
            providerExecuted: false,
            dynamic: false,
          });
        }

        controller.enqueue({
          type: 'finish',
          finishReason: { unified: 'stop', raw: 'stop' },
          usage: { inputTokens: 0, outputTokens: 0, totalTokens: 0 },
        });
        controller.close();
      },
    });
  }

  private mapError(error: unknown): Error {
    // Google GenAI errors need to be handled based on actual error types
    if (error instanceof Error) {
      const message = error.message;
      if (message.includes('429') || message.includes('rate limit') || message.includes('quota')) {
        return new LanguageModelRateLimitError(this.provider, this.modelId);
      }
      if (message.includes('401') || message.includes('API key') || message.includes('unauthorized')) {
        return new LanguageModelAuthenticationError(this.provider, this.modelId);
      }
      if (message.includes('context length') || message.includes('token limit')) {
        return new LanguageModelContextLengthError(this.provider, this.modelId);
      }
      return new LanguageModelError(message, this.provider, this.modelId);
    }
    return new LanguageModelError(String(error), this.provider, this.modelId);
  }
}

/**
 * Factory function to create Google provider
 */
export function createGoogleProvider(apiKey: string, options?: Partial<GoogleClientOptions>) {
  return {
    specificationVersion: 'v4' as const,
    languageModel: (modelId: string) => new GoogleLanguageModel(modelId, { apiKey, ...options }),
  };
}