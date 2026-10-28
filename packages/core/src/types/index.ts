/**
 * UAIP Core Types
 * Based on Vercel AI SDK v5 LanguageModelV4 interface
 * All providers must implement these interfaces exactly
 */

// JSON Schema type - zod doesn't export JSONSchema directly, use a generic object type
export type JSONSchema = Record<string, unknown>;

/**
 * Unified finish reason across all providers
 */
export type LanguageModelV4FinishReason = {
  unified: 'stop' | 'length' | 'content-filter' | 'tool-calls' | 'error' | 'interrupted' | 'other';
  raw: string | undefined;
};

/**
 * Token usage details
 */
export interface LanguageModelV4Usage {
  inputTokens: number;
  outputTokens: number;
  totalTokens: number;
  inputTokensDetails?: {
    text?: number;
    cacheRead?: number;
    cacheWrite?: number;
    reasoning?: number;
  };
  outputTokensDetails?: {
    text?: number;
    reasoning?: number;
  };
}

/**
 * Content types in model responses
 */
export type LanguageModelV4Content =
  | { type: 'text'; text: string }
  | { type: 'reasoning'; reasoning: string; details?: unknown }
  | { type: 'tool-call'; toolCallId: string; toolName: string; input: unknown; providerExecuted: boolean; dynamic: boolean }
  | { type: 'tool-result'; toolCallId: string; toolName: string; result: unknown; isError: boolean; preliminary: boolean; dynamic: boolean }
  | { type: 'source'; sourceId: string; url: string; title?: string; providerMetadata?: Record<string, unknown> }
  | { type: 'file'; fileId: string; filename: string; mimeType: string; size: number }
  | { type: 'custom'; [key: string]: unknown };

/**
 * Stream parts (discriminated union)
 * Matches Vercel AI SDK v5 stream part types
 */
export type LanguageModelV4StreamPart =
  | { type: 'text-start'; id: string }
  | { type: 'text-delta'; id: string; delta: string }
  | { type: 'text-end'; id: string }
  | { type: 'reasoning-start'; id: string }
  | { type: 'reasoning-delta'; id: string; delta: string }
  | { type: 'reasoning-end'; id: string }
  | { type: 'tool-input-start'; id: string; toolCallId: string; toolName: string }
  | { type: 'tool-input-delta'; id: string; toolCallId: string; inputTextDelta: string }
  | { type: 'tool-input-end'; id: string; toolCallId: string; toolName: string; input: unknown }
  | { type: 'tool-call'; id: string; toolCallId: string; toolName: string; input: unknown; providerExecuted: boolean; dynamic: boolean }
  | { type: 'tool-result'; id: string; toolCallId: string; toolName: string; result: unknown; isError: boolean; preliminary: boolean; dynamic: boolean }
  | { type: 'source'; id: string; sourceId: string; url: string; title?: string; providerMetadata?: Record<string, unknown> }
  | { type: 'file'; id: string; fileId: string; filename: string; mimeType: string; size: number }
  | { type: 'stream-start' }
  | { type: 'finish'; finishReason: LanguageModelV4FinishReason; usage: LanguageModelV4Usage }
  | { type: 'error'; error: Error }
  | { type: 'raw'; provider: string; data: unknown };

/**
 * Tool definitions
 */
export interface LanguageModelV4FunctionTool {
  type: 'function';
  name: string;
  description?: string;
  inputSchema: JSONSchema;
}

export interface LanguageModelV4ProviderTool {
  type: 'provider';
  id: string;
  name: string;
  inputSchema: JSONSchema;
}

export type LanguageModelV4Tool = LanguageModelV4FunctionTool | LanguageModelV4ProviderTool;

export type LanguageModelV4ToolChoice =
  | 'auto'
  | 'none'
  | 'required'
  | { type: 'function'; function: { name: string } }
  | { type: 'provider'; id: string };

/**
 * Call options passed to providers
 */
export interface LanguageModelV4CallOptions {
  prompt: LanguageModelV4Prompt;
  maxOutputTokens?: number;
  temperature?: number;
  topP?: number;
  topK?: number;
  presencePenalty?: number;
  frequencyPenalty?: number;
  stopSequences?: string[];
  responseFormat?:
    | { type: 'text' }
    | { type: 'json'; schema?: JSONSchema; name?: string; description?: string };
  seed?: number;
  tools?: LanguageModelV4Tool[];
  toolChoice?: LanguageModelV4ToolChoice;
  includeRawChunks?: boolean;
  abortSignal?: AbortSignal;
  headers?: Record<string, string>;
  reasoning?: 'provider-default' | 'none' | 'minimal' | 'low' | 'medium' | 'high' | 'xhigh' | 'max';
  providerOptions?: Record<string, unknown>;
}

/**
 * Prompt format (standardized across providers)
 */
export type LanguageModelV4Prompt =
  | string
  | Array<
      | { role: 'system'; content: string }
      | { role: 'user'; content: string }
      | { role: 'assistant'; content: string }
      | { role: 'tool'; content: string; toolCallId: string; toolName: string }
    >;

/**
 * Generate result (non-streaming)
 */
export interface LanguageModelV4GenerateResult {
  content: LanguageModelV4Content[];
  structuredOutputText?: string;
  finishReason: LanguageModelV4FinishReason;
  usage: LanguageModelV4Usage;
  providerMetadata?: Record<string, unknown>;
  request?: { body?: unknown };
  response?: {
    headers?: Record<string, string>;
    body?: unknown;
  };
  warnings: SharedV4Warning[];
}

/**
 * Stream result
 */
export interface LanguageModelV4StreamResult {
  stream: ReadableStream<LanguageModelV4StreamPart>;
  request?: { body?: unknown };
  response?: { headers?: Record<string, string> };
}

/**
 * Unified language model interface - ALL providers must implement this
 */
export interface LanguageModelV4 {
  readonly specificationVersion: 'v4';
  readonly provider: string;
  readonly modelId: string;
  readonly supportedUrls: Record<string, RegExp[]>;

  doGenerate(options: LanguageModelV4CallOptions): Promise<LanguageModelV4GenerateResult>;
  doStream(options: LanguageModelV4CallOptions): Promise<LanguageModelV4StreamResult>;
}

/**
 * Provider factory interface
 */
export interface ProviderV4 {
  readonly specificationVersion: 'v4';
  languageModel(modelId: string): LanguageModelV4;
  embeddingModel?(modelId: string): EmbeddingModelV4;
  imageModel?(modelId: string): ImageModelV4;
  transcriptionModel?(modelId: string): TranscriptionModelV4;
  speechModel?(modelId: string): SpeechModelV4;
  rerankingModel?(modelId: string): RerankingModelV4;
  files?(): FilesV4;
  skills?(): SkillsV4;
}

/**
 * Additional model types (optional)
 */
export interface EmbeddingModelV4 {
  readonly provider: string;
  readonly modelId: string;
  doEmbed(values: string[], options?: { abortSignal?: AbortSignal }): Promise<number[][]>;
}

export interface ImageModelV4 {
  readonly provider: string;
  readonly modelId: string;
  doGenerate(prompt: string, options?: ImageGenerateOptions): Promise<ImageGenerateResult>;
}

export interface ImageGenerateOptions {
  n?: number;
  size?: string;
  quality?: 'standard' | 'hd';
  style?: 'vivid' | 'natural';
  responseFormat?: 'url' | 'b64_json';
  abortSignal?: AbortSignal;
}

export interface ImageGenerateResult {
  images: Array<{ url?: string; b64_json?: string }>;
  usage?: { totalTokens: number };
}

export interface TranscriptionModelV4 {
  readonly provider: string;
  readonly modelId: string;
  doTranscribe(audio: Uint8Array, options?: TranscriptionOptions): Promise<TranscriptionResult>;
}

export interface TranscriptionOptions {
  language?: string;
  prompt?: string;
  responseFormat?: 'json' | 'text' | 'srt' | 'verbose_json' | 'vtt';
  abortSignal?: AbortSignal;
}

export interface TranscriptionResult {
  text: string;
  segments?: Array<{ start: number; end: number; text: string }>;
  language?: string;
}

export interface SpeechModelV4 {
  readonly provider: string;
  readonly modelId: string;
  doGenerate(text: string, options?: SpeechOptions): Promise<SpeechResult>;
}

export interface SpeechOptions {
  voice?: string;
  format?: 'mp3' | 'opus' | 'aac' | 'flac' | 'wav' | 'pcm';
  speed?: number;
  abortSignal?: AbortSignal;
}

export interface SpeechResult {
  audio: Uint8Array;
  format: string;
}

export interface RerankingModelV4 {
  readonly provider: string;
  readonly modelId: string;
  doRerank(query: string, documents: string[], options?: RerankOptions): Promise<RerankResult>;
}

export interface RerankOptions {
  topK?: number;
  abortSignal?: AbortSignal;
}

export interface RerankResult {
  results: Array<{ index: number; score: number; document: string }>;
}

export interface FilesV4 {
  upload(file: File | Blob | Uint8Array, options?: { purpose?: string }): Promise<FileObject>;
  list(purpose?: string): Promise<FileObject[]>;
  delete(fileId: string): Promise<void>;
  download(fileId: string): Promise<Uint8Array>;
}

export interface FileObject {
  id: string;
  filename: string;
  bytes: number;
  createdAt: number;
  purpose: string;
}

export interface SkillsV4 {
  list(): Promise<SkillObject[]>;
  get(skillId: string): Promise<SkillObject>;
  create(skill: Omit<SkillObject, 'id' | 'createdAt'>): Promise<SkillObject>;
  update(skillId: string, skill: Partial<SkillObject>): Promise<SkillObject>;
  delete(skillId: string): Promise<void>;
}

export interface SkillObject {
  id: string;
  name: string;
  description: string;
  createdAt: number;
}

/**
 * Middleware system (composable)
 */
export interface LanguageModelMiddleware {
  transformParams?(params: LanguageModelV4CallOptions): LanguageModelV4CallOptions;
  wrapGenerate?(
    doGenerate: LanguageModelV4['doGenerate']
  ): LanguageModelV4['doGenerate'];
  wrapStream?(doStream: LanguageModelV4['doStream']): LanguageModelV4['doStream'];
}

/**
 * Wrap a language model with middleware
 */
export function wrapLanguageModel(options: {
  model: LanguageModelV4;
  middleware: LanguageModelMiddleware;
}): LanguageModelV4 {
  const { model, middleware } = options;

  return {
    ...model,
    specificationVersion: model.specificationVersion,
    provider: model.provider,
    modelId: model.modelId,
    supportedUrls: model.supportedUrls,

    async doGenerate(options: LanguageModelV4CallOptions): Promise<LanguageModelV4GenerateResult> {
      const transformedParams = middleware.transformParams ? middleware.transformParams(options) : options;
      const doGenerate = middleware.wrapGenerate
        ? middleware.wrapGenerate(model.doGenerate.bind(model))
        : model.doGenerate.bind(model);
      return doGenerate(transformedParams);
    },

    async doStream(options: LanguageModelV4CallOptions): Promise<LanguageModelV4StreamResult> {
      const transformedParams = middleware.transformParams ? middleware.transformParams(options) : options;
      const doStream = middleware.wrapStream
        ? middleware.wrapStream(model.doStream.bind(model))
        : model.doStream.bind(model);
      return doStream(transformedParams);
    },
  };
}

export interface SharedV4Warning {
  type: string;
  message: string;
}

export interface SharedV4ProviderMetadata {
  [provider: string]: unknown;
}

export interface SharedV4Headers {
  [key: string]: string;
}

export interface LanguageModelV4ResponseMetadata {
  id: string;
  timestamp: Date;
  modelId: string;
}

/**
 * Error types for provider errors
 */
export class LanguageModelError extends Error {
  constructor(
    message: string,
    public readonly provider: string,
    public readonly modelId: string,
    public readonly statusCode?: number,
    public readonly body?: unknown
  ) {
    super(message);
    this.name = 'LanguageModelError';
  }
}

export class LanguageModelRateLimitError extends LanguageModelError {
  constructor(
    provider: string,
    modelId: string,
    public readonly retryAfterMs?: number
  ) {
    super('Rate limit exceeded', provider, modelId, 429);
    this.name = 'LanguageModelRateLimitError';
  }
}

export class LanguageModelAuthenticationError extends LanguageModelError {
  constructor(provider: string, modelId: string) {
    super('Authentication failed', provider, modelId, 401);
    this.name = 'LanguageModelAuthenticationError';
  }
}

export class LanguageModelContextLengthError extends LanguageModelError {
  constructor(provider: string, modelId: string) {
    super('Context length exceeded', provider, modelId, 400);
    this.name = 'LanguageModelContextLengthError';
  }
}

/**
 * Model capabilities
 */
export interface ModelCapability {
  tools: boolean;
  streaming: boolean;
  structuredOutput: boolean;
  vision: boolean;
  audio: boolean;
  reasoning: boolean;
  maxContextTokens: number;
  maxOutputTokens: number;
}

export interface ModelCandidate {
  provider: string;
  modelId: string;
  displayName: string;
  capabilities: ModelCapability;
  estimatedCostPer1kTokens: { input: number; output: number };
  avgLatencyMs: number;
  health: 'healthy' | 'degraded' | 'down';
}

export interface RoutingContext {
  prompt: LanguageModelV4Prompt;
  tools?: LanguageModelV4Tool[];
  requiredCapabilities?: Partial<ModelCapability>;
  maxCost?: number;
  maxLatencyMs?: number;
  userTier?: 'free' | 'pro' | 'enterprise';
}

export interface RoutingStrategy {
  name: 'fallback' | 'cost-optimize' | 'latency-optimize' | 'round-robin' | 'priority';
  decide(models: ModelCandidate[], context: RoutingContext): ModelCandidate;
}