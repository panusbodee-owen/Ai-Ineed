/**
 * @uaip/core - Universal AI Platform Core SDK
 * Unified interface for all AI providers with smart routing
 */

// Types
export * from './types/index.js';

// Providers
export {
  AnthropicLanguageModel,
  createAnthropicProvider,
  type AnthropicClientOptions,
} from './providers/anthropic.js';

export {
  OpenAIChatLanguageModel,
  OpenAIResponsesLanguageModel,
  createOpenAIProvider,
  type OpenAIClientOptions,
} from './providers/openai.js';

export {
  GoogleLanguageModel,
  createGoogleProvider,
  type GoogleClientOptions,
} from './providers/google.js';

export {
  OllamaLanguageModel,
  createOllamaProvider,
  listOllamaModels,
  type OllamaClientOptions,
} from './providers/ollama.js';

// Routing
export {
  SmartRouter,
  ModelRegistry,
  type ModelRegistryEntry,
  routingStrategies,
  type RoutingStrategy,
  createDefaultRouter,
} from './routing/smart-router.js';

// Middleware
export {
  wrapLanguageModel,
  type LanguageModelMiddleware,
} from './types/index.js';

// Convenience function to create a configured router from environment variables
import type { SmartRouter } from './routing/smart-router.js';

export async function createRouterFromEnv(): Promise<SmartRouter> {
  const { createDefaultRouter } = await import('./routing/smart-router.js');

  return createDefaultRouter({
    anthropic: process.env.ANTHROPIC_API_KEY,
    openai: process.env.OPENAI_API_KEY,
    google: process.env.GOOGLE_API_KEY,
    ollama: process.env.OLLAMA_BASE_URL ? { baseURL: process.env.OLLAMA_BASE_URL } : undefined,
  });
}

// Re-export commonly used types
export type {
  LanguageModelV4,
  LanguageModelV4CallOptions,
  LanguageModelV4GenerateResult,
  LanguageModelV4StreamResult,
  LanguageModelV4StreamPart,
  LanguageModelV4Content,
  LanguageModelV4FinishReason,
  LanguageModelV4Usage,
  LanguageModelV4Tool,
  LanguageModelV4ToolChoice,
  LanguageModelV4Prompt,
  ModelCandidate,
  RoutingContext,
  ProviderV4,
} from './types/index.js';