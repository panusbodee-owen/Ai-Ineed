/**
 * Smart Routing for UAIP Core
 * Routes requests to optimal provider/model based on strategy
 */

import {
  type ModelCandidate,
  type RoutingContext,
  type RoutingStrategy,
  type LanguageModelV4,
  type LanguageModelV4CallOptions,
  type LanguageModelV4GenerateResult,
  type LanguageModelV4StreamResult,
} from '../types/index.js';

// Re-export RoutingStrategy type for consumers
export type { RoutingStrategy } from '../types/index.js';

/**
 * Built-in routing strategies
 */
export const routingStrategies: Record<string, RoutingStrategy> = {
  fallback: {
    name: 'fallback',
    decide(models: ModelCandidate[], context: RoutingContext): ModelCandidate {
      const healthy = models.filter(m => m.health === 'healthy');
      if (healthy.length === 0) throw new Error('No healthy models available');
      // Sort by priority (first in list = highest priority)
      return healthy[0];
    },
  },

  'cost-optimize': {
    name: 'cost-optimize',
    decide(models: ModelCandidate[], context: RoutingContext): ModelCandidate {
      const healthy = models.filter(m => m.health === 'healthy');
      if (healthy.length === 0) throw new Error('No healthy models available');

      const filtered = healthy.filter(m => {
        if (context.maxCost) {
          const estCost = m.estimatedCostPer1kTokens.input + m.estimatedCostPer1kTokens.output;
          return estCost <= context.maxCost!;
        }
        return true;
      });

      if (filtered.length === 0) throw new Error('No models within cost budget');

      // Sort by cost (cheapest first)
      filtered.sort((a, b) => {
        const costA = a.estimatedCostPer1kTokens.input + a.estimatedCostPer1kTokens.output;
        const costB = b.estimatedCostPer1kTokens.input + b.estimatedCostPer1kTokens.output;
        return costA - costB;
      });

      return filtered[0];
    },
  },

  'latency-optimize': {
    name: 'latency-optimize',
    decide(models: ModelCandidate[], context: RoutingContext): ModelCandidate {
      const healthy = models.filter(m => m.health === 'healthy');
      if (healthy.length === 0) throw new Error('No healthy models available');

      const filtered = healthy.filter(m => {
        if (context.maxLatencyMs) {
          return m.avgLatencyMs <= context.maxLatencyMs!;
        }
        return true;
      });

      if (filtered.length === 0) throw new Error('No models within latency budget');

      // Sort by latency (fastest first)
      filtered.sort((a, b) => a.avgLatencyMs - b.avgLatencyMs);
      return filtered[0];
    },
  },

  'round-robin': {
    name: 'round-robin',
    decide(models: ModelCandidate[], context: RoutingContext): ModelCandidate {
      const healthy = models.filter(m => m.health === 'healthy');
      if (healthy.length === 0) throw new Error('No healthy models available');

      // Use a simple hash of the prompt for consistent routing
      const promptHash = hashPrompt(context.prompt);
      const index = promptHash % healthy.length;
      return healthy[index];
    },
  },

  priority: {
    name: 'priority',
    decide(models: ModelCandidate[], context: RoutingContext): ModelCandidate {
      const healthy = models.filter(m => m.health === 'healthy');
      if (healthy.length === 0) throw new Error('No healthy models available');

      // Sort by provider priority (can be customized via model metadata)
      const priorityOrder = ['anthropic.messages', 'openai.chat', 'google.generative', 'ollama'];
      healthy.sort((a, b) => {
        const pa = priorityOrder.indexOf(a.provider);
        const pb = priorityOrder.indexOf(b.provider);
        return (pa === -1 ? 999 : pa) - (pb === -1 ? 999 : pb);
      });

      return healthy[0];
    },
  },
};

function hashPrompt(prompt: RoutingContext['prompt']): number {
  const str = typeof prompt === 'string' ? prompt : JSON.stringify(prompt);
  let hash = 0;
  for (let i = 0; i < str.length; i++) {
    const char = str.charCodeAt(i);
    hash = ((hash << 5) - hash) + char;
    hash = hash & hash;
  }
  return Math.abs(hash);
}

/**
 * Model registry with health tracking
 */
export interface ModelRegistryEntry {
  candidate: ModelCandidate;
  model: LanguageModelV4;
  lastHealthCheck: number;
  consecutiveFailures: number;
}

export class ModelRegistry {
  private models: Map<string, ModelRegistryEntry> = new Map();
  private healthCheckInterval: NodeJS.Timeout | null = null;

  register(provider: string, modelId: string, model: LanguageModelV4, candidate: ModelCandidate): void {
    const key = `${provider}:${modelId}`;
    this.models.set(key, {
      candidate: { ...candidate, provider, modelId },
      model,
      lastHealthCheck: Date.now(),
      consecutiveFailures: 0,
    });
  }

  unregister(provider: string, modelId: string): void {
    const key = `${provider}:${modelId}`;
    this.models.delete(key);
  }

  getAll(): ModelRegistryEntry[] {
    return Array.from(this.models.values());
  }

  get(provider: string, modelId: string): ModelRegistryEntry | undefined {
    return this.models.get(`${provider}:${modelId}`);
  }

  getHealthy(): ModelRegistryEntry[] {
    return this.getAll().filter(m => m.candidate.health === 'healthy');
  }

  markFailure(provider: string, modelId: string): void {
    const key = `${provider}:${modelId}`;
    const entry = this.models.get(key);
    if (entry) {
      entry.consecutiveFailures++;
      if (entry.consecutiveFailures >= 3) {
        entry.candidate.health = 'degraded';
      }
      if (entry.consecutiveFailures >= 5) {
        entry.candidate.health = 'down';
      }
    }
  }

  markSuccess(provider: string, modelId: string): void {
    const key = `${provider}:${modelId}`;
    const entry = this.models.get(key);
    if (entry) {
      entry.consecutiveFailures = 0;
      entry.candidate.health = 'healthy';
    }
  }

  startHealthChecks(intervalMs = 60000): void {
    if (this.healthCheckInterval) return;
    this.healthCheckInterval = setInterval(() => this.runHealthChecks(), intervalMs);
  }

  stopHealthChecks(): void {
    if (this.healthCheckInterval) {
      clearInterval(this.healthCheckInterval);
      this.healthCheckInterval = null;
    }
  }

  private async runHealthChecks(): Promise<void> {
    for (const entry of this.models.values()) {
      try {
        // Simple health check - try a minimal generation
        await entry.model.doGenerate({
          prompt: 'ping',
          maxOutputTokens: 1,
        });
        this.markSuccess(entry.candidate.provider, entry.candidate.modelId);
        entry.lastHealthCheck = Date.now();
      } catch {
        this.markFailure(entry.candidate.provider, entry.candidate.modelId);
      }
    }
  }
}

/**
 * Smart Router - Main routing interface
 */
export class SmartRouter {
  private registry = new ModelRegistry();
  private strategies: Map<string, RoutingStrategy> = new Map();
  private defaultStrategy = 'fallback';

  constructor() {
    // Register built-in strategies
    for (const [name, strategy] of Object.entries(routingStrategies)) {
      this.strategies.set(name, strategy);
    }
  }

  registerModel(
    provider: string,
    modelId: string,
    model: LanguageModelV4,
    candidate: Omit<ModelCandidate, 'provider' | 'modelId'>
  ): void {
    this.registry.register(provider, modelId, model, { ...candidate, provider, modelId });
  }

  unregisterModel(provider: string, modelId: string): void {
    this.registry.unregister(provider, modelId);
  }

  registerStrategy(name: string, strategy: RoutingStrategy): void {
    this.strategies.set(name, strategy);
  }

  setDefaultStrategy(name: string): void {
    if (!this.strategies.has(name)) {
      throw new Error(`Strategy '${name}' not registered`);
    }
    this.defaultStrategy = name;
  }

  async route(context: RoutingContext): Promise<ModelRegistryEntry> {
    const strategy = this.strategies.get(this.defaultStrategy);
    if (!strategy) throw new Error(`Default strategy '${this.defaultStrategy}' not found`);

    const candidates = this.registry.getHealthy().map(e => e.candidate);
    if (candidates.length === 0) {
      throw new Error('No healthy models available');
    }

    const selected = strategy.decide(candidates, context);
    const entry = this.registry.get(selected.provider, selected.modelId);
    if (!entry) throw new Error(`Selected model not found: ${selected.provider}:${selected.modelId}`);

    return entry;
  }

  async generate(options: LanguageModelV4CallOptions, context?: Partial<RoutingContext>): Promise<LanguageModelV4GenerateResult> {
    const routingContext: RoutingContext = {
      prompt: options.prompt,
      tools: options.tools,
      requiredCapabilities: this.extractRequiredCapabilities(options),
      ...context,
    };

    const entry = await this.route(routingContext);

    try {
      const result = await entry.model.doGenerate(options);
      this.registry.markSuccess(entry.candidate.provider, entry.candidate.modelId);
      return result;
    } catch (error) {
      this.registry.markFailure(entry.candidate.provider, entry.candidate.modelId);

      // Try fallback if using fallback strategy
      if (this.defaultStrategy === 'fallback') {
        return this.generateWithFallback(options, routingContext, entry);
      }
      throw error;
    }
  }

  async stream(options: LanguageModelV4CallOptions, context?: Partial<RoutingContext>): Promise<LanguageModelV4StreamResult> {
    const routingContext: RoutingContext = {
      prompt: options.prompt,
      tools: options.tools,
      requiredCapabilities: this.extractRequiredCapabilities(options),
      ...context,
    };

    const entry = await this.route(routingContext);

    try {
      const result = await entry.model.doStream(options);
      this.registry.markSuccess(entry.candidate.provider, entry.candidate.modelId);
      return result;
    } catch (error) {
      this.registry.markFailure(entry.candidate.provider, entry.candidate.modelId);

      if (this.defaultStrategy === 'fallback') {
        return this.streamWithFallback(options, routingContext, entry);
      }
      throw error;
    }
  }

  private extractRequiredCapabilities(options: LanguageModelV4CallOptions): Partial<ModelCandidate['capabilities']> {
    const caps: Partial<ModelCandidate['capabilities']> = {};
    if (options.tools && options.tools.length > 0) caps.tools = true;
    if (options.responseFormat?.type === 'json') caps.structuredOutput = true;
    // Could add vision, audio, reasoning detection based on prompt
    return caps;
  }

  private async generateWithFallback(
    options: LanguageModelV4CallOptions,
    context: RoutingContext,
    failedEntry: ModelRegistryEntry
  ): Promise<LanguageModelV4GenerateResult> {
    const remaining = this.registry.getHealthy().filter(
      e => e.candidate.provider !== failedEntry.candidate.provider || e.candidate.modelId !== failedEntry.candidate.modelId
    );

    for (const entry of remaining) {
      try {
        const result = await entry.model.doGenerate(options);
        this.registry.markSuccess(entry.candidate.provider, entry.candidate.modelId);
        return result;
      } catch {
        this.registry.markFailure(entry.candidate.provider, entry.candidate.modelId);
      }
    }

    throw new Error('All fallback models failed');
  }

  private async streamWithFallback(
    options: LanguageModelV4CallOptions,
    context: RoutingContext,
    failedEntry: ModelRegistryEntry
  ): Promise<LanguageModelV4StreamResult> {
    const remaining = this.registry.getHealthy().filter(
      e => e.candidate.provider !== failedEntry.candidate.provider || e.candidate.modelId !== failedEntry.candidate.modelId
    );

    for (const entry of remaining) {
      try {
        const result = await entry.model.doStream(options);
        this.registry.markSuccess(entry.candidate.provider, entry.candidate.modelId);
        return result;
      } catch {
        this.registry.markFailure(entry.candidate.provider, entry.candidate.modelId);
      }
    }

    throw new Error('All fallback models failed');
  }

  getRegistry(): ModelRegistry {
    return this.registry;
  }

  getStrategies(): string[] {
    return Array.from(this.strategies.keys());
  }
}

/**
 * Create a pre-configured router with common models
 */
export async function createDefaultRouter(apiKeys: {
  anthropic?: string;
  openai?: string;
  google?: string;
  ollama?: { baseURL?: string };
}): Promise<SmartRouter> {
  const router = new SmartRouter();

  // Register Anthropic models
  if (apiKeys.anthropic) {
    const { createAnthropicProvider } = await import('../providers/anthropic.js');
    const anthropicProvider = createAnthropicProvider(apiKeys.anthropic);
    for (const modelId of ['claude-opus-5', 'claude-sonnet-5', 'claude-fable-5', 'claude-haiku-4-5-20251001']) {
      router.registerModel('anthropic.messages', modelId, anthropicProvider.languageModel(modelId), {
        displayName: modelId,
        capabilities: { tools: true, streaming: true, structuredOutput: true, vision: true, audio: false, reasoning: true, maxContextTokens: 200000, maxOutputTokens: 64000 },
        estimatedCostPer1kTokens: { input: 0.003, output: 0.015 }, // Approximate
        avgLatencyMs: 2000,
        health: 'healthy',
      });
    }
  }

  // Register OpenAI models
  if (apiKeys.openai) {
    const { createOpenAIProvider } = await import('../providers/openai.js');
    const openaiProvider = createOpenAIProvider(apiKeys.openai);
    for (const modelId of ['gpt-4o', 'gpt-4o-mini', 'gpt-4-turbo', 'o1-preview', 'o1-mini']) {
      router.registerModel('openai.chat', modelId, openaiProvider.languageModel(modelId), {
        displayName: modelId,
        capabilities: { tools: true, streaming: true, structuredOutput: true, vision: true, audio: true, reasoning: modelId.startsWith('o1'), maxContextTokens: modelId.startsWith('o1') ? 200000 : 128000, maxOutputTokens: 16384 },
        estimatedCostPer1kTokens: { input: modelId.includes('mini') ? 0.00015 : 0.005, output: modelId.includes('mini') ? 0.0006 : 0.015 },
        avgLatencyMs: modelId.startsWith('o1') ? 10000 : 2000,
        health: 'healthy',
      });
    }
  }

  // Register Google models
  if (apiKeys.google) {
    const { createGoogleProvider } = await import('../providers/google.js');
    const googleProvider = createGoogleProvider(apiKeys.google);
    for (const modelId of ['gemini-1.5-pro', 'gemini-1.5-flash', 'gemini-1.5-flash-8b']) {
      router.registerModel('google.generative', modelId, googleProvider.languageModel(modelId), {
        displayName: modelId,
        capabilities: { tools: true, streaming: true, structuredOutput: true, vision: true, audio: true, reasoning: false, maxContextTokens: 1000000, maxOutputTokens: 8192 },
        estimatedCostPer1kTokens: { input: 0.000125, output: 0.0005 },
        avgLatencyMs: 1500,
        health: 'healthy',
      });
    }
  }

  // Register Ollama models
  if (apiKeys.ollama) {
    const { createOllamaProvider } = await import('../providers/ollama.js');
    const ollamaProvider = createOllamaProvider({ baseURL: apiKeys.ollama.baseURL });
    // Would need to fetch available models dynamically
    router.registerModel('ollama', 'llama3.1', ollamaProvider.languageModel('llama3.1'), {
      displayName: 'Llama 3.1 (Local)',
      capabilities: { tools: true, streaming: true, structuredOutput: false, vision: false, audio: false, reasoning: false, maxContextTokens: 128000, maxOutputTokens: 4096 },
      estimatedCostPer1kTokens: { input: 0, output: 0 },
      avgLatencyMs: 5000,
      health: 'healthy',
    });
  }

  return router;
}