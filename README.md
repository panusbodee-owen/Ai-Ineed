# Universal AI Platform (UAIP)

> **One API. Any AI Provider. Build Once, Run Everywhere.**

[![npm version](https://img.shields.io/npm/v/@uaip/core?style=flat-square)](https://www.npmjs.com/package/@uaip/core)
[![TypeScript](https://img.shields.io/badge/TypeScript-5.0+-blue?style=flat-square&logo=typescript)](https://www.typescriptlang.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green?style=flat-square)](https://opensource.org/licenses/MIT)
[![Build Status](https://img.shields.io/github/actions/workflow/status/panusbodee-owen/Ai-Ineed/ci.yml?style=flat-square)](https://github.com/panusbodee-owen/Ai-Ineed/actions)

**UAIP (Universal AI Platform)** is an open-source, vendor-agnostic AI platform that lets you write code once and run it with **any AI provider** — Anthropic Claude, OpenAI (GPT-4o, o1), Google Gemini, local models via Ollama, and more — without changing a single line of application code.

---

## 🎯 Who Is This For?

| Audience | Why UAIP? |
|----------|-----------|
| **AI App Builders** | Switch providers without rewrites; A/B test models in production |
| **CLI/IDE Tool Authors** | Build vendor-agnostic coding assistants (like Claude Code but for any provider) |
| **Teams & Orgs** | Avoid vendor lock-in; enforce cost/latency policies centrally |
| **Researchers** | Benchmark models across providers with identical prompts |
| **Enterprise** | Governance, audit trails, compliance — all in one platform |
| **Open Source Contributors** | Extensible plugin system; contribute providers, tools, memory backends |

---

## ✨ Key Features

### 🔌 **Universal AI Client** (`@uaip/core`)
```typescript
// One interface, any provider
import { createDefaultRouter } from '@uaip/core';

const router = await createDefaultRouter({
  anthropic: process.env.ANTHROPIC_API_KEY,
  openai: process.env.OPENAI_API_KEY,
  google: process.env.GOOGLE_API_KEY,
  ollama: { baseURL: 'http://localhost:11434/v1' }
});

// Use it — provider selected automatically by strategy
const result = await router.generate({
  prompt: 'Write a TypeScript function to parse CSV',
  maxOutputTokens: 2000
});
```

### 🧠 **Smart Routing Strategies**
| Strategy | Use Case |
|----------|----------|
| `fallback` | High availability — try next model on failure |
| `cost-optimize` | Minimize spend — cheapest healthy model |
| `latency-optimize` | Real-time apps — fastest response |
| `round-robin` | Load balance across providers |
| `priority` | Enforce org policy (e.g., "always prefer Claude for coding") |

### 🤖 **Agent Orchestration** (`@uaip/orchestrator`)
- DAG-based workflow engine
- Human-in-the-loop interrupts
- Sub-agent spawning
- Pluggable checkpointers (Memory, SQLite, Postgres)

### 🧠 **Context & Memory** (`@uaip/memory`)
- Cross-session persistence
- RAG-ready vector storage
- Conversation summarization
- Multi-tenant isolation

### 🧩 **Plugin System** (`@uaip/plugins`)
- Custom providers, tools, memory backends
- Capability-based sandboxing (Worker Threads)
- Registry with versioning & signatures

### 💻 **CLI Coding Assistant** (`@uaip/cli`)
```bash
# Vendor-agnostic coding assistant
uaip chat --model claude-opus-5
uaip chat --model gpt-4o
uaip chat --model ollama:llama3.1
```

### 🌐 **Web Dashboard** (`@uaip/web`)
- Team collaboration
- Cost tracking & budgets
- Model performance analytics
- Governance policies

---

## 🚀 Quick Start

### Installation
```bash
# Core package
npm install @uaip/core

# Or with specific providers
npm install @uaip/core @anthropic-ai/sdk openai @google/genai
```

### Basic Usage
```typescript
import { createDefaultRouter } from '@uaip/core';

// 1. Configure with your API keys
const router = await createDefaultRouter({
  anthropic: process.env.ANTHROPIC_API_KEY,
  openai: process.env.OPENAI_API_KEY,
  google: process.env.GOOGLE_API_KEY,
  ollama: { baseURL: 'http://localhost:11434/v1' }
});

// 2. Generate — smart routing picks the best model
const result = await router.generate({
  prompt: 'Explain TypeScript generics with examples',
  maxOutputTokens: 1500,
  temperature: 0.7
});

console.log(result.content[0].text);
```

### Streaming
```typescript
for await (const chunk of router.stream({ prompt: 'Write a React component' })) {
  if (chunk.type === 'text-delta') process.stdout.write(chunk.delta);
  if (chunk.type === 'finish') console.log('\nDone:', chunk.finishReason);
}
```

### Custom Routing
```typescript
const router = await createDefaultRouter({ /* keys */ });

// Override strategy per-request
const result = await router.generate(
  { prompt: 'Quick answer needed', maxOutputTokens: 100 },
  { strategy: 'latency-optimize', maxLatencyMs: 2000 }
);
```

---

## 🔧 Provider Setup

| Provider | Env Var | Models | Notes |
|----------|---------|--------|-------|
| **Anthropic** | `ANTHROPIC_API_KEY` | Opus 5, Sonnet 5, Fable 5, Haiku 4.5 | Full streaming, tools, vision |
| **OpenAI** | `OPENAI_API_KEY` | GPT-4o, GPT-4o-mini, o1-preview, o1-mini | Chat + Responses API |
| **Google** | `GOOGLE_API_KEY` | Gemini 1.5 Pro, Flash, Flash-8B | New `@google/genai` SDK |
| **Ollama** | (local) | Llama 3.1, Mistral, CodeLlama, etc. | `ollama pull llama3.1` first |

> **No API keys?** Use Ollama locally — zero cost, full privacy, runs on your machine.

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      @uaip/core                              │
├─────────────────────────────────────────────────────────────┤
│  LanguageModelV4 (Unified Interface)                        │
│  ├─ Providers: Anthropic │ OpenAI │ Google │ Ollama        │
│  ├─ SmartRouter (5 strategies + health checks)             │
│  ├─ ModelRegistry (failover, metrics)                      │
│  └─ Middleware (logging, caching, guardrails)              │
├─────────────────────────────────────────────────────────────┤
│  @uaip/orchestrator  │  @uaip/memory  │  @uaip/plugins      │
│  (DAG workflows)     │  (RAG, cache)   │  (extensions)      │
├─────────────────────────────────────────────────────────────┤
│  @uaip/cli (REPL)    │  @uaip/web (Dashboard)               │
└─────────────────────────────────────────────────────────────┘
```

---

## 📦 Monorepo Packages

| Package | Description | Status |
|---------|-------------|--------|
| [`@uaip/core`](packages/core) | Unified client, providers, routing | ✅ v0.0.1 |
| [`@uaip/cli`](packages/cli) | Coding assistant REPL | 🚧 WIP |
| [`@uaip/orchestrator`](packages/orchestrator) | Multi-agent workflows | 🚧 WIP |
| [`@uaip/memory`](packages/memory) | Context persistence & RAG | 🚧 WIP |
| [`@uaip/plugins`](packages/plugins) | Extension system | 🚧 WIP |
| [`@uaip/web`](packages/web) | React dashboard | 🚧 WIP |

---

## 🛠️ Development

### Prerequisites
- Node.js 20+
- npm 10+ (or pnpm/yarn)
- Ollama (optional, for local models)

### Setup
```bash
git clone https://github.com/panusbodee-owen/Ai-Ineed.git
cd uaip
npm install
npm run build       # Build all packages
npm run test        # Run tests
npm run lint        # Lint code
```

### Project Structure
```
uaip/
├── packages/
│   ├── core/           # ✅ Main library
│   ├── cli/            # CLI coding assistant
│   ├── orchestrator/   # Agent workflows
│   ├── memory/         # Context & RAG
│   ├── plugins/        # Plugin system
│   └── web/            # React dashboard
├── nx.json             # Nx workspace config
├── tsconfig.base.json  # Shared TypeScript config
└── package.json        # Root workspace
```

---

## 🤝 Contributing

We welcome contributions! UAIP is designed to be extensible:

1. **Add a Provider** — Implement `LanguageModelV4` in `packages/core/src/providers/`
2. **Add a Routing Strategy** — Extend `RoutingStrategy` in `packages/core/src/routing/`
3. **Add a Memory Backend** — Implement `MemoryBackend` in `packages/memory/`
4. **Add a Plugin** — Create capability-based plugins in `packages/plugins/`
5. **Improve Docs** — Fix typos, add examples, translate

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgments

- **Vercel AI SDK v5** — Inspiration for `LanguageModelV4` interface
- **Anthropic, OpenAI, Google, Ollama** — For excellent APIs
- **Nx** — Monorepo tooling
- **Community** — For feedback, issues, and PRs

---

## 🔗 Links

- **Documentation**: [github.com/panusbodee-owen/Ai-Ineed/wiki](https://github.com/panusbodee-owen/Ai-Ineed/wiki)
- **Issues**: [github.com/panusbodee-owen/Ai-Ineed/issues](https://github.com/panusbodee-owen/Ai-Ineed/issues)
- **Discussions**: [github.com/panusbodee-owen/Ai-Ineed/discussions](https://github.com/panusbodee-owen/Ai-Ineed/discussions)
- **NPM**: [@uaip/core](https://www.npmjs.com/package/@uaip/core)

---

<div align="center">
  <strong>Built with ❤️ for the AI community</strong><br>
  <sub>Star ⭐ this repo if UAIP helps you build better AI apps!</sub>
</div>