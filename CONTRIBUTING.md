# Contributing to UAIP

Thank you for your interest in contributing to the Universal AI Platform! 🎉

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Workflow](#development-workflow)
- [Adding a Provider](#adding-a-provider)
- [Adding a Routing Strategy](#adding-a-routing-strategy)
- [Testing](#testing)
- [Code Style](#code-style)
- [Pull Request Process](#pull-request-process)
- [Release Process](#release-process)

---

## Code of Conduct

This project follows the [Contributor Covenant Code of Conduct](CODE_OF_CONDUCT.md). By participating, you agree to uphold this code.

---

## Getting Started

### Prerequisites

- Node.js 20+
- npm 10+ (or pnpm/yarn)
- Git

### Setup

```bash
# 1. Fork the repository on GitHub
# 2. Clone your fork
git clone https://github.com/YOUR_USERNAME/Ai-Ineed.git
cd uaip

# 3. Install dependencies
npm install

# 4. Build all packages
npm run build

# 5. Run tests to verify setup
npm run test
```

### Environment Variables

Create a `.env` file for local development:

```bash
# Optional - for testing with real providers
ANTHROPIC_API_KEY=your_key_here
OPENAI_API_KEY=your_key_here
GOOGLE_API_KEY=your_key_here
```

---

## Development Workflow

### Branch Naming

- `feat/` — New features (e.g., `feat/add-google-provider`)
- `fix/` — Bug fixes (e.g., `fix/streaming-error`)
- `docs/` — Documentation changes (e.g., `docs/update-readme`)
- `refactor/` — Code refactoring (e.g., `refactor/router-logic`)
- `test/` — Test additions (e.g., `test/memory-backend`)

### Commit Messages

Follow [Conventional Commits](https://www.conventionalcommits.org/):

```
feat(core): add Ollama provider adapter
fix(routing): handle null usage in stream chunks
docs: update quick start guide
refactor(types): simplify LanguageModelV4Content union
```

---

## Adding a Provider

UAIP providers implement the `LanguageModelV4` interface. Here's how to add one:

### 1. Create Provider File

```bash
touch packages/core/src/providers/myprovider.ts
```

### 2. Implement LanguageModelV4

```typescript
// packages/core/src/providers/myprovider.ts
import { MyProviderSDK } from 'myprovider-sdk';
import {
  type LanguageModelV4,
  type LanguageModelV4CallOptions,
  type LanguageModelV4GenerateResult,
  type LanguageModelV4StreamResult,
  // ... other types
} from '../types/index.js';

export class MyProviderLanguageModel implements LanguageModelV4 {
  readonly specificationVersion = 'v4' as const;
  readonly provider = 'myprovider';
  readonly modelId: string;
  readonly supportedUrls = {};

  private client: MyProviderSDK;

  constructor(modelId: string, options: MyProviderClientOptions) {
    this.modelId = modelId;
    this.client = new MyProviderSDK(options);
  }

  async doGenerate(options: LanguageModelV4CallOptions): Promise<LanguageModelV4GenerateResult> {
    // Map UAIP options to provider SDK
    const providerOptions = this.mapCallOptions(options);

    try {
      const response = await this.client.generate(providerOptions);
      return this.mapGenerateResult(response);
    } catch (error) {
      throw this.mapError(error);
    }
  }

  async doStream(options: LanguageModelV4CallOptions): Promise<LanguageModelV4StreamResult> {
    const providerOptions = this.mapCallOptions(options);
    providerOptions.stream = true;

    try {
      const stream = await this.client.generateStream(providerOptions);
      return { stream: this.transformStream(stream) };
    } catch (error) {
      throw this.mapError(error);
    }
  }

  // Implement mapping methods:
  // - mapCallOptions()
  // - mapGenerateResult()
  // - transformStream()
  // - mapError()
  // - mapPrompt(), mapTool(), mapToolChoice(), etc.
}

export function createMyProvider(apiKey: string, options?: Partial<MyProviderClientOptions>) {
  return {
    specificationVersion: 'v4' as const,
    languageModel: (modelId: string) => new MyProviderLanguageModel(modelId, { apiKey, ...options }),
  };
}
```

### 3. Export from Barrel File

```typescript
// packages/core/src/index.ts
export { createMyProvider } from './providers/myprovider.js';
```

### 4. Add to Default Router (Optional)

```typescript
// packages/core/src/routing/smart-router.ts
export async function createDefaultRouter(apiKeys: { ... }) {
  // ... existing providers
  if (apiKeys.myprovider) {
    const { createMyProvider } = await import('../providers/myprovider.js');
    // Register models...
  }
}
```

### 5. Add Tests

```typescript
// packages/core/src/providers/myprovider.spec.ts
import { describe, it, expect } from 'vitest';
import { createMyProvider } from './myprovider.js';

describe('MyProvider', () => {
  it('should generate text', async () => {
    const provider = createMyProvider('test-key');
    const model = provider.languageModel('my-model');
    const result = await model.doGenerate({ prompt: 'Hello', maxOutputTokens: 10 });
    expect(result.content).toBeDefined();
  });
});
```

---

## Adding a Routing Strategy

### 1. Define Strategy

```typescript
// packages/core/src/routing/my-strategy.ts
import { type RoutingStrategy, type ModelCandidate, type RoutingContext } from '../types/index.js';

export const myCustomStrategy: RoutingStrategy = {
  name: 'my-custom',
  decide(models: ModelCandidate[], context: RoutingContext): ModelCandidate {
    // Your custom logic here
    // Return the selected model candidate
    return models[0]; // Example
  },
};
```

### 2. Register Strategy

```typescript
// In your application code
import { SmartRouter } from '@uaip/core';
import { myCustomStrategy } from '@uaip/core/routing/my-strategy.js';

const router = new SmartRouter();
router.registerStrategy('my-custom', myCustomStrategy);
router.setDefaultStrategy('my-custom');
```

---

## Testing

### Run All Tests

```bash
npm run test
```

### Run Specific Package Tests

```bash
# Core package
npm run test -- --filter=@uaip/core

# With coverage
npm run test:coverage
```

### Writing Tests

- Place tests next to source files: `*.spec.ts`
- Use Vitest (already configured)
- Mock external APIs — don't make real network calls in unit tests
- Integration tests go in `tests/integration/`

---

## Code Style

### TypeScript

- **ESM modules** with explicit `.js` extensions in imports
- **Strict mode** enabled — no `any` unless absolutely necessary
- **Type inference** preferred over explicit annotations
- **Discriminated unions** for variant types (see `LanguageModelV4Content`)

### Formatting

```bash
# Format code
npm run format

# Lint
npm run lint

# Type check
npm run typecheck
```

### Conventions

- `async/await` over Promises
- Early returns for guard clauses
- Descriptive variable names (`modelRegistry` not `mr`)
- JSDoc for public APIs
- No default exports — use named exports

---

## Pull Request Process

1. **Create a feature branch** from `main`
2. **Make your changes** with tests
3. **Run checks locally**:
   ```bash
   npm run lint && npm run typecheck && npm run test
   ```
4. **Push and open PR** against `main`
5. **Fill out PR template** — describe what, why, and how
6. **Wait for CI** — all checks must pass
7. **Address review feedback**
8. **Squash and merge** (maintainers will merge)

### PR Requirements

- [ ] All CI checks pass
- [ ] Tests added/updated
- [ ] Documentation updated (if applicable)
- [ ] No breaking changes without discussion
- [ ] Conventional commit messages

---

## Release Process

Releases are automated via GitHub Actions on merge to `main`:

1. Version bump determined by commit messages:
   - `feat:` → minor version
   - `fix:` → patch version
   - `BREAKING CHANGE:` → major version
2. Changelog generated from commits
3. Packages published to npm
4. GitHub Release created

---

## Need Help?

- **Questions**: Open a [Discussion](https://github.com/panusbodee-owen/Ai-Ineed/discussions)
- **Bugs**: File an [Issue](https://github.com/panusbodee-owen/Ai-Ineed/issues)
- **Security**: Email security@uaip.dev (or use GitHub Security Advisories)

---

## Recognition

Contributors are recognized in:
- [AUTHORS.md](AUTHORS.md)
- Release notes
- GitHub contributors graph

Thank you for making UAIP better! 🚀