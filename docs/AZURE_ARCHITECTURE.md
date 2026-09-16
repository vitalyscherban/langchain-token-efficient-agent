# Azure Architecture — Token-Efficient CI Triage Agent

This document describes how to host the token-efficient LangGraph CI triage agent in
Azure. It covers the full architecture, component choices, data flows, network design,
security model, CI/CD integration, and operations.

For the agent's internal design (LangGraph graph, token-saving levers, module map),
see [ARCHITECTURE.md](ARCHITECTURE.md). For the benchmark numbers that motivate this
work, see the [README](../README.md).

---

## Contents

1. [Architecture overview](#1-architecture-overview)
2. [Component decisions](#2-component-decisions)
3. [Request lifecycle](#3-request-lifecycle)
4. [Agent internals mapped to Azure](#4-agent-internals-mapped-to-azure)
5. [Network topology](#5-network-topology)
6. [Security model](#6-security-model)
7. [CI/CD integration](#7-cicd-integration)
8. [Agent deployment pipeline](#8-agent-deployment-pipeline)
9. [Observability](#9-observability)
10. [Cost model](#10-cost-model)
11. [Code changes required](#11-code-changes-required)
12. [Phased rollout](#12-phased-rollout)

---

## 1. Architecture overview

The agent takes a CI log and a PR diff, runs a multi-turn LangGraph loop to identify
the root cause, and returns a structured diagnosis. In Azure, this becomes an
event-driven HTTP service triggered by CI failures.

```mermaid
graph TB
    subgraph cicd ["CI/CD Systems"]
        ADO["Azure DevOps\nPipeline"]
        GH["GitHub Actions"]
    end

    subgraph azure ["Azure"]
        subgraph ingress ["Ingress Layer"]
            APIM["Azure API Management\n(auth · rate-limit · TLS)"]
        end

        subgraph compute ["Compute Layer"]
            ACA["Azure Container Apps\ntriage-agent\n(min=0, max=10 replicas)"]
        end

        subgraph ai ["AI Layer"]
            AOAI["Azure OpenAI Service\nGPT-4o · GPT-4o-mini"]
        end

        subgraph data ["Data Layer"]
            Redis["Azure Cache for Redis\n(LLM response cache)"]
            Blob["Azure Blob Storage\n(logs · diffs · results)"]
            Search["Azure AI Search\n(incident knowledge base)\n[optional]"]
        end

        subgraph ops ["Operations"]
            KV["Azure Key Vault\n(secrets)"]
            ACR["Azure Container Registry\n(image store)"]
            AI["Application Insights\n(traces · token metrics)"]
            Monitor["Azure Monitor\n(alerts · dashboards)"]
        end
    end

    ADO -->|"POST /triage\n(log + diff + repo archive)"| APIM
    GH -->|"POST /triage\n(log + diff + repo archive)"| APIM
    APIM -->|"internal VNet"| ACA
    ACA -->|"chat completions"| AOAI
    ACA -->|"exact-match cache"| Redis
    ACA -->|"store results"| Blob
    ACA -->|"vector search\n[optional]"| Search
    ACA -->|"pull image"| ACR
    ACA -->|"get secrets\n(managed identity)"| KV
    ACA -->|"emit telemetry"| AI
    AI --> Monitor
    AOAI -->|"auth\n(managed identity)"| KV
```

### Key design choices at a glance

| Decision | Choice | Reason |
|---|---|---|
| Compute | Azure Container Apps | Scales to zero; no cold-start on Functions for 30–60 s loops |
| LLM | Azure OpenAI | Same API surface as OpenAI; managed identity auth; data residency |
| Cache | Azure Cache for Redis | Shared across replicas; SQLite is single-replica only |
| Auth | Managed Identity → Key Vault | No secrets in environment variables or container images |
| Source access | Bundled in request body | No filesystem; CI already has the repo checked out |
| Trigger | HTTP webhook from pipeline | Decouples agent from CI system type |

---

## 2. Component decisions

### 2.1 Azure Container Apps

The agent runs as a Python process executing a LangGraph loop of up to 12 recursive
turns. Each turn makes an HTTP call to Azure OpenAI. Total wall time per request is
typically 20–90 seconds depending on model latency and number of tool calls.

Azure Functions Consumption plan is unsuitable here: the default 5-minute timeout is
sufficient, but the invocation model creates friction for streaming and stateful
in-process loops. Container Apps provides a clean HTTP server model with autoscaling.

```
min replicas = 0    → zero cost when idle
max replicas = 10   → burst capacity for parallel CI failures
scale trigger       → HTTP concurrent requests (threshold: 5 per replica)
CPU / memory        → 1.0 vCPU / 2 GiB per replica
```

### 2.2 Azure OpenAI

Drop-in replacement for the current `ChatOpenAI` via `AzureChatOpenAI`. The existing
`stable_system_prompt()` function already satisfies the byte-identical prefix
requirement for provider-side **prompt prefix caching** — no code changes needed to
benefit from it.

Deploy two model versions in one Azure OpenAI resource:

| Deployment name | Model | Used for |
|---|---|---|
| `gpt-4o` | GPT-4o | Final diagnosis (`SMART_MODEL`) |
| `gpt-4o-mini` | GPT-4o-mini | History summarisation (`CHEAP_MODEL`) |

### 2.3 Azure Cache for Redis (replacing SQLite)

`cache.py` installs a `SQLiteCache` that writes to `.langchain.db` on the local
filesystem. With multiple Container App replicas, each replica would maintain its own
isolated cache — re-running identical prompts would still hit the model on other
replicas.

Redis provides a single shared cache across all replicas. The LangChain
`RedisCache` adapter is a one-line swap (see [section 11](#11-code-changes-required)).

Tier: **C1 Basic** (1 GB). At 200 runs/day × 10 LLM calls/run, each cached response
is typically 500–2,000 bytes of JSON. C1 is comfortably under capacity.

TTL: 7 days. CI triage for the same failure on a retry is semantically identical, but
results for a fixed bug are stale after the fix merges.

### 2.4 Azure Blob Storage

Three containers:

| Container | Contents | Retention |
|---|---|---|
| `triage-inputs` | Raw CI log + diff per run | 30 days |
| `triage-results` | Full agent output (JSON) | 90 days |
| `triage-sources` | Repo source archives (optional) | 1 day |

Run ID (UUID) is the blob name prefix so a result is always traceable to its input.

### 2.5 Azure API Management (Consumption tier)

Responsibilities:
- TLS termination
- Subscription-key authentication for the pipeline caller
- Rate limiting: 60 requests/minute per subscription to protect the OpenAI quota
- Request size validation (reject payloads > 10 MB before they reach the container)
- Backend URL points to the Container App's internal ingress — the Container App is
  never directly reachable from the internet

### 2.6 Azure Key Vault

All secrets and configuration that must not appear in environment variables or image
layers. Container App reads them via **Key Vault references** — secrets are projected
as environment variables at runtime using the app's system-assigned managed identity.

| Secret name | Value |
|---|---|
| `aoai-endpoint` | `https://<resource>.openai.azure.com/` |
| `redis-connection-string` | Full Redis connection string with auth |
| `storage-connection-string` | Blob Storage connection string |
| `appinsights-connection-string` | Application Insights connection string |

### 2.7 Application Insights + Azure Monitor

The `prompt_tokens` field accumulated in `TriageState` is emitted as a custom metric
after every run. This makes the token budget a first-class observable quantity —
not just a stdout line in a log.

---

## 3. Request lifecycle

The sequence from a CI failure to a PR comment.

```mermaid
sequenceDiagram
    autonumber
    participant CI as CI Pipeline<br/>(ADO / GitHub)
    participant APIM as API Management
    participant ACA as Container App<br/>(triage-agent)
    participant KV as Key Vault
    participant Redis as Redis Cache
    participant AOAI as Azure OpenAI
    participant Blob as Blob Storage
    participant AI as App Insights

    CI->>APIM: POST /triage {log, diff, source_archive}
    APIM->>APIM: validate subscription key, check rate limit
    APIM->>ACA: forward request (internal VNet)

    ACA->>KV: fetch secrets (managed identity, once at startup)
    KV-->>ACA: connection strings

    ACA->>Blob: store raw log + diff (run_id)
    ACA->>ACA: triage node — prune log (regex, zero tokens)

    loop LangGraph agent loop (≤12 turns)
        ACA->>Redis: check exact-match cache
        alt cache hit
            Redis-->>ACA: cached LLM response
        else cache miss
            ACA->>AOAI: chat completion (pruned history)
            AOAI-->>ACA: response (possibly with tool calls)
            ACA->>Redis: store response in cache
        end

        opt model requested a tool call
            ACA->>ACA: execute tool (read_source_window / grep_symbol / file_outline)
            ACA->>ACA: compact node — digest tool result in place
        end
    end

    ACA->>Blob: store result JSON (run_id)
    ACA->>AI: emit prompt_tokens metric, trace
    ACA-->>APIM: 200 {analysis, prompt_tokens, run_id}
    APIM-->>CI: 200 {analysis, prompt_tokens, run_id}

    CI->>CI: post analysis as PR comment via ADO/GitHub API
```

---

## 4. Agent internals mapped to Azure

The LangGraph graph runs entirely in-process inside the Container App. This diagram
shows which Azure services each graph node touches.

```mermaid
flowchart LR
    subgraph container ["Container App (single request, in-process)"]
        direction TB

        START([start]) --> triage

        subgraph triage_node ["triage node"]
            T1["prune_ci_log\n(regex, zero tokens)"]
            T2["filter_diff\n(denylist, zero tokens)"]
        end

        triage --> agent

        subgraph agent_node ["agent node (per turn)"]
            A1["strip_stale_tool_payloads"]
            A2["trim_history\n(token ceiling: 3,000)"]
            A3["model.invoke\n→ AzureChatOpenAI"]
        end

        agent -->|"tool_calls present"| tools

        subgraph tools_node ["tools node"]
            TL1["read_source_window\n(±40 lines)"]
            TL2["grep_symbol\n(locations only)"]
            TL3["file_outline\n(defs map)"]
        end

        tools --> compact

        subgraph compact_node ["compact node"]
            C1["compact_tool_result\n(head/tail digest)"]
            C2["rewrite ToolMessage in place\n(same id → add_messages replaces)"]
        end

        compact --> agent
        agent -->|"no tool_calls"| END([END])
    end

    subgraph azure_services ["Azure Services touched"]
        AOAI2["Azure OpenAI\nGPT-4o / 4o-mini"]
        Redis2["Redis Cache\n(exact-match hit/miss)"]
        Src["Source archive\n(extracted from request body)"]
    end

    A3 <-->|"cache check\nthen completion"| Redis2
    A3 <-->|"chat completion"| AOAI2
    TL1 & TL2 & TL3 <-->|"reads .py files"| Src
```

### Node token costs

| Node | Azure OpenAI calls | Token cost |
|---|---|---|
| `triage` | 0 | Zero — pure Python regex |
| `agent` | 1 per turn | ~9,205 prompt tokens across all turns (optimised) |
| `tools` | 0 | Zero — local file reads |
| `compact` | 0 | Zero — string truncation |

---

## 5. Network topology

All traffic between Azure services stays on the Microsoft backbone. The Container App
is never reachable from the public internet directly.

```mermaid
graph TB
    Internet["Public Internet\n(CI runner)"]

    subgraph rg ["Resource Group: rg-triage-agent-prod"]
        subgraph vnet ["Virtual Network: vnet-triage (10.0.0.0/16)"]
            subgraph snet_apim ["Subnet: snet-apim (10.0.1.0/24)"]
                APIM2["API Management\n(external-facing)"]
            end

            subgraph snet_aca ["Subnet: snet-aca (10.0.2.0/24)"]
                ACA2["Container Apps\nEnvironment\n(internal ingress only)"]
            end

            subgraph snet_pe ["Subnet: snet-pe (10.0.3.0/24)\nPrivate Endpoints"]
                PE_AOAI["PE → Azure OpenAI"]
                PE_Redis["PE → Redis Cache"]
                PE_Blob["PE → Blob Storage"]
                PE_KV["PE → Key Vault"]
            end
        end

        DNS["Private DNS Zones\n(privatelink.openai.azure.com\nprivatelink.redis.cache.windows.net\nprivatelink.blob.core.windows.net\nprivatelink.vaultcore.azure.net)"]

        AOAI3["Azure OpenAI"]
        Redis3["Redis Cache"]
        Blob3["Blob Storage"]
        KV3["Key Vault"]
        ACR3["Container Registry\n(ACR private endpoint)"]
        AI3["Application Insights\n(public endpoint — telemetry only)"]
    end

    Internet -->|"HTTPS :443"| APIM2
    APIM2 -->|"HTTP :8000\ninternal"| ACA2
    ACA2 --> PE_AOAI --> AOAI3
    ACA2 --> PE_Redis --> Redis3
    ACA2 --> PE_Blob --> Blob3
    ACA2 --> PE_KV --> KV3
    ACA2 -->|"ACR private endpoint"| ACR3
    ACA2 -->|"telemetry\n(public)"| AI3
    DNS -.->|"name resolution"| PE_AOAI & PE_Redis & PE_Blob & PE_KV
```

### Network rules summary

| From | To | Protocol | Port | Note |
|---|---|---|---|---|
| Internet | APIM | HTTPS | 443 | Only entry point |
| APIM | Container Apps | HTTP | 8000 | Internal VNet only |
| Container Apps | All Azure services | HTTPS | 443 | Via private endpoints |
| Container Apps | Application Insights | HTTPS | 443 | Public endpoint (telemetry only) |
| All other | Container Apps | — | — | Denied |

---

## 6. Security model

```mermaid
graph LR
    subgraph identities ["Managed Identities"]
        MI_ACA["System-assigned MI\nContainer App"]
        MI_Build["User-assigned MI\nBuild Pipeline"]
    end

    subgraph kv ["Key Vault RBAC"]
        KV_Read["Key Vault Secrets User\n(runtime secrets)"]
        KV_Admin["Key Vault Administrator\n(secret rotation — ops only)"]
    end

    subgraph aoai_rbac ["Azure OpenAI RBAC"]
        AOAI_User["Cognitive Services OpenAI User\n(send completions)"]
    end

    subgraph acr_rbac ["ACR RBAC"]
        ACR_Pull["AcrPull\n(Container App pulls image)"]
        ACR_Push["AcrPush\n(Build pipeline pushes image)"]
    end

    subgraph storage_rbac ["Storage RBAC"]
        Blob_Contrib["Storage Blob Data Contributor\n(read/write logs and results)"]
    end

    MI_ACA --> KV_Read
    MI_ACA --> AOAI_User
    MI_ACA --> ACR_Pull
    MI_ACA --> Blob_Contrib

    MI_Build --> ACR_Push
    MI_Build --> KV_Read
```

### Security controls

| Control | Implementation |
|---|---|
| No secrets in code or env vars | Key Vault references projected as env vars at runtime |
| No secret rotation downtime | Key Vault reference re-reads on Container App restart |
| LLM API auth | Managed identity (`DefaultAzureCredential`) — no API keys |
| Inbound auth | APIM subscription key checked before any request reaches ACA |
| Network isolation | Container App on internal ingress; all services on private endpoints |
| Image signing | ACR with Content Trust; Container App rejects unsigned images |
| Supply chain | Dependabot on `requirements.txt`; base image pinned by digest |
| Audit log | Azure Monitor activity log on Key Vault, APIM, Container Apps |

---

## 7. CI/CD integration

### 7.1 Azure DevOps

Add a pipeline step that runs **after** the test stage and only on failure. The step
sends the CI log and PR diff to the triage endpoint and posts the result as a PR
comment.

```yaml
# azure-pipelines.yml

stages:
  - stage: Test
    jobs:
      - job: RunTests
        steps:
          - script: pytest --tb=long 2>&1 | tee ci.log
            displayName: Run tests
            continueOnError: true

          - task: Bash@3
            displayName: Triage CI failure
            condition: failed()
            env:
              TRIAGE_API_KEY: $(TRIAGE_API_KEY)     # pipeline variable (secret)
              TRIAGE_ENDPOINT: $(TRIAGE_ENDPOINT)   # pipeline variable
              ADO_TOKEN: $(System.AccessToken)
            inputs:
              script: |
                # Bundle source .py files (already checked out)
                find . -name "*.py" \
                  -not -path "./.venv/*" \
                  -not -path "./node_modules/*" \
                  | tar czf source.tar.gz -T -

                RESPONSE=$(curl -sf -X POST "$TRIAGE_ENDPOINT/triage" \
                  -H "Ocp-Apim-Subscription-Key: $TRIAGE_API_KEY" \
                  -H "Content-Type: application/json" \
                  -d "{
                    \"raw_log\":         $(cat ci.log | jq -Rs .),
                    \"raw_diff\":        $(git diff origin/$(System.PullRequest.TargetBranch) | jq -Rs .),
                    \"source_archive\":  $(base64 -w0 source.tar.gz | jq -Rs .)
                  }")

                ANALYSIS=$(echo "$RESPONSE" | jq -r '.analysis')
                TOKENS=$(echo "$RESPONSE"   | jq -r '.prompt_tokens')
                RUN_ID=$(echo "$RESPONSE"   | jq -r '.run_id')

                # Post to PR comment via ADO REST API
                curl -sf -X POST \
                  "$(System.TeamFoundationCollectionUri)$(System.TeamProject)/_apis/git/repositories/$(Build.Repository.Name)/pullRequests/$(System.PullRequest.PullRequestId)/threads?api-version=7.1" \
                  -H "Authorization: Bearer $ADO_TOKEN" \
                  -H "Content-Type: application/json" \
                  -d "{
                    \"comments\": [{
                      \"parentCommentId\": 0,
                      \"content\": \"## CI Triage\n\n$ANALYSIS\n\n---\n_prompt tokens: $TOKENS · run: $RUN_ID_\",
                      \"commentType\": 1
                    }],
                    \"status\": 1
                  }"
```

### 7.2 GitHub Actions

```yaml
# .github/workflows/triage.yml

name: CI Triage
on:
  workflow_run:
    workflows: ["CI"]
    types: [completed]

jobs:
  triage:
    if: ${{ github.event.workflow_run.conclusion == 'failure' }}
    runs-on: ubuntu-latest
    permissions:
      pull-requests: write

    steps:
      - uses: actions/checkout@v4
        with:
          ref: ${{ github.event.workflow_run.head_sha }}
          fetch-depth: 0

      - name: Download CI log artifact
        uses: actions/download-artifact@v4
        with:
          name: ci-log
          run-id: ${{ github.event.workflow_run.id }}
          github-token: ${{ secrets.GITHUB_TOKEN }}

      - name: Bundle source files
        run: |
          find . -name "*.py" \
            -not -path "./.venv/*" \
            -not -path "./node_modules/*" \
            | tar czf source.tar.gz -T -

      - name: Triage failure
        env:
          TRIAGE_API_KEY: ${{ secrets.TRIAGE_API_KEY }}
          TRIAGE_ENDPOINT: ${{ secrets.TRIAGE_ENDPOINT }}
        run: |
          RESPONSE=$(curl -sf -X POST "$TRIAGE_ENDPOINT/triage" \
            -H "Ocp-Apim-Subscription-Key: $TRIAGE_API_KEY" \
            -H "Content-Type: application/json" \
            -d "{
              \"raw_log\":         $(cat ci.log | jq -Rs .),
              \"raw_diff\":        $(git diff origin/${{ github.event.workflow_run.head_branch }} | jq -Rs .),
              \"source_archive\":  $(base64 -w0 source.tar.gz | jq -Rs .)
            }")

          echo "ANALYSIS<<EOF" >> $GITHUB_ENV
          echo "$RESPONSE" | jq -r '.analysis' >> $GITHUB_ENV
          echo "EOF" >> $GITHUB_ENV
          echo "RUN_ID=$(echo "$RESPONSE" | jq -r '.run_id')" >> $GITHUB_ENV
          echo "TOKENS=$(echo "$RESPONSE" | jq -r '.prompt_tokens')" >> $GITHUB_ENV

      - name: Post PR comment
        uses: actions/github-script@v7
        with:
          script: |
            const prNumber = context.payload.workflow_run.pull_requests[0]?.number;
            if (!prNumber) { core.warning('No PR found for this run'); return; }
            await github.rest.issues.createComment({
              owner: context.repo.owner,
              repo: context.repo.repo,
              issue_number: prNumber,
              body: `## CI Triage\n\n${process.env.ANALYSIS}\n\n---\n_prompt tokens: ${process.env.TOKENS} · run: ${process.env.RUN_ID}_`
            });
```

---

## 8. Agent deployment pipeline

This is the pipeline that builds and deploys the triage agent itself — separate from
the CI integration in section 7.

```mermaid
flowchart LR
    subgraph dev ["Developer"]
        PR["Pull Request\nto main"]
    end

    subgraph build ["Build Stage"]
        Lint["ruff / mypy"]
        Test["pytest\n(16 tests, offline)"]
        Bench["benchmarks/compare.py\n(token regression gate)"]
        Docker["docker build\n(pin base image by digest)"]
        Push["docker push\nto ACR"]
    end

    subgraph deploy ["Deploy Stage"]
        DevDeploy["Deploy to\ntriage-agent-dev\n(Container App)"]
        SmokeTest["Smoke test\nPOST /triage --demo"]
        ProdApproval["Manual approval\n(required for prod)"]
        ProdDeploy["Deploy to\ntriage-agent-prod\n(Container App)"]
        Rollback["Rollback trigger\n(alert → revision rollback)"]
    end

    PR --> Lint --> Test --> Bench
    Bench -->|"passes"| Docker --> Push
    Push --> DevDeploy --> SmokeTest
    SmokeTest -->|"passes"| ProdApproval --> ProdDeploy
    ProdDeploy -.->|"token metric spike"| Rollback
```

### Token regression gate

`benchmarks/compare.py` is runnable with no API key. Add it as a build step and fail
the pipeline if the optimised token total exceeds a threshold:

```bash
python benchmarks/compare.py --assert-optimised-below 12000
```

This makes the README's 9,205-token number a hard CI gate — a change that regresses
token efficiency fails before it can reach production.

### Container App revision strategy

Container Apps supports **revision-based deployments**. The deploy step creates a new
revision with zero traffic, runs the smoke test against it, then shifts 100% of traffic
to it. If the `prompt_tokens_per_run` metric spikes above the alert threshold within
10 minutes of the traffic shift, the revision is rolled back automatically via an Azure
Monitor action group calling the Container Apps revision deactivation API.

---

## 9. Observability

### 9.1 Custom metrics

The graph accumulates `prompt_tokens` in `TriageState`. Emit it to Application
Insights after every run alongside the raw log token count so the compression ratio is
directly observable.

```python
# In the FastAPI wrapper, after run() returns:
from applicationinsights import TelemetryClient

tc = TelemetryClient(os.environ["APPINSIGHTS_CONNECTION_STRING"])
tc.track_metric("prompt_tokens_per_run",   result["prompt_tokens"])
tc.track_metric("raw_log_tokens",          count_tokens(raw_log))
tc.track_metric("compression_ratio",
    count_tokens(raw_log) / max(result["prompt_tokens"], 1))
tc.track_metric("cache_hit",               1 if cache_hit else 0)
tc.track_metric("agent_turns",             turn_count)
tc.flush()
```

### 9.2 Alerts

| Alert name | Condition | Action |
|---|---|---|
| Token budget regression | `prompt_tokens_per_run > 20,000` (avg over 5 min) | Email + Teams webhook |
| High error rate | HTTP 5xx > 5% over 5 min | Page on-call |
| Redis cache miss rate | `cache_hit < 0.3` (avg over 1 hour) | Email — investigate model or prompt change |
| Cost spike | Azure OpenAI spend > $50/day | Email |
| Revision rollback trigger | `prompt_tokens_per_run > 15,000` within 10 min of deploy | Container App revision rollback |

### 9.3 Dashboard panels

Recommended Azure Monitor workbook layout:

```
┌──────────────────────────┬──────────────────────────┬──────────────────────┐
│  Runs today              │  Avg prompt tokens/run   │  Cache hit rate      │
│  [count]                 │  [gauge vs 9,205 target] │  [percentage]        │
├──────────────────────────┼──────────────────────────┼──────────────────────┤
│  Prompt tokens over time (line chart, 7-day rolling average)              │
├──────────────────────────┬──────────────────────────┬──────────────────────┤
│  Compression ratio       │  Avg agent turns/run     │  Azure OpenAI spend  │
│  (raw / prompt tokens)   │  [gauge vs limit 12]     │  [cost this month]   │
├──────────────────────────┴──────────────────────────┴──────────────────────┤
│  P95 request latency (ms) — line chart                                    │
│  Error rate (%) — line chart                                              │
└────────────────────────────────────────────────────────────────────────────┘
```

### 9.4 Distributed tracing

LangSmith tracing is available by setting two environment variables. Add them as Key
Vault references on the Container App for production tracing without code changes:

```
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=<langsmith-key>
LANGCHAIN_PROJECT=triage-agent-prod
```

LangSmith gives per-turn token counts, tool call traces, and latency breakdowns — the
complement to the aggregate Azure Monitor metrics.

---

## 10. Cost model

Based on 200 runs/day. All prices are approximate USD, East US region, as of 2025.

| Service | Tier | Monthly cost |
|---|---|---|
| Azure OpenAI — GPT-4o (input) | 9,205 tokens × 200/day × 30 days × $2.50/M | ~$14 |
| Azure OpenAI — GPT-4o (output) | ~500 tokens × 200/day × 30 days × $10/M | ~$30 |
| Azure OpenAI — GPT-4o-mini (summaries) | ~1,000 tokens × 200/day × 30 days × $0.15/M | ~$1 |
| Azure Cache for Redis — C1 Basic | Fixed | ~$55 |
| Azure Container Apps | ~1 vCPU-hour/day × 30 × $0.036 | ~$1 |
| Azure API Management — Consumption | 6,000 calls/month × $0.0035/1k | ~$0.02 |
| Azure Blob Storage — LRS Hot | ~5 GB × $0.018/GB | ~$0.09 |
| Azure Container Registry — Basic | Fixed | ~$5 |
| Azure Key Vault | ~20,000 operations × $0.03/10k | ~$0.06 |
| Application Insights | ~500 MB/month × $2.30/GB | ~$1 |
| **Total** | | **~$107/month** |

### Comparison with naive implementation

| Scenario | Monthly cost |
|---|---|
| Naive (201,233 tokens/run, GPT-4o) | ~$3,018 |
| Optimised on Azure (9,205 tokens/run) | ~$107 |
| **Saving** | **~$2,911/month (96.5%)** |

The token-efficiency work pays for the entire Azure hosting cost (including Redis, APIM,
Container Registry, etc.) with **~27× headroom** against a naive GPT-4o implementation.

### Scaling to higher volume

At 1,000 runs/day (5× current):

- Azure OpenAI cost scales linearly: ~$225/month
- Redis, APIM, Container Registry costs are flat
- Container Apps scales automatically; compute cost ~$5/month
- Total: ~$290/month (vs. naive ~$15,090/month)

---

## 11. Code changes required

The five token-saving levers deploy as-is. Only three changes are needed to run in
Azure.

### 11.1 Redis cache (`cache.py`)

```python
def enable_response_cache(path: str | None = ".langchain.db") -> None:
    redis_url = os.getenv("REDIS_URL")
    if redis_url:
        from langchain_community.cache import RedisCache
        import redis as _redis
        set_llm_cache(RedisCache(redis_=_redis.from_url(redis_url), ttl=604800))
        return
    # existing SQLite / InMemoryCache fallback unchanged below
    ...
```

### 11.2 Azure OpenAI client (`run_triage.py` and the FastAPI wrapper)

```python
from langchain_openai import AzureChatOpenAI
from azure.identity import DefaultAzureCredential, get_bearer_token_provider

token_provider = get_bearer_token_provider(
    DefaultAzureCredential(), "https://cognitiveservices.azure.com/.default"
)

llm = AzureChatOpenAI(
    azure_deployment=os.environ["SMART_MODEL"],
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    api_version="2024-08-01-preview",
    azure_ad_token_provider=token_provider,
    temperature=0,
)
```

### 11.3 Source file access

The `tools.py` functions call `Path.rglob("*.py")` on the local filesystem. In the
Container App, there is no local checkout. The FastAPI wrapper receives a `source_archive`
field (base64-encoded tar.gz from the CI pipeline), extracts it to a per-request temp
directory, and passes that path as the working root.

```python
import base64, tarfile, tempfile

@app.post("/triage")
async def triage(req: TriageRequest) -> TriageResponse:
    with tempfile.TemporaryDirectory() as source_root:
        if req.source_archive:
            archive_bytes = base64.b64decode(req.source_archive)
            with tarfile.open(fileobj=io.BytesIO(archive_bytes)) as tar:
                tar.extractall(source_root)

        os.environ["SOURCE_ROOT"] = source_root   # tools.py reads this
        result = run(llm, req.raw_log, req.raw_diff)
    ...
```

These three changes are the only ones that affect agent behaviour. Everything in
`log_pruner.py`, `context.py`, `windows.py`, `tokens.py`, `prompts.py`, `config.py`,
and `graph.py` is unchanged.

---

## 12. Phased rollout

```mermaid
gantt
    dateFormat  YYYY-MM-DD
    title       Deployment Phases

    section Phase 1 — Containerise
    Dockerfile + FastAPI wrapper       :p1a, 2025-10-01, 3d
    Local Docker smoke test            :p1b, after p1a, 1d
    Deploy to Container Apps (dev)     :p1c, after p1b, 1d
    OpenAI API key via Key Vault ref   :p1d, after p1c, 1d

    section Phase 2 — Azure-native
    Switch to AzureChatOpenAI          :p2a, after p1d, 2d
    Replace SQLite with Redis          :p2b, after p2a, 1d
    Private endpoints + VNet           :p2c, after p2b, 2d
    Multi-replica smoke test           :p2d, after p2c, 1d

    section Phase 3 — CI integration
    ADO pipeline step                  :p3a, after p2d, 2d
    GitHub Actions workflow            :p3b, after p2d, 2d
    App Insights metrics + alerts      :p3c, after p3a, 2d
    Token regression gate in build     :p3d, after p3c, 1d
    Prod deployment + approval gate    :p3e, after p3d, 1d

    section Phase 4 — Vector retrieval (optional)
    Azure AI Search index (incidents)  :p4a, after p3e, 5d
    Wire retrieval.py to AI Search     :p4b, after p4a, 3d
    Recall vs cost evaluation          :p4c, after p4b, 2d
```

### Phase gates

| Gate | Criterion | Blocks |
|---|---|---|
| Phase 1 → 2 | `POST /triage --demo` returns a valid diagnosis in dev | Phase 2 start |
| Phase 2 → 3 | Multi-replica test: same prompt from 2 replicas returns identical result (Redis cache hit on second call) | Phase 3 start |
| Phase 3 → prod | Token regression gate passes: `benchmarks/compare.py` reports < 12,000 optimised tokens | Prod deployment |
| Prod → Phase 4 | 2 weeks stable in prod with no budget regression alerts | Phase 4 start |
