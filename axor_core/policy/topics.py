"""
axor_core.policy.topics
───────────────────────

Domain vocabulary used by `keyword_relevance` for tool selection scoring.
Pure data — zero runtime dependencies, zero provider SDKs.

Why this lives in core:
  Keyword overlap on tool names alone fails the moment task language and
  tool-description language diverge ("vulnerable code paths" vs "security
  findings"). This module bridges that gap with hand-curated topic
  vocabularies and one-hop implication edges. Provider adapters
  (axor-langchain, axor-claude) import from here so domain knowledge
  stays in one place across providers.

Maintenance:
  • Add new words to existing topics rather than inventing new topics.
  • Multi-topic membership is fine and intentional ("incident" lives in
    both `incident` and `tickets`).
  • One-hop only on `TOPIC_IMPLICATIONS` — propagation is computed in
    `keyword_relevance.compute_topic_strength`, not encoded here.
"""
from __future__ import annotations


# Words filtered out from query-keyword extraction. Includes generic
# English stopwords plus filler that adds no signal in production-agent
# prompts ("please", "evidence", "task").
STOPWORDS: frozenset[str] = frozenset({
    "a", "an", "the", "and", "or", "for", "to", "in", "of", "is", "it", "this",
    "that", "with", "from", "by", "on", "be", "are", "was", "were", "as",
    "at", "but", "if", "then", "than", "into", "over", "under", "out", "up",
    "down", "do", "does", "did", "have", "has", "had", "can", "could", "would",
    "should", "may", "might", "will", "shall", "must", "your", "you", "we",
    "our", "i", "me", "my", "they", "them", "their", "his", "her", "its",
    "use", "using", "make", "made", "get", "got", "set", "any", "all", "some",
    "what", "which", "who", "how", "why", "when", "where",
    "please", "task", "tasks", "production", "concrete", "evidence",
})


# Tokens that almost always mark a destructive/write tool. Used by
# adapters to drop write-tools when a classifier says READONLY. Token-
# based matching only — `\b...\b` regexes do NOT see underscore as a
# word boundary, so `\bdeploy\b` would slip past `deploy_service`.
DESTRUCTIVE_TOKENS: frozenset[str] = frozenset({
    "write", "delete", "drop", "push", "exec", "kill", "stop", "terminate",
    "deploy", "deployment", "publish", "shutdown", "reset", "create",
    "update", "insert", "patch", "put", "post", "send", "email", "notify",
    "charge", "refund", "migrate", "rollback", "remove", "edit", "modify",
    "destroy", "purge", "wipe", "rm",
})


# Domain topics: each topic = a set of semantically-related terms.
# Keywords expand by topic membership; multi-topic words split their
# contribution evenly across topics so ambiguous tokens don't unfairly
# dominate any single topic.
DOMAIN_TOPICS: dict[str, set[str]] = {
    "observability": {
        "latency", "trace", "traces", "tracing", "log", "logs", "logging",
        "metric", "metrics", "monitoring", "monitor", "alert", "alerting",
        "dashboard", "dashboards", "span", "spans", "timing", "timings",
        "p50", "p90", "p95", "p99", "ms", "duration", "slow", "spike",
        "perf", "performance", "telemetry", "observability", "gauge",
        "counter", "histogram", "rate", "throughput", "saturation", "qps",
        "rps", "tps", "tail", "percentile", "slo", "sli", "sla", "uptime",
        "availability", "downtime", "burn",
    },
    "incident": {
        "incident", "incidents", "outage", "outages", "rca", "postmortem",
        "regression", "regressions", "spike", "alert", "page", "paged",
        "oncall", "downtime", "sev", "sev1", "sev2", "severity", "mitigation",
        "mitigate", "mitigations", "blast", "radius", "escalation",
        "escalate", "rollback", "rolledback", "timeline", "cause", "causes",
        "root", "trigger", "triggered", "evidence", "factor", "factors",
        "contributing", "war", "room", "fire", "firefight", "drill",
        "remediation", "remediate", "prevention", "prevent",
    },
    "code": {
        "code", "codebase", "source", "sources", "file", "files", "module",
        "modules", "function", "functions", "class", "classes", "method",
        "methods", "repo", "repository", "repositories", "package", "packages",
        "import", "imports", "library", "libraries", "implementation",
        "implementations", "grep", "search", "find", "lookup", "snippet",
        "snippets", "line", "lines", "ast", "syntax", "symbol", "symbols",
        "definition", "definitions", "callsite", "callsites", "branch",
        "branches", "diff", "patch",
    },
    "testing": {
        "test", "tests", "testing", "spec", "specs", "unittest", "pytest",
        "integration", "e2e", "smoke", "regression", "suite", "failing",
        "flaky", "coverage", "fixture", "fixtures", "mock", "mocks", "stub",
        "stubs", "assertion", "assertions", "passed", "failed", "skipped",
        "harness", "tdd", "bdd", "qa",
    },
    "security": {
        "security", "secure", "vulnerability", "vulnerabilities", "vulnerable",
        "exploit", "exploits", "cve", "cves", "audit", "audits", "auditing",
        "auth", "authn", "authz", "oauth", "oidc", "saml", "sso", "token",
        "tokens", "jwt", "jwts", "credential", "credentials", "permission",
        "permissions", "scope", "scopes", "role", "roles", "leak", "leaked",
        "breach", "breached", "spoofing", "spoof", "tampering", "tamper",
        "validate", "validation", "validator", "principal", "subject",
        "tenant", "tenants", "tenancy", "encrypt", "encryption", "decrypt",
        "decryption", "ssl", "tls", "mtls", "cipher", "ciphers", "key",
        "keys", "secret", "secrets", "cors", "csrf", "xss", "ssrf",
        "injection", "rbac", "iam", "policy", "policies", "acl", "acls",
        "rotation", "rotate", "expiry", "expire", "expired", "audience",
        "issuer", "claims", "session", "sessions", "trust", "trusted",
        "untrusted", "harden", "hardening", "compliance", "regulatory",
        "gdpr", "soc2", "pii",
    },
    "dependencies": {
        "dependency", "dependencies", "dep", "deps", "package", "packages",
        "library", "libraries", "version", "versions", "pin", "pinned",
        "upgrade", "upgrades", "downgrade", "lockfile", "vendor", "vendored",
        "supply", "transitive", "outdated", "deprecated", "deprecation",
        "vendoring", "manifest", "requirements", "pyproject", "package_json",
        "npm", "pypi", "cargo", "gem",
    },
    "cost": {
        "cost", "costs", "costly", "spend", "spending", "spent", "bill",
        "billed", "billing", "expense", "expenses", "expensive", "budget",
        "budgets", "money", "monthly", "savings", "save", "saving",
        "optimization", "optimize", "optimizing", "efficiency", "efficient",
        "throughput", "input_tokens", "output_tokens",
        # Note: bare "tokens" / "model" intentionally NOT here — they are
        # security/agentic ambiguous and would leak cost-topic boosts into
        # security tasks ("OAuth tokens"). The compound forms above carry
        # unambiguous cost intent.
        "compute", "gpu", "cpu", "ram", "memory", "storage", "egress",
        "ingress", "amortize", "amortized", "tco", "roi", "burn",
    },
    "migration": {
        "migration", "migrations", "migrate", "migrating", "rollout",
        "rollouts", "phases", "phase", "phased", "rollback", "rollbacks",
        "backout", "revert", "cutover", "transition", "transitioning",
        "refactor", "refactoring", "rewrite", "rewriting", "sunset",
        "sunsetting", "deprecate", "deprecating", "compatibility",
        "compat", "bridge", "adapter", "shim",
    },
    "ops": {
        "deploy", "deployed", "deployment", "deployments", "release",
        "released", "releases", "canary", "rollout", "rollouts", "rollback",
        "production", "prod", "staging", "stage", "ops", "devops", "ci",
        "cd", "cicd", "pipeline", "pipelines", "build", "builds", "infra",
        "infrastructure", "kubernetes", "k8s", "docker", "container",
        "containers", "helm", "terraform", "ansible",
    },
    "customer": {
        "customer", "customers", "user", "users", "tenant", "tenants",
        "impact", "impacted", "affected", "lost", "revenue", "churn",
        "support", "escalation", "escalations", "client", "clients",
        "segment", "segments", "cohort", "cohorts", "enterprise",
        "consumer", "consumers", "account", "accounts",
    },
    "docs": {
        "design", "doc", "docs", "document", "documents", "documentation",
        "spec", "specs", "specification", "rfc", "rfcs", "adr", "adrs",
        "proposal", "proposals", "plan", "plans", "playbook", "playbooks",
        "runbook", "runbooks", "guide", "guides", "manual", "readme",
        "wiki", "page", "pages", "kb", "knowledge",
    },
    "architecture": {
        "service", "services", "microservice", "microservices", "module",
        "modules", "component", "components", "system", "systems",
        "infrastructure", "platform", "platforms", "stack", "layer",
        "layers", "tier", "tiers", "topology", "boundary", "boundaries",
        "domain", "subdomain", "context", "bounded",
    },
    "errors": {
        "error", "errors", "failure", "failures", "fail", "failed",
        "broken", "crash", "crashed", "crashing", "exception", "exceptions",
        "bug", "bugs", "fault", "faults", "panic", "traceback", "stacktrace",
        "stack_trace", "abort", "aborted", "timeout", "timeouts",
        "deadline", "deadlines", "retries", "retry", "retried", "5xx",
        "4xx",
    },
    "tickets": {
        "ticket", "tickets", "issue", "issues", "jira", "linear", "story",
        "stories", "epic", "epics", "kanban", "backlog", "sprint",
    },
    "data": {
        "database", "databases", "db", "dbs", "table", "tables", "query",
        "queries", "sql", "nosql", "schema", "schemas", "row", "rows",
        "index", "indexes", "indexing", "transaction", "transactions",
        "txn", "commit", "rollback", "replica", "replication", "shard",
        "sharding", "partition", "partitioning", "lock", "deadlock",
        "consistency", "isolation", "acid", "olap", "oltp", "etl",
        "warehouse", "lake",
    },
    "api": {
        "api", "apis", "endpoint", "endpoints", "request", "requests",
        "response", "responses", "header", "headers", "rest", "grpc",
        "http", "https", "url", "urls", "route", "routes", "router",
        "routing", "webhook", "webhooks", "graphql", "rpc",
    },
    "read_actions": {
        "read", "reads", "reading", "inspect", "inspecting", "inspection",
        "view", "views", "show", "showing", "dump", "list", "listing",
        "tree", "browse", "browsing", "fetch", "fetching", "get", "load",
        "loading", "lookup", "examine", "examining", "audit", "auditing",
        "investigate", "investigating", "investigation", "explore",
        "exploring", "analyze", "analyzing", "analysis", "diagnose",
        "diagnosing", "diagnostic", "review", "reviewing",
    },
    "write_actions": {
        "write", "writing", "create", "creating", "modify", "modifying",
        "update", "updating", "edit", "editing", "patch", "patching",
        "insert", "inserting", "delete", "deleting", "remove", "removing",
        "drop", "dropping", "push", "pushing", "commit", "committing",
        "merge", "merging",
    },
    "process": {
        "workflow", "workflows", "pipeline", "pipelines", "agent", "agents",
        "graph", "graphs", "node", "nodes", "checkpoint", "checkpoints",
        "state", "session", "sessions", "turn", "turns", "step", "steps",
        "stage", "stages",
    },
    "billing_domain": {
        "billing", "invoice", "invoices", "subscription", "subscriptions",
        "charge", "charges", "refund", "refunds", "payment", "payments",
        "checkout", "cart", "order", "orders", "purchase", "purchases",
        "transaction", "transactions", "inventory", "stock", "fulfillment",
        "shipping", "tax", "taxes", "currency", "discount", "discounts",
        "coupon", "coupons", "pricing", "price",
    },
    "agentic": {
        "agent", "agents", "agentic", "llm", "llms", "model", "models",
        "prompt", "prompts", "prompting", "context", "contexts", "memory",
        "tool", "tools", "tooluse", "reasoning", "reason", "thought",
        "thoughts", "completion", "completions", "inference", "research",
        "researcher", "writer", "synthesis",
    },
    "investigation": {
        "investigate", "investigation", "investigating", "diagnose",
        "diagnosis", "diagnosing", "diagnostic", "analyze", "analysis",
        "analyzing", "examine", "examining", "audit", "review",
        "reviewing", "find", "identify", "identifying", "discover",
        "discovery", "uncover", "trace", "tracing", "explore", "exploring",
        "deep", "deeper", "drill",
    },
    "planning": {
        "plan", "planning", "plans", "roadmap", "roadmaps", "strategy",
        "strategic", "tactical", "phase", "phases", "phased", "milestone",
        "milestones", "sequence", "sequencing", "ordering", "priority",
        "prioritize", "prioritized", "rollout", "scope", "scoping",
        "estimate", "estimation", "estimating",
    },
    "quality": {
        "quality", "qualitative", "quantitative", "metric", "metrics",
        "kpi", "kpis", "measure", "measurement", "evaluate", "evaluation",
        "evaluating", "assess", "assessment", "verify", "verification",
        "verifying", "validate", "validation",
    },
}


# One-hop topic implications: when the task is strongly about topic X,
# tools in implied topics get partial alignment too. Encodes domain
# knowledge that pure word-overlap can't recover (e.g., RCA prompt that
# says "blast radius" but never "code" still needs source-file tools).
TOPIC_IMPLICATIONS: dict[str, list[str]] = {
    # Incident RCA forensics: code+test+log+impact, NOT design docs.
    "incident": [
        "observability", "errors", "code", "testing", "ops", "customer",
        "investigation", "tickets",
    ],
    # Generic investigation: observability + code + tests + tickets.
    # Excludes docs intentionally — design-doc reading is a security/
    # migration concern, not an incident-RCA one.
    "investigation": [
        "observability", "incident", "code", "testing", "errors", "tickets",
    ],
    # Security review touches design docs and tickets heavily.
    "security": [
        "code", "dependencies", "docs", "tickets", "errors", "data", "api",
    ],
    # Migration planning needs design docs.
    "migration": [
        "code", "testing", "docs", "ops", "dependencies", "customer",
    ],
    # Cost analysis: workflows, agent runs, observability counters,
    # plus design docs and tests (architectural cost drivers + test
    # workflows that bloat token spend).
    "cost": [
        "observability", "agentic", "code", "ops", "process", "docs",
        "testing",
    ],
    "observability": [
        "incident", "errors", "code",
    ],
    "errors": [
        "observability", "code", "testing", "incident",
    ],
    "dependencies": [
        "security", "code", "errors",
    ],
    "ops": [
        "incident", "observability", "errors",
    ],
    "billing_domain": [
        "data", "api", "customer", "errors",
    ],
}


def _build_word_topics(topics: dict[str, set[str]]) -> dict[str, set[str]]:
    """Inverted index: word → set of topics it belongs to."""
    inv: dict[str, set[str]] = {}
    for topic, words in topics.items():
        for w in words:
            inv.setdefault(w, set()).add(topic)
    return inv


def _build_synonym_map(
    topics: dict[str, set[str]],
    implications: dict[str, list[str]],
) -> dict[str, set[str]]:
    """
    Per-word synonym universe. Each word resolves to the union of:
      (a) every topic it directly belongs to, plus
      (b) every topic implied by those topics (one hop).
    """
    word_topics = _build_word_topics(topics)
    m: dict[str, set[str]] = {}
    for word, ts in word_topics.items():
        universe: set[str] = set()
        for t in ts:
            universe |= topics[t]
            for implied in implications.get(t, []):
                universe |= topics.get(implied, set())
        m[word] = universe
    return m


# Precomputed at import time — these are pure-data and cheap.
WORD_TOPICS: dict[str, set[str]] = _build_word_topics(DOMAIN_TOPICS)
SYNONYM_MAP: dict[str, set[str]] = _build_synonym_map(
    DOMAIN_TOPICS, TOPIC_IMPLICATIONS,
)
