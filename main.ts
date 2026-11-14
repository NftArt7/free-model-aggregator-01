// Free-model aggregator for the make.fun free-model-aggregator-01 competition.
//
// One Node (stdlib only, no dependencies) server that answers OpenAI Chat
// Completions (POST /v1/chat/completions) and Anthropic Messages
// (POST /v1/messages) by forwarding each request to the best available free
// model on one of ten simulated upstream providers, and that fails over to
// the next-best free model whenever an upstream call is refused (402/429)
// or otherwise fails.
//
// How the server learns its world:
//   * The trial brief (MAKEFUN_PROMPT_FILE) is a Markdown document that
//     embeds three JSON documents: a provider directory, a free-model
//     ratings table, and a workload manifest of the twenty requests the
//     trial will send. The server finds them by scanning for JSON objects
//     and matching their shape, so it never depends on their order.
//   * At startup the server probes every provider's catalog route (in that
//     provider's own wire) and builds one catalog. That catalog is the
//     document registered with the trial environment and the document
//     `--list` prints, byte for byte.
//   * At runtime each request is routed greedily: free models that meet the
//     request's requirements (tools, vision, JSON, minimum context) are
//     ranked by their rating, and the server tries them in that order until
//     one answers. A provider that refuses repeatedly is set aside for the
//     rest of the trial; a provider that refused once or twice is kept,
//     because a refusal can be a scheduled single-call failure rather than
//     a spent budget.
//
// Determinism is a first-class property here: no wall clock, no unseeded
// randomness, fixed response ids and timestamps, and a fixed candidate
// order, so replaying a trial produces byte-identical state.
//
// Usage:
//   node main.ts --list          print the model catalog and exit
//   node main.ts                 serve the trial (MAKEFUN_ENV_URL and
//                                MAKEFUN_PROMPT_FILE set by the harness)

import { createServer } from "node:http";
import { readFileSync } from "node:fs";

// ---------------------------------------------------------------------------
// Small shared helpers
// ---------------------------------------------------------------------------

const ANTHROPIC_VERSION = "2023-06-01";
const FIXED_CREATED = 1767225600; // 2026-01-01T00:00:00Z, fixed for replay
const FIXED_CREATED_AT = "2026-01-01T00:00:00Z";
const RESPONSE_ID = "chatcmpl-agg";
const MESSAGE_ID = "msg_agg";
const PROTOCOL_VERSION = 1;
const POLL_MS = 100;
const PROBE_TIMEOUT_MS = 5000;
const CALL_TIMEOUT_MS = 10000;
const MAX_PROVIDER_FAILURES = 3; // failures before a provider is set aside

type Json = Record<string, unknown>;

function isRecord(v: unknown): v is Json {
  return typeof v === "object" && v !== null && !Array.isArray(v);
}

function asArray(v: unknown): unknown[] {
  return Array.isArray(v) ? v : [];
}

function asString(v: unknown): string {
  return typeof v === "string" ? v : "";
}

function asNumber(v: unknown, fallback = 0): number {
  return typeof v === "number" && Number.isFinite(v) ? v : fallback;
}

function log(...parts: unknown[]): void {
  console.error("[aggregator]", ...parts);
}

// Join the URL path of a provider base (an absolute path such as /p/groq)
// onto the environment root, without ever producing a double slash.
function joinBase(root: string, path: string): string {
  const cleanRoot = root.replace(/\/+$/, "");
  const cleanPath = path.startsWith("/") ? path : "/" + path;
  return cleanRoot + cleanPath;
}

// ---------------------------------------------------------------------------
// Brief parsing: find the three embedded JSON documents by their shape
// ---------------------------------------------------------------------------

interface ProviderDir {
  id: string;
  base_url: string;
  key_env: string;
  wire: string;
  // The auth scheme names the header: "bearer", "x-goog-api-key",
  // "x-api-key", or "none". It is independent of the wire: zai speaks the
  // Anthropic wire but authenticates with a Bearer token.
  auth: string;
  // The brief embeds each provider's key as well; the environment variable
  // is preferred at runtime and this is only a fallback.
  key: string;
}

interface RatingRow {
  provider: string;
  model: string;
  free: boolean;
  context_length: number;
  rating_milli: number;
  tools: boolean;
  vision: boolean;
  json: boolean;
}

interface WorkloadReq {
  id: string;
  wire: string;
  stream: boolean;
  weight: number;
  requires_tools: boolean;
  requires_vision: boolean;
  requires_json: boolean;
  min_context: number;
}

interface Brief {
  providers: ProviderDir[];
  ratings: RatingRow[];
  workload: WorkloadReq[];
}

// Scan a text for balanced JSON objects, yielding each one in order. The
// scan only tracks brace depth and string escapes, which is enough for the
// brief's embedded documents.
function* jsonObjects(text: string): IterableIterator<Json> {
  let i = 0;
  while (i < text.length) {
    const start = text.indexOf("{", i);
    if (start === -1) return;
    let depth = 0;
    let inString = false;
    let escaped = false;
    let j = start;
    for (; j < text.length; j++) {
      const ch = text[j];
      if (inString) {
        if (escaped) escaped = false;
        else if (ch === "\\") escaped = true;
        else if (ch === '"') inString = false;
      } else if (ch === '"') {
        inString = true;
      } else if (ch === "{") {
        depth++;
      } else if (ch === "}") {
        depth--;
        if (depth === 0) break;
      }
    }
    if (depth !== 0) return;
    try {
      const parsed: unknown = JSON.parse(text.slice(start, j + 1));
      if (isRecord(parsed)) yield parsed;
    } catch {
      // Not a standalone object; keep scanning after it.
    }
    i = j + 1;
  }
}

function readBrief(path: string | undefined): Brief {
  const brief: Brief = { providers: [], ratings: [], workload: [] };
  if (!path) return brief;
  let text: string;
  try {
    text = readFileSync(path, "utf8");
  } catch {
    log("brief unreadable:", path);
    return brief;
  }
  for (const doc of jsonObjects(text)) {
    if (Array.isArray(doc.providers)) {
      brief.providers = asArray(doc.providers)
        .filter(isRecord)
        .map((p) => ({
          id: asString(p.id),
          // The brief names the provider's path "path"; the catalog and the
          // rules call the same value base_url.
          base_url: asString(p.base_url) || asString(p.path),
          key_env: asString(p.key_env),
          wire: asString(p.wire),
          auth: asString(p.auth),
          key: asString(p.key),
        }))
        .filter((p) => p.id && p.base_url);
    } else if (Array.isArray(doc.ratings) || Array.isArray(doc.models)) {
      brief.ratings = asArray(doc.ratings ?? doc.models)
        .filter(isRecord)
        .map((m) => ({
          provider: asString(m.provider),
          model: asString(m.model),
          free: m.free === true,
          context_length: asNumber(m.context_length),
          rating_milli: asNumber(m.rating_milli),
          tools: m.tools === true,
          vision: m.vision === true,
          json: m.json === true,
        }))
        .filter((m) => m.provider && m.model);
    } else if (Array.isArray(doc.manifest) || Array.isArray(doc.requests)) {
      brief.workload = asArray(doc.manifest ?? doc.requests)
        .filter(isRecord)
        .map((r) => {
          // Requirements live in a nested "requires" object; flat fields
          // are accepted too for robustness.
          const req = isRecord(r.requires) ? r.requires : r;
          return {
            id: asString(r.id),
            wire: asString(r.api) || asString(r.wire),
            stream: r.stream === true,
            weight: asNumber(r.weight, 1),
            requires_tools: req.tools === true,
            requires_vision: req.vision === true,
            requires_json: req.json === true,
            min_context: asNumber(req.min_context),
          };
        })
        .filter((r) => r.id);
    }
  }
  return brief;
}

// ---------------------------------------------------------------------------
// Catalog: probe each provider in its own wire and build the registered
// document once. The same document is used for --list and for register.
// ---------------------------------------------------------------------------

interface CatalogModel {
  id: string;
  free: boolean;
  context_length: number;
}

interface CatalogProvider {
  id: string;
  base_url: string;
  key_env: string;
  wire: string;
  models: CatalogModel[];
}

interface Catalog {
  providers: CatalogProvider[];
}

interface ProbedModel {
  id: string;
  free: boolean | null;
  context_length: number;
}

function authHeaders(provider: ProviderDir, env: NodeJS.ProcessEnv): Record<string, string> {
  // The key comes from the environment first; the brief embeds the same key
  // and is the fallback, so the server also works when the harness only
  // hands over the brief.
  const key = env[provider.key_env] ?? provider.key ?? "";
  const headers: Record<string, string> = { "content-type": "application/json" };
  // The header is decided by the auth scheme, not the wire: zai speaks the
  // Anthropic wire but authenticates with a Bearer token.
  if (provider.auth === "bearer") {
    if (key) headers["authorization"] = "Bearer " + key;
  } else if (provider.auth === "x-goog-api-key") {
    if (key) headers["x-goog-api-key"] = key;
  } else if (provider.auth === "x-api-key") {
    if (key) headers["x-api-key"] = key;
  }
  if (provider.wire === "anthropic") headers["anthropic-version"] = ANTHROPIC_VERSION;
  return headers;
}

// One provider's catalog route, in its own wire. Returns the model rows the
// provider reports (id, free flag, context length) or null when the probe
// fails; a failed probe simply leaves that provider out of the catalog.
async function probeProvider(
  provider: ProviderDir,
  envUrl: string,
  env: NodeJS.ProcessEnv,
): Promise<ProbedModel[] | null> {
  const root = joinBase(envUrl, provider.base_url);
  let url: string;
  if (provider.wire === "gemini") url = root + "/v1beta/models";
  else url = root + "/v1/models";
  try {
    const res = await fetch(url, {
      method: "GET",
      headers: authHeaders(provider, env),
      signal: AbortSignal.timeout(PROBE_TIMEOUT_MS),
    });
    if (res.status !== 200) {
      log("probe", provider.id, "HTTP", res.status);
      return null;
    }
    const body: unknown = await res.json();
    if (!isRecord(body)) return null;
    const rows: ProbedModel[] = [];
    if (provider.wire === "gemini") {
      for (const m of asArray(body.models)) {
        if (!isRecord(m)) continue;
        const name = asString(m.name).replace(/^models\//, "");
        if (!name) continue;
        rows.push({
          id: name,
          free: typeof m.free === "boolean" ? m.free : null,
          context_length: asNumber(m.inputTokenLimit),
        });
      }
    } else {
      for (const m of asArray(body.data)) {
        if (!isRecord(m)) continue;
        const id = asString(m.id);
        if (!id) continue;
        rows.push({
          id,
          free: typeof m.free === "boolean" ? m.free : null,
          context_length: asNumber(m.context_length),
        });
      }
    }
    rows.sort((a, b) => (a.id < b.id ? -1 : a.id > b.id ? 1 : 0));
    return rows;
  } catch (err) {
    log("probe", provider.id, "failed:", err instanceof Error ? err.message : String(err));
    return null;
  }
}

// The ratings table is the sealed truth for free flags and context lengths;
// a probe listing is only a fallback when a row is absent.
function ratingFor(brief: Brief, providerId: string, modelId: string): RatingRow | undefined {
  return brief.ratings.find((r) => r.provider === providerId && r.model === modelId);
}

async function buildCatalog(brief: Brief, envUrl: string, env: NodeJS.ProcessEnv): Promise<Catalog> {
  const catalog: Catalog = { providers: [] };
  if (!envUrl) return catalog;
  for (const provider of brief.providers) {
    const probed = await probeProvider(provider, envUrl, env);
    if (!probed) continue;
    const models: CatalogModel[] = probed.map((p) => {
      const row = ratingFor(brief, provider.id, p.id);
      return {
        id: p.id,
        free: row ? row.free : p.free === true,
        context_length: row && row.context_length > 0 ? row.context_length : p.context_length,
      };
    });
    catalog.providers.push({
      id: provider.id,
      base_url: provider.base_url,
      key_env: provider.key_env,
      wire: provider.wire,
      models,
    });
  }
  return catalog;
}

// The set of free model ids across the whole catalog, deduplicated and
// sorted. Both /v1/models wires answer with exactly this set.
function freeModelIds(catalog: Catalog): string[] {
  const ids = new Set<string>();
  for (const p of catalog.providers) {
    for (const m of p.models) if (m.free) ids.add(m.id);
  }
  return [...ids].sort();
}

// Does a free model meet a request's requirements?
function meetsRequirements(
  row: RatingRow,
  req: WorkloadReq,
  jsonMode: boolean,
  wire: string,
): boolean {
  if (!row.free) return false;
  if (req.requires_tools && !row.tools) return false;
  if (req.requires_vision && !row.vision) return false;
  if (req.requires_json && !row.json) return false;
  if (req.min_context > 0 && row.context_length > 0 && row.context_length < req.min_context) return false;
  // A JSON-mode request cannot go to an Anthropic-wire provider: that wire
  // cannot carry a JSON mode, so the provider would echo json_mode false.
  if (jsonMode && wire === "anthropic") return false;
  return true;
}

// ---------------------------------------------------------------------------
// Client requests: extract the fields the router and the usage formula need
// ---------------------------------------------------------------------------

interface ClientRequest {
  wire: "openai" | "anthropic";
  stream: boolean;
  jsonMode: boolean;
  systemText: string;
  userText: string;
  toolNames: string[];
  maxTokens: number | null;
  stop: string[];
  temperature: number | null;
  topP: number | null;
  body: Json;
}

// The text of a message's content: a plain string as-is, or the text blocks
// of a content array joined with newlines (images contribute nothing).
function blocksText(content: unknown): string {
  if (typeof content === "string") return content;
  if (!Array.isArray(content)) return "";
  const texts: string[] = [];
  for (const block of content) {
    if (isRecord(block) && block.type === "text" && typeof block.text === "string") {
      texts.push(block.text);
    }
  }
  return texts.join("\n");
}

function numberOrNull(v: unknown): number | null {
  return typeof v === "number" && Number.isFinite(v) ? v : null;
}

function stopList(v: unknown): string[] {
  if (typeof v === "string") return [v];
  if (Array.isArray(v)) return v.filter((s): s is string => typeof s === "string");
  return [];
}

function extractClientRequest(body: Json, wire: "openai" | "anthropic"): ClientRequest {
  const req: ClientRequest = {
    wire,
    stream: body.stream === true,
    jsonMode: false,
    systemText: "",
    userText: "",
    toolNames: [],
    maxTokens: null,
    stop: [],
    temperature: numberOrNull(body.temperature),
    topP: numberOrNull(body.top_p),
    body,
  };
  if (wire === "openai") {
    const messages = asArray(body.messages).filter(isRecord);
    const systemParts: string[] = [];
    for (const m of messages) {
      if (m.role === "system") systemParts.push(blocksText(m.content));
    }
    req.systemText = systemParts.join("\n");
    for (let i = messages.length - 1; i >= 0; i--) {
      if (messages[i].role === "user") {
        req.userText = blocksText(messages[i].content);
        break;
      }
    }
    for (const t of asArray(body.tools)) {
      if (isRecord(t) && isRecord(t.function) && typeof t.function.name === "string") {
        req.toolNames.push(t.function.name);
      }
    }
    if (isRecord(body.response_format)) {
      const t = body.response_format.type;
      req.jsonMode = t === "json_object" || t === "json_schema";
    }
    req.stop = stopList(body.stop);
    req.maxTokens = numberOrNull(body.max_tokens) ?? numberOrNull(body.max_completion_tokens);
  } else {
    const system = body.system;
    if (typeof system === "string") req.systemText = system;
    else if (Array.isArray(system)) req.systemText = blocksText(system);
    const messages = asArray(body.messages).filter(isRecord);
    for (let i = messages.length - 1; i >= 0; i--) {
      if (messages[i].role === "user") {
        req.userText = blocksText(messages[i].content);
        break;
      }
    }
    for (const t of asArray(body.tools)) {
      if (isRecord(t) && typeof t.name === "string") req.toolNames.push(t.name);
    }
    req.stop = stopList(body.stop_sequences);
    req.maxTokens = numberOrNull(body.max_tokens);
  }
  return req;
}

// The usage the trial's checker expects: input from the request's system
// text, last user text, and tool names; output from the answer text.
function usageOf(client: ClientRequest, answerText: string): { input: number; output: number } {
  const input = 1 + Math.floor((client.systemText.length + client.userText.length + client.toolNames.join(",").length) / 4);
  const output = 1 + Math.floor(answerText.length / 4);
  return { input, output };
}

// ---------------------------------------------------------------------------
// Translation: client wire -> provider wire. The request's content (system,
// user text and images, tools, stop sequences, max tokens) is passed through
// untouched in the provider's own shape, so the provider's echo of it
// matches what the client sent.
// ---------------------------------------------------------------------------

function parseDataUrl(url: string): { mediaType: string; data: string } | null {
  const m = /^data:([^;]+);base64,(.*)$/s.exec(url);
  return m ? { mediaType: m[1], data: m[2] } : null;
}

function openAIImagePart(url: string): Json {
  return { type: "image_url", image_url: { url } };
}

function anthropicImageBlock(url: string): Json {
  const parsed = parseDataUrl(url);
  if (parsed) {
    return { type: "image", source: { type: "base64", media_type: parsed.mediaType, data: parsed.data } };
  }
  return { type: "image", source: { type: "url", url } };
}

// Anthropic client -> OpenAI provider.
function toOpenAIProvider(client: ClientRequest, model: string): Json {
  const body = client.body;
  const out: Json = { model };
  const messages: Json[] = [];
  if (client.systemText) messages.push({ role: "system", content: client.systemText });
  for (const m of asArray(body.messages)) {
    if (!isRecord(m)) continue;
    if (m.role === "user") {
      const content = m.content;
      if (typeof content === "string") {
        messages.push({ role: "user", content });
      } else if (Array.isArray(content)) {
        const parts: Json[] = [];
        for (const block of content) {
          if (!isRecord(block)) continue;
          if (block.type === "text" && typeof block.text === "string") {
            parts.push({ type: "text", text: block.text });
          } else if (block.type === "image" && isRecord(block.source)) {
            if (block.source.type === "base64") {
              const url = `data:${asString(block.source.media_type)};base64,${asString(block.source.data)}`;
              parts.push(openAIImagePart(url));
            } else if (block.source.type === "url") {
              parts.push(openAIImagePart(asString(block.source.url)));
            }
          }
        }
        messages.push({ role: "user", content: parts });
      }
    } else if (m.role === "assistant") {
      messages.push({ role: "assistant", content: blocksText(m.content) });
    }
  }
  out.messages = messages;
  if (client.toolNames.length > 0) {
    out.tools = asArray(body.tools)
      .filter(isRecord)
      .map((t) => ({
        type: "function",
        function: { name: t.name, description: asString(t.description), parameters: t.input_schema ?? {} },
      }));
    const choice = body.tool_choice;
    if (isRecord(choice)) {
      if (choice.type === "auto") out.tool_choice = "auto";
      else if (choice.type === "any") out.tool_choice = "required";
      else if (choice.type === "none") out.tool_choice = "none";
      else if (choice.type === "tool" && typeof choice.name === "string") {
        out.tool_choice = { type: "function", function: { name: choice.name } };
      }
    }
  }
  if (client.stop.length > 0) out.stop = client.stop;
  if (client.maxTokens !== null) out.max_tokens = client.maxTokens;
  if (client.temperature !== null) out.temperature = client.temperature;
  if (client.topP !== null) out.top_p = client.topP;
  if (client.stream) {
    out.stream = true;
    out.stream_options = { include_usage: true };
  }
  return out;
}

// OpenAI client -> Anthropic provider.
function toAnthropicProvider(client: ClientRequest, model: string): Json {
  const body = client.body;
  const out: Json = { model, max_tokens: client.maxTokens ?? 1024 };
  if (client.systemText) out.system = client.systemText;
  const messages: Json[] = [];
  for (const m of asArray(body.messages)) {
    if (!isRecord(m)) continue;
    if (m.role === "system") continue;
    if (m.role === "user") {
      const content = m.content;
      if (typeof content === "string") {
        messages.push({ role: "user", content });
      } else if (Array.isArray(content)) {
        const blocks: Json[] = [];
        for (const part of content) {
          if (!isRecord(part)) continue;
          if (part.type === "text" && typeof part.text === "string") {
            blocks.push({ type: "text", text: part.text });
          } else if (part.type === "image_url" && isRecord(part.image_url)) {
            blocks.push(anthropicImageBlock(asString(part.image_url.url)));
          }
        }
        messages.push({ role: "user", content: blocks });
      }
    } else if (m.role === "assistant") {
      messages.push({ role: "assistant", content: blocksText(m.content) });
    }
  }
  out.messages = messages;
  if (client.toolNames.length > 0) {
    out.tools = asArray(body.tools)
      .filter(isRecord)
      .map((t) => {
        const fn = isRecord(t.function) ? t.function : {};
        return { name: asString(fn.name), description: asString(fn.description), input_schema: fn.parameters ?? {} };
      });
    const choice = body.tool_choice;
    if (choice === "auto" || choice === undefined) out.tool_choice = { type: "auto" };
    else if (choice === "required") out.tool_choice = { type: "any" };
    else if (choice === "none") out.tool_choice = { type: "none" };
    else if (isRecord(choice) && choice.type === "function" && isRecord(choice.function)) {
      out.tool_choice = { type: "tool", name: asString(choice.function.name) };
    }
  }
  if (client.stop.length > 0) out.stop_sequences = client.stop;
  if (client.temperature !== null) out.temperature = client.temperature;
  if (client.topP !== null) out.top_p = client.topP;
  if (client.stream) out.stream = true;
  return out;
}

// Any client -> Gemini provider.
function toGeminiProvider(client: ClientRequest, model: string): Json {
  const body = client.body;
  const out: Json = {};
  if (client.systemText) out.systemInstruction = { parts: [{ text: client.systemText }] };
  const contents: Json[] = [];
  const pushUser = (content: unknown): void => {
    const parts: Json[] = [];
    if (typeof content === "string") {
      parts.push({ text: content });
    } else if (Array.isArray(content)) {
      for (const block of content) {
        if (!isRecord(block)) continue;
        if (block.type === "text" && typeof block.text === "string") {
          parts.push({ text: block.text });
        } else if (block.type === "image_url" && isRecord(block.image_url)) {
          const parsed = parseDataUrl(asString(block.image_url.url));
          if (parsed) parts.push({ inlineData: { mimeType: parsed.mediaType, data: parsed.data } });
        } else if (block.type === "image" && isRecord(block.source)) {
          if (block.source.type === "base64") {
            parts.push({ inlineData: { mimeType: asString(block.source.media_type), data: asString(block.source.data) } });
          }
        }
      }
    }
    contents.push({ role: "user", parts });
  };
  if (client.wire === "openai") {
    for (const m of asArray(body.messages)) {
      if (!isRecord(m)) continue;
      if (m.role === "system") continue;
      if (m.role === "user") pushUser(m.content);
      else if (m.role === "assistant") contents.push({ role: "model", parts: [{ text: blocksText(m.content) }] });
    }
  } else {
    for (const m of asArray(body.messages)) {
      if (!isRecord(m)) continue;
      if (m.role === "user") pushUser(m.content);
      else if (m.role === "assistant") contents.push({ role: "model", parts: [{ text: blocksText(m.content) }] });
    }
  }
  out.contents = contents;
  if (client.toolNames.length > 0) {
    const declarations: Json[] = [];
    if (client.wire === "openai") {
      for (const t of asArray(body.tools)) {
        if (!isRecord(t)) continue;
        const fn = isRecord(t.function) ? t.function : {};
        declarations.push({ name: asString(fn.name), description: asString(fn.description), parameters: fn.parameters ?? {} });
      }
    } else {
      for (const t of asArray(body.tools)) {
        if (!isRecord(t)) continue;
        declarations.push({ name: asString(t.name), description: asString(t.description), parameters: t.input_schema ?? {} });
      }
    }
    out.tools = [{ functionDeclarations: declarations }];
  }
  const config: Json = {};
  if (client.maxTokens !== null) config.maxOutputTokens = client.maxTokens;
  if (client.stop.length > 0) config.stopSequences = client.stop;
  if (client.jsonMode) config.responseMimeType = "application/json";
  if (client.temperature !== null) config.temperature = client.temperature;
  out.generationConfig = config;
  return out;
}

// Build the provider request (url, body, streaming flag) for one candidate.
function providerRequest(
  provider: ProviderDir,
  model: string,
  client: ClientRequest,
  envUrl: string,
): { url: string; body: Json; stream: boolean } {
  const root = joinBase(envUrl, provider.base_url);
  if (provider.wire === "openai") {
    const body =
      client.wire === "openai"
        ? { ...client.body, model, ...(client.stream ? { stream_options: { include_usage: true } } : {}) }
        : toOpenAIProvider(client, model);
    return { url: root + "/v1/chat/completions", body, stream: client.stream };
  }
  if (provider.wire === "anthropic") {
    const body =
      client.wire === "anthropic"
        ? { ...client.body, model }
        : toAnthropicProvider(client, model);
    return { url: root + "/v1/messages", body, stream: client.stream };
  }
  // gemini
  const body = toGeminiProvider(client, model);
  const suffix = client.stream ? ":streamGenerateContent?alt=sse" : ":generateContent";
  return { url: `${root}/v1beta/models/${model}${suffix}`, body, stream: client.stream };
}

// ---------------------------------------------------------------------------
// Provider answers: parse each wire back into plain text. Every provider
// answers with the same "ANSWER ECHO{...}" text; the wire only changes how
// that text is carried, so parsing is just unwrapping the envelope.
// ---------------------------------------------------------------------------

// Split an SSE body into (event, data) pairs. A blank line ends an event;
// data lines are joined with newlines; an event with no data is skipped.
function parseSSEEvents(text: string): { event: string; data: string }[] {
  const events: { event: string; data: string }[] = [];
  for (const chunk of text.split(/\r?\n\r?\n/)) {
    let event = "message";
    let data = "";
    for (const line of chunk.split(/\r?\n/)) {
      if (line.startsWith("event:")) event = line.slice(6).trim();
      else if (line.startsWith("data:")) data += (data ? "\n" : "") + line.slice(5).trimStart();
    }
    if (data) events.push({ event, data });
  }
  return events;
}

function parseOpenAIAnswer(bodyText: string, stream: boolean): string | null {
  if (!stream) {
    let parsed: unknown;
    try {
      parsed = JSON.parse(bodyText);
    } catch {
      return null;
    }
    if (!isRecord(parsed)) return null;
    const choices = asArray(parsed.choices);
    if (choices.length === 0 || !isRecord(choices[0])) return null;
    const message = choices[0].message;
    if (!isRecord(message)) return null;
    return blocksText(message.content);
  }
  let text = "";
  for (const { data } of parseSSEEvents(bodyText)) {
    if (data === "[DONE]") continue;
    let frame: unknown;
    try {
      frame = JSON.parse(data);
    } catch {
      continue;
    }
    if (!isRecord(frame)) continue;
    if (isRecord(frame.usage)) continue; // usage chunk, no content
    const choices = asArray(frame.choices);
    if (choices.length === 0 || !isRecord(choices[0])) continue;
    const delta = choices[0].delta;
    if (isRecord(delta) && typeof delta.content === "string") text += delta.content;
  }
  return text;
}

function parseAnthropicAnswer(bodyText: string, stream: boolean): string | null {
  if (!stream) {
    let parsed: unknown;
    try {
      parsed = JSON.parse(bodyText);
    } catch {
      return null;
    }
    if (!isRecord(parsed)) return null;
    const content = asArray(parsed.content);
    const texts: string[] = [];
    for (const block of content) {
      if (isRecord(block) && block.type === "text" && typeof block.text === "string") {
        texts.push(block.text);
      }
    }
    return texts.join("");
  }
  let text = "";
  for (const { event, data } of parseSSEEvents(bodyText)) {
    if (event !== "content_block_delta") continue;
    let frame: unknown;
    try {
      frame = JSON.parse(data);
    } catch {
      continue;
    }
    if (!isRecord(frame) || !isRecord(frame.delta)) continue;
    if (frame.delta.type === "text_delta" && typeof frame.delta.text === "string") {
      text += frame.delta.text;
    }
  }
  return text;
}

function geminiPartsText(parts: unknown[]): string {
  const texts: string[] = [];
  for (const part of parts) {
    if (isRecord(part) && typeof part.text === "string") texts.push(part.text);
  }
  return texts.join("");
}

function parseGeminiAnswer(bodyText: string, stream: boolean): string | null {
  if (!stream) {
    let parsed: unknown;
    try {
      parsed = JSON.parse(bodyText);
    } catch {
      return null;
    }
    if (!isRecord(parsed)) return null;
    const candidates = asArray(parsed.candidates);
    if (candidates.length === 0 || !isRecord(candidates[0])) return null;
    const content = candidates[0].content;
    if (!isRecord(content)) return null;
    return geminiPartsText(asArray(content.parts));
  }
  let text = "";
  for (const { data } of parseSSEEvents(bodyText)) {
    let frame: unknown;
    try {
      frame = JSON.parse(data);
    } catch {
      continue;
    }
    if (!isRecord(frame)) continue;
    const candidates = asArray(frame.candidates);
    if (candidates.length === 0 || !isRecord(candidates[0])) continue;
    const content = candidates[0].content;
    if (!isRecord(content)) continue;
    text += geminiPartsText(asArray(content.parts));
  }
  return text;
}

type ProviderResult =
  | { ok: true; text: string }
  | { ok: false; status: number; reason: string };

// One call to one provider for one client request. A non-200 status, a
// timeout, or an unparseable body is a failure the router can fail over
// from; only a 200 with a readable answer text is a success.
async function callProvider(
  provider: ProviderDir,
  model: string,
  client: ClientRequest,
  envUrl: string,
  env: NodeJS.ProcessEnv,
): Promise<ProviderResult> {
  const { url, body, stream } = providerRequest(provider, model, client, envUrl);
  let res: Response;
  try {
    res = await fetch(url, {
      method: "POST",
      headers: authHeaders(provider, env),
      body: JSON.stringify(body),
      signal: AbortSignal.timeout(CALL_TIMEOUT_MS),
    });
  } catch (err) {
    return { ok: false, status: 0, reason: err instanceof Error ? err.message : String(err) };
  }
  if (res.status !== 200) {
    let detail = "";
    try {
      detail = (await res.text()).slice(0, 200);
    } catch {
      detail = "";
    }
    return { ok: false, status: res.status, reason: detail };
  }
  const bodyText = await res.text();
  let text: string | null = null;
  if (provider.wire === "openai") text = parseOpenAIAnswer(bodyText, stream);
  else if (provider.wire === "anthropic") text = parseAnthropicAnswer(bodyText, stream);
  else text = parseGeminiAnswer(bodyText, stream);
  if (text === null || text.length === 0) {
    return { ok: false, status: 200, reason: "unparseable provider answer" };
  }
  return { ok: true, text };
}

// ---------------------------------------------------------------------------
// Client-wire emitters: answer text + usage, in the wire the client speaks.
// The providers always finish with stop/end_turn/STOP, so the response
// always carries finish "stop" / "end_turn".
// ---------------------------------------------------------------------------

function openAIJsonResponse(model: string, text: string, usage: { input: number; output: number }): Json {
  return {
    id: RESPONSE_ID,
    object: "chat.completion",
    created: FIXED_CREATED,
    model,
    choices: [{ index: 0, message: { role: "assistant", content: text }, finish_reason: "stop" }],
    usage: { prompt_tokens: usage.input, completion_tokens: usage.output, total_tokens: usage.input + usage.output },
  };
}

function openAIStreamBody(model: string, text: string, usage: { input: number; output: number }): string {
  const frames: string[] = [];
  const chunk = (choices: unknown, extra?: Json): string =>
    "data: " +
    JSON.stringify({
      id: RESPONSE_ID,
      object: "chat.completion.chunk",
      created: FIXED_CREATED,
      model,
      choices,
      ...(extra ?? {}),
    }) +
    "\n\n";
  frames.push(chunk([{ index: 0, delta: { role: "assistant", content: "" }, finish_reason: null }]));
  for (let i = 0; i < text.length; i += 13) {
    frames.push(chunk([{ index: 0, delta: { content: text.slice(i, i + 13) }, finish_reason: null }]));
  }
  frames.push(chunk([{ index: 0, delta: {}, finish_reason: "stop" }]));
  frames.push(
    chunk([], {
      usage: {
        prompt_tokens: usage.input,
        completion_tokens: usage.output,
        total_tokens: usage.input + usage.output,
      },
    }),
  );
  frames.push("data: [DONE]\n\n");
  return frames.join("");
}

function anthropicJsonResponse(model: string, text: string, usage: { input: number; output: number }): Json {
  return {
    id: MESSAGE_ID,
    type: "message",
    role: "assistant",
    model,
    content: [{ type: "text", text }],
    stop_reason: "end_turn",
    stop_sequence: null,
    usage: { input_tokens: usage.input, output_tokens: usage.output },
  };
}

function anthropicStreamBody(model: string, text: string, usage: { input: number; output: number }): string {
  const frames: string[] = [];
  const ev = (event: string, data: unknown): string => `event: ${event}\ndata: ${JSON.stringify(data)}\n\n`;
  frames.push(
    ev("message_start", {
      type: "message_start",
      message: {
        id: MESSAGE_ID,
        type: "message",
        role: "assistant",
        model,
        content: [],
        stop_reason: null,
        stop_sequence: null,
        usage: { input_tokens: usage.input, output_tokens: 0 },
      },
    }),
  );
  frames.push(
    ev("content_block_start", {
      type: "content_block_start",
      index: 0,
      content_block: { type: "text", text: "" },
    }),
  );
  for (let i = 0; i < text.length; i += 13) {
    frames.push(
      ev("content_block_delta", {
        type: "content_block_delta",
        index: 0,
        delta: { type: "text_delta", text: text.slice(i, i + 13) },
      }),
    );
  }
  frames.push(ev("content_block_stop", { type: "content_block_stop", index: 0 }));
  frames.push(
    ev("message_delta", {
      type: "message_delta",
      delta: { stop_reason: "end_turn", stop_sequence: null },
      usage: { output_tokens: usage.output },
    }),
  );
  frames.push(ev("message_stop", { type: "message_stop" }));
  return frames.join("");
}

// ---------------------------------------------------------------------------
// Router: rank free models by rating, try in order, fail over on refusal.
// A provider that refuses three times is set aside for the rest of the
// trial; one or two refusals are treated as scheduled single-call failures
// (the provider may still have budget left), so it stays in the pool.
// ---------------------------------------------------------------------------

interface Candidate {
  provider: ProviderDir;
  model: string;
  rating: number;
}

interface RouterState {
  failures: Map<string, number>;
  exhausted: Set<string>;
}

function buildCandidates(brief: Brief, catalog: Catalog, req: WorkloadReq, jsonMode: boolean): Candidate[] {
  const candidates: Candidate[] = [];
  for (const provider of brief.providers) {
    const catalogProvider = catalog.providers.find((p) => p.id === provider.id);
    if (!catalogProvider) continue;
    for (const cm of catalogProvider.models) {
      const row = ratingFor(brief, provider.id, cm.id);
      if (!row) continue;
      if (!meetsRequirements(row, req, jsonMode, provider.wire)) continue;
      candidates.push({ provider, model: cm.id, rating: row.rating_milli });
    }
  }
  candidates.sort(
    (a, b) =>
      b.rating - a.rating ||
      (a.provider.id < b.provider.id ? -1 : a.provider.id > b.provider.id ? 1 : 0) ||
      (a.model < b.model ? -1 : a.model > b.model ? 1 : 0),
  );
  return candidates;
}

// Requirements derived from the request body itself, used when the workload
// manifest has no row for this request (it always does in a trial).
function hasVision(body: Json, wire: string): boolean {
  const messages = asArray(body.messages);
  const haystack = JSON.stringify(messages);
  if (wire === "openai") return haystack.includes("image_url");
  return haystack.includes('"type":"image"');
}

function deriveRequirements(client: ClientRequest): WorkloadReq {
  return {
    id: "",
    wire: client.wire,
    stream: client.stream,
    weight: 1,
    requires_tools: client.toolNames.length > 0,
    requires_vision: hasVision(client.body, client.wire),
    requires_json: client.jsonMode,
    min_context: 0,
  };
}

async function serveRequest(
  client: ClientRequest,
  brief: Brief,
  catalog: Catalog,
  state: RouterState,
  envUrl: string,
  env: NodeJS.ProcessEnv,
  manifestIndex: number,
): Promise<{ model: string; text: string } | null> {
  const req = brief.workload[manifestIndex] ?? deriveRequirements(client);
  const candidates = buildCandidates(brief, catalog, req, client.jsonMode);
  for (const cand of candidates) {
    if (state.exhausted.has(cand.provider.id)) continue;
    const result = await callProvider(cand.provider, cand.model, client, envUrl, env);
    if (result.ok) {
      log("served", req.id || "?", "via", cand.provider.id, cand.model);
      return { model: cand.model, text: result.text };
    }
    const failures = (state.failures.get(cand.provider.id) ?? 0) + 1;
    state.failures.set(cand.provider.id, failures);
    log("failover", cand.provider.id, cand.model, "->", result.status, result.reason.slice(0, 120));
    if (failures >= MAX_PROVIDER_FAILURES) {
      state.exhausted.add(cand.provider.id);
      log("exhausted", cand.provider.id, "set aside for the rest of the trial");
    }
  }
  return null;
}

// ---------------------------------------------------------------------------
// HTTP server: the two client wires plus GET /v1/models in both wires.
// ---------------------------------------------------------------------------

function ownerFor(catalog: Catalog, modelId: string): string {
  for (const p of catalog.providers) {
    if (p.models.some((m) => m.id === modelId && m.free)) return p.id;
  }
  return "aggregator";
}

function modelsResponse(catalog: Catalog, anthropicWire: boolean): Json {
  const ids = freeModelIds(catalog);
  if (anthropicWire) {
    return {
      data: ids.map((id) => ({
        type: "model",
        id,
        display_name: id,
        created_at: FIXED_CREATED_AT,
        free: true,
      })),
      has_more: false,
    };
  }
  return {
    object: "list",
    data: ids.map((id) => ({
      id,
      object: "model",
      created: FIXED_CREATED,
      owned_by: ownerFor(catalog, id),
      free: true,
    })),
  };
}

function readBody(req: import("node:http").IncomingMessage): Promise<string> {
  return new Promise((resolve, reject) => {
    const chunks: Buffer[] = [];
    let size = 0;
    req.on("data", (chunk: Buffer) => {
      size += chunk.length;
      if (size > 1_000_000) {
        reject(new Error("request body too large"));
        return;
      }
      chunks.push(chunk);
    });
    req.on("end", () => resolve(Buffer.concat(chunks).toString("utf8")));
    req.on("error", reject);
  });
}

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

async function main(): Promise<void> {
  const env = process.env;
  const envUrl = (env.MAKEFUN_ENV_URL ?? "").replace(/\/+$/, "");
  const promptFile = env.MAKEFUN_PROMPT_FILE;
  const brief = readBrief(promptFile);
  const catalog = await buildCatalog(brief, envUrl, env);
  const args = process.argv.slice(2);

  if (args.includes("--list")) {
    process.stdout.write(JSON.stringify(catalog) + "\n");
    return;
  }

  if (!envUrl) {
    log("MAKEFUN_ENV_URL is required to serve a trial");
    process.exit(1);
  }

  const state: RouterState = { failures: new Map(), exhausted: new Set() };
  let manifestIndex = 0;

  const server = createServer(async (req, res) => {
    try {
      const url = new URL(req.url ?? "/", "http://127.0.0.1");
      if (req.method === "GET" && url.pathname === "/v1/models") {
        const anthropicWire = req.headers["x-api-key"] !== undefined;
        const body = modelsResponse(catalog, anthropicWire);
        res.writeHead(200, { "content-type": "application/json" });
        res.end(JSON.stringify(body));
        return;
      }
      const isOpenAI = req.method === "POST" && url.pathname === "/v1/chat/completions";
      const isAnthropic = req.method === "POST" && url.pathname === "/v1/messages";
      if (!isOpenAI && !isAnthropic) {
        res.writeHead(404, { "content-type": "application/json" });
        res.end(JSON.stringify({ error: { message: "not found", type: "not_found_error" } }));
        return;
      }
      const wire: "openai" | "anthropic" = isOpenAI ? "openai" : "anthropic";
      let parsed: unknown;
      try {
        parsed = JSON.parse(await readBody(req));
      } catch {
        res.writeHead(400, { "content-type": "application/json" });
        res.end(JSON.stringify({ error: { message: "invalid JSON body", type: "invalid_request_error" } }));
        return;
      }
      if (!isRecord(parsed)) {
        res.writeHead(400, { "content-type": "application/json" });
        res.end(JSON.stringify({ error: { message: "request body must be a JSON object", type: "invalid_request_error" } }));
        return;
      }
      const client = extractClientRequest(parsed, wire);
      const index = manifestIndex;
      manifestIndex++;
      const answer = await serveRequest(client, brief, catalog, state, envUrl, env, index);
      if (!answer) {
        res.writeHead(503, { "content-type": "application/json" });
        if (wire === "openai") {
          res.end(JSON.stringify({ error: { message: "no upstream provider could serve this request", type: "server_error" } }));
        } else {
          res.end(JSON.stringify({ type: "error", error: { type: "overloaded_error", message: "no upstream provider could serve this request" } }));
        }
        return;
      }
      const usage = usageOf(client, answer.text);
      if (wire === "openai") {
        if (client.stream) {
          res.writeHead(200, { "content-type": "text/event-stream", "cache-control": "no-cache", connection: "keep-alive" });
          res.end(openAIStreamBody(answer.model, answer.text, usage));
        } else {
          res.writeHead(200, { "content-type": "application/json" });
          res.end(JSON.stringify(openAIJsonResponse(answer.model, answer.text, usage)));
        }
      } else {
        if (client.stream) {
          res.writeHead(200, { "content-type": "text/event-stream", "cache-control": "no-cache", connection: "keep-alive" });
          res.end(anthropicStreamBody(answer.model, answer.text, usage));
        } else {
          res.writeHead(200, { "content-type": "application/json" });
          res.end(JSON.stringify(anthropicJsonResponse(answer.model, answer.text, usage)));
        }
      }
    } catch (err) {
      log("handler error:", err instanceof Error ? err.message : String(err));
      if (!res.headersSent) {
        res.writeHead(500, { "content-type": "application/json" });
        res.end(JSON.stringify({ error: { message: "internal error", type: "server_error" } }));
      } else {
        res.end();
      }
    }
  });

  const port = await new Promise<number>((resolve, reject) => {
    server.once("error", reject);
    server.listen(0, "0.0.0.0", () => {
      const address = server.address();
      if (address && typeof address === "object") resolve(address.port);
      else reject(new Error("could not determine listen port"));
    });
  });
  log("listening on port", port);

  const registerRes = await fetch(joinBase(envUrl, "/register"), {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ protocol_version: PROTOCOL_VERSION, port, catalog }),
  });
  const registerBody: unknown = await registerRes.json().catch(() => null);
  if (!isRecord(registerBody) || registerBody.ok !== true) {
    log("register failed:", JSON.stringify(registerBody));
    process.exit(1);
  }
  log("registered; trial expects", asNumber(registerBody.requests), "requests");

  let stopping = false;
  const stop = (): void => {
    if (stopping) return;
    stopping = true;
    server.close(() => process.exit(0));
    setTimeout(() => process.exit(0), 1000).unref();
  };
  process.on("SIGTERM", stop);
  process.on("SIGINT", stop);

  for (;;) {
    await sleep(POLL_MS);
    let work: unknown = null;
    try {
      const res = await fetch(joinBase(envUrl, "/work"), {
        signal: AbortSignal.timeout(5000),
      });
      work = await res.json();
    } catch {
      work = null;
    }
    if (isRecord(work) && work.done === true) {
      log("trial done; exiting");
      server.close(() => process.exit(0));
      setTimeout(() => process.exit(0), 1000).unref();
      return;
    }
  }
}

main().catch((err) => {
  log("fatal:", err instanceof Error ? err.message : String(err));
  process.exit(1);
});
