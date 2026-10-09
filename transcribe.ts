#!/usr/bin/env node
import { execFile } from "node:child_process";
import { promisify } from "node:util";
import { createWriteStream, readFileSync } from "node:fs";
import { mkdir, mkdtemp, readFile, readdir, rm, stat, writeFile } from "node:fs/promises";
import { cpus, homedir, tmpdir } from "node:os";
import { basename, dirname, extname, join } from "node:path";
import { Readable } from "node:stream";
import { pipeline } from "node:stream/promises";

const run = promisify(execFile);

// ---- config (env vars) ----
const WHISPER_BIN = process.env.WHISPER_BIN ?? "whisper-cli";
const DEFAULT_MODEL = "ggml-large-v3-turbo.bin"; // the unquantized f16 file
const DEFAULT_VAD_MODEL = "ggml-silero-v5.1.2.bin"; // Silero VAD, as in aTrain
const LANGUAGE = process.env.WHISPER_LANG ?? "auto"; // NB: whisper-cli's own default is "en"
const PROMPT = process.env.WHISPER_PROMPT; // aTrain's --prompt
const THREADS = process.env.WHISPER_THREADS ?? String(Math.max(1, cpus().length - 1)); // aTrain: cores - 1
// merge/summary switches: can be overridden per-run via CLI flags (see main())
const INCLUDE_TIMESTAMPS = process.env.INCLUDE_TIMESTAMPS === "1";
const MERGE_MODE: "lines" | "paragraphs" = process.env.MERGE_MODE === "paragraphs" ? "paragraphs" : "lines";

const AUDIO_EXT = new Set([".flac", ".wav", ".mp3", ".ogg", ".opus", ".m4a", ".aac"]);

interface Track { name: string; path: string }
interface Word { word: string; start: number; end: number }
interface Seg { speaker: string; start: number; end: number; text: string; words: Word[] }

// ---------- speaker name from file name ----------
/** "1-alice_0.flac" -> "alice_0", "1-Bob_Smith.flac" -> "Bob_Smith" (underscores are kept) */
function speakerFromFile(path: string): string {
  const raw = basename(path, extname(path));
  let n = raw;
  n = n.replace(/^\d+\s*[-_.]\s*/, ""); // leading track index: "1-", "02_"
  n = n.replace(/#\d{1,4}$/, "");       // trailing Discord discriminator: "#1234"
  return n.trim() || raw;
}

async function resolveTracks(args: string[]): Promise<Track[]> {
  const found: Track[] = [];
  for (const a of args) {
    const eq = a.indexOf("=");
    if (eq > 0 && !(await stat(a).then(() => true, () => false))) {
      found.push({ name: a.slice(0, eq), path: a.slice(eq + 1) });
      continue;
    }
    const st = await stat(a);
    if (st.isDirectory()) {
      for (const f of (await readdir(a)).sort()) {
        if (AUDIO_EXT.has(extname(f).toLowerCase())) {
          const p = join(a, f);
          found.push({ name: speakerFromFile(p), path: p });
        }
      }
    } else {
      found.push({ name: speakerFromFile(a), path: a });
    }
  }
  const seen = new Map<string, number>();
  for (const t of found) {
    const c = (seen.get(t.name) ?? 0) + 1;
    seen.set(t.name, c);
    if (c > 1) t.name = `${t.name} (${c})`;
  }
  return found;
}

// ---------- Craig recordings (https://craig.horse/rec/<id>?key=<key>) ----------
interface CraigRecording { origin: string; host: string; id: string; key: string }

const CRAIG_JOB_TIMEOUT_MS = 60 * 60 * 1000;
const CRAIG_POLL_MS = 3000;

const sleep = (ms: number) => new Promise<void>((r) => setTimeout(r, ms));

/** Accepts /rec/<id>?key=... and the older /rec/?id=<id>&key=... form. */
function parseCraigUrl(arg: string): CraigRecording | null {
  let u: URL;
  try {
    u = new URL(arg);
  } catch {
    return null;
  }
  if (u.protocol !== "http:" && u.protocol !== "https:") return null;

  const path = u.pathname.replace(/\/+$/, "");
  const id = path.match(/^\/rec\/([^/]+)$/)?.[1] ?? (path === "/rec" ? u.searchParams.get("id") ?? "" : "");
  if (!id) return null;
  return { origin: u.origin, host: u.host, id, key: u.searchParams.get("key") ?? "" };
}

class CraigApiError extends Error {
  code: string;
  constructor(code: string, message: string) {
    super(message);
    this.code = code;
  }
}

async function craigApi(url: string, init?: any): Promise<any> {
  let res: any;
  try {
    res = await fetch(url, init);
  } catch (e) {
    throw new Error(`Cannot reach Craig (${new URL(url).origin}): ${(e as Error).message}`);
  }
  const body = await res.json().catch(() => null);
  if (!res.ok) {
    const code = body?.code ?? res.status;
    throw new CraigApiError(String(code), `Craig API error (${code}): ${body?.error ?? res.statusText}`);
  }
  if (body === null) throw new Error(`Unexpected non-JSON response from ${new URL(url).origin}`);
  return body;
}

const craigJobUrl = (rec: CraigRecording) =>
  `${rec.origin}/api/v1/recordings/${rec.id}/job?key=${encodeURIComponent(rec.key)}`;

/** Start a multitrack FLAC (zip) download job, or attach to one that is already running. */
async function startCraigJob(rec: CraigRecording): Promise<any> {
  try {
    const { job } = await craigApi(craigJobUrl(rec), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ type: "recording", options: { container: "zip", format: "flac" } }),
    });
    return job;
  } catch (e) {
    if (!(e instanceof CraigApiError) || e.code !== "JOB_ALREADY_EXISTS") throw e;
    // e.g. the recording is being downloaded from the web page: wait for that job instead
    const { job } = await craigApi(craigJobUrl(rec));
    if (!job) throw e;
    console.error(`Attaching to existing Craig job ${job.id}`);
    return job;
  }
}

async function waitForCraigJob(rec: CraigRecording, job: any): Promise<any> {
  const started = Date.now();
  let last = "";
  while (job.status === "idle" || job.status === "queued" || job.status === "running") {
    const label = `${job.status}${job.state?.type ? ` - ${job.state.type}` : ""}`;
    if (label !== last) {
      console.error(`Craig job ${job.id}: ${label}`);
      last = label;
    }
    if (Date.now() - started > CRAIG_JOB_TIMEOUT_MS) throw new Error(`Craig job ${job.id} timed out`);
    await sleep(CRAIG_POLL_MS);
    const next = (await craigApi(craigJobUrl(rec))).job;
    if (!next) throw new Error(`Craig job ${job.id} disappeared`);
    job = next;
  }
  if (job.status !== "complete") {
    throw new Error(`Craig job ${job.id} ${job.status}${job.failReason ? `: ${job.failReason}` : ""}`);
  }
  return job;
}

/** True if the command can be executed; only ENOENT counts as "missing". */
async function binaryAvailable(cmd: string): Promise<boolean> {
  try {
    await run(cmd, ["--version"], { maxBuffer: 1024 * 1024 * 16 });
    return true;
  } catch (e: any) {
    return e?.code !== "ENOENT";
  }
}

/** Extract a zip with unzip, falling back to bsdtar (Windows/macOS). */
async function extractZip(zip: string, dest: string): Promise<void> {
  await mkdir(dest, { recursive: true });
  try {
    await run("unzip", ["-o", "-q", zip, "-d", dest]);
    return;
  } catch (e: any) {
    if (e?.code !== "ENOENT") throw e;
  }
  try {
    await run("tar", ["-xf", zip, "-C", dest]);
  } catch (e: any) {
    if (e?.code === "ENOENT") throw new Error("no unzip or tar found on PATH: install unzip to extract Craig archives");
    throw new Error(`cannot extract ${basename(zip)}: install unzip (the tar fallback failed)`);
  }
}

/** Download the multitrack FLAC zip for a Craig link, extract it, and return its tracks. */
async function craigTracks(rec: CraigRecording, tmp: string): Promise<Track[]> {
  const meta = await craigApi(`${rec.origin}/api/v1/recordings/${rec.id}?key=${encodeURIComponent(rec.key)}`);
  if (meta.live) throw new Error(`Craig recording ${rec.id} is still being recorded`);
  console.error(`Craig recording ${rec.id}: ${meta.users.length} tracks (${rec.host})`);

  const job = await waitForCraigJob(rec, await startCraigJob(rec));

  const dir = await mkdtemp(join(tmp, `craig-${rec.id}-`));
  const zip = join(dir, "recording.zip");
  const size = job.outputSize ? ` (${(job.outputSize / 1e6).toFixed(1)} MB)` : "";
  console.error(`Downloading ${job.outputFileName}${size}...`);
  const res: any = await fetch(`${rec.origin}/dl/${encodeURIComponent(job.outputFileName)}`);
  if (!res.ok || !res.body) throw new Error(`Craig download failed: HTTP ${res.status} ${res.statusText}`);
  await pipeline(Readable.fromWeb(res.body as any), createWriteStream(zip));

  const tracksDir = join(dir, "tracks");
  await extractZip(zip, tracksDir);
  await rm(zip, { force: true });
  return resolveTracks([tracksDir]);
}

// ---------- model files (whisper + VAD) ----------
const exists = (p: string) => stat(p).then(() => true, () => false);

const MODEL_HELP = [
  "",
  "Download a whisper model with whisper.cpp's model downloader, e.g.:",
  "  whisper-cpp-download-model large-v3-turbo    (or models/download-ggml-model.sh)",
  "  (run it without arguments to list all available models)",
  "",
  `It is saved to ~/.local/share/whisper-cpp/models/${DEFAULT_MODEL}, where it is picked up automatically.`,
  "Or point to any model file explicitly:",
  `  WHISPER_MODEL=/path/to/${DEFAULT_MODEL} whisper-multitrack <audio>`,
  "",
];

const VAD_HELP = [
  "",
  "Download the VAD model (aTrain always runs with VAD), e.g.:",
  `  curl -L -o ${DEFAULT_VAD_MODEL} https://huggingface.co/ggml-org/whisper-vad/resolve/main/${DEFAULT_VAD_MODEL}`,
  "",
  "It is picked up automatically from ./models or ~/.local/share/whisper-cpp/models,",
  "or point to it explicitly:",
  `  WHISPER_VAD_MODEL=/path/to/${DEFAULT_VAD_MODEL} whisper-multitrack <audio>`,
  "",
];

/** <envVar> wins (and must exist), then ./models/<name>, then ~/.local/share/whisper-cpp/models/<name>. */
async function resolveModelFile(envVar: string, defaultName: string, help: string[]): Promise<string> {
  const explicit = process.env[envVar];
  if (explicit) {
    if (await exists(explicit)) return explicit;
    console.error(`${envVar} is set to "${explicit}", but that file does not exist.`);
  } else {
    for (const p of [`./models/${defaultName}`, join(homedir(), ".local", "share", "whisper-cpp", "models", defaultName)]) {
      if (await exists(p)) return p;
    }
  }
  console.error(help.join("\n"));
  throw new Error(explicit ? `${envVar} not found at ${explicit}` : `No ${envVar} configured and no default model found`);
}

// ---------- whisper-cli, with aTrain's faster-whisper settings ----------
async function transcribeWords(t: Track, tmp: string, idx: number, model: string, vadModel: string): Promise<Word[]> {
  const wav = join(tmp, `${idx}.wav`);
  const outPrefix = join(tmp, String(idx));

  // 16 kHz mono, as aTrain's decode_audio
  await run("ffmpeg", ["-y", "-i", t.path, "-ar", "16000", "-ac", "1", wav]);

  const args = [
    "-m", model,
    "-f", wav,
    "-l", LANGUAGE,
    "-t", THREADS,
    // decoding: aTrain's transcribe() call
    "-bs", "5",                      // beam_size=5
    "-nth", "0.6",                   // no_speech_threshold=0.6
    "-tp", "0.0", "-tpi", "0.2",     // temperature=[0.0, 0.2, ... 1.0]
    "-et", "2.4", "-lpt", "-1.0",    // faster-whisper defaults for the fallback triggers
    // (no -mc: previous text stays as context, like condition_on_previous_text=True)
    // vad_filter=True with faster-whisper's default VadOptions
    "--vad", "-vm", vadModel,
    "-vt", "0.5",                    // threshold
    "-vspd", "0",                    // min_speech_duration_ms
    "-vsd", "2000",                  // min_silence_duration_ms
    "-vp", "400",                    // speech_pad_ms
    // word_timestamps=True: one segment per word
    "-ml", "1", "-sow",
    "-oj", "-of", outPrefix,
  ];
  if (PROMPT) args.push("--prompt", PROMPT);
  await run(WHISPER_BIN, args, { maxBuffer: 1024 * 1024 * 64 });

  const json = JSON.parse(await readFile(`${outPrefix}.json`, "utf8"));
  return (json.transcription as any[])
    .map((s) => ({
      word: String(s.text),
      start: s.offsets.from / 1000,
      end: s.offsets.to / 1000,
    }))
    // drop empties and non-speech tags like [BLANK_AUDIO], (music)
    .filter((w) => w.word.trim() && !/^\s*[\[(].*[\])]\s*$/.test(w.word));
}

// ---------- port of aTrain_core/backends/common.py ----------
const SENTENCE_END = [".", "?", "!", "…", "。", "？", "！"];
const CLOSERS = new Set([..."\"'”’»)]」』"]);

function endsSentence(text: string): boolean {
  let end = text.length;
  while (end > 0 && CLOSERS.has(text[end - 1])) end--;
  const t = text.slice(0, end);
  return SENTENCE_END.some((s) => t.endsWith(s));
}

/** words_to_segments: one segment per timestamped word */
function wordsToSegments(speaker: string, words: Word[]): Seg[] {
  return words
    .filter((w) => w.start != null && w.end != null)
    .map((w) => ({ speaker, start: w.start, end: w.end, text: w.word.trim(), words: [w] }));
}

/** group_word_segments(join_raw=True): sentence-aware grouping, max_duration 20 s */
function groupWordSegments(segments: Seg[], maxDuration = 20.0): Seg[] {
  const grouped: Seg[] = [];
  for (const seg of segments) {
    const cur = grouped[grouped.length - 1];
    if (!cur) {
      grouped.push({ ...seg, words: [...seg.words] });
      continue;
    }
    const gap = seg.start - cur.end;
    const sameSpeaker = seg.speaker === cur.speaker;
    const withinDuration = seg.end - cur.start <= maxDuration;
    const sentenceDone = endsSentence(cur.text);
    const longEnough = cur.end - cur.start >= 3.0;
    if (!sameSpeaker || gap >= 2.0 || !withinDuration || (sentenceDone && longEnough)) {
      grouped.push({ ...seg, words: [...seg.words] });
      continue;
    }
    cur.end = seg.end;
    cur.text += seg.words.map((w) => w.word).join("");
    cur.words.push(...seg.words);
  }
  return grouped;
}

const ts = (x: number) => {
  const h = Math.floor(x / 3600), m = Math.floor((x % 3600) / 60), s = Math.floor(x % 60);
  return `[${[h, m, s].map((n) => String(n).padStart(2, "0")).join(":")}]`;
};

/** 73.2 -> "1m 13s" */
const fmtElapsed = (x: number) => {
  const total = Math.round(x);
  const h = Math.floor(total / 3600), m = Math.floor((total % 3600) / 60), s = total % 60;
  return h ? `${h}h ${m}m ${s}s` : m ? `${m}m ${s}s` : `${s}s`;
};

const HELP = `
whisper-multitrack — transcribe multi-track recordings (e.g. Craig Discord
meetings) or single audio files with whisper.cpp. Each track is transcribed
separately, then merged into one speaker-labelled transcript.

Usage:
  whisper-multitrack [options] <dir | file ... | craig-url>
  whisper-multitrack [options] Name=file.flac            # override the speaker name

Options:
  --timestamps              prefix each line with a [hh:mm:ss] timestamp
  --merge=lines|paragraphs  output layout (default: lines)
  --out=file                transcript path (default: transcript.txt next to
                            the audio, or transcript-<id>.txt for Craig links)
  -h, --help                show this help
  -v, --version             show the version

Craig links:
  https://craig.horse/rec/<id>?key=<key> downloads the multi-track FLAC zip,
  extracts it to a temporary folder and transcribes it.

Config (environment variables):
  WHISPER_BIN         whisper.cpp binary (default: whisper-cli)
  WHISPER_MODEL       whisper model; auto-detected in ./models and
                      ~/.local/share/whisper-cpp/models if unset
  WHISPER_VAD_MODEL   Silero VAD model; auto-detected the same way
  WHISPER_LANG        language (default: auto)
  WHISPER_PROMPT      initial prompt passed to whisper
  WHISPER_THREADS     number of threads (default: cores - 1)
  INCLUDE_TIMESTAMPS  1 = same as --timestamps
  MERGE_MODE          same as --merge
  OUTPUT              same as --out

Requirements:
  ffmpeg, whisper.cpp (whisper-cli) and unzip need to be installed.
  If no whisper/VAD model is found, download instructions are printed.

Examples:
  whisper-multitrack ./recording-tracks
  whisper-multitrack --timestamps --merge=paragraphs --out=meeting.txt ./tracks
  whisper-multitrack Alice=1-alice.flac Bob=2-bob.flac
  whisper-multitrack "https://craig.horse/rec/XXXXXXXXXXXX?key=YYYY"
`;

const VERSION = (() => {
  // works both from dist/ (published) and from the repo root (running the source)
  for (const p of ["../package.json", "./package.json"]) {
    try {
      return JSON.parse(readFileSync(new URL(p, import.meta.url), "utf8")).version as string;
    } catch {}
  }
  return "0.0.0";
})();

async function main() {
  // runtime switches: CLI flags override the env-var defaults
  let includeTimestamps = INCLUDE_TIMESTAMPS;
  let mergeMode = MERGE_MODE;
  let outFile: string | null = null;
  const raw = process.argv.slice(2);
  if (raw.includes("-h") || raw.includes("--help")) {
    console.log(HELP);
    return;
  }
  if (raw.includes("-v") || raw.includes("--version")) {
    console.log(VERSION);
    return;
  }
  const args = raw.filter((a) => {
    if (a === "--timestamps") {
      includeTimestamps = true;
      return false;
    }
    if (a.startsWith("--merge=")) {
      const v = a.slice("--merge=".length);
      if (v === "paragraphs" || v === "lines") mergeMode = v;
      else throw new Error(`unknown --merge value "${v}" (expected "paragraphs" or "lines")`);
      return false;
    }
    if (a.startsWith("--out=") || a.startsWith("--output=")) {
      outFile = a.slice(a.indexOf("=") + 1) || null;
      if (!outFile) throw new Error(`--out requires a file path (got "${a}")`);
      return false;
    }
    return true;
  });
  if (args.length === 0) {
    console.error(
      "usage: whisper-multitrack [--timestamps] [--merge=lines|paragraphs] [--out=file] <dir | file ... | craig-url>   (override name: Name=file.flac)",
    );
    console.error("       whisper-multitrack --help for details");
    process.exit(1);
  }
  const model = await resolveModelFile("WHISPER_MODEL", DEFAULT_MODEL, MODEL_HELP);
  const vadModel = await resolveModelFile("WHISPER_VAD_MODEL", DEFAULT_VAD_MODEL, VAD_HELP);

  // Craig links are downloaded and unzipped; everything else resolves as file/dir/Name=file
  const craig: CraigRecording[] = [];
  const local: string[] = [];
  for (const a of args) {
    const rec = parseCraigUrl(a);
    if (rec) {
      if (!rec.key) throw new Error(`Craig URL is missing the ?key=... parameter: ${a}`);
      craig.push(rec);
    } else if (/^https?:\/\//i.test(a)) {
      throw new Error(`Unsupported URL (expected a Craig recording link): ${a}`);
    } else {
      local.push(a);
    }
  }

  // required external tools (otherwise the first use fails with a raw ENOENT)
  if (!(await binaryAvailable("ffmpeg"))) {
    throw new Error('"ffmpeg" not found on PATH. Install ffmpeg: https://ffmpeg.org/download.html');
  }
  if (!(await binaryAvailable(WHISPER_BIN))) {
    throw new Error(`whisper binary "${WHISPER_BIN}" not found on PATH. Install whisper.cpp (whisper-cli), or set WHISPER_BIN to its path.`);
  }
  if (craig.length && !(await binaryAvailable("unzip")) && !(await binaryAvailable("tar"))) {
    throw new Error('neither "unzip" nor "tar" found on PATH: install unzip to extract Craig archives');
  }

  const tmp = await mkdtemp(join(tmpdir(), "multitrack-"));
  try {
    const tracks: Track[] = [];
    for (const rec of craig) tracks.push(...(await craigTracks(rec, tmp)));
    if (local.length) tracks.push(...(await resolveTracks(local)));
    if (tracks.length === 0) throw new Error("No audio files found");
    console.error("Speakers:", tracks.map((t) => `${t.name} <- ${basename(t.path)}`).join(", "));

    // transcript goes next to the audio files (or to cwd for Craig links) unless OUTPUT/--out is set
    const outputPath =
      outFile ??
      process.env.OUTPUT ??
      (craig.length ? `transcript-${craig[0].id}.txt` : join(dirname(tracks[0].path), "transcript.txt"));
    // aTrain's sentence-aware grouping, applied to each speaker's own track,
    // then all cues interleaved by start time.
    const cues: Seg[] = [];
    for (const [i, t] of tracks.entries()) {
      console.error(`Transcribing ${t.name}...`);
      const t0 = Date.now();
      const words = await transcribeWords(t, tmp, i, model, vadModel);
      cues.push(...groupWordSegments(wordsToSegments(t.name, words)));
      console.error(`Transcribed ${t.name} in ${fmtElapsed((Date.now() - t0) / 1000)}`);
    }
    cues.sort((a, b) => a.start - b.start);

    // same layout as aTrain's transcription_timestamps.txt
    const firstCraig = parseCraigUrl(args[0]);
    const sourceLabel = firstCraig ? `${firstCraig.host}/rec/${firstCraig.id}` : basename(args[0]);
    let out = "Transcription for " + sourceLabel + "\n";
    if (mergeMode === "paragraphs") {
      // flowing text: paragraphs break on speaker change or a >= 2 s pause
      const blocks: { speaker: string; paras: { start: number; text: string }[] }[] = [];
      for (let i = 0; i < cues.length; i++) {
        const c = cues[i];
        let block = blocks[blocks.length - 1];
        if (!block || c.speaker !== block.speaker) {
          block = { speaker: c.speaker, paras: [] };
          blocks.push(block);
        }
        const lastPara = block.paras[block.paras.length - 1];
        const gap = i > 0 ? c.start - cues[i - 1].end : 0;
        if (!lastPara || gap >= 2.0) {
          block.paras.push({ start: c.start, text: c.text.trim() });
        } else {
          lastPara.text += " " + c.text.trim();
        }
      }
      for (const block of blocks) {
        out += "\n" + block.speaker + "\n";
        out += block.paras
          .map((p) => (includeTimestamps ? `${ts(p.start)} - ` : "") + p.text)
          .join("\n\n") + "\n";
      }
    } else {
      // lines mode: one line per cue
      let current: string | null = null;
      for (const c of cues) {
        if (c.speaker !== current) {
          out += "\n" + c.speaker + "\n";
          current = c.speaker;
        }
        out += includeTimestamps ? `${ts(c.start)} - ${c.text.trimStart()}\n` : `${c.text.trimStart()}\n`;
      }
    }
    await writeFile(outputPath, out, "utf8");
    console.error(`Wrote ${outputPath}`);
  } finally {
    await rm(tmp, { recursive: true, force: true });
  }
}

main().catch((e) => { console.error(e); process.exit(1); });
