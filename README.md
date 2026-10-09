# whisper-multitrack

Transcribe multi-track recordings — e.g. meetings recorded with the [Craig](https://craig.chat) Discord bot — or plain audio files, with [whisper.cpp](https://github.com/ggml-org/whisper.cpp).

Each speaker's track is transcribed separately, so the transcript is automatically labelled per speaker, and all tracks are interleaved by timestamp into one readable document. Craig recording links are accepted directly: the multi-track FLAC archive is downloaded, unzipped into a temporary folder and transcribed.

## Requirements

- Node.js >= 20
- [ffmpeg](https://ffmpeg.org/download.html) on `PATH`
- [whisper.cpp](https://github.com/ggml-org/whisper.cpp) (`whisper-cli`) on `PATH`
- A whisper model and the Silero VAD model — see [Models](#models)
- `unzip` (needed for Craig links; if it is missing the built-in `tar` fallback is used where available)

## Install

```sh
npm install -g whisper-multitrack
```

Or run it without installing:

```sh
npx whisper-multitrack --help
```

## Models

Models are auto-detected in `./models` and `~/.local/share/whisper-cpp/models`, so anything downloaded with whisper.cpp's standard downloader just works:

```sh
whisper-cpp-download-model large-v3-turbo     # or models/download-ggml-model.sh
```

The Silero VAD model is also required (matching aTrain's defaults):

```sh
curl -L -o ggml-silero-v5.1.2.bin \
  https://huggingface.co/ggml-org/whisper-vad/resolve/main/ggml-silero-v5.1.2.bin
```

If either model is missing, the tool prints exactly what to download.

## Usage

```sh
whisper-multitrack [options] <dir | file ... | craig-url>
```

Transcribe a folder of per-speaker tracks (e.g. an extracted Craig zip):

```sh
whisper-multitrack ./recording-tracks
```

Transcribe specific files, optionally overriding the speaker names:

```sh
whisper-multitrack Alice=1-alice.flac Bob=2-bob.flac
```

Transcribe a Craig recording link directly:

```sh
whisper-multitrack "https://craig.horse/rec/XXXXXXXXXXXX?key=YYYY"
```

Paragraph output with timestamps, written to a specific file:

```sh
whisper-multitrack --timestamps --merge=paragraphs --out=meeting.txt ./tracks
```

### Try it with npx

No install required — npx fetches the CLI and runs it in one go:

```sh
# a folder of tracks
npx -y whisper-multitrack ./recording-tracks

# paragraph output with timestamps, explicit output file
npx -y whisper-multitrack --timestamps --merge=paragraphs --out=transcript.txt ./tracks

# straight from a Craig recording URL
npx -y whisper-multitrack "https://craig.horse/rec/XXXXXXXXXXXX?key=YYYY"
```

`npx` only provides the CLI itself — `ffmpeg`, `whisper-cli` and the models still need to be installed locally. The `-y` flag skips npx's first-run install prompt.

### Options

| Option | Description |
| --- | --- |
| `--timestamps` | Prefix each line with a `[hh:mm:ss]` timestamp |
| `--merge=lines` | One line per cue (default) |
| `--merge=paragraphs` | Flowing text; paragraphs break on speaker change or a ≥ 2 s pause |
| `--out=file` | Transcript path (default: `transcript.txt` next to the audio, or `transcript-<id>.txt` in the current directory for Craig links) |
| `-h`, `--help` | Show help |
| `-v`, `--version` | Show version |

### Environment variables

| Variable | Description |
| --- | --- |
| `WHISPER_BIN` | whisper.cpp binary (default: `whisper-cli`) |
| `WHISPER_MODEL` | whisper model path (default: auto-detected) |
| `WHISPER_VAD_MODEL` | Silero VAD model path (default: auto-detected) |
| `WHISPER_LANG` | Language (default: `auto`) |
| `WHISPER_PROMPT` | Initial prompt passed to whisper |
| `WHISPER_THREADS` | Number of threads (default: CPU cores − 1) |
| `INCLUDE_TIMESTAMPS` | `1` = same as `--timestamps` |
| `MERGE_MODE` | Same as `--merge` |
| `OUTPUT` | Same as `--out` |

## How it works

- Every input track is converted to 16 kHz mono with ffmpeg and transcribed by `whisper-cli` with word timestamps and VAD, using aTrain's faster-whisper decoding settings.
- Word timestamps are grouped into sentence-like segments (aTrain-style rules: pauses, sentence ends, a max duration), then all speakers' segments are interleaved by time.
- Speaker names come from file names (`1-alice_0.flac` → `alice_0`), or from `Name=file` arguments.

## Author

Svetlozar Argirov <zarrro@gmail.com> [Broken By Design](https://broken-by-design.art/)

## License

MIT
