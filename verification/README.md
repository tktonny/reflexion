# Verification

This directory contains an audio-only verification pipeline that mirrors the main clinic prompt and schema while staying separate from the product routing code.

## What is included

- `run_audio_only_verification.py`
  Batch ASR plus transcript-classification runner.
- `run_qwen_omni_verification.py`
  Batch native-audio Qwen Omni runner for dialogue-only verification samples.
- `qwen_audio_only.py`
  DashScope-backed Qwen ASR plus Qwen 3.5 classifier.
- `qwen_omni_audio.py`
  Verification-only Qwen Omni native audio classifier with patient-versus-examiner guidance.
- `download_talkbank.py`
  Pulls public ADReSSo labels, downloads configured TalkBank assets, and exports a corpus catalog.
- `build_talkbank_catalog.py`
  Summarizes all authorized TalkBank dementia corpora into a reusable JSON catalog.
- `prepare_talkbank_verification_data.py`
  Downloads the selected verification corpora and generates per-corpus plus combined manifests.
- `build_adresso_manifest.py`
  Builds a manifest after ADReSSo audio files are available locally.
- `download_google_drive_folder.py`
  Lists or downloads a shared Google Drive folder into `verification/data`.
- `prepare_smoke_dataset.py`
  Creates a tiny local smoke manifest from bundled YT-DemTalk clips.

## Prompt and schema consistency

The classifier reuses:

- `clinic.intelligence.prompts.build_provider_prompt(..., provider_mode="audio_only")`
- `backend.src.app.models.ProviderAssessmentPayload`

This keeps the verification output aligned with the clinic app's `audio_only` branch.

For the native-audio `qwen_omni` verification branch, the prompt also tells the model that:

- the patient only provides conversational audio
- video is missing rather than broken
- examiner prompts may appear briefly
- the model must separate examiner speech from patient speech and assess only the patient

## API key routing

Verification now uses two credential lanes:

- `QWEN_API_KEY`
  For standard Qwen / Model Studio endpoints such as `qwen3-asr-flash`.
- `DASHSCOPE_API_KEY`
  For Coding Plan models such as `qwen3.5-plus`, `qwen3-max-2026-01-23`, `qwen3-coder-next`, `qwen3-coder-plus`, `glm-5`, `glm-4.7`, `kimi-k2.5`, and `MiniMax-M2.5`.

Default base URLs:

- Standard Qwen: `https://dashscope.aliyuncs.com/compatible-mode/v1`
- Coding Plan: `https://coding.dashscope.aliyuncs.com/v1`

## Recommended flow

Store local TalkBank credentials in `.secret/.env` before running the access probe:

```bash
TALKBANK_EMAIL=<your-email>
TALKBANK_PASSWORD=<your-password>
```

If TalkBank gives you a non-default auth host, override it with:

```bash
REFLEXION_TALKBANK_AUTH_BASE_URL=<your-talkbank-auth-base-url>
```

1. Download public labels and check TalkBank access:

```bash
python3 verification/download_talkbank.py
```

This also exports a cross-language corpus catalog to:

```bash
verification/data/talkbank/corpus_catalog.json
```

To download the current default verification corpora and build manifests:

```bash
python3 verification/prepare_talkbank_verification_data.py
```

If your dataset is shared through Google Drive instead, you can inspect it first without downloading:

```bash
python3 verification/download_google_drive_folder.py 'https://drive.google.com/drive/folders/<folder-id>?usp=sharing' --list-only --remaining-ok
```

2. If TalkBank grants media access later, place the downloaded audio under `verification/data/adresso/audio/` or `verification/data/talkbank/...`, then build a manifest:

```bash
python3 verification/build_adresso_manifest.py
```

3. Run a local smoke check first:

```bash
python3 verification/prepare_smoke_dataset.py
python3 verification/run_audio_only_verification.py --manifest verification/data/smoke/manifest.jsonl
```

4. Run the real benchmark once the dataset is present:

```bash
python3 verification/run_audio_only_verification.py --manifest verification/data/adresso/manifest.jsonl --output-dir verification/results/adresso_qwen35_audio_only
```

To run the native-audio Qwen Omni verification branch on selected audio cases:

```bash
python3 verification/run_qwen_omni_verification.py \
  --manifest verification/data/talkbank/corpora/english/pitt/manifest.jsonl \
  --case-id english__pitt__control__cookie__054_0 \
  --case-id english__pitt__dementia__cookie__001_0 \
  --output-dir verification/results/qwen_omni_prompt_probe
```

## Outputs

- Per-case outputs: `verification/results/<run>/cases/<case_id>.json`
- Run summary and metrics: `verification/results/<run>/summary.json`

The summary includes:

- accuracy
- precision
- recall
- specificity
- f1
- confusion matrix

## Current constraint

TalkBank media access is separate from basic website login. If `download_talkbank.py` reports `notAuthorized`, the account still needs dataset-level media permission before audio files can be downloaded automatically.
