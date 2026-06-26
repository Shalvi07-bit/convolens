# Sync Architecture Design

## Goal

ConvoLens should keep private conversation intelligence usable on-device while allowing a hosted demo to sync only safe, compact artifacts. The design assumes raw chats are sensitive and may contain personal relationships, emotions, and reminders.

## Architecture

```
Mobile/Desktop Device
  |
  |-- Local encrypted store
  |     - raw conversations.csv
  |     - message index
  |     - persona.json
  |     - persona_drift.json
  |     - intent model
  |
  |-- Local processors
  |     - topic splitter
  |     - persona drift detector
  |     - offline intent classifier
  |     - conflict resolver
  |
  | syncs only compact records
  v
Sync API
  |
  |-- Auth + device identity
  |-- Conflict merge service
  v
Cloud metadata store
  - topic summaries
  - checkpoint summaries
  - redacted persona facts
  - artifact versions
```

## What Stays Local

Raw messages, raw CSV uploads, full message embeddings, the offline intent classifier input text, and unresolved private chunks stay on-device. The app can answer sensitive questions locally because the FAISS message index and resolver run on CPU without calling OpenAI/Gemini.

## What Syncs

The device syncs topic IDs, checkpoint summaries, persona fields approved for backup, daily drift labels, drift trigger labels, model/artifact versions, and timestamps. For hosted demos, the repo can include generated JSON/index artifacts, but a production app would encrypt synced summaries per user.

## Conflict Resolution

Each generated artifact has `{artifact_type, source_range, device_id, updated_at, version}`. If two devices edit the same artifact, newer versions win for deterministic fields such as labels, while append-only evidence lists are merged by message ID. Contradictory facts are not overwritten; they are stored as competing claims with confidence, recency, and emotional weight. The UI/API returns a `contradiction_flag` and ranked evidence instead of pretending there is one perfect memory.

## Tradeoffs

Keeping raw data local protects privacy and enables offline use, but cloud search is less powerful because it only sees summaries. Syncing compact summaries is cheaper and safer, but bad local summaries can propagate unless artifacts are versioned and rebuildable. The chosen design prioritizes privacy, explainability, and graceful conflict handling over centralized model power.
