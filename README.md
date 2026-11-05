# Ultillama

A browser-based chat workspace for local Ollama models.

## Features
- Multiple saved chats: create, switch, rename (double-click a chat), and delete
- Streaming responses, model selection, and a stop-generation button
- Short-term memory: choose how many recent messages from the active chat are sent, or turn it off
- Long-term memory: add, edit, and delete saved facts
- Automatic memory extraction using your selected local Ollama model
- Relevance-ranked memory retrieval: only the most relevant saved facts are added to a prompt
- Settings for context size, memory, and response temperature
- Export chats, memories, and settings as a JSON backup

## Run it
1. Start Ollama on your computer.
2. Open the site through a local static web server or deploy it to a static host.
3. Choose an installed model and start chatting.

Ultillama connects directly to `http://localhost:11434`. Configure Ollama's allowed origins if your browser blocks the site.

## How memory works
- Chats, settings, and memories are saved in this browser's localStorage.
- Short-term context sends a configurable number of recent messages from the active chat.
- Long-term memory is shared across chats. Ultillama uses the configured Ollama embedding model (default: `nomic-embed-text`) to find up to six semantically relevant memories. If embeddings are unavailable, it falls back to keyword matching.
- Automatic memory makes an additional local Ollama request after a response to identify stable, useful facts. You can turn it off or edit/delete saved items.
- The current message and selected recent context are sent to Ollama. Irrelevant saved memories are left out.

## Privacy and limitations
This is browser-local storage, not encrypted cloud sync. Data stays in this browser profile unless you export it. Clearing site data removes chats and memories. When you send a message, selected context and memories are sent to your local Ollama instance; nothing is sent to an Ultillama server.
