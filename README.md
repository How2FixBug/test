# Ultillama

A browser-based chat workspace for local Ollama models.

## Features
- Multiple chats with local persistence
- Create and delete chats
- Short-term context: choose how many recent messages are sent to the model
- Long-term memory: add, edit, and delete saved facts
- Optional automatic memory extraction after replies
- Settings for context size, memory, and temperature
- Streaming responses from Ollama

## Run it
1. Start Ollama on your computer.
2. Open the site through a local static web server or deploy it to a static host.
3. Choose an installed model and start chatting.

Ultillama connects directly to `http://localhost:11434`. Configure Ollama's allowed origins if your browser blocks the site.

## How memory works
- Chats and memories are saved in this browser's localStorage.
- Short-term context sends a configurable number of recent messages from the active chat.
- Long-term memory is shared across chats and can be managed in the Memory panel.
- Automatic memory asks your selected Ollama model to extract stable, useful facts. You can turn it off or edit/delete saved items.
- Only the active chat's selected context and enabled memories are sent to Ollama. Nothing is sent to an Ultillama server.

This is browser-local storage, not encrypted cloud sync. Clearing site data will remove chats and memories.
