(() => {
  "use strict";

  const SPECIAL_TOKENS = {
    "<user>": 256,
    "</user>": 257,
    "<astra>": 258,
    "</astra>": 259
  };

  // Matches tokenizer.py's Unicode-aware pre-tokenization pattern.
  const PRETOKEN_RE = /\p{White_Space}+|[\p{L}\p{N}]+|_+|[^\p{L}\p{N}_\p{White_Space}]+/gu;
  const SPECIAL_RE = /(<user>|<\/user>|<astra>|<\/astra>)/g;
  const BASE_VOCAB_SIZE = 256;
  const MERGE_TOKEN_OFFSET = 260;

  class MinHeap {
    constructor() { this.items = []; }

    less(a, b) {
      return a.rank < b.rank ||
        (a.rank === b.rank && a.left < b.left) ||
        (a.rank === b.rank && a.left === b.left && a.right < b.right);
    }

    push(item) {
      const items = this.items;
      items.push(item);
      let i = items.length - 1;
      while (i > 0) {
        const parent = (i - 1) >> 1;
        if (!this.less(items[i], items[parent])) break;
        [items[i], items[parent]] = [items[parent], items[i]];
        i = parent;
      }
    }

    pop() {
      const items = this.items;
      if (!items.length) return null;
      const first = items[0];
      const last = items.pop();
      if (items.length) {
        items[0] = last;
        let i = 0;
        while (true) {
          const left = i * 2 + 1;
          const right = left + 1;
          let smallest = i;
          if (left < items.length && this.less(items[left], items[smallest])) smallest = left;
          if (right < items.length && this.less(items[right], items[smallest])) smallest = right;
          if (smallest === i) break;
          [items[i], items[smallest]] = [items[smallest], items[i]];
          i = smallest;
        }
      }
      return first;
    }
  }

  class AstraTokenizer {
    constructor(config) {
      if (!config || config.type !== "byte_level_bpe" || !Array.isArray(config.merges)) {
        throw new Error("Invalid ASTRA tokenizer config.");
      }
      if (config.fingerprint !== "90084574c2cb150e25edb930f342774e14c09f25920b13e53858933d6301d71f" ||\n          config.vocab_size !== 8192 || config.merges.length !== 7932 || config.vocab_hex?.length !== 8192) {
        throw new Error("ASTRA tokenizer config does not match the expected 8,192-token vocabulary.");
      }
      this.config = config;
      this.mergeRanks = new Map();
      config.merges.forEach(([left, right], rank) => {
        this.mergeRanks.set(left + "," + right, rank);
      });
      this.decoder = new TextDecoder("utf-8", { fatal: false });
      this.vocabBytes = config.vocab_hex.map(hex => {
        if (hex === null) return null;
        const bytes = new Uint8Array(hex.length / 2);
        for (let i = 0; i < bytes.length; i++) {
          bytes[i] = parseInt(hex.slice(i * 2, i * 2 + 2), 16);
        }
        return bytes;
      });
      this.specialById = new Map(
        Object.entries(config.special_tokens).map(([token, id]) => [id, token])
      );
    }

    static async load(url) {
      const response = await fetch(url);
      if (!response.ok) throw new Error("Could not load ASTRA tokenizer config (" + response.status + ").");
      const config = await response.json();
      return new AstraTokenizer(config);
    }

    encodePlain(text) {
      const output = [];
      const pieces = text.match(PRETOKEN_RE) || [];

      for (const piece of pieces) {
        const values = Array.from(new TextEncoder().encode(piece));
        const size = values.length;
        if (size < 2) {
          output.push(...values);
          continue;
        }

        const previous = new Int32Array(size);
        const following = new Int32Array(size);
        const alive = new Uint8Array(size);
        for (let i = 0; i < size; i++) {
          previous[i] = i - 1;
          following[i] = i + 1 < size ? i + 1 : -1;
          alive[i] = 1;
        }

        const heap = new MinHeap();
        const addPair = left => {
          const right = following[left];
          if (right === -1) return;
          const rank = this.mergeRanks.get(values[left] + "," + values[right]);
          if (rank !== undefined) heap.push({ rank, left, right });
        };

        for (let i = 0; i < size - 1; i++) addPair(i);

        while (heap.items.length) {
          const item = heap.pop();
          const { rank, left, right } = item;
          if (!alive[left] || !alive[right] || following[left] !== right) continue;
          if (this.mergeRanks.get(values[left] + "," + values[right]) !== rank) continue;

          values[left] = MERGE_TOKEN_OFFSET + rank;
          alive[right] = 0;
          const after = following[right];
          following[left] = after;
          if (after !== -1) previous[after] = left;

          const before = previous[left];
          if (before !== -1) addPair(before);
          addPair(left);
        }

        let index = 0;
        while (index !== -1) {
          if (alive[index]) output.push(values[index]);
          index = following[index];
        }
      }
      return output;
    }

    encode(text) {
      const ids = [];
      for (const part of text.split(SPECIAL_RE)) {
        if (!part) continue;
        if (Object.prototype.hasOwnProperty.call(SPECIAL_TOKENS, part)) {
          ids.push(SPECIAL_TOKENS[part]);
        } else {
          ids.push(...this.encodePlain(part));
        }
      }
      return ids;
    }

    decode(ids) {
      const output = [];
      let byteChunks = [];

      const flush = () => {
        if (!byteChunks.length) return;
        const length = byteChunks.reduce((sum, chunk) => sum + chunk.length, 0);
        const bytes = new Uint8Array(length);
        let offset = 0;
        for (const chunk of byteChunks) {
          bytes.set(chunk, offset);
          offset += chunk.length;
        }
        output.push(this.decoder.decode(bytes));
        byteChunks = [];
      };

      for (const tokenId of ids) {
        if (this.specialById.has(tokenId)) {
          flush();
          output.push(this.specialById.get(tokenId));
          continue;
        }
        const bytes = this.vocabBytes[tokenId];
        if (!bytes) throw new Error("Unknown ASTRA token ID: " + tokenId);
        byteChunks.push(bytes);
      }
      flush();
      return output.join("");
    }
  }

  window.AstraTokenizer = AstraTokenizer;
})();
