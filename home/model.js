(() => {
  "use strict";

  const MODEL_PATH = "../Main/Models/Astra%200.1/astra-0.1.onnx";
  const TOKENIZER_PATH = "../Main/Models/Astra%200.1/tokenizer_config.json";
  const MAX_CONTEXT = 512;
  const VOCAB_SIZE = 8192;
  const END_TOKEN = 259;

  class AstraBrowserModel {
    static async load() {
      if (!window.ort) throw new Error("ONNX Runtime Web did not load.");
      if (!window.AstraTokenizer) throw new Error("ASTRA tokenizer did not load.");

      window.ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/";
      window.ort.env.wasm.numThreads = 1;

      const tokenizerUrl = new URL(TOKENIZER_PATH, window.location.href).toString();
      const modelUrl = new URL(MODEL_PATH, window.location.href).toString();

      const [tokenizer, session] = await Promise.all([
        window.AstraTokenizer.load(tokenizerUrl),
        window.ort.InferenceSession.create(modelUrl, {
          executionProviders: ["webgpu", "wasm"],
          graphOptimizationLevel: "all"
        })
      ]);

      return new AstraBrowserModel(tokenizer, session);
    }

    constructor(tokenizer, session) {
      this.tokenizer = tokenizer;
      this.session = session;
    }

    async generate(inputIds, options = {}) {
      const temperature = options.temperature ?? 0.85;
      const topK = options.topK ?? 40;
      const maxNewTokens = options.maxNewTokens ?? 120;
      let ids = inputIds.slice(-MAX_CONTEXT);
      const generated = [];

      for (let step = 0; step < maxNewTokens; step++) {
        if (ids.length > MAX_CONTEXT) ids = ids.slice(-MAX_CONTEXT);

        const inputTensor = new window.ort.Tensor(
          "int64",
          BigInt64Array.from(ids, token => BigInt(token)),
          [1, ids.length]
        );
        const result = await this.session.run({ input_ids: inputTensor });
        const logits = result.logits.data;
        const lastOffset = (ids.length - 1) * VOCAB_SIZE;
        const recent = ids.slice(-64);
        const candidates = [];

        for (let token = 0; token < VOCAB_SIZE; token++) {
          if (token >= 256 && token <= 258) continue;
          let score = logits[lastOffset + token];
          if (recent.includes(token) && token !== END_TOKEN) {
            score = score > 0 ? score / 1.12 : score * 1.12;
          }
          candidates.push({ token, score });
        }

        candidates.sort((a, b) => b.score - a.score);
        const shortlist = candidates.slice(0, Math.max(1, topK));
        const maxScore = shortlist[0].score;
        const weights = shortlist.map(item => Math.exp((item.score - maxScore) / temperature));
        const total = weights.reduce((sum, weight) => sum + weight, 0);
        let choice = Math.random() * total;
        let selected = shortlist[0].token;

        for (let i = 0; i < shortlist.length; i++) {
          choice -= weights[i];
          if (choice <= 0) {
            selected = shortlist[i].token;
            break;
          }
        }

        if (selected === END_TOKEN) break;
        generated.push(selected);
        ids.push(selected);
      }

      return this.tokenizer.decode(generated).trim();
    }
  }

  window.AstraBrowserModel = AstraBrowserModel;
})();
