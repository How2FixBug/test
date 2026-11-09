import json, base64
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F

SOURCE = Path("checkpoint-v9-instruct.pt")
if not SOURCE.exists():
    raise FileNotFoundError("checkpoint-v9-instruct.pt is missing. Keep the Phase 2 checkpoint in this folder.")

ck = torch.load(SOURCE, map_location="cpu", weights_only=False)
if ck.get("format") != "littleGPT-v9-instruct-checkpoint":
    raise RuntimeError("This is not a compatible Phase 2 checkpoint.")
cfg, vocab = ck["config"], ck["vocab"]
BLOCK_SIZE = cfg["block_size"]
N_EMBD, N_HEAD, N_LAYER, VOCAB_SIZE = cfg["n_embd"], cfg["n_head"], cfg["n_layer"], cfg["vocab_size"]

class Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.qkv = nn.Linear(N_EMBD, 3*N_EMBD)
        self.proj = nn.Linear(N_EMBD, N_EMBD)
    def forward(self, x):
        b,t,c=x.shape
        q,k,v=self.qkv(x).split(N_EMBD,dim=2); hd=c//N_HEAD
        q=q.view(b,t,N_HEAD,hd).transpose(1,2)
        k=k.view(b,t,N_HEAD,hd).transpose(1,2)
        v=v.view(b,t,N_HEAD,hd).transpose(1,2)
        y=F.scaled_dot_product_attention(q,k,v,is_causal=True)
        return self.proj(y.transpose(1,2).contiguous().view(b,t,c))

class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1=nn.LayerNorm(N_EMBD); self.attn=Attention(); self.ln2=nn.LayerNorm(N_EMBD)
        self.fc1=nn.Linear(N_EMBD,4*N_EMBD); self.fc2=nn.Linear(4*N_EMBD,N_EMBD)
    def forward(self,x):
        x=x+self.attn(self.ln1(x))
        return x+self.fc2(F.gelu(self.fc1(self.ln2(x)),approximate="tanh"))

class GPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.wte=nn.Embedding(VOCAB_SIZE,N_EMBD); self.wpe=nn.Embedding(BLOCK_SIZE,N_EMBD)
        self.drop=nn.Dropout(.10); self.blocks=nn.ModuleList([Block() for _ in range(N_LAYER)])
        self.ln_f=nn.LayerNorm(N_EMBD); self.lm_head=nn.Linear(N_EMBD,VOCAB_SIZE)
    def forward(self,idx):
        _,t=idx.shape; pos=torch.arange(t,device=idx.device)
        x=self.drop(self.wte(idx)+self.wpe(pos))
        for block in self.blocks: x=block(x)
        return self.lm_head(self.ln_f(x))

model=GPT()
weights=ck.get("best_model") or ck["model"]
model.load_state_dict(weights)
model.eval()
params=sum(p.numel() for p in model.parameters())

def arr(t):
    packed=t.detach().to(device="cpu",dtype=torch.float16).contiguous()
    return {"f16":base64.b64encode(bytes(packed.untyped_storage())).decode("ascii")}

state={"wte":arr(model.wte.weight),"wpe":arr(model.wpe.weight),
       "ln_f":{"weight":arr(model.ln_f.weight),"bias":arr(model.ln_f.bias)},
       "lm_head":{"weight":arr(model.lm_head.weight),"bias":arr(model.lm_head.bias)},"blocks":[]}
for b in model.blocks:
    state["blocks"].append({
      "ln1":{"weight":arr(b.ln1.weight),"bias":arr(b.ln1.bias)},
      "attn":{"qkv":{"weight":arr(b.attn.qkv.weight),"bias":arr(b.attn.qkv.bias)},
              "proj":{"weight":arr(b.attn.proj.weight),"bias":arr(b.attn.proj.bias)}},
      "ln2":{"weight":arr(b.ln2.weight),"bias":arr(b.ln2.bias)},
      "mlp":{"fc1":{"weight":arr(b.fc1.weight),"bias":arr(b.fc1.bias)},
             "fc2":{"weight":arr(b.fc2.weight),"bias":arr(b.fc2.bias)}}})

payload={"format":"littleGPT-v7",
 "created_at":__import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
 "config":{"block_size":BLOCK_SIZE,"n_embd":N_EMBD,"n_head":N_HEAD,"n_layer":N_LAYER,
 "vocab_size":VOCAB_SIZE,"unk_id":0,"bos_id":1,"eos_id":2,
 "tokenizer":"byte-subword-greedy-v1","max_token_bytes":max(len(x.encode("latin1")) for x in vocab[3:])},
 "vocab":vocab,"parameters":params,
 "training":{"steps":ck["step"],"loss":ck["best"],"best_validation_loss":round(ck["best"],4),
 "dataset":"TinyStories instruction pairs (synthetic story-writing prompts paired with original stories)",
 "instruction_tuned":True,"base_model":"littleGPT V8","base_training_steps":30000,
 "instruction_pairs":"synthetic TinyStories prompt/story pairs","device":"mps"},
 "state":state}
Path("model.json").write_text(json.dumps(payload,separators=(",",":")),encoding="utf-8")
print(f"Exported Phase 2 littleGPT ({Path('model.json').stat().st_size/1024/1024:.1f} MB)")
print(f"Best instruction validation loss: {ck['best']:.4f}")
print(f"Parameters: {params:,} | vocab: {len(vocab)}")
