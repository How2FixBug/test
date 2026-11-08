import json, os, time, math, random, urllib.parse, urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F

# littleGPT v7: byte-subword tokenizer, boundary-safe windows, ~9.8M parameters.
SEED=1337
BLOCK_SIZE=256
BATCH_SIZE=16
N_EMBD=288
N_HEAD=6
N_LAYER=5
DROPOUT=0.10
VOCAB_SIZE=8192
DATASET_ROWS=int(os.environ.get("DATASET_ROWS","250000"))
MAX_STEPS=int(os.environ.get("TRAIN_STEPS","30000"))
CHUNK_STEPS=int(os.environ.get("CHUNK_STEPS",str(MAX_STEPS)))
EVAL_INTERVAL=500
EVAL_BATCHES=12
LEARNING_RATE=8e-4
DATASET="roneneldan/TinyStories"
ROWS_API="https://datasets-server.huggingface.co/rows"
CACHE=Path("training_stories_v7.json")
CHECKPOINT=Path("checkpoint-v7.pt")
random.seed(SEED); torch.manual_seed(SEED)
if torch.backends.mps.is_available():
    device=torch.device("mps"); print("MPS acceleration enabled.",flush=True)
else:
    device=torch.device("cpu"); torch.set_num_threads(min(4,os.cpu_count() or 2)); torch.set_num_interop_threads(1)
    print("Using CPU. For M1/M2/M3 Mac, MPS should be available in a PyTorch build with MPS support.",flush=True)

def fetch_page(offset,length):
    params=urllib.parse.urlencode({"dataset":DATASET,"config":"default","split":"train","offset":offset,"length":length})
    req=urllib.request.Request(ROWS_API+"?"+params,headers={"User-Agent":"littleGPT-v7/1.0"})
    for attempt in range(7):
        try:
            with urllib.request.urlopen(req,timeout=120) as r: data=json.loads(r.read().decode())
            rows=[x.get("row",{}).get("text","") for x in data.get("rows",[])]
            return offset,[s.strip() for s in rows if isinstance(s,str) and len(s.strip())>30]
        except Exception as e:
            if attempt==6: raise RuntimeError(f"Download failed at offset {offset}: {e}") from e
            wait=min(90,5*(attempt+1))
            if getattr(e,"headers",None):
                try: wait=max(wait,int(e.headers.get("Retry-After","0")))
                except Exception: pass
            print(f"Retry page {offset} in {wait}s: {e}",flush=True); time.sleep(wait)

def download():
    stories=[]
    if CACHE.exists():
        try: stories=json.loads(CACHE.read_text(encoding="utf-8"))
        except Exception: stories=[]
    if len(stories)>=DATASET_ROWS: return stories[:DATASET_ROWS]
    # Fetch in bounded concurrent batches; save after each batch for safe resume.
    offsets=list(range(len(stories),DATASET_ROWS,100))
    for base in range(0,len(offsets),32):
        batch_offsets=offsets[base:base+32]
        with ThreadPoolExecutor(max_workers=6) as pool:
            futures=[pool.submit(fetch_page,o,min(100,DATASET_ROWS-o)) for o in batch_offsets]
            pages=[f.result() for f in as_completed(futures)]
        page_map=dict(pages)
        for o in sorted(page_map): stories.extend(page_map[o])
        stories=stories[:DATASET_ROWS]
        CACHE.write_text(json.dumps(stories,ensure_ascii=False),encoding="utf-8")
        print(f"Downloaded {len(stories):,}/{DATASET_ROWS:,} stories",flush=True)
        time.sleep(1.0)
    if len(stories)<1000: raise RuntimeError(f"Only {len(stories)} stories downloaded.")
    return stories[:DATASET_ROWS]

def clean(s):
    table={"\u201c":'"',"\u201d":'"',"\u201e":'"',"\u201f":'"',"\u2018":"'","\u2019":"'","\u201a":"'","\u201b":"'","\u00a0":" "}
    for a,b in table.items(): s=s.replace(a,b)
    return s

stories=download()
random.Random(SEED).shuffle(stories)
cut=int(len(stories)*.95)
train_stories,val_stories=stories[:cut],stories[cut:]

# Learn a deterministic byte-subword vocabulary. Every UTF-8 byte is always representable,
# so unseen words never collapse to <unk>. Tokens are byte strings stored as Latin-1 in JSON.
def raw_bytes(s): return clean(s).encode("utf-8")
counts=Counter()
for n,s in enumerate(train_stories):
    b=raw_bytes(s)
    for width in range(2,9):
        counts.update(b[i:i+width] for i in range(max(0,len(b)-width+1)))
    if (n+1)%50000==0: print(f"Tokenizer scan: {n+1:,}/{len(train_stories):,}",flush=True)
special=["<unk>","<bos>","<eos>"]
byte_tokens=[bytes([i]) for i in range(256)]
extra=[x for x,_ in counts.most_common(VOCAB_SIZE-259) if len(x)>1]
tokens=byte_tokens+extra
# Unique and exactly bounded; one-byte fallback guarantees full coverage.
seen=set(tokens); tokens=[bytes([i]) for i in range(256)]+[x for x in extra if x not in set(bytes([i]) for i in range(256))]
tokens=tokens[:VOCAB_SIZE-3]
vocab=special+[x.decode("latin1") for x in tokens]
stoi={x:i+3 for i,x in enumerate(tokens)}
max_width=max(map(len,tokens))
by_first={}
for token,idx in stoi.items():
    by_first.setdefault(token[0],[]).append((token,idx))
for key in by_first: by_first[key].sort(key=lambda x:len(x[0]),reverse=True)

def encode_bytes(data):
    out=[]; i=0
    while i<len(data):
        tail=data[i:i+max_width].decode("latin1")
        chosen=None
        for token,idx in by_first.get(tail[0],[]):
            if tail.startswith(token): chosen=idx; break
        if chosen is None: raise RuntimeError("Missing byte fallback")
        out.append(chosen); i+=len(vocab[chosen].encode("latin1"))
    return out

def encode_stories(source):
    result=[]
    for i,s in enumerate(source):
        ids=encode_bytes(raw_bytes(s))
        if ids: result.append(torch.tensor([1]+ids+[2],dtype=torch.long))
        if (i+1)%50000==0: print(f"Encoded {i+1:,}/{len(source):,} stories",flush=True)
    return result

train_docs=encode_stories(train_stories)
val_docs=encode_stories(val_stories)
def pack(docs):
    stream=torch.cat(docs).to(device)
    starts=[]; offset=0
    for doc in docs:
        length=doc.numel()
        if length>BLOCK_SIZE+1:
            starts.extend(range(offset,offset+length-BLOCK_SIZE))
        offset+=length
    if not starts: raise RuntimeError("No stories long enough for a training window.")
    return stream,torch.tensor(starts,dtype=torch.long,device=device)
train_stream,train_starts=pack(train_docs)
val_stream,val_starts=pack(val_docs)
OFFSETS=torch.arange(BLOCK_SIZE,device=device).unsqueeze(0)
def get_batch(stream,starts):
    chosen=starts[torch.randint(starts.numel(),(BATCH_SIZE,),device=device)].unsqueeze(1)
    ix=chosen+OFFSETS
    return stream[ix],stream[ix+1]

class Attention(nn.Module):
    def __init__(self):
        super().__init__(); self.qkv=nn.Linear(N_EMBD,3*N_EMBD); self.proj=nn.Linear(N_EMBD,N_EMBD)
    def forward(self,x):
        b,t,c=x.shape; q,k,v=self.qkv(x).split(N_EMBD,dim=2); hd=c//N_HEAD
        q=q.view(b,t,N_HEAD,hd).transpose(1,2); k=k.view(b,t,N_HEAD,hd).transpose(1,2); v=v.view(b,t,N_HEAD,hd).transpose(1,2)
        y=F.scaled_dot_product_attention(q,k,v,is_causal=True)
        return self.proj(y.transpose(1,2).contiguous().view(b,t,c))
class Block(nn.Module):
    def __init__(self):
        super().__init__(); self.ln1=nn.LayerNorm(N_EMBD); self.attn=Attention(); self.ln2=nn.LayerNorm(N_EMBD)
        self.fc1=nn.Linear(N_EMBD,4*N_EMBD); self.fc2=nn.Linear(4*N_EMBD,N_EMBD)
    def forward(self,x):
        x=x+self.attn(self.ln1(x)); return x+self.fc2(F.gelu(self.fc1(self.ln2(x))))
class GPT(nn.Module):
    def __init__(self):
        super().__init__(); self.wte=nn.Embedding(VOCAB_SIZE,N_EMBD); self.wpe=nn.Embedding(BLOCK_SIZE,N_EMBD)
        self.drop=nn.Dropout(DROPOUT); self.blocks=nn.ModuleList([Block() for _ in range(N_LAYER)])
        self.ln_f=nn.LayerNorm(N_EMBD); self.lm_head=nn.Linear(N_EMBD,VOCAB_SIZE)
    def forward(self,idx,targets=None):
        _,t=idx.shape; pos=torch.arange(t,device=idx.device)
        x=self.drop(self.wte(idx)+self.wpe(pos))
        for block in self.blocks: x=block(x)
        logits=self.lm_head(self.ln_f(x)); loss=None
        if targets is not None: loss=F.cross_entropy(logits.reshape(-1,VOCAB_SIZE),targets.reshape(-1))
        return logits,loss

base=GPT().to(device)
params=sum(p.numel() for p in base.parameters())
assert params==9796448, f"Expected 9,796,448 params, got {params}"
optimizer=torch.optim.AdamW(base.parameters(),lr=LEARNING_RATE,weight_decay=.1,foreach=(device.type=="cpu"))
config={"block_size":BLOCK_SIZE,"n_embd":N_EMBD,"n_head":N_HEAD,"n_layer":N_LAYER,"vocab_size":VOCAB_SIZE,"dataset_rows":DATASET_ROWS,"tokenizer_version":1,"boundary_safe":True}
step0=0; best=float("inf"); best_state=None; metrics={"train":None,"val":None}
if CHECKPOINT.exists():
    ck=torch.load(CHECKPOINT,map_location=device,weights_only=False)
    if ck.get("format")!="littleGPT-v7-checkpoint" or ck.get("config")!=config or ck.get("vocab")!=vocab:
        raise RuntimeError("Checkpoint does not match v7 data/tokenizer/config; refusing to resume.")
    base.load_state_dict(ck["model"]); optimizer.load_state_dict(ck["optimizer"]); step0=ck["step"]
    best=ck["best"]; best_state=ck["best_model"]; metrics=ck["metrics"]
    torch.set_rng_state(ck["rng"])
    if device.type=="mps" and ck.get("mps_rng") is not None: torch.mps.set_rng_state(ck["mps_rng"])
print(f"littleGPT v7 | {params:,} parameters | vocab={VOCAB_SIZE} byte-subwords | stories={len(stories):,} | train tokens={train_stream.numel():,} | validation tokens={val_stream.numel():,} | device={device}",flush=True)

@torch.no_grad()
def evaluate():
    base.eval(); out={}
    for name,stream,starts in (("train",train_stream,train_starts),("val",val_stream,val_starts)):
        vals=[]
        for _ in range(EVAL_BATCHES):
            x,y=get_batch(stream,starts); _,loss=base(x,y); vals.append(loss.item())
        out[name]=round(sum(vals)/len(vals),4)
    base.train(); return out
def clone_state(): return {k:v.detach().cpu().clone() for k,v in base.state_dict().items()}
def save(step):
    data={"format":"littleGPT-v7-checkpoint","step":step,"model":{k:v.detach().cpu() for k,v in base.state_dict().items()},
          "best_model":best_state,"best":best,"optimizer":optimizer.state_dict(),"vocab":vocab,"config":config,"metrics":metrics,"rng":torch.get_rng_state()}
    if device.type=="mps": data["mps_rng"]=torch.mps.get_rng_state()
    torch.save(data,CHECKPOINT); print(f"CHECKPOINT SAVED {step}/{MAX_STEPS} best_val={best:.4f}",flush=True)

def lr_at(step):
    warm=min(1.,(step+1)/500); progress=min(1.,step/max(1,MAX_STEPS-1))
    return LEARNING_RATE*warm*(.08+.92*.5*(1+math.cos(math.pi*progress)))

base.train(); started=time.perf_counter(); start_step=step0
for step in range(step0,min(step0+CHUNK_STEPS,MAX_STEPS)):
    for group in optimizer.param_groups: group["lr"]=lr_at(step)
    x,y=get_batch(train_stream,train_starts); optimizer.zero_grad(set_to_none=True)
    _,loss=base(x,y); loss.backward(); torch.nn.utils.clip_grad_norm_(base.parameters(),1.0); optimizer.step()
    if (step+1)%EVAL_INTERVAL==0 or step==0 or step+1==MAX_STEPS:
        metrics=evaluate(); improved=metrics["val"]<best
        if improved: best=metrics["val"]; best_state=clone_state()
        elapsed=time.perf_counter()-started
        print(f"step {step+1}/{MAX_STEPS} train={metrics['train']:.4f} val={metrics['val']:.4f} best_val={best:.4f}{' BEST' if improved else ''} speed={(step+1-start_step)/max(elapsed,1e-9):.2f} steps/s",flush=True)
    if (step+1)%500==0 or step+1==MAX_STEPS: save(step+1)
completed=min(step0+CHUNK_STEPS,MAX_STEPS)
if completed%500 and completed<MAX_STEPS: save(completed)
if completed<MAX_STEPS: print("Chunk complete; checkpoint saved.",flush=True); raise SystemExit(0)
if best_state: base.load_state_dict(best_state)
base.eval()

def arr(t): return t.detach().cpu().tolist()
state={"wte":arr(base.wte.weight),"wpe":arr(base.wpe.weight),"ln_f":{"weight":arr(base.ln_f.weight),"bias":arr(base.ln_f.bias)},"lm_head":{"weight":arr(base.lm_head.weight),"bias":arr(base.lm_head.bias)},"blocks":[]}
for b in base.blocks:
    state["blocks"].append({"ln1":{"weight":arr(b.ln1.weight),"bias":arr(b.ln1.bias)},"attn":{"qkv":{"weight":arr(b.attn.qkv.weight),"bias":arr(b.attn.qkv.bias)},"proj":{"weight":arr(b.attn.proj.weight),"bias":arr(b.attn.proj.bias)}},"ln2":{"weight":arr(b.ln2.weight),"bias":arr(b.ln2.bias)},"mlp":{"fc1":{"weight":arr(b.fc1.weight),"bias":arr(b.fc1.bias)},"fc2":{"weight":arr(b.fc2.weight),"bias":arr(b.fc2.bias)}}})
payload={"format":"littleGPT-v7","created_at":__import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
"config":{"block_size":BLOCK_SIZE,"n_embd":N_EMBD,"n_head":N_HEAD,"n_layer":N_LAYER,"vocab_size":VOCAB_SIZE,"unk_id":0,"bos_id":1,"eos_id":2,"tokenizer":"byte-subword-greedy-v1","max_token_bytes":max_width},
"vocab":vocab,"parameters":params,"training":{"steps":MAX_STEPS,"loss":best,"train_loss":metrics["train"],"dataset":f"TinyStories ({len(stories)} stories; story-boundary-safe windows)","seed":SEED,"train_tokens":train_stream.numel(),"validation_tokens":val_stream.numel(),"learning_rate":LEARNING_RATE,"batch_size":BATCH_SIZE,"dropout":DROPOUT,"best_validation_loss":best,"device":str(device)},"state":state}
Path("model.json").write_text(json.dumps(payload,separators=(",",":")),encoding="utf-8")
print(f"Exported littleGPT v7 ({Path('model.json').stat().st_size/1024/1024:.1f} MB)",flush=True)
