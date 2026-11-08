import json, os, time, math, random, base64, urllib.parse, urllib.request
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
DATASET_ROWS=int(os.environ.get("DATASET_ROWS","25000"))
VALIDATION_ROWS=int(os.environ.get("VALIDATION_ROWS","21990"))
MAX_STEPS=int(os.environ.get("TRAIN_STEPS","30000"))
CHUNK_STEPS=int(os.environ.get("CHUNK_STEPS",str(MAX_STEPS)))
EVAL_INTERVAL=500
EVAL_BATCHES=32
LEARNING_RATE=8e-4
DATASET="roneneldan/TinyStories"
ROWS_API="https://datasets-server.huggingface.co/rows"
CACHE=Path("training_stories_v7.json")
VAL_CACHE=Path("validation_stories_v7.json")
CHECKPOINT=Path("checkpoint-v7-25k.pt")
random.seed(SEED); torch.manual_seed(SEED)
if torch.backends.mps.is_available():
    device=torch.device("mps"); print("MPS acceleration enabled.",flush=True)
else:
    device=torch.device("cpu"); torch.set_num_threads(min(4,os.cpu_count() or 2)); torch.set_num_interop_threads(1)
    print("Using CPU. For M1/M2/M3 Mac, MPS should be available in a PyTorch build with MPS support.",flush=True)

def fetch_page(offset,length,split):
    params=urllib.parse.urlencode({"dataset":DATASET,"config":"default","split":split,"offset":offset,"length":length})
    req=urllib.request.Request(ROWS_API+"?"+params,headers={"User-Agent":"littleGPT-v7/1.1 (sequential-resumable)"})
    for attempt in range(12):
        try:
            with urllib.request.urlopen(req,timeout=120) as r: data=json.loads(r.read().decode())
            rows=[x.get("row",{}).get("text","") for x in data.get("rows",[])]
            return [s.strip() for s in rows if isinstance(s,str) and len(s.strip())>0],len(data.get("rows",[]))
        except Exception as e:
            if attempt==11: raise RuntimeError(f"Download failed at offset {offset}: {e}") from e
            # Be gentle with the public dataset API, especially after HTTP 429.
            wait=min(300,15*(2**attempt))
            if getattr(e,"headers",None):
                try: wait=max(wait,int(e.headers.get("Retry-After","0")))
                except Exception: pass
            print(f"Retry page {offset} in {wait}s: {e}",flush=True); time.sleep(wait)

def download_split(split,target_rows,cache_path):
    stories=[]; next_offset=0
    if cache_path.exists():
        try:
            cached=json.loads(cache_path.read_text(encoding="utf-8"))
            if isinstance(cached,dict) and "stories" in cached:
                if cached.get("split",split)==split:
                    stories=cached["stories"]; next_offset=int(cached.get("next_offset",0))
            elif isinstance(cached,list) and split=="train":
                # Migrate the previous list-based v7 training cache.
                stories=cached; next_offset=len(stories)
                print(f"Resuming legacy train cache at approximately row {next_offset:,}.",flush=True)
        except Exception as e:
            raise RuntimeError(f"Could not read {split} cache {cache_path}: {e}") from e
    if len(stories)>=target_rows: return stories[:target_rows]
    print(f"Loading TinyStories {split} split: target {target_rows:,} stories; cached {len(stories):,}.",flush=True)
    # Fetch one page at a time and atomically cache every successful page.
    # The train and official validation split have independent caches and offsets.
    while len(stories)<target_rows:
        requested=100
        page,row_count=fetch_page(next_offset,requested,split)
        if row_count<=0: break
        stories.extend(page)
        next_offset+=row_count
        temp=cache_path.with_suffix(".tmp")
        temp.write_text(json.dumps({"split":split,"next_offset":next_offset,"stories":stories},ensure_ascii=False),encoding="utf-8")
        temp.replace(cache_path)
        print(f"{split}: cached {len(stories):,}/{target_rows:,} stories (source row {next_offset:,})",flush=True)
    if len(stories)<min(1000,target_rows):
        raise RuntimeError(f"Only {len(stories)} stories downloaded from TinyStories {split}.")
    return stories[:target_rows]

def clean(s):
    table={"\u201c":'"',"\u201d":'"',"\u201e":'"',"\u201f":'"',"\u2018":"'","\u2019":"'","\u201a":"'","\u201b":"'","\u00a0":" "}
    for a,b in table.items(): s=s.replace(a,b)
    return s

train_stories=download_split("train",DATASET_ROWS,CACHE)
val_stories=download_split("validation",VALIDATION_ROWS,VAL_CACHE)
stories=train_stories

# Learn a deterministic byte-subword vocabulary. Every UTF-8 byte is always representable,
# so unseen words never collapse to <unk>. Tokens are byte strings stored as Latin-1 in JSON.
def raw_bytes(s): return clean(s).encode("utf-8")
counts=Counter()
tokenizer_sample=random.Random(SEED).sample(train_stories,min(50000,len(train_stories)))
for n,s in enumerate(tokenizer_sample):
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
if len(tokens)<VOCAB_SIZE-3: raise RuntimeError(f"Tokenizer built only {len(tokens)+3} tokens; need {VOCAB_SIZE}.")
vocab=special+[x.decode("latin1") for x in tokens]
stoi={x:i+3 for i,x in enumerate(tokens)}
max_width=max(map(len,tokens))
token_trie={}
for token,idx in stoi.items():
    node=token_trie
    # Tokens are already bytes; iterating yields integer byte values.
    for byte in token: node=node.setdefault(byte,{})
    node[None]=idx
max_width=max(map(len,tokens))

def encode_bytes(data):
    out=[]; i=0
    while i<len(data):
        node=token_trie; j=i; best=None; best_end=i
        while j<len(data) and j<i+max_width:
            node=node.get(data[j])
            if node is None: break
            j+=1
            if None in node: best=node[None]; best_end=j
        if best is None: raise RuntimeError("Missing byte fallback")
        out.append(best); i=best_end
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
    doc_offsets=[]; doc_spans=[]; offset=0
    for doc in docs:
        length=doc.numel()
        if length>BLOCK_SIZE+1:
            doc_offsets.append(offset)
            doc_spans.append(length-BLOCK_SIZE)
        offset+=length
    if not doc_offsets: raise RuntimeError("No stories long enough for a training window.")
    return stream,torch.tensor(doc_offsets,dtype=torch.long,device=device),torch.tensor(doc_spans,dtype=torch.long,device=device)
train_stream,train_offsets,train_spans=pack(train_docs)
val_stream,val_offsets,val_spans=pack(val_docs)
OFFSETS=torch.arange(BLOCK_SIZE,device=device).unsqueeze(0)
def get_batch(stream,doc_offsets,doc_spans):
    which=torch.randint(doc_offsets.numel(),(BATCH_SIZE,),device=device)
    starts=doc_offsets[which]+(torch.rand(BATCH_SIZE,device=device)*doc_spans[which]).long()
    ix=starts.unsqueeze(1)+OFFSETS
    return stream[ix],stream[ix+1]

# Fixed evaluation windows make loss comparable between checkpoints.
# Separate Python RNGs keep evaluation sampling independent of training RNG state.
def make_fixed_eval_batches(stream,doc_offsets,doc_spans,seed):
    rng=random.Random(seed)
    offsets=doc_offsets.detach().cpu().tolist()
    spans=doc_spans.detach().cpu().tolist()
    batches=[]
    for _ in range(EVAL_BATCHES):
        starts=[offsets[j]+rng.randrange(spans[j]) for j in (rng.randrange(len(offsets)) for _ in range(BATCH_SIZE))]
        ix=torch.tensor(starts,dtype=torch.long,device=device).unsqueeze(1)+OFFSETS
        batches.append((stream[ix],stream[ix+1]))
    return batches

train_eval_batches=make_fixed_eval_batches(train_stream,train_offsets,train_spans,SEED+101)
val_eval_batches=make_fixed_eval_batches(val_stream,val_offsets,val_spans,SEED+202)

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
config={"block_size":BLOCK_SIZE,"n_embd":N_EMBD,"n_head":N_HEAD,"n_layer":N_LAYER,"vocab_size":VOCAB_SIZE,"dataset_rows":DATASET_ROWS,"validation_rows":VALIDATION_ROWS,"tokenizer_version":1,"boundary_safe":True,"eval_version":2}
step0=0; best=float("inf"); best_state=None; metrics={"train":None,"val":None}
if CHECKPOINT.exists():
    ck=torch.load(CHECKPOINT,map_location=device,weights_only=False)
    old_config=ck.get("config",{})
    legacy_eval_config={k:v for k,v in config.items() if k!="eval_version"}
    legacy_eval=(old_config==legacy_eval_config)
    if ck.get("format")!="littleGPT-v7-checkpoint" or (old_config!=config and not legacy_eval) or ck.get("vocab")!=vocab:
        raise RuntimeError("Checkpoint does not match v7 data/tokenizer/config; refusing to resume.")
    base.load_state_dict(ck["model"]); optimizer.load_state_dict(ck["optimizer"]); step0=ck["step"]
    if legacy_eval:
        # Old best scores came from changing random samples and cannot be compared
        # with the new fixed evaluation set. Keep model/optimizer progress, reset best.
        best=float("inf"); best_state=None; metrics={"train":None,"val":None}
        print("Migrated checkpoint: preserved training progress; recalibrating best checkpoint with fixed evaluation batches.",flush=True)
    else:
        best=ck["best"]; best_state=ck["best_model"]; metrics=ck["metrics"]
    torch.set_rng_state(ck["rng"].cpu())
    if device.type=="mps" and ck.get("mps_rng") is not None: torch.mps.set_rng_state(ck["mps_rng"].cpu())
print(f"littleGPT v7 | {params:,} parameters | vocab={VOCAB_SIZE} byte-subwords | stories={len(stories):,} | train tokens={train_stream.numel():,} | validation tokens={val_stream.numel():,} | device={device}",flush=True)

@torch.no_grad()
def evaluate():
    base.eval(); out={}
    for name,batches in (("train",train_eval_batches),("val",val_eval_batches)):
        vals=[]
        for x,y in batches:
            _,loss=base(x,y); vals.append(loss.item())
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
    x,y=get_batch(train_stream,train_offsets,train_spans); optimizer.zero_grad(set_to_none=True)
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

def arr(t):
    packed=t.detach().to(device="cpu",dtype=torch.float16).contiguous()
    return {"f16":base64.b64encode(bytes(packed.untyped_storage())).decode("ascii") }
state={"wte":arr(base.wte.weight),"wpe":arr(base.wpe.weight),"ln_f":{"weight":arr(base.ln_f.weight),"bias":arr(base.ln_f.bias)},"lm_head":{"weight":arr(base.lm_head.weight),"bias":arr(base.lm_head.bias)},"blocks":[]}
for b in base.blocks:
    state["blocks"].append({"ln1":{"weight":arr(b.ln1.weight),"bias":arr(b.ln1.bias)},"attn":{"qkv":{"weight":arr(b.attn.qkv.weight),"bias":arr(b.attn.qkv.bias)},"proj":{"weight":arr(b.attn.proj.weight),"bias":arr(b.attn.proj.bias)}},"ln2":{"weight":arr(b.ln2.weight),"bias":arr(b.ln2.bias)},"mlp":{"fc1":{"weight":arr(b.fc1.weight),"bias":arr(b.fc1.bias)},"fc2":{"weight":arr(b.fc2.weight),"bias":arr(b.fc2.bias)}}})
payload={"format":"littleGPT-v7","created_at":__import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
"config":{"block_size":BLOCK_SIZE,"n_embd":N_EMBD,"n_head":N_HEAD,"n_layer":N_LAYER,"vocab_size":VOCAB_SIZE,"unk_id":0,"bos_id":1,"eos_id":2,"tokenizer":"byte-subword-greedy-v1","max_token_bytes":max_width},
"vocab":vocab,"parameters":params,"training":{"steps":MAX_STEPS,"loss":best,"train_loss":metrics["train"],"dataset":f"TinyStories official train split ({len(train_stories)} stories; official validation split ({len(val_stories)} stories); story-boundary-safe windows)","seed":SEED,"train_tokens":train_stream.numel(),"validation_tokens":val_stream.numel(),"learning_rate":LEARNING_RATE,"batch_size":BATCH_SIZE,"dropout":DROPOUT,"best_validation_loss":best,"device":str(device)},"state":state}
Path("model.json").write_text(json.dumps(payload,separators=(",",":")),encoding="utf-8")
print(f"Exported littleGPT v7 ({Path('model.json').stat().st_size/1024/1024:.1f} MB)",flush=True)
