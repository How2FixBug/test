import json, os, time, math, random, re, base64
from pathlib import Path
from collections import Counter
import torch
import torch.nn as nn
import torch.nn.functional as F

# littleGPT Phase 2: instruction fine-tuning from the best V8 checkpoint.
SEED = 2026
BLOCK_SIZE = 256
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "16"))
TRAIN_STEPS = int(os.environ.get("INSTRUCT_STEPS", "2500"))
EVAL_INTERVAL = 250
EVAL_BATCHES = 32
LEARNING_RATE = 2e-5
CHECKPOINT_IN = Path("checkpoint-v8.pt")
CHECKPOINT_OUT = Path("checkpoint-v9-instruct.pt")
TRAIN_CACHE = Path("training_stories_v7.json")
VAL_CACHE = Path("validation_stories_v7.json")
random.seed(SEED)
torch.manual_seed(SEED)

if not CHECKPOINT_IN.exists():
    raise FileNotFoundError("checkpoint-v8.pt is missing. Put your V8 checkpoint in this folder; do not upload it to GitHub.")

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
if device.type == "cpu":
    torch.set_num_threads(min(4, os.cpu_count() or 2))
    torch.set_num_interop_threads(1)
print(f"Loading V8 checkpoint on {device}…", flush=True)
ck = torch.load(CHECKPOINT_IN, map_location="cpu", weights_only=False)
if ck.get("format") != "littleGPT-v8-checkpoint":
    raise RuntimeError("This is not a compatible littleGPT V8 checkpoint.")
vocab = ck["vocab"]
cfg = ck["config"]
N_EMBD, N_HEAD, N_LAYER, VOCAB_SIZE = cfg["n_embd"], cfg["n_head"], cfg["n_layer"], cfg["vocab_size"]
assert len(vocab) == VOCAB_SIZE and cfg["block_size"] == BLOCK_SIZE

def clean(s):
    table = {"\u201c":'"',"\u201d":'"',"\u201e":'"',"\u201f":'"',"\u2018":"'","\u2019":"'","\u201a":"'","\u201b":"'","\u00a0":" "}
    for a,b in table.items(): s=s.replace(a,b)
    return s

tokens = [x.encode("latin1") for x in vocab[3:]]
trie = {}
for i, token in enumerate(tokens, 3):
    node = trie
    for byte in token: node = node.setdefault(byte, {})
    node[None] = i
max_width = max(map(len, tokens))
def encode(text):
    data = clean(text).encode("utf-8")
    out=[]; i=0
    while i<len(data):
        node=trie; j=i; best=None; end=i
        while j<len(data) and j<i+max_width:
            node=node.get(data[j])
            if node is None: break
            j+=1
            if None in node: best=node[None]; end=j
        if best is None: raise RuntimeError("Tokenizer byte fallback is missing.")
        out.append(best); i=end
    return out

def read_cache(path, split):
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing. The V8 training cache is needed for instruction examples.")
    data=json.loads(path.read_text(encoding="utf-8"))
    stories=data.get("stories",[]) if isinstance(data,dict) else data
    if not isinstance(stories,list) or len(stories)<1000:
        raise RuntimeError(f"{path} does not contain enough {split} stories.")
    return [s.strip() for s in stories if isinstance(s,str) and s.strip()]

train_stories=read_cache(TRAIN_CACHE,"training")
val_stories=read_cache(VAL_CACHE,"validation")

# TinyStories commonly introduces its protagonists with "named Lily".
STOP={"Once","One","The","A","An","Then","After","Before","When","There","He","She","They","His","Her","It","Mom","Mommy","Dad","Daddy","But","And","So","Finally","Suddenly","Every","This","That"}
def protagonist(story):
    patterns=[
        r"\bnamed\s+([A-Z][a-z]{1,15})",
        r"\bcalled\s+([A-Z][a-z]{1,15})",
        r"\bname\s+was\s+([A-Z][a-z]{1,15})",
    ]
    for pattern in patterns:
        m=re.search(pattern,story)
        if m and m.group(1) not in STOP: return m.group(1)
    words=re.findall(r"\b[A-Z][a-z]{2,15}\b",story)
    for word in words:
        if word not in STOP: return word
    return "a little animal"

PROMPTS=[
    "Write a story about {name}.",
    "Tell me a story about {name}.",
    "Make up a short story featuring {name}.",
    "Create a story where {name} is the main character.",
    "I want a story about {name}.",
    "Please write a story about {name}.",
    "Can you make up an adventure about {name}?",
    "Write me a little story featuring {name}.",
    "Come up with a story about {name}.",
    "Tell a fun story starring {name}.",
    "Write an imaginative story about {name}.",
    "Make a story with {name} as the main character."
]
# Explicit constraint-following examples. TinyStories alone does not reliably teach
# the model to preserve requested relationships, species, and character details.
CONSTRAINED_PROMPTS = [
    "Write a story about {name} {detail}.",
    "Tell me a story featuring {name} {detail}.",
    "Make up a short story about {name} {detail}.",
    "Create an adventure about {name} {detail}.",
]
STORY_FRAMES = [
    "{name} {identity} {detail}. One morning, {name} {action1}. At first, things did not go as planned. {name} {action2}. Soon, {name} {resolution}. By the end of the day, {name} felt happy and proud. It had been a wonderful adventure.",
    "{name} {identity} {detail}. One sunny day, {name} {action1}. A little problem appeared, but {name} stayed calm. {name} {action2}. After that, {name} {resolution}. Everyone was glad, and {name} went home feeling happy.",
    "Everyone in town knew that {name} {identity} {detail}. One day, {name} {action1}. Then something unexpected happened. Instead of giving up, {name} {action2}. In the end, {name} {resolution}. It was a day {name} would always remember.",
    "{name} {identity} {detail}. {name} loved ordinary days, but today was different. First, {name} {action1}. Next, {name} {action2}. The problem was finally solved when {name} {resolution}. That evening, {name} was glad to be home.",
    "One day, {name} wanted to try something new. {name} {identity} {detail}. During the adventure, {name} {action1}. A small mistake made the plan harder, so {name} {action2}. Thankfully, {name} {resolution}. The day ended with a big smile."
]
def make_constrained_examples(seed):
    rng = random.Random(seed)
    examples = []
    people = [
        ("Lily", "was a little girl", "who loved her mom", "who loves her mom", "visited the garden", "picked a flower for her mom", "gave the flower to her mom, who thanked her"),
        ("Mia", "was a little girl", "who loved her dad", "who loves her dad", "built a tall block tower", "asked her dad to help with the last block", "finished the tower together with her dad"),
        ("Max", "was a little boy", "who loved his grandma", "who loves his grandma", "baked cookies", "saved the biggest cookie for his grandma", "shared the cookies with his grandma"),
        ("Sam", "was a little boy", "who loved his family", "who loves his family", "went on a walk", "collected pretty leaves to show them", "showed the leaves to his family"),
        ("Lily", "was a curious girl", "who loved her little brother", "who loves her little brother", "made a paper airplane", "helped her brother fold one too", "flew the airplanes with her brother"),
        ("Mia", "was a kind girl", "who loved her mom", "who loves her mom", "noticed her mom was busy", "set the table for dinner", "sat down to eat with her mom")
    ]
    animals = [
        ("Rex", "was a friendly dog", "who loved his owner", "the dog who loves his owner", "followed a trail through the park", "found a lost ball near a bench", "carried the ball back to its owner"),
        ("Rex", "was a playful dog", "who loved making friends", "the dog who loves making friends", "met a small rabbit", "helped the rabbit find its way home", "walked back to his owner"),
        ("Buddy", "was a gentle dog", "who loved his family", "the dog who loves his family", "heard a kitten meowing", "stayed nearby while the kitten came down", "led the kitten safely back to its family"),
        ("Pip", "was a curious rabbit", "who loved his garden", "the rabbit who loves his garden", "found a tiny hole in the fence", "asked his animal friends for help", "fixed the fence with them"),
        ("Milo", "was a clever cat", "who loved his owner", "the cat who loves his owner", "noticed a toy under the sofa", "carefully reached under the sofa", "brought the toy back to his owner"),
        ("Daisy", "was a brave little dog", "who loved her family", "the dog who loves her family", "explored the backyard", "found her family's missing hat", "carried the hat back to the house")
    ]
    scenarios = people + animals
    for name, identity, detail, prompt_detail, action1, action2, resolution in scenarios:
        for prompt_template in CONSTRAINED_PROMPTS:
            for frame in STORY_FRAMES:
                prompt = prompt_template.format(name=name, detail=prompt_detail)
                story = frame.format(
                    name=name, identity=identity, detail=detail,
                    action1=action1, action2=action2, resolution=resolution
                )
                pids = encode("User: " + prompt + "\nAssistant:\n")
                answer = encode(story)
                room = BLOCK_SIZE + 1 - 1 - len(pids) - 1
                if room < 24:
                    continue
                answer = answer[:room]
                ids = [1] + pids + answer + [2]
                labels = [-100] * (1 + len(pids)) + answer + [2]
                examples.append((torch.tensor(ids[:-1], dtype=torch.long), torch.tensor(labels[1:], dtype=torch.long)))
    rng.shuffle(examples)
    split = int(len(examples) * 0.9)
    return examples[:split], examples[split:]

def make_examples(stories, seed):
    rng=random.Random(seed); examples=[]
    for story in stories:
        name=protagonist(story)
        prompt=rng.choice(PROMPTS).format(name=name)
        pids=encode("User: "+prompt+"\nAssistant:\n")
        answer=encode(story)
        # Reserve one position for BOS and one for EOS; retain a meaningful answer.
        room=BLOCK_SIZE+1-1-len(pids)-1
        if room<24: continue
        answer=answer[:room]
        ids=[1]+pids+answer+[2]
        labels=[-100]*(1+len(pids))+answer+[2]
        examples.append((torch.tensor(ids[:-1],dtype=torch.long),torch.tensor(labels[1:],dtype=torch.long)))
    return examples

train_examples=make_examples(train_stories,SEED)
val_examples=make_examples(val_stories,SEED+1)
extra_train, extra_val = make_constrained_examples(SEED + 77)
# Oversample the smaller, carefully checked set so explicit constraints matter
# during fine-tuning without replacing the broad TinyStories examples.
train_examples.extend(extra_train * 8)
val_examples.extend(extra_val * 8)
random.Random(SEED + 78).shuffle(train_examples)
random.Random(SEED + 79).shuffle(val_examples)
if len(train_examples)<1000 or len(val_examples)<500:
    raise RuntimeError("Not enough usable instruction examples were created.")
print(f"Instruction pairs: {len(train_examples):,} train / {len(val_examples):,} validation",flush=True)

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
        x=x+self.attn(self.ln1(x))
        return x+self.fc2(F.gelu(self.fc1(self.ln2(x)),approximate="tanh"))
class GPT(nn.Module):
    def __init__(self):
        super().__init__(); self.wte=nn.Embedding(VOCAB_SIZE,N_EMBD); self.wpe=nn.Embedding(BLOCK_SIZE,N_EMBD)
        self.drop=nn.Dropout(.10); self.blocks=nn.ModuleList([Block() for _ in range(N_LAYER)])
        self.ln_f=nn.LayerNorm(N_EMBD); self.lm_head=nn.Linear(N_EMBD,VOCAB_SIZE)
    def forward(self,idx,targets=None):
        _,t=idx.shape; pos=torch.arange(t,device=idx.device)
        x=self.drop(self.wte(idx)+self.wpe(pos))
        for block in self.blocks: x=block(x)
        logits=self.lm_head(self.ln_f(x)); loss=None
        if targets is not None: loss=F.cross_entropy(logits.reshape(-1,VOCAB_SIZE),targets.reshape(-1),ignore_index=-100)
        return logits,loss

model=GPT()
initial=ck.get("best_model") or ck["model"]
model.load_state_dict(initial)
model.to(device)
params=sum(p.numel() for p in model.parameters())
optimizer=torch.optim.AdamW(model.parameters(),lr=LEARNING_RATE,weight_decay=.01)
print(f"Loaded V8 best weights: {params:,} parameters | {ck['step']:,} original steps",flush=True)

def batch(examples, indices):
    selected=[examples[i] for i in indices]
    width=max(x.numel() for x,y in selected)
    xs=torch.zeros((len(selected),width),dtype=torch.long)
    ys=torch.full((len(selected),width),-100,dtype=torch.long)
    for row,(x,y) in enumerate(selected):
        xs[row,:x.numel()]=x
        ys[row,:y.numel()]=y
    return xs.to(device),ys.to(device)

@torch.inference_mode()
def evaluate():
    model.eval(); vals=[]
    rng=random.Random(SEED+909)
    for _ in range(EVAL_BATCHES):
        ids=[rng.randrange(len(val_examples)) for _ in range(BATCH_SIZE)]
        x,y=batch(val_examples,ids)
        _,loss=model(x,y); vals.append(loss.item())
    model.train()
    return sum(vals)/len(vals)

best=float("inf"); best_state=None
if CHECKPOINT_OUT.exists():
    print("An existing Phase 2 checkpoint was found; starting a fresh fine-tune from V8 to keep this run reproducible.",flush=True)
start=time.perf_counter()
model.train()
for step in range(1,TRAIN_STEPS+1):
    indices=[random.randrange(len(train_examples)) for _ in range(BATCH_SIZE)]
    x,y=batch(train_examples,indices)
    optimizer.zero_grad(set_to_none=True)
    _,loss=model(x,y)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
    optimizer.step()
    if step%EVAL_INTERVAL==0 or step==1 or step==TRAIN_STEPS:
        val=evaluate()
        if val<best:
            best=val
            best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        print(f"step {step}/{TRAIN_STEPS} train={loss.item():.4f} instruction_val={val:.4f} best={best:.4f} speed={step/max(time.perf_counter()-start,1e-9):.2f} steps/s",flush=True)
        if best_state:
            torch.save({"format":"littleGPT-v9-instruct-checkpoint","step":step,"model":model.state_dict(),"best_model":best_state,"best":best,"vocab":vocab,"config":cfg},CHECKPOINT_OUT)
if best_state is None: raise RuntimeError("No best instruction checkpoint was produced.")
model.load_state_dict(best_state); model.eval()

def arr(t):
    packed=t.detach().to(device="cpu",dtype=torch.float16).contiguous()
    return {"f16":base64.b64encode(bytes(packed.untyped_storage())).decode("ascii")}
state={"wte":arr(model.wte.weight),"wpe":arr(model.wpe.weight),
       "ln_f":{"weight":arr(model.ln_f.weight),"bias":arr(model.ln_f.bias)},
       "lm_head":{"weight":arr(model.lm_head.weight),"bias":arr(model.lm_head.bias)},"blocks":[]}
for b in model.blocks:
    state["blocks"].append({"ln1":{"weight":arr(b.ln1.weight),"bias":arr(b.ln1.bias)},
      "attn":{"qkv":{"weight":arr(b.attn.qkv.weight),"bias":arr(b.attn.qkv.bias)},
              "proj":{"weight":arr(b.attn.proj.weight),"bias":arr(b.attn.proj.bias)}},
      "ln2":{"weight":arr(b.ln2.weight),"bias":arr(b.ln2.bias)},
      "mlp":{"fc1":{"weight":arr(b.fc1.weight),"bias":arr(b.fc1.bias)},
             "fc2":{"weight":arr(b.fc2.weight),"bias":arr(b.fc2.bias)}}})
payload={"format":"littleGPT-v7","created_at":__import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
 "config":{"block_size":BLOCK_SIZE,"n_embd":N_EMBD,"n_head":N_HEAD,"n_layer":N_LAYER,"vocab_size":VOCAB_SIZE,
 "unk_id":0,"bos_id":1,"eos_id":2,"tokenizer":"byte-subword-greedy-v1","max_token_bytes":max_width},
 "vocab":vocab,"parameters":params,
 "training":{"steps":TRAIN_STEPS,"loss":best,"best_validation_loss":round(best,4),
 "dataset":"TinyStories instruction pairs (synthetic story-writing prompts paired with original stories)",
 "instruction_tuned":True,"base_model":"littleGPT V8","base_training_steps":ck["step"],
 "instruction_pairs":len(train_examples),"validation_pairs":len(val_examples),"learning_rate":LEARNING_RATE,
 "batch_size":BATCH_SIZE,"device":str(device)},"state":state}
Path("model.json").write_text(json.dumps(payload,separators=(",",":")),encoding="utf-8")
print(f"Exported Phase 2 littleGPT ({Path('model.json').stat().st_size/1024/1024:.1f} MB)",flush=True)
print(f"Best instruction validation loss: {best:.4f}",flush=True)
