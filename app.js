const MODEL_URL = "./model.json";
const $ = (id) => document.getElementById(id);
let model = null;
let generating = false;
let cancelled = false;

function setStatus(ready, text) {
  $("model-state").textContent = text;
  $("status-dot").classList.toggle("ready", ready);
  $("weight-label").textContent = ready ? "WEIGHTS LOADED" : "NO WEIGHTS";
  $("weight-label").classList.toggle("loaded", ready);
  $("generate").disabled = !ready || generating || !$("prompt").value.trim();
}
function fmt(n) { return Number(n).toLocaleString("en-US"); }
async function loadModel() {
  setStatus(false, "Looking for trained weights…");
  try {
    const response = await fetch(MODEL_URL + "?v=" + Date.now(), {cache:"no-store"});
    if (!response.ok) throw new Error("No trained model yet");
    const data = await response.json();
    if (data.format !== "littleGPT-v1" || !data.state) throw new Error("Model file is invalid");
    model = data;
    $("param-count").textContent = fmt(data.parameters);
    $("loss-value").textContent = Number(data.training.loss).toFixed(3);
    $("steps-value").textContent = fmt(data.training.steps);
    $("vocab-value").textContent = fmt(data.vocab.length);
    $("context-value").textContent = data.config.block_size;
    $("model-state").textContent = "Trained · " + data.training.steps + " steps";
    $("runtime-note").textContent = "Model ready · " + fmt(data.parameters) + " parameters loaded into this tab.";
    $("welcome-title").textContent = "Your model is ready.";
    $("welcome-copy").textContent = "These weights were learned from random initialization by a GitHub Actions CPU runner. Type a prompt and see what it learned.";
    setStatus(true, "Trained · " + data.training.steps + " steps");
  } catch (error) {
    model = null;
    $("param-count").textContent = "—"; $("loss-value").textContent = "—";
    $("steps-value").textContent = "—"; $("vocab-value").textContent = "—"; $("context-value").textContent = "—";
    $("runtime-note").textContent = "No trained weights found yet. Run the GitHub Actions workflow.";
    setStatus(false, "Not trained yet");
  }
}
function matvec(x, weight, bias) {
  const out = new Array(bias.length);
  for (let j=0;j<bias.length;j++) { let sum=bias[j]; for(let i=0;i<x.length;i++) sum += x[i]*weight[j][i]; out[j]=sum; }
  return out;
}
function layerNorm(x, scale, bias, eps=1e-5) {
  let mean=0; for(const v of x) mean+=v; mean/=x.length;
  let variance=0; for(const v of x) variance+=(v-mean)*(v-mean); variance/=x.length;
  const inv=1/Math.sqrt(variance+eps);
  return x.map((v,i)=>(v-mean)*inv*scale[i]+bias[i]);
}
function gelu(x) { return 0.5*x*(1+Math.tanh(0.7978845608*(x+0.044715*x*x*x))); }
function softmax(values) {
  const max=Math.max(...values), exps=values.map(v=>Math.exp(v-max));
  const sum=exps.reduce((a,b)=>a+b,0); return exps.map(v=>v/sum);
}
function linearRows(rows, weight, bias) { return rows.map(row=>matvec(row,weight,bias)); }
function forward(tokenIds) {
  const s=model.state, c=model.config, T=tokenIds.length, D=c.n_embd, H=c.n_head, HD=D/H;
  let x=tokenIds.map((id,pos)=>s.wte[id].map((v,j)=>v+s.wpe[pos][j]));
  for(let layer=0;layer<c.n_layer;layer++) {
    const p=s.blocks[layer], residual=x;
    const norm=x.map(row=>layerNorm(row,p.ln1.weight,p.ln1.bias));
    const qkv=linearRows(norm,p.attn.qkv.weight,p.attn.qkv.bias);
    const attended=Array.from({length:T},()=>new Array(D).fill(0));
    for(let h=0;h<H;h++) for(let t=0;t<T;t++) {
      const scores=[];
      for(let j=0;j<=t;j++) {
        let dot=0; for(let k=0;k<HD;k++) dot+=qkv[t][h*HD+k]*qkv[j][D+h*HD+k];
        scores.push(dot/Math.sqrt(HD));
      }
      const probs=softmax(scores);
      for(let k=0;k<HD;k++) { let value=0; for(let j=0;j<=t;j++) value+=probs[j]*qkv[j][2*D+h*HD+k]; attended[t][h*HD+k]=value; }
    }
    x=linearRows(attended,p.attn.proj.weight,p.attn.proj.bias).map((row,t)=>row.map((v,j)=>v+residual[t][j]));
    const residual2=x;
    x=x.map(row=>layerNorm(row,p.ln2.weight,p.ln2.bias));
    x=linearRows(x,p.mlp.fc1.weight,p.mlp.fc1.bias).map(row=>row.map(gelu));
    x=linearRows(x,p.mlp.fc2.weight,p.mlp.fc2.bias).map((row,t)=>row.map((v,j)=>v+residual2[t][j]));
  }
  const last=layerNorm(x[T-1],s.ln_f.weight,s.ln_f.bias);
  return matvec(last,s.lm_head.weight,s.lm_head.bias);
}
function sample(logits, temperature, topK) {
  const scaled=logits.map(v=>v/Math.max(0.05,temperature));
  if(topK>0 && topK<scaled.length) {
    const keep=new Set(scaled.map((v,i)=>[v,i]).sort((a,b)=>b[0]-a[0]).slice(0,topK).map(x=>x[1]));
    for(let i=0;i<scaled.length;i++) if(!keep.has(i)) scaled[i]=-Infinity;
  }
  const probs=softmax(scaled);
  let r=Math.random(); for(let i=0;i<probs.length;i++){r-=probs[i];if(r<=0)return i;} return probs.length-1;
}
async function generate(prompt, maxTokens=240, temperature=0.8, topK=24) {
  const stoi=new Map(model.vocab.map((ch,i)=>[ch,i]));
  const unknown=stoi.has(" ") ? stoi.get(" ") : 0;
  let ids=Array.from(prompt).map(ch=>stoi.has(ch)?stoi.get(ch):unknown);
  if(!ids.length) ids=[unknown];
  let output="";
  for(let i=0;i<maxTokens;i++) {
    if(cancelled) break;
    const context=ids.slice(-model.config.block_size);
    const logits=forward(context);
    const next=sample(logits,temperature,topK);
    ids.push(next);
    const ch=model.vocab[next];
    output+=ch;
    if(i%4===0) {
      const node=document.querySelector(".message.assistant:last-child .message-body");
      if(node) node.textContent=output;
      await new Promise(resolve=>setTimeout(resolve,0));
    }
  }
  return output;
}
function addMessage(role, content) {
  const wrap=document.createElement("article"); wrap.className="message "+role;
  const label=document.createElement("div"); label.className="message-label"; label.textContent=role==="user"?"YOU":"LITTLEGPT";
  const body=document.createElement("div"); body.className="message-body"; body.textContent=content;
  wrap.append(label,body); $("messages").append(wrap); $("messages").scrollTop=$("messages").scrollHeight;
  return body;
}
async function handleGenerate(prompt) {
  if(!model||generating||!prompt.trim()) return;
  generating=true; cancelled=false; $("generate").disabled=true; $("generate").innerHTML="<span>Generating…</span><b>Ⅱ</b>";
  $("welcome").hidden=true;
  addMessage("user",prompt);
  addMessage("assistant","…");
  try {
    const result=await generate(prompt,240,0.8,24);
    const node=document.querySelector(".message.assistant:last-child .message-body");
    if(node) node.textContent=result || "(Generation stopped.)";
  } catch(error) {
    const node=document.querySelector(".message.assistant:last-child .message-body");
    if(node) node.textContent="Inference error: "+error.message;
    console.error(error);
  } finally {
    generating=false; $("generate").innerHTML="<span>Generate</span><b>↑</b>"; $("generate").disabled=!model||!$("prompt").value.trim();
  }
}
$("chat-form").addEventListener("submit",async event=>{
  event.preventDefault(); const prompt=$("prompt").value; $("prompt").value=""; $("prompt").style.height="auto"; $("generate").disabled=true; await handleGenerate(prompt);
});
$("prompt").addEventListener("input",()=>{$("prompt").style.height="auto";$("prompt").style.height=Math.min($("prompt").scrollHeight,120)+"px";$("generate").disabled=!model||generating||!$("prompt").value.trim();});
$("prompt").addEventListener("keydown",event=>{if(event.key==="Enter"&&!event.shiftKey){event.preventDefault();$("chat-form").requestSubmit();}});
document.querySelectorAll("[data-prompt]").forEach(button=>button.addEventListener("click",()=>{$("prompt").value=button.dataset.prompt;$("prompt").dispatchEvent(new Event("input"));$("prompt").focus();}));
$("reload-model").addEventListener("click",loadModel);
$("generate").addEventListener("click",()=>{if(generating){cancelled=true;}});
loadModel();