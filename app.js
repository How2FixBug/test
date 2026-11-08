const MODEL_URL="./model.json";
const $=id=>document.getElementById(id);
let model=null,generating=false,cancelled=false;
let chats=[],activeChatId=null;
const STORAGE_KEY="littlegpt-chats-v1";
function makeId(){return Date.now().toString(36)+Math.random().toString(36).slice(2,7)}
function newChat(){const chat={id:makeId(),title:"New chat",messages:[]};chats.unshift(chat);activeChatId=chat.id;saveChats();renderChatList();renderActiveChat();$("prompt").focus()}
function saveChats(){try{localStorage.setItem(STORAGE_KEY,JSON.stringify(chats))}catch(e){console.warn("Could not save chat history",e)}}
function loadChats(){try{const saved=JSON.parse(localStorage.getItem(STORAGE_KEY)||"[]");if(Array.isArray(saved))chats=saved.filter(c=>c&&typeof c.id==="string"&&Array.isArray(c.messages))}catch(e){chats=[]}if(!chats.length)chats=[{id:makeId(),title:"New chat",messages:[]}];activeChatId=chats[0].id;renderChatList();renderActiveChat()}
function renderChatList(){
 const list=$("chat-list");list.replaceChildren();
 chats.forEach(chat=>{
  const row=document.createElement("div");row.className="chat-row"+(chat.id===activeChatId?" active":"");
  const select=document.createElement("button");select.type="button";select.className="chat-item";select.textContent=chat.title||"New chat";select.title=select.textContent;
  select.addEventListener("click",()=>{if(generating)return;activeChatId=chat.id;renderChatList();renderActiveChat()});
  const actions=document.createElement("div");actions.className="chat-actions";
  const rename=document.createElement("button");rename.type="button";rename.className="chat-action";rename.textContent="✎";rename.title="Rename chat";rename.setAttribute("aria-label","Rename chat");
  rename.addEventListener("click",event=>{event.stopPropagation();if(generating)return;const value=window.prompt("Rename chat:",chat.title||"New chat");if(value===null)return;const title=value.trim();if(!title)return;chat.title=title.slice(0,80);saveChats();renderChatList();if(chat.id===activeChatId)$("current-chat-title").textContent=chat.title});
  const remove=document.createElement("button");remove.type="button";remove.className="chat-action delete-chat";remove.textContent="×";remove.title="Delete chat";remove.setAttribute("aria-label","Delete chat");
  remove.addEventListener("click",event=>{event.stopPropagation();if(generating)return;if(!window.confirm("Delete this chat? This cannot be undone."))return;const index=chats.findIndex(item=>item.id===chat.id);if(index<0)return;chats.splice(index,1);if(activeChatId===chat.id){if(chats.length)activeChatId=chats[Math.min(index,chats.length-1)].id;else{const fresh={id:makeId(),title:"New chat",messages:[]};chats.push(fresh);activeChatId=fresh.id}}saveChats();renderChatList();renderActiveChat()});
  actions.append(rename,remove);row.append(select,actions);list.append(row);
 });
}
function activeChat(){return chats.find(c=>c.id===activeChatId)}
function renderActiveChat(){
 const chat=activeChat();if(!chat)return;$("current-chat-title").textContent=chat.title||"New chat";const messages=$("messages");messages.replaceChildren();
 if(!chat.messages.length){const welcome=document.createElement("div");welcome.className="welcome";welcome.id="welcome";const glyph=document.createElement("div");glyph.className="welcome-glyph";glyph.textContent="✳";const h=document.createElement("h1");h.id="welcome-title";h.textContent=model?"Your model is ready.":"What can I help you explore?";const p=document.createElement("p");p.id="welcome-copy";p.textContent=model?"littleGPT is trained on simple English stories. Try a story prompt and watch it continue.":"A tiny language model trained from random weights on a GitHub CPU runner.";const chips=document.createElement("div");chips.className="prompt-chips";[["Once upon a time,","Start a story"],["A little robot wanted to","Meet a robot"],["One sunny morning,","Start an adventure"]].forEach(([prompt,label])=>{const b=document.createElement("button");b.className="prompt-chip";b.type="button";b.textContent=label;b.addEventListener("click",()=>{$("prompt").value=prompt;$("prompt").dispatchEvent(new Event("input"));$("prompt").focus()});chips.append(b)});welcome.append(glyph,h,p,chips);messages.append(welcome)}
 else chat.messages.forEach(m=>addMessage(m.role,m.content,false));messages.scrollTop=messages.scrollHeight;
}
function setStatus(ready,text){$("model-state").textContent=text;$("status-dot").classList.toggle("ready",ready);$("weight-label").textContent=ready?"WEIGHTS LOADED":"NO WEIGHTS";$("generate").disabled=!ready||generating||!$("prompt").value.trim()}
function fmt(n){return Number(n).toLocaleString("en-US")}
async function loadModel(){
 setStatus(false,"Looking for trained weights…");
 try{const response=await fetch(MODEL_URL+"?v="+Date.now(),{cache:"no-store"});if(!response.ok)throw new Error("No trained model yet");const data=await response.json();if(!["littleGPT-v3","littleGPT-v4","littleGPT-v5","littleGPT-v6"].includes(data.format)||!data.state)throw new Error("Model file is invalid");model=data;model.vocabToId=new Map(data.vocab.map((token,index)=>[token,index]));$("param-count").textContent=fmt(data.parameters);$("loss-value").textContent=Number(data.training.loss).toFixed(3);$("steps-value").textContent=fmt(data.training.steps);$("vocab-value").textContent=fmt(data.vocab.length);$("context-value").textContent=data.config.block_size;$("runtime-note").textContent="Model ready · "+fmt(data.parameters)+" parameters · "+data.training.dataset;setStatus(true,"Trained · "+fmt(data.training.steps)+" steps");renderActiveChat()}
 catch(error){model=null;$("param-count").textContent="—";$("loss-value").textContent="—";$("steps-value").textContent="—";$("vocab-value").textContent="—";$("context-value").textContent="—";$("runtime-note").textContent="No trained weights yet. Run the littleGPT v6 training workflow.";setStatus(false,"Not trained yet")}
}
function matvec(x,w,b){const out=new Array(b.length);for(let j=0;j<b.length;j++){let sum=b[j];for(let i=0;i<x.length;i++)sum+=x[i]*w[j][i];out[j]=sum}return out}
function layerNorm(x,scale,bias,eps=1e-5){let mean=0;for(const v of x)mean+=v;mean/=x.length;let variance=0;for(const v of x)variance+=(v-mean)*(v-mean);variance/=x.length;const inv=1/Math.sqrt(variance+eps);return x.map((v,i)=>(v-mean)*inv*scale[i]+bias[i])}
function gelu(x){return .5*x*(1+Math.tanh(.7978845608*(x+.044715*x*x*x)))}
function softmax(values){const max=Math.max(...values),exps=values.map(v=>Math.exp(v-max)),sum=exps.reduce((a,b)=>a+b,0);return exps.map(v=>v/sum)}
function linearRows(rows,w,b){return rows.map(row=>matvec(row,w,b))}
function forward(tokenIds){
 const s=model.state,c=model.config,T=tokenIds.length,D=c.n_embd,H=c.n_head,HD=D/H;let x=tokenIds.map((id,pos)=>s.wte[id].map((v,j)=>v+s.wpe[pos][j]));
 for(let layer=0;layer<c.n_layer;layer++){const p=s.blocks[layer],residual=x,norm=x.map(row=>layerNorm(row,p.ln1.weight,p.ln1.bias)),qkv=linearRows(norm,p.attn.qkv.weight,p.attn.qkv.bias),attended=Array.from({length:T},()=>new Array(D).fill(0));
  for(let h=0;h<H;h++)for(let t=0;t<T;t++){const scores=[];for(let j=0;j<=t;j++){let dot=0;for(let k=0;k<HD;k++)dot+=qkv[t][h*HD+k]*qkv[j][D+h*HD+k];scores.push(dot/Math.sqrt(HD))}const probs=softmax(scores);for(let k=0;k<HD;k++){let value=0;for(let j=0;j<=t;j++)value+=probs[j]*qkv[j][2*D+h*HD+k];attended[t][h*HD+k]=value}}
  x=linearRows(attended,p.attn.proj.weight,p.attn.proj.bias).map((row,t)=>row.map((v,j)=>v+residual[t][j]));const residual2=x;x=x.map(row=>layerNorm(row,p.ln2.weight,p.ln2.bias));x=linearRows(x,p.mlp.fc1.weight,p.mlp.fc1.bias).map(row=>row.map(gelu));x=linearRows(x,p.mlp.fc2.weight,p.mlp.fc2.bias).map((row,t)=>row.map((v,j)=>v+residual2[t][j]));
 }
 return matvec(layerNorm(x[T-1],s.ln_f.weight,s.ln_f.bias),s.lm_head.weight,s.lm_head.bias)
}
function sample(logits,temp,topK){const scaled=logits.map(v=>v/Math.max(.05,temp));if(topK>0&&topK<scaled.length){const keep=new Set(scaled.map((v,i)=>[v,i]).sort((a,b)=>b[0]-a[0]).slice(0,topK).map(x=>x[1]));for(let i=0;i<scaled.length;i++)if(!keep.has(i))scaled[i]=-Infinity}const probs=softmax(scaled);let r=Math.random();for(let i=0;i<probs.length;i++){r-=probs[i];if(r<=0)return i}return probs.length-1}
function encodePrompt(prompt){const normalized=prompt.replace(/\u00e2\u20ac\u0153|\u00e2\u20ac\u009d/g,'"').replace(/\u00e2\u20ac\u02dc|\u00e2\u20ac\u2122/g,"'").replace(/[“”„‟]/g,'"').replace(/[‘’‚‛]/g,"'").replace(/€/g,'"').replace(/™/g,"'").replace(/\u00a0/g," ");const pieces=normalized.toLowerCase().match(/[a-z0-9]+(?:'[a-z0-9]+)?|[.,!?;:"()\-]/g)||[];const ids=[model.config.bos_id];for(const piece of pieces)ids.push(model.vocabToId.get(piece)??model.config.unk_id);return ids}
function decodeTokens(ids){let out="",quoteOpen=false;for(const id of ids){if(id===model.config.eos_id)break;const token=model.vocab[id];if(!token||token==="<unk>"||token==="<bos>")continue;if(token==='"'){if(quoteOpen)out=out.replace(/\s+$/,"")+'"';else out+=(out?" ":"")+'"';quoteOpen=!quoteOpen;continue}const punctuation=/^[.,!?;:%)}\]]$/.test(token);const opening=/^[([{]$/.test(token);if(!out)out=token;else if(punctuation)out+=token;else if(opening)out+=" "+token;else out+=" "+token}return out.replace(/(^|[.!?]\s+)([a-z])/g,(_,prefix,ch)=>prefix+ch.toUpperCase())}
async function generate(prompt,maxTokens=120,temp=.8,topK=32,onUpdate=()=>{}){
 const ids=encodePrompt(prompt);let generated=[];
 for(let i=0;i<maxTokens;i++){if(cancelled)break;const context=ids.slice(-model.config.block_size);const next=sample(forward(context),temp,topK);if(next===model.config.eos_id)break;ids.push(next);generated.push(next);if(i%2===0){onUpdate(decodeTokens(generated));await new Promise(resolve=>setTimeout(resolve,0))}}
 onUpdate(decodeTokens(generated));return decodeTokens(generated)
}
function addMessage(role,content,store=true){const wrap=document.createElement("article");wrap.className="message "+role;const label=document.createElement("div");label.className="message-label";label.textContent=role==="user"?"YOU":"LITTLEGPT";const body=document.createElement("div");body.className="message-body";body.textContent=content;wrap.append(label,body);$("messages").append(wrap);if(store){const chat=activeChat();chat.messages.push({role,content});saveChats()}$("messages").scrollTop=$("messages").scrollHeight;return body}
async function handleGenerate(prompt){
 if(!model||generating||!prompt.trim())return;generating=true;cancelled=false;$("generate").disabled=true;$("generate").textContent="…";const chat=activeChat();$("welcome")?.remove();
 if(chat.title==="New chat"){chat.title=prompt.trim().replace(/\s+/g," ").slice(0,34);$("current-chat-title").textContent=chat.title;renderChatList()}
 addMessage("user",prompt);const reply=addMessage("assistant","…");const record=chat.messages[chat.messages.length-1];
 try{const result=await generate(prompt,100,.76,28,text=>{reply.textContent=text;$("messages").scrollTop=$("messages").scrollHeight});reply.textContent=result||"(No text generated.)";record.content=reply.textContent;saveChats()}
 catch(error){reply.textContent="Inference error: "+error.message;record.content=reply.textContent;saveChats();console.error(error)}
 finally{generating=false;$("generate").textContent="↑";$("generate").disabled=!model||!$("prompt").value.trim()}
}
$("new-chat").addEventListener("click",()=>{if(!generating)newChat()});
$("chat-form").addEventListener("submit",async event=>{event.preventDefault();const prompt=$("prompt").value;$("prompt").value="";$("prompt").style.height="auto";$("generate").disabled=true;await handleGenerate(prompt)});
$("prompt").addEventListener("input",()=>{$("prompt").style.height="auto";$("prompt").style.height=Math.min($("prompt").scrollHeight,120)+"px";$("generate").disabled=!model||generating||!$("prompt").value.trim()});
$("prompt").addEventListener("keydown",event=>{if(event.key==="Enter"&&!event.shiftKey){event.preventDefault();$("chat-form").requestSubmit()}});
$("reload-model").addEventListener("click",loadModel);$("generate").addEventListener("click",()=>{if(generating)cancelled=true});
loadChats();loadModel();
