/* Local Internet storage and website runtime. No server required. */
(() => {
  const DB_NAME='local-internet-db', DB_VERSION=1;
  let dbPromise;
  function openDB(){
    if(dbPromise)return dbPromise;
    dbPromise=new Promise((resolve,reject)=>{
      const req=indexedDB.open(DB_NAME,DB_VERSION);
      req.onupgradeneeded=()=>{
        const db=req.result;
        for(const name of ['sites','users','transactions','posts','visits','notifications'])if(!db.objectStoreNames.contains(name))db.createObjectStore(name,{keyPath:'id'});
      };
      req.onsuccess=()=>resolve(req.result);req.onerror=()=>reject(req.error);
    });return dbPromise;
  }
  async function all(store){const db=await openDB();return new Promise((res,rej)=>{const r=db.transaction(store).objectStore(store).getAll();r.onsuccess=()=>res(r.result||[]);r.onerror=()=>rej(r.error)})}
  async function get(store,id){const db=await openDB();return new Promise((res,rej)=>{const r=db.transaction(store).objectStore(store).get(id);r.onsuccess=()=>res(r.result||null);r.onerror=()=>rej(r.error)})}
  async function put(store,value){const db=await openDB();return new Promise((res,rej)=>{const tx=db.transaction(store,'readwrite');tx.objectStore(store).put(value);tx.oncomplete=()=>res(value);tx.onerror=()=>rej(tx.error)})}
  async function del(store,id){const db=await openDB();return new Promise((res,rej)=>{const tx=db.transaction(store,'readwrite');tx.objectStore(store).delete(id);tx.oncomplete=()=>res();tx.onerror=()=>rej(tx.error)})}
  const uid=()=>crypto.randomUUID?crypto.randomUUID():Date.now().toString(36)+Math.random().toString(36).slice(2);
  const safeDomain=s=>String(s||'').trim().toLowerCase().replace(/^https?:\/\//,'').replace(/\/$/,'');
  const esc=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const starterSites=[
    {domain:'search.local',name:'Findit',description:'A small search engine for this very large little internet.',icon:'⌕',color:'#d8e9ff',files:{'index.html':`<!doctype html><html><head><title>Findit</title><style>*{box-sizing:border-box}body{font:16px Arial;background:#f7f9fc;color:#18243a;margin:0}.wrap{max-width:720px;margin:13vh auto;padding:24px}.logo{font-size:48px;font-weight:800;letter-spacing:-3px;color:#3269bd;text-align:center}.tag{text-align:center;color:#71809a;margin:8px 0 28px}.search{display:flex;gap:8px}input{flex:1;padding:15px;border:1px solid #c9d5e5;border-radius:9px;font-size:16px}button{background:#3269bd;color:white;border:0;border-radius:9px;padding:0 20px;font-weight:700}.results{margin-top:30px}.result{padding:15px 0;border-bottom:1px solid #e1e7ef}.result a{color:#245aa9;font-size:18px}.result p{color:#5d6c82;font-size:13px;line-height:1.5}</style></head><body><main class="wrap"><div class="logo">findit.</div><div class="tag">Search the Local Internet</div><form class="search" id="f"><input id="q" placeholder="Search websites, people, ideas..." autofocus><button>Search</button></form><div class="results" id="results"></div></main><script>document.getElementById('f').onsubmit=async e=>{e.preventDefault();const q=document.getElementById('q').value;const r=await Internet.search(q);document.getElementById('results').innerHTML=r.map(x=>'<div class="result"><a href="'+x.url+'">'+x.name+'</a><p>'+x.description+'</p></div>').join('')||'No results yet.'};</script></body></html>`}},
    {domain:'forum.local',name:'Town Square',description:'A community board for conversations, questions, and discoveries.',icon:'☷',color:'#e6ddff',files:{'index.html':`<!doctype html><html><head><title>Town Square</title><style>body{margin:0;background:#f4f3fb;color:#25243b;font:15px Arial}.top{background:#51448e;color:white;padding:28px max(calc((100% - 850px)/2),24px)}main{max-width:850px;margin:30px auto;padding:0 20px}.post{background:white;border:1px solid #e4e1f0;border-radius:12px;padding:18px;margin:12px 0}.post small{color:#827c9e}.post h3{margin:8px 0}.post p{color:#59566d;line-height:1.5}button{background:#594b9d;color:white;border:0;border-radius:8px;padding:10px 14px}textarea{width:100%;min-height:65px;border:1px solid #ddd8ed;border-radius:8px;padding:10px}</style></head><body><header class="top"><h1>Town Square</h1><p>A place for the people of this internet.</p></header><main><h2>Community board</h2><div class="post"><small>Welcome · Town Square team</small><h3>Welcome to Town Square!</h3><p>Introduce yourself, share something interesting, or ask the neighborhood a question.</p></div><form id="f"><textarea id="body" placeholder="Start a discussion..."></textarea><p><button>Publish post</button></p></form><div id="posts"></div></main><script>async function load(){const p=await Internet.posts.list();document.getElementById('posts').innerHTML=p.map(x=>'<article class="post"><small>'+x.author+'</small><p>'+x.body+'</p></article>').join('')}document.getElementById('f').onsubmit=async e=>{e.preventDefault();await Internet.posts.create(document.getElementById('body').value);document.getElementById('body').value='';load()};load();</script></body></html>`}},
    {domain:'shop.local',name:'Little Market',description:'A tiny marketplace with products and a working local checkout.',icon:'◇',color:'#ffe8d7',files:{'index.html':`<!doctype html><html><head><title>Little Market</title><style>body{margin:0;background:#fffaf5;color:#392b25;font:15px Arial}.head{padding:35px max(calc((100% - 900px)/2),24px);background:#f4d7c4}.head h1{margin:0;font-size:32px}main{max-width:900px;margin:35px auto;padding:0 20px}.products{display:grid;grid-template-columns:repeat(3,1fr);gap:15px}.product{background:white;border:1px solid #eee0d5;border-radius:12px;padding:18px}.emoji{font-size:42px}.product h3{margin:12px 0 5px}.product p{color:#7c6b60;font-size:13px}.price{font-weight:bold;margin:14px 0}button{background:#9c5b38;color:white;border:0;border-radius:8px;padding:10px 13px}@media(max-width:600px){.products{grid-template-columns:1fr}}</style></head><body><header class="head"><h1>Little Market</h1><p>Good finds from around the neighborhood.</p></header><main><h2>Featured finds</h2><div class="products"><article class="product"><div class="emoji">🎧</div><h3>Cloud headphones</h3><p>Comfy headphones for long browsing sessions.</p><div class="price">35 CR</div><button onclick="buy('Cloud headphones',35)">Buy now</button></article><article class="product"><div class="emoji">📓</div><h3>Idea notebook</h3><p>A fresh place for your next big idea.</p><div class="price">12 CR</div><button onclick="buy('Idea notebook',12)">Buy now</button></article><article class="product"><div class="emoji">🪴</div><h3>Desk plant</h3><p>A little green friend for your workspace.</p><div class="price">18 CR</div><button onclick="buy('Desk plant',18)">Buy now</button></article></div><p id="msg"></p></main><script>async function buy(name,price){try{await Internet.wallet.pay({to:'shop.local',amount:price,description:name});document.getElementById('msg').textContent='Purchase complete! Thanks for shopping.'}catch(e){document.getElementById('msg').textContent=e.message}}</script></body></html>`}}
  ];
  const InternetEngine={
    uid,safeDomain,esc,all,get,put,del,starterSites,
    async seed(){
      let me=await get('users','human');if(!await get('users','system'))await put('users',{id:'system',name:'Local Internet',kind:'system',balance:0,bio:'Built-in website publisher',created:Date.now()});
      if(!me){me={id:'human',name:'Roger',kind:'human',balance:500,created:Date.now(),bio:'The human who owns this corner of the internet.'};await put('users',me)}
      const sites=await all('sites');
      if(!sites.length){for(const s of starterSites)await put('sites',{id:uid(),...s,owner:'system',created:Date.now(),visitors:0,earnings:0,files:s.files})}
      const agents=await all('users');
      if(!agents.some(u=>u.kind==='ai'))await put('users',{id:'ai-sam',name:'Sam',kind:'ai',balance:500,bio:'Curious about technology, small communities, and interesting websites.',interests:['technology','community','shopping'],created:Date.now(),memory:[],active:true});
    },
    async me(){return get('users','human')},
    async balance(id='human'){const u=await get('users',id);return u?.balance||0},
    async transfer(from,to,amount,description='Payment'){
      amount=Number(amount);if(!Number.isFinite(amount)||amount<=0||amount>1000000)throw Error('Enter a valid payment amount.');
      const sender=await get('users',from);let recipient=await get('users',to);if(!recipient){const site=(await all('sites')).find(s=>s.domain===safeDomain(to));if(site)recipient=await get('users',site.owner)}
      if(!sender)throw Error('Payer account not found.');
      if(sender.balance<amount)throw Error('Not enough credits.');
      if(!recipient)throw Error('Recipient account not found.');
      sender.balance=Math.round((sender.balance-amount)*100)/100;recipient.balance=Math.round((recipient.balance+amount)*100)/100;
      await put('users',sender);await put('users',recipient);
      await put('transactions',{id:uid(),from,to,amount,description,created:Date.now()});
      return {balance:sender.balance};
    },
    async visit(domain,userId='human'){
      const site=(await all('sites')).find(s=>s.domain===safeDomain(domain));if(!site)return null;
      site.visitors=(site.visitors||0)+1;site.lastVisit=Date.now();site.earnings=(site.earnings||0)+0.05;await put('sites',site);
      const owner=await get('users',site.owner);if(owner){owner.balance=Math.round((owner.balance+.05)*100)/100;await put('users',owner);await put('transactions',{id:uid(),from:'system',to:owner.id,amount:.05,description:'Visitor revenue · '+site.domain,created:Date.now()})}
      await put('visits',{id:uid(),domain:site.domain,userId,created:Date.now()});return site;
    },
    async api(siteId,userId){
      const user=await get('users',userId)||await get('users','human');
      return {
        me:async()=>({id:user.id,name:user.name,balance:user.balance,kind:user.kind}),
        balance:async()=>user.balance,
        pay:async({to,amount,description})=>{if(userId==='human'&&!window.confirm('Pay '+amount+' CR to '+to+' for '+(description||'purchase')+'?'))throw Error('Payment cancelled.');return this.transfer(userId,to,amount,description)},
        search:async(q)=>{const term=String(q||'').toLowerCase();return (await all('sites')).filter(s=>(s.name+' '+s.domain+' '+s.description).toLowerCase().includes(term)).map(s=>({name:s.name,url:s.domain,description:s.description}))},
        posts:{list:async()=> (await all('posts')).filter(p=>p.domain===siteId).sort((a,b)=>b.created-a.created),create:async(body)=>{const p={id:uid(),domain:siteId,author:user.name,body:String(body||'').slice(0,4000),created:Date.now()};await put('posts',p);return p}},
        notifications:{send:async({title,message})=>{const n={id:uid(),userId,title:String(title||'Notification'),message:String(message||''),created:Date.now()};await put('notifications',n);return n}},
        analytics:{visitors:async()=>{const s=(await all('sites')).find(x=>x.domain===siteId);return s?.visitors||0}},
        users:{list:async()=> (await all('users')).map(u=>({name:u.name,kind:u.kind,bio:u.bio}))}
      };
    }
  };
  window.InternetEngine=InternetEngine;
  window.addEventListener('message',async e=>{
    const d=e.data;if(!d||d.source!=='local-internet-site'||!d.id)return;
    const frames=[...document.querySelectorAll('iframe[data-internet-frame]')];
    const frame=frames.find(f=>f.contentWindow===e.source);if(!frame)return;
    try{
      const api=await InternetEngine.api(frame.dataset.domain,frame.dataset.user||'human');
      let value;
      const path=d.method.split('.');
      if(path[0]==='me')value=await api.me();
      else if(path[0]==='wallet'&&path[1]==='balance')value=await api.balance();
      else if(path[0]==='wallet'&&path[1]==='pay')value=await api.pay(d.args[0]||{});
      else if(path[0]==='search')value=await api.search(d.args[0]);
      else if(path[0]==='posts'&&path[1]==='list')value=await api.posts.list();
      else if(path[0]==='posts'&&path[1]==='create')value=await api.posts.create(d.args[0]);
      else if(path[0]==='notifications'&&path[1]==='send')value=await api.notifications.send(d.args[0]||{});
      else if(path[0]==='analytics'&&path[1]==='visitors')value=await api.analytics.visitors();
      else if(path[0]==='users'&&path[1]==='list')value=await api.users.list();
      else throw Error('Unknown Internet API method: '+d.method);
      e.source.postMessage({source:'local-internet-response',id:d.id,ok:true,value},'*');
    }catch(err){e.source.postMessage({source:'local-internet-response',id:d.id,ok:false,error:err.message},'*')}
  });
})();