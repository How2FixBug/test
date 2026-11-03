(() => {
  const $ = (s) => document.querySelector(s);
  const E = window.InternetEngine;
  if (!E) return;

  const nav = document.createElement('button');
  nav.className = 'side-link';
  nav.dataset.view = 'activity';
  nav.innerHTML = '<span>◷</span> Activity';
  $('.sidebar-spacer')?.before(nav);

  const escape = E.esc;
  const money = n => Number(n || 0).toLocaleString(undefined,{minimumFractionDigits:2,maximumFractionDigits:2});
  const date = n => new Date(n).toLocaleString();

  async function renderActivity() {
    document.querySelectorAll('.side-link').forEach(b => b.classList.toggle('active',b===nav));
    const [sites,users,visits,transactions,posts] = await Promise.all([
      E.all('sites'),E.all('users'),E.all('visits'),E.all('transactions'),E.all('posts')
    ]);
    const recentVisits = [...visits].sort((a,b)=>b.created-a.created).slice(0,30);
    const recentTransactions = [...transactions].sort((a,b)=>b.created-a.created).slice(0,20);
    const recentPosts = [...posts].sort((a,b)=>b.created-a.created).slice(0,12);
    const topSites = [...sites].sort((a,b)=>(b.visitors||0)-(a.visitors||0)).slice(0,6);
    const totalVisits = visits.length;
    const human = users.find(u=>u.id==='human');
    $('#main').innerHTML = `
      <div class="activity-head"><div><div class="eyebrow">LIVE WORLD LEDGER</div><h1 class="page-title">Network activity</h1><p class="page-subtitle">A running record of what people are doing across your local internet.</p></div><button class="secondary" id="activity-refresh">↻ Refresh</button></div>
      <div class="activity-stats">
        <div class="activity-stat"><span>Total visits recorded</span><strong>${totalVisits.toLocaleString()}</strong></div>
        <div class="activity-stat"><span>Websites</span><strong>${sites.length}</strong></div>
        <div class="activity-stat"><span>Residents</span><strong>${users.length}</strong></div>
        <div class="activity-stat"><span>Your balance</span><strong>${money(human?.balance)} <small>CR</small></strong></div>
      </div>
      <div class="activity-columns">
        <section class="activity-panel"><div class="activity-panel-head"><div><div class="eyebrow">MOST VISITED</div><h2>Popular websites</h2></div></div>
          <div class="traffic-list">${topSites.map((s,i)=>`<div class="traffic-row"><span class="traffic-rank">${i+1}</span><div class="traffic-info"><strong>${escape(s.name)}</strong><small>${escape(s.domain)}</small><div class="traffic-track"><i style="width:${Math.max(2,100*(s.visitors||0)/Math.max(1,topSites[0]?.visitors||1))}%"></i></div></div><b>${(s.visitors||0).toLocaleString()}</b></div>`).join('')||'<div class="empty">No traffic yet. Visit a website to get things moving.</div>'}</div>
        </section>
        <section class="activity-panel"><div class="activity-panel-head"><div><div class="eyebrow">RECENT VISITS</div><h2>Browsing history</h2></div></div>
          <div class="activity-feed">${recentVisits.map(v=>{const site=sites.find(s=>s.domain===v.domain);const user=users.find(u=>u.id===v.userId);return `<div class="activity-item"><span class="activity-dot visit"></span><div><strong>${escape(user?.name||'Unknown visitor')} visited ${escape(site?.name||v.domain)}</strong><small>${escape(v.domain)} · ${date(v.created)}</small></div></div>`}).join('')||'<div class="empty">No visits have been recorded.</div>'}</div>
        </section>
      </div>
      <div class="activity-columns">
        <section class="activity-panel"><div class="activity-panel-head"><div><div class="eyebrow">THE ECONOMY</div><h2>Recent transactions</h2></div></div>
          <div class="activity-feed">${recentTransactions.map(t=>{const from=users.find(u=>u.id===t.from),to=users.find(u=>u.id===t.to);return `<div class="activity-item"><span class="activity-dot cash"></span><div><strong>${escape(t.description||'Transaction')}</strong><small>${escape(from?.name||t.from)} → ${escape(to?.name||t.to)} · ${date(t.created)}</small></div><b>${money(t.amount)} CR</b></div>`}).join('')||'<div class="empty">The economy is quiet. For now.</div>'}</div>
        </section>
        <section class="activity-panel"><div class="activity-panel-head"><div><div class="eyebrow">COMMUNITY</div><h2>Town Square posts</h2></div></div>
          <div class="activity-feed">${recentPosts.map(p=>`<div class="activity-item post-item"><span class="activity-dot post"></span><div><strong>${escape(p.author)}</strong><p>${escape(p.body)}</p><small>${date(p.created)}</small></div></div>`).join('')||'<div class="empty">Nobody has posted yet.</div>'}</div>
        </section>
      </div>`;
    $('#activity-refresh').onclick=renderActivity;
  }

  nav.addEventListener('click',renderActivity);
  document.addEventListener('click',e=>{
    if(e.target.closest('.side-link') && e.target.closest('.side-link')!==nav) nav.classList.remove('active');
  });
})();