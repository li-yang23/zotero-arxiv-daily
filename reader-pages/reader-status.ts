export type Paper = {title:string,url:string,summary?:string,group?:string};
export type Digest = {id:string,title:string,papers:Paper[],created:number};
export type Job = {id:string,title:string,url:string,prompt:string,status:string,result?:string,error?:string,model?:string,created:number,updated?:number};
export type Library = {digests:Digest[],jobs:Job[],prompt:string,repository:string,updated:number};
export type Run = {status:string,conclusion:string|null,created_at:string,updated_at?:string,html_url:string};
export type ReaderIssue = {title:string,body:string|null,state:string,user:{login:string},created_at:string,updated_at?:string};

export function publicLiveLibrary(snapshot: Library, raw: unknown): Library {
  const state = raw as Partial<Library> & {version?:number};
  if (!state || state.version !== 1 || !Array.isArray(state.jobs) ||
      !Array.isArray(state.digests) || typeof state.prompt !== 'string' ||
      typeof state.updated !== 'number') throw Error('后台文库格式无效');
  if (state.updated < snapshot.updated) return snapshot;
  return {
    repository: snapshot.repository, prompt: state.prompt, updated: state.updated,
    digests: [...state.digests].sort((a,b)=>b.created-a.created),
    jobs: [...state.jobs].sort((a,b)=>b.created-a.created).map(j=>({
      id:j.id,title:j.title,url:j.url,prompt:j.prompt,status:j.status,
      result:j.result,error:j.error,model:j.model,created:j.created,updated:j.updated,
    })),
  };
}

export function workflowNotice(runs: Run[], label: string, maxHours: number, now = Date.now()): string {
  const latest = runs.find(r=>r.conclusion !== 'skipped');
  if (!latest) return `${label}还没有运行记录，请检查 Actions。`;
  const age = now - Date.parse(latest.created_at);
  if (!Number.isFinite(age)) return `${label}的运行时间无法核实。`;
  if (age > maxHours * 3600000) return `${label}已超过 ${maxHours} 小时没有启动，后台可能已停跑。`;
  if (latest.status !== 'completed') return '';
  if (latest.conclusion !== 'success') return `${label}最近一次未成功，请查看运行日志。`;
  return '';
}

export function readerWorkflowNotice(runs: Run[], jobs: Job[], now = Date.now()): string {
  const pending = jobs.filter(j=>j.status==='queued'||j.status==='running');
  if (!pending.length) return ''; // Event-driven readers are normally idle.
  if (pending.some(j=>j.status==='running' && now-(j.updated||j.created)>40*60000))
    return '有论文阅读超过 40 分钟没有更新，请检查精读任务和自动续接日志。';
  const latest = runs.find(r=>r.conclusion!=='skipped');
  if (!latest) return '有待处理论文，但还没有精读运行记录。';
  if (latest.status==='completed') {
    if (latest.conclusion!=='success') return '精读任务最近一次未成功，请检查运行日志；异常中断最多自动重试三次。';
    if (now-Date.parse(latest.updated_at||latest.created_at)>15*60000)
      return '仍有论文排队，但上一轮结束后超过 15 分钟未启动下一轮，请检查自动续接日志。';
  } else if (now-Date.parse(latest.created_at)>45*60000) {
    return '精读任务启动后超过 45 分钟未结束，请检查运行日志。';
  }
  return '';
}

export function unreceivedIssueNotice(issues: ReaderIssue[], jobs: Job[], repository: string, now = Date.now()): string {
  const owner = repository.split('/')[0].toLowerCase();
  const imported = new Set(jobs.map(j=>j.id));
  let missing = 0;
  for (const issue of issues) {
    if (issue.state!=='open'||issue.user.login.toLowerCase()!==owner||!issue.title.startsWith('[Paper Reader]')) continue;
    if (now-Date.parse(issue.updated_at||issue.created_at)<=30*60000) continue;
    try {
      const body = issue.body||'';
      const request = JSON.parse(body.match(/```json\s*\n([\s\S]*?)\n```/)?.[1]||body);
      if (request.version!==1||typeof request.request_id!=='string'||
          !/^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i.test(request.request_id)||
          !Array.isArray(request.papers)||!request.papers.length||request.papers.length>10) continue;
      const id = request.request_id.toLowerCase();
      if (request.papers.some((_:unknown,index:number)=>!imported.has(`${id}:${index}`))) missing++;
    } catch { /* Invalid Issue text is not evidence of a stopped worker. */ }
  }
  return missing ? `有 ${missing} 个精读请求提交后超过 30 分钟未被后台接收，请检查 Actions 是否可运行。` : '';
}
