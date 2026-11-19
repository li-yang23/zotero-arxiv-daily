export type Paper = {title:string,url:string,summary?:string,group?:string};
export type Digest = {id:string,title:string,papers:Paper[],created:number};
export type Job = {id:string,title:string,url:string,prompt:string,status:string,result?:string,error?:string,model?:string,created:number,updated?:number};
export type Library = {digests:Digest[],jobs:Job[],prompt:string,repository:string,updated:number};
export type Run = {status:string,conclusion:string|null,created_at:string,html_url:string};

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
