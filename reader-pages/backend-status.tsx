import {useEffect,useState} from 'react';
import {workflowNotice,readerWorkflowNotice,unreceivedIssueNotice,type Run,type Job,type ReaderIssue} from './reader-status';

// This check runs in the browser, so it still works when Actions cannot start.
export default function BackendStatus({repository,jobs}:{repository:string,jobs:Job[]}) {
  const [warnings,setWarnings] = useState<string[]>([]);
  const [readerRuns,setReaderRuns] = useState<Run[]>([]);
  const [issues,setIssues] = useState<ReaderIssue[]>([]);
  const [checked,setChecked] = useState(false);
  useEffect(()=>{
    if (!repository) return;
    let active = true;
    const check = async()=>{
      const get = async(path:string)=>{
        const response = await fetch(`https://api.github.com/repos/${repository}${path}`,
          {signal:AbortSignal.timeout(15000),headers:{Accept:'application/vnd.github+json'}});
        if (!response.ok) throw Error('无法核实后台状态');
        return response.json();
      };
      const results = await Promise.allSettled([
        get('/actions/workflows/main.yml/runs?per_page=10'),
        get('/actions/workflows/paper-reader.yml/runs?per_page=10'),
        get(`/issues?state=open&creator=${encodeURIComponent(repository.split('/')[0])}&per_page=100`),
      ]);
      if (!active) return;
      const messages:string[] = [];
      if (results[0].status==='fulfilled') {
        const warning = workflowNotice(results[0].value.workflow_runs,'日报任务',36);
        if (warning) messages.push(warning);
      }
      if (results[1].status==='fulfilled') {setReaderRuns(results[1].value.workflow_runs);setChecked(true)}
      if (results[2].status==='fulfilled') setIssues(results[2].value);
      if (results.some(r=>r.status==='rejected')) messages.push('暂时无法核实后台运行状态，当前显示的阅读记录仍可查看。');
      setWarnings(messages);
    };
    void check();
    const timer = setInterval(check,300000);
    return ()=>{active=false;clearInterval(timer)};
  },[repository]);
  const messages = [...warnings,checked?readerWorkflowNotice(readerRuns,jobs):'',unreceivedIssueNotice(issues,jobs,repository)].filter(Boolean);
  if (!messages.length) return null;
  return <div role="alert" className="alert"><div>{messages.map(w=><p key={w}>{w}</p>)}</div>
    <a href={`https://github.com/${repository}/actions`} target="_blank" rel="noreferrer">检查后台任务 ↗</a></div>;
}
