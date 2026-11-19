import {useEffect,useState} from 'react';
import {workflowNotice, type Run} from './reader-status';

// This check runs in the browser, so it still works when Actions cannot start.
export default function BackendStatus({repository}:{repository:string}) {
  const [warnings,setWarnings] = useState<string[]>([]);
  useEffect(()=>{
    if (!repository) return;
    let active = true;
    const check = async()=>{
      const results = await Promise.allSettled([
        ['main.yml','日报任务',36], ['paper-reader.yml','精读任务',3],
      ].map(async([workflow,label,hours])=>{
        const response = await fetch(`https://api.github.com/repos/${repository}/actions/workflows/${workflow}/runs?per_page=10`,
          {signal:AbortSignal.timeout(15000),headers:{Accept:'application/vnd.github+json'}});
        if (!response.ok) throw Error('无法核实后台状态');
        const data:{workflow_runs:Run[]} = await response.json();
        return workflowNotice(data.workflow_runs,String(label),Number(hours));
      }));
      if (!active) return;
      const messages = results.flatMap(r=>r.status==='fulfilled' && r.value ? [r.value] : []);
      if (results.some(r=>r.status==='rejected')) messages.push('暂时无法核实后台运行状态，当前显示的阅读记录仍可查看。');
      setWarnings(messages);
    };
    void check();
    const timer = setInterval(check,300000);
    return ()=>{active=false;clearInterval(timer)};
  },[repository]);
  if (!warnings.length) return null;
  return <div role="alert" className="alert"><div>{warnings.map(w=><p key={w}>{w}</p>)}</div>
    <a href={`https://github.com/${repository}/actions`} target="_blank" rel="noreferrer">检查后台任务 ↗</a></div>;
}
