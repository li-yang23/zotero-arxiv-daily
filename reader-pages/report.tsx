'use client';
import {useState} from 'react';
import Markdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';
import 'katex/dist/katex.min.css';
// Model content is rendered as React text/SVG; never execute model HTML or scripts.
export function reportMarkdown(raw:string){try{const d=JSON.parse(raw);return (d.review_notes?'核对说明：'+d.review_notes+'\n\n':'')+d.sections.map((s:any)=>'## '+s.title+'\n\n'+s.blocks.map((b:any)=>b.type==='text'?b.text:b.type==='flow'?'流程：'+b.nodes.join(' → ')+'\n\n'+b.caption:'图表：'+b.caption+'\n\n'+b.points.map((p:any)=>p.x+', '+p.y).join('\n')).join('\n\n')).join('\n\n')}catch{return raw}}
function Plot({block}:{block:any}){
 const [hover,setHover]=useState<number|null>(null);
 const points=block.points as {x:number,y:number}[];
 const xs=points.map(p=>p.x),ys=points.map(p=>p.y),x0=Math.min(...xs),x1=Math.max(...xs),y0=Math.min(...ys),y1=Math.max(...ys);
 const sx=(v:number)=>65+((v-x0)/(x1-x0||1))*485,sy=(v:number)=>245-((v-y0)/(y1-y0||1))*190;
 return <figure><svg viewBox="0 0 600 300" role="img" aria-label={block.caption}><line x1="65" y1="245" x2="550" y2="245" stroke="#798d7c"/><line x1="65" y1="55" x2="65" y2="245" stroke="#798d7c"/>{Array.from({length:5},(_,i)=>{const x=x0+(x1-x0)*i/4,y=y0+(y1-y0)*i/4;return <g key={i}><text x={65+i*485/4} y="265" textAnchor="middle" fontSize="11" fill="#738272">{Number(x.toPrecision(3))}</text><text x="54" y={249-i*190/4} textAnchor="end" fontSize="11" fill="#738272">{Number(y.toPrecision(3))}</text><line x1="65" x2="550" y1={245-i*190/4} y2={245-i*190/4} stroke="#e0e6db"/></g>})}<text x="310" y="291" fontSize="12" textAnchor="middle" fill="#526b54">{block.xLabel}</text><text x="65" y="27" fontSize="12" fill="#526b54">{block.yLabel}</text>{block.kind==='line'&&<polyline fill="none" stroke="#428760" strokeWidth="2" points={points.map(p=>sx(p.x)+','+sy(p.y)).join(' ')}/>} {points.map((p,i)=><circle key={i} cx={sx(p.x)} cy={sy(p.y)} r={hover===i?6:4} fill="#376c50" tabIndex={0} onFocus={()=>setHover(i)} onBlur={()=>setHover(null)} onMouseEnter={()=>setHover(i)} onMouseLeave={()=>setHover(null)} aria-label={`${p.x}, ${p.y}`}><title>{`${p.x}, ${p.y}`}</title></circle>)}</svg>{hover!==null&&<div className="plot-value">{block.xLabel}: {points[hover].x} · {block.yLabel}: {points[hover].y}</div>}<figcaption>{block.caption}</figcaption></figure>
}
function Flow({block}:{block:any}){
 const heights=block.nodes.map((n:string)=>Math.max(68,Math.ceil([...n].reduce((w,c)=>w+(/[^\x00-\x7F]/.test(c)?1:0.55),0)/29)*20+28));
 let offset=4;
 const positions=heights.map((h:number)=>{const y=offset;offset+=h+26;return y});
 return <figure><svg viewBox={`0 0 600 ${offset}`} role="img" aria-label={block.caption}>{block.nodes.map((node:string,i:number)=><g key={i}><rect x="70" y={positions[i]} width="460" height={heights[i]} rx="8" fill={i===0?'#e0ecdc':'#f7faf3'} stroke="#a3b69d"/><foreignObject x="86" y={positions[i]+10} width="428" height={heights[i]-20}><div style={{fontSize:13,lineHeight:'20px',height:'100%',display:'flex',alignItems:'center',justifyContent:'center',textAlign:'center',overflowWrap:'anywhere',whiteSpace:'pre-wrap'}}>{node}</div></foreignObject>{i<block.nodes.length-1&&<path d={`M300 ${positions[i]+heights[i]+2}v19m-5 -5l5 5l5 -5`} fill="none" stroke="#6b8d69" strokeWidth="2"/>}</g>)}</svg><figcaption>{block.caption}</figcaption></figure>
}
export default function Report({raw}:{raw:string}){
 let d:any;try{d=JSON.parse(raw)}catch{return <div className="text-output">{raw}</div>}
 if(!Array.isArray(d.sections))return <div className="text-output">{raw}</div>;
 return <div className="report">{d.review_notes&&<p className="notice">核对说明：{d.review_notes}</p>}{d.sections.map((s:any,i:number)=><section key={i}><h3>{s.title}</h3>{s.blocks.map((b:any,j:number)=>b.type==='text'?<div className="prose" key={j}><Markdown remarkPlugins={[remarkGfm,remarkMath]} rehypePlugins={[rehypeKatex]} components={{a:({children,href})=><a href={href} target="_blank" rel="noreferrer">{children}</a>,img:()=>null}}>{b.text}</Markdown></div>:b.type==='flow'?<Flow key={j} block={b}/>:b.type==='chart'?<Plot key={j} block={b}/>:null)}</section>)}{d.sources&&<details><summary>背景文献检索依据（{d.sources.sources?.length||0} 篇摘要）</summary><p>检索范围为该论文引用文献中的有限摘要，不代表完整文献综述。{d.sources.status==='unavailable'?'本次外部检索未成功，报告须据此限定背景结论。':''}</p>{d.sources.sources?.map((s:any,i:number)=><p key={i}><a href={s.url} target="_blank" rel="noreferrer">{s.title}</a>（{s.year||'年份未知'}，摘要证据）</p>)}</details>}</div>
}
