import test from 'node:test';
import assert from 'node:assert/strict';
import {publicLiveLibrary,workflowNotice,readerWorkflowNotice,unreceivedIssueNotice} from './reader-status.ts';

const snapshot={repository:'owner/repo',prompt:'read',updated:1,digests:[],jobs:[]};
test('live queue appears before Pages deploys and internal fields are excluded',()=>{
  const live=publicLiveLibrary(snapshot,{version:1,prompt:'read',updated:2,digests:[],
    jobs:[{id:'job',created:2,title:'Paper',url:'https://example.org',prompt:'read',status:'running',lease:'secret',draft:'draft'}]});
  assert.equal(live.jobs[0].status,'running');
  assert.equal('lease' in live.jobs[0],false);
  assert.equal('draft' in live.jobs[0],false);
  assert.equal(live.repository,snapshot.repository);
});
test('stale CDN state cannot overwrite a newer published snapshot',()=>{
  const newer={...snapshot,updated:5};
  assert.equal(publicLiveLibrary(newer,{...snapshot,version:1}),newer);
});
test('malformed state is rejected so the caller can retain the snapshot',()=>{
  assert.throws(()=>publicLiveLibrary(snapshot,{jobs:[]}));
});
const now=Date.parse('2026-11-19T04:00:00Z');
const run={status:'completed',conclusion:'success',created_at:'2026-11-19T03:00:00Z',html_url:'https://github.com/owner/repo/actions/runs/1'};
test('recent healthy and actively running workflows do not warn',()=>{
  assert.equal(workflowNotice([run],'精读任务',3,now),'');
  assert.equal(workflowNotice([{...run,status:'in_progress',conclusion:null}],'精读任务',3,now),'');
});
test('stalled schedules, stuck runs, failures and missing runs warn',()=>{
  for(const status of ['completed','queued','in_progress']) assert.match(workflowNotice([{...run,status,created_at:'2026-11-18T00:00:00Z'}],'精读任务',3,now),/停跑/);
  assert.match(workflowNotice([{...run,conclusion:'failure'}],'精读任务',3,now),/未成功/);
  assert.match(workflowNotice([],'精读任务',3,now),/没有运行记录/);
});

test('unrelated skipped Issue events do not hide the latest real reader run',()=>{
  assert.equal(workflowNotice([{...run,conclusion:'skipped'},run],'精读任务',3,now),'');
});

test('an event-driven reader may be idle indefinitely without an outage warning',()=>{
  assert.equal(readerWorkflowNotice([],[{id:'one',status:'completed',created:0}],now),'');
  assert.equal(readerWorkflowNotice([{...run,created_at:'2026-01-01T00:00:00Z'}],[],now),'');
});
test('queued work warns when continuation never started but not during active reading',()=>{
  const jobs=[{id:'one',status:'queued',created:now-3600000}];
  assert.match(readerWorkflowNotice([run],jobs,now),/未启动下一轮/);
  assert.equal(readerWorkflowNotice([{...run,status:'in_progress',created_at:'2026-11-19T03:50:00Z'}],jobs,now),'');
  assert.match(readerWorkflowNotice([run],[{id:'one',status:'running',created:now-3600000,updated:now-3600000}],now),/没有更新/);
});
const requestId='3a5bb2fc-81d1-48aa-bbf6-ea75bb3295df';
const issue={title:'[Paper Reader] 1 paper',state:'open',user:{login:'owner'},created_at:'2026-11-19T03:00:00Z',
  body:'```json\n'+JSON.stringify({version:1,request_id:requestId,papers:[0]})+'\n```'};
test('submitted Issues missing from the queue remain detectable when Actions is disabled',()=>{
  assert.match(unreceivedIssueNotice([issue],[],'owner/repo',now),/未被后台接收/);
  assert.equal(unreceivedIssueNotice([issue],[{id:requestId+':0',status:'failed'}],'owner/repo',now),'');
  assert.equal(unreceivedIssueNotice([{...issue,updated_at:'2026-11-19T03:50:00Z'}],[],'owner/repo',now),'');
  assert.equal(unreceivedIssueNotice([{...issue,user:{login:'stranger'}}],[],'owner/repo',now),'');
  assert.equal(unreceivedIssueNotice([{...issue,body:'not JSON'}],[],'owner/repo',now),'');
});
