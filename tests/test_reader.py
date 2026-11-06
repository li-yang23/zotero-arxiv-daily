import json
from types import SimpleNamespace
from unittest.mock import Mock

import httpx
import pytest
from omegaconf import OmegaConf

from zotero_arxiv_daily.reader import ReaderClient, attach_reader
from zotero_arxiv_daily.reader_main import analyze, validate_result, background_sources


def test_upload_failure_does_not_prevent_email(monkeypatch):
    monkeypatch.setenv('PAPER_READER_URL','https://reader.example')
    monkeypatch.setenv('PAPER_READER_TOKEN','secret')
    monkeypatch.setattr(ReaderClient,'post',Mock(side_effect=RuntimeError('network')))
    html='<html><body>daily</body></html>'
    assert attach_reader(html,[{'title':'x','url':'https://arxiv.org/abs/1234.12345'}])==html


def test_upload_links_only_after_success(monkeypatch):
    monkeypatch.setenv('PAPER_READER_URL','https://reader.example')
    monkeypatch.setenv('PAPER_READER_TOKEN','secret')
    post=Mock(return_value={'ok':True}); monkeypatch.setattr(ReaderClient,'post',post)
    result=attach_reader('<body>papers</body>',[{'title':'x','url':'https://arxiv.org/abs/1234.12345'}])
    assert '选择论文并精读' in result
    assert 'secret' not in result
    assert post.call_args.args[1]['id'] in result


def test_transport_does_not_follow_auth_redirects(monkeypatch):
    monkeypatch.setenv('PAPER_READER_URL','https://reader.example')
    monkeypatch.setenv('PAPER_READER_TOKEN','secret')
    calls=[]
    def handle(request):
        calls.append(request)
        return httpx.Response(302,headers={'Location':'https://elsewhere.example'})
    client=ReaderClient(httpx.Client(transport=httpx.MockTransport(handle)))
    with pytest.raises(httpx.HTTPStatusError): client.post('claim',{})
    assert len(calls)==1


def test_full_text_is_required():
    resolver=Mock();resolver.resolve.return_value=SimpleNamespace(paper=SimpleNamespace(full_text=None))
    llm=Mock()
    with pytest.raises(ValueError,match='全文'):
        analyze({'title':'x','url':'https://arxiv.org/abs/x','prompt':'read'},resolver,llm,None,None)
    llm.chat.completions.create.assert_not_called()


def test_background_failure_is_explicit():
    http=Mock(); http.get.side_effect=httpx.ConnectError('offline')
    result=background_sources(SimpleNamespace(url='https://arxiv.org/abs/2601.12345'),http)
    assert result['status']=='unavailable' and not result['sources']


def test_validate_report_and_reject_invalid_graph():
    valid={'sections':[{'title':'Method','blocks':[{'type':'text','text':'grounded claim'}, {'type':'flow','nodes':['input','output'],'caption':'illustration'}]}]}
    assert json.loads(validate_result(json.dumps(valid)))==valid
    valid['sections'][0]['blocks'].append({'type':'chart','kind':'line','points':[{'x':0,'y':float('nan')},{'x':1,'y':2}],'caption':'bad'})
    with pytest.raises(ValueError): validate_result(json.dumps(valid))


def test_model_fallback_keeps_prompt_and_sources(monkeypatch):
    import zotero_arxiv_daily.reader_main as module
    monkeypatch.setattr(module,'truncate_text_by_tokens',lambda text,n:text)
    monkeypatch.setattr(module,'background_sources',lambda p,c:{'status':'retrieved','sources':[{'title':'Prior','url':'https://doi.org/example'}]})
    paper=SimpleNamespace(full_text='full paper '*100,title='Paper',url='https://arxiv.org/abs/2601.12345')
    resolver=Mock();resolver.resolve.return_value=SimpleNamespace(paper=paper)
    valid=json.dumps({'sections':[{'title':'Method','blocks':[{'type':'text','text':'analysis'}]}]})
    llm=Mock();llm.chat.completions.create.side_effect=[RuntimeError('provider'),[SimpleNamespace(choices=[SimpleNamespace(finish_reason='stop',delta=SimpleNamespace(content=valid))])]]
    config=OmegaConf.create({'llm':{'generation_kwargs':{'model':['bad','good']}}})
    result,model=analyze({'title':'Paper','url':paper.url,'prompt':'my exact prompt'},resolver,llm,config,None)
    assert model=='good' and json.loads(result)['sources']['status']=='retrieved'
    assert 'my exact prompt' in llm.chat.completions.create.call_args.kwargs['messages'][1]['content']


def test_reader_rejects_private_redirect(monkeypatch):
    from zotero_arxiv_daily.reader_main import PublicPaperClient
    monkeypatch.setattr('zotero_arxiv_daily.reader_main.socket.getaddrinfo',lambda host,*a,**k:[(None,None,None,None,('127.0.0.1' if host=='internal.example' else '8.8.8.8',443))])
    seen=[]
    def handle(req):
        seen.append(str(req.url))
        return httpx.Response(302,headers={'Location':'https://internal.example/private'})
    with PublicPaperClient(transport=httpx.MockTransport(handle)) as client:
        with pytest.raises(ValueError,match='公网'): client.get('https://arxiv.org/pdf/test')
    assert seen==['https://arxiv.org/pdf/test']


def test_exact_arxiv_reader_never_calls_export_api():
    from zotero_arxiv_daily.reader_main import ReadingPaperResolver
    from zotero_arxiv_daily.request_models import PaperRequest
    resolver=ReadingPaperResolver(OmegaConf.create({}))
    paper=SimpleNamespace(full_text='body')
    resolver._paper_from_web_page=Mock(return_value=paper)
    resolver._lookup_arxiv_id=Mock(side_effect=AssertionError('must not call export API'))
    try:
        result=resolver.resolve(PaperRequest('Title','https://arxiv.org/abs/2610.07212',0))
        assert result.paper is paper and result.status=='matched'
        resolver._lookup_arxiv_id.assert_not_called()
    finally: resolver.close()


def test_exact_arxiv_reader_falls_back_to_official_pdf():
    from zotero_arxiv_daily.reader_main import ReadingPaperResolver
    from zotero_arxiv_daily.request_models import PaperRequest
    resolver=ReadingPaperResolver(OmegaConf.create({}))
    resolver._paper_from_web_page=Mock(side_effect=httpx.TimeoutException('timeout'))
    resolver._download_pdf_text=Mock(return_value='full document')
    try:
        result=resolver.resolve(PaperRequest('Title','https://arxiv.org/abs/2610.07212',0))
        assert result.paper.full_text=='full document'
        resolver._download_pdf_text.assert_called_once_with('https://arxiv.org/pdf/2610.07212')
    finally: resolver.close()
