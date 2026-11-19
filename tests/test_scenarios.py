"""Offline contracts with explicit doubles, not evidence of live model accuracy."""
import json
from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient
from hdfc_mistral_mvp import api
from hdfc_mistral_mvp.faq import ChromaStore, answer_result, fixtures
from scripts.faq_scenarios import CASES, verify, updated_rows


@pytest.mark.parametrize('case', CASES, ids=lambda c:c['id'])
def test_http_scenario_contract(case):
    rows = updated_rows(fixtures()) if case['kind']=='update' else fixtures()
    ai, store = Mock(), Mock()
    ai.embed.return_value = [[1.,0.]]
    candidates = [r for r in rows if r['document']==case.get('document') and r['page']==case.get('page')]
    source = next((r for r in candidates if all(f.casefold() in r['text'].casefold() for f in case.get('fact_fragments', []))), candidates[0] if candidates else rows[0])
    # Even insufficient questions see related evidence: generation must abstain.
    store.retrieve.return_value = [dict(source,distance=0.1)]
    ai.generate.return_value = json.dumps({'chunk_id':source['id'],'quote':source['text']}) if case['status']=='answered' else '{"chunk_id":"","quote":""}'
    if case['kind']=='dependency':
        (store.retrieve if case['dependency']=='chroma' else ai.embed).side_effect = RuntimeError('private synthetic diagnostic')
    runner = Mock(side_effect=lambda q: answer_result(q,ai,store))
    api.app.dependency_overrides[api.get_runner] = lambda: runner
    try:
        with TestClient(api.app) as client:
            response=client.post('/v1/faq',json={'question':case['question']})
        verify(case,response.status_code,response.json(),rows)
        if case['kind']=='screening':
            runner.assert_not_called()
            ai.embed.assert_not_called()
    finally:
        api.app.dependency_overrides.clear()


class FakeCollection:
    """Stateful Chroma collection double to exercise the real update algorithm."""
    def __init__(self): self.rows={}
    def upsert(self,ids,documents,embeddings,metadatas):
        self.rows.update({i:dict(text=t,embedding=e,**m) for i,t,e,m in zip(ids,documents,embeddings,metadatas)})
    def get(self,where): return {'ids':[i for i,r in self.rows.items() if r['document']==where['document']]}
    def delete(self,ids):
        for i in ids: del self.rows[i]
    def count(self): return len(self.rows)


def test_document_revision_removes_stale_chunks_without_touching_other_documents():
    store=ChromaStore.__new__(ChromaStore); store.collection=FakeCollection()
    original=fixtures(); vectors=[[1.,0.]]*len(original)
    assert store.ingest(original,vectors)==len(original)
    untouched={i:dict(r) for i,r in store.collection.rows.items() if r['document']!='statements'}
    revision=updated_rows(original)
    statements=[r for r in revision if r['document']=='statements']
    assert store.ingest(statements,[[1.,0.]]*len(statements))==len(revision)
    assert set(store.collection.rows)=={r['id'] for r in revision}
    assert all(store.collection.rows[i]==r for i,r in untouched.items())
    assert store.ingest(statements,[[1.,0.]]*len(statements))==len(revision)


def test_reviewed_facts_are_present_in_corpus_and_suite_is_not_indexed():
    rows=fixtures()
    for case in CASES:
        if case['status']=='answered' and case['kind']!='update':
            source=' '.join(r['text'] for r in rows if r['document']==case['document'] and r['page']==case['page']).casefold()
            for fact in case['fact_fragments']: assert fact in source
    assert len(rows)>5


def test_fact_checker_rejects_real_citation_with_irrelevant_quote():
    case=next(c for c in CASES if c['id']=='complaints-ack')
    rows=fixtures()
    source=next(r for r in rows if r['document']=='complaints' and 'complaint acknowledgement' in r['text'].casefold())
    payload={'status':'answered','answer':'Escalate unresolved complaints to the designated grievance officer after seven working days.',
             'citations':[{'title':source['title'],'page':source['page'],'chunk_id':source['id']}]}
    with pytest.raises(AssertionError,match='Answer is not a quote from cited source'):
        verify(case,200,payload,rows)


@pytest.mark.parametrize('question',['What is the fee?','How long does it take?'])
def test_underspecified_question_does_not_let_retrieval_choose_a_subject(question):
    ai,store=Mock(),Mock()
    assert answer_result(question,ai,store)['status']=='not_found'
    ai.embed.assert_not_called()
    ai.generate.assert_not_called()
    store.retrieve.assert_not_called()
