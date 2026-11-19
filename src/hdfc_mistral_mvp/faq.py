"""Small read-only FAQ pipeline: approved extraction -> vectors -> cited quote."""
import hashlib
import json
import os
import re
import unicodedata
import textwrap
from pathlib import Path
from .sdk import ROOT, DemoError

FALLBACK = 'I couldn’t find that in the approved documents.'
SUPPORTED = 'How can I download electronic statements?'
UNSUPPORTED = 'What is the home loan interest rate?'

def screen(text):
    normalized = ''.join(c for c in unicodedata.normalize('NFKC', text) if unicodedata.category(c) != 'Cf')
    compact = re.sub(r'[\s-]', '', normalized)
    if any(ord(c)<32 and c not in '\n\t\r' for c in text) or len(text)>16000 or re.search(r'\d{6,}|[\w.+-]+@[\w.-]+|[A-Z]{5}\d{4}[A-Z]', compact, re.I) or re.search(r'(?i)\b(password|otp|cvv|pin|api[_ -]?key|my balance|my account|my card)\b', normalized):
        raise DemoError('Sensitive or oversized input rejected. Use public synthetic policy questions only.')
    return text

def approved(document, manifest_path=None):
    for item in json.loads((Path(manifest_path) if manifest_path else ROOT/'demo/approved.json').read_text()):
        if item['document']==document:
            pdf=ROOT/item['file']
            if not pdf.is_file(): raise DemoError('Approved synthetic PDF is missing; rebuild demo documents.')
            if hashlib.sha256(pdf.read_bytes()).hexdigest()!=item['sha256']:
                raise DemoError('Synthetic PDF approval hash mismatch.')
            return item,pdf
    raise DemoError('Document is not in the approved synthetic PDF manifest.')

def chunks(data, manifest_path=None):
    item,_=approved(data['document'], manifest_path)
    if data['sha256']!=item['sha256'] or data['title']!=item['title']:
        raise DemoError('Extraction does not match approved document metadata.')
    result=[]; seen=set()
    for page in data['pages']:
        number=page['page']; text=screen(page['text']).strip()
        if type(number)!=int or number<1 or number in seen or not text: raise DemoError('Invalid extraction pages.')
        seen.add(number)
        # Keep short, complete sentences together as separate passages. This
        # improves retrieval granularity while retaining page-level citations.
        sentences = re.split(r'(?<=[.!?])\s+', text)
        parts=[]
        for sentence in sentences:
            if len(sentence) <= 1000:
                parts.append(sentence)
            else:
                parts.extend(textwrap.wrap(sentence, width=1000, replace_whitespace=False, drop_whitespace=True))
        for i,part in enumerate(parts):
            part=part.strip()
            if not part: continue
            cid=f"{data['document']}-p{number}-s{i}-"+hashlib.sha256(part.encode()).hexdigest()[:12]
            result.append({'id':cid,'text':part,'title':item['title'],'page':number,'document':item['document'],'sha256':item['sha256']})
    if not result or len(result)>100: raise DemoError('Empty or oversized extraction.')
    return result

def fixtures():
    return [c for p in sorted((ROOT/'demo/extractions').glob('*.json')) for c in chunks(json.loads(p.read_text()))]

class ChromaStore:
    def __init__(self, model, *, create=True, collection_name=None):
        import chromadb
        from chromadb.config import Settings
        self.client=chromadb.HttpClient(host=os.getenv('CHROMA_HOST','127.0.0.1'),port=int(os.getenv('CHROMA_PORT','8015')), settings=Settings(anonymized_telemetry=False))
        name = collection_name or os.getenv('FAQ_COLLECTION', 'approved-demo-faq')
        if create:
            self.collection=self.client.get_or_create_collection(name,embedding_function=None,metadata={'embedding_model':model,'hnsw:space':'cosine'})
        else:
            self.collection=self.client.get_collection(name,embedding_function=None)
        if self.collection.metadata.get('embedding_model')!=model:
            raise DemoError('Collection uses another embedding model. Restore that model or reset the demo volume explicitly.')
    def ingest(self, rows, vectors):
        self.collection.upsert(ids=[r['id'] for r in rows],documents=[r['text'] for r in rows],embeddings=vectors,metadatas=[{k:v for k,v in r.items() if k not in ('id','text')} for r in rows])
        # Remove superseded chunks only after new chunks are stored.
        for doc in {r['document'] for r in rows}:
            wanted={r['id'] for r in rows if r['document']==doc}
            stale=set(self.collection.get(where={'document':doc})['ids'])-wanted
            if stale: self.collection.delete(ids=sorted(stale))
        return self.collection.count()
    def retrieve(self, vector):
        count=self.collection.count()
        if not count: return []
        result=self.collection.query(query_embeddings=[vector],n_results=min(3,count),include=['documents','metadatas','distances'])
        return [dict(m,id=i,text=t,distance=d) for i,t,m,d in zip(result['ids'][0],result['documents'][0],result['metadatas'][0],result['distances'][0])]

def answer_result(question, ai, store):
    if not question.strip() or len(question)>2000: raise DemoError('Enter a public policy question of at most 2,000 characters.')
    screen(question)
    missing = {'status': 'not_found', 'message': FALLBACK, 'citations': []}
    # A generic attribute/pronoun alone identifies no service. Do not let retrieved
    # context invent the missing subject in this stateless English FAQ endpoint.
    generic = set('what is the a an how long does do it take takes fee fees cost costs price time rate this that they are much when where can i you please tell me about would will'.split())
    words = set(re.findall(r"[a-z]+", question.casefold()))
    if words and not (words - generic):
        return missing
    passages=store.retrieve(ai.embed([question])[0])
    passages=[p for p in passages if p['distance']<=0.55]
    for p in passages: screen(p['text']); screen(p['title'])
    if not passages: return missing
    raw=ai.generate(question,[{'chunk_id':p['id'],'text':p['text']} for p in passages])
    try:
        value=json.loads(raw)
        if not isinstance(value,dict) or set(value)!={'chunk_id','quote'} or not all(isinstance(v,str) for v in value.values()): return missing
        quote=screen(value['quote']).strip()
        match=next((p for p in passages if p['id']==value['chunk_id']),None)
        if not match or len(quote)<20 or len(quote)>1400 or quote not in match['text']: return missing
        return {'status': 'answered', 'answer': quote, 'citations': [{'title': match['title'], 'page': match['page'], 'chunk_id': match['id']}]}
    except (ValueError,TypeError,DemoError):
        return missing


def answer(question, ai, store):
    """Compatibility renderer for the retained CLI utilities."""
    result = answer_result(question, ai, store)
    if result['status'] == 'not_found':
        return result['message']
    citation = result['citations'][0]
    return f"{result['answer']}\nSource: {citation['title']}, page {citation['page']} [{citation['chunk_id']}]"
