"""Deterministic test doubles. Never presented as actual model inference."""
import hashlib
import json
import math
import re

class FixtureAI:
    embedding_model='fixture-hash-v1'
    def embed(self,texts):
        vectors=[]
        for text in texts:
            v=[0.0]*256
            for word in re.findall('[a-z]+',text.lower()):
                if word in {'the','a','in','is','can','how','i','what','of','to','from'}: continue
                v[int(hashlib.sha256(word.encode()).hexdigest()[:8],16)%256]+=1
            norm=math.sqrt(sum(x*x for x in v)) or 1
            vectors.append([x/norm for x in v])
        return vectors
    def generate(self,question,passages):
        if 'riverside' in question.lower() and 'saturdays' in question.lower():
            p=next((p for p in passages if 'Riverside is open on Saturdays' in p['text']),None)
            if p: return json.dumps({'chunk_id':p['chunk_id'],'quote':'Riverside is open on Saturdays from 10 AM to 2 PM under this synthetic demo schedule.'})
        if 'electronic statements' in question.lower():
            p=next((p for p in passages if 'download electronic statements' in p['text']),None)
            if p: return json.dumps({'chunk_id':p['chunk_id'],'quote':'Customers can download electronic statements from the secure banking document centre.'})
        return json.dumps({'chunk_id':'','quote':''})

class MemoryStore:
    def __init__(self,rows,ai):
        self.rows=rows; self.vectors=ai.embed([r['text'] for r in rows])
    def retrieve(self,vector):
        return sorted([dict(r,distance=1-sum(a*b for a,b in zip(vector,v))) for r,v in zip(self.rows,self.vectors)],key=lambda r:r['distance'])[:3]
