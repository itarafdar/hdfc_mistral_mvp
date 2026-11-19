import argparse
import json
import sys
from pathlib import Path
from .faq import ChromaStore, SUPPORTED, UNSUPPORTED, answer, approved, chunks, fixtures
from .sdk import ROOT, DemoError, LocalAI, hosted_ocr

def main():
    parser=argparse.ArgumentParser(description='Synthetic policy FAQ. No customer data or banking operations.')
    sub=parser.add_subparsers(dest='command',required=True)
    demo=sub.add_parser('demo'); demo.add_argument('--offline',action='store_true')
    ingest=sub.add_parser('ingest'); ingest.add_argument('--extraction',type=Path); ingest.add_argument('--manifest',type=Path,help='Reviewed approval manifest; requires --extraction')
    ocr=sub.add_parser('ocr',help='Explicitly send one approved synthetic PDF to hosted OCR 4.1')
    ocr.add_argument('--document',default='statements'); ocr.add_argument('--allow-hosted-ocr',action='store_true',required=True)
    ocr.add_argument('--manifest',type=Path,help='Reviewed approval manifest; default demo/approved.json')
    ask=sub.add_parser('ask'); ask.add_argument('question')
    args=parser.parse_args()
    if args.command=='ingest' and args.manifest and not args.extraction:
        parser.error('--manifest requires --extraction')
    try:
        if args.command=='ocr':
            item,pdf=approved(args.document, args.manifest)
            data=hosted_ocr(pdf,item['title'],item['document'],item['sha256'],args.manifest)
            chunks(data, args.manifest)  # Validate before writing or indexing hosted output.
            out=ROOT/'artifacts'/f'{item["document"]}.json'; out.parent.mkdir(exist_ok=True)
            out.write_text(json.dumps(data,indent=2)+'\n')
            print(f'Hosted OCR 4.1 extracted {len(data["pages"])} pages. Saved {out.relative_to(ROOT)}. Review before ingesting.')
            return
        if args.command=='demo' and args.offline:
            from .fixture import FixtureAI, MemoryStore
            ai=FixtureAI(); store=MemoryStore(fixtures(),ai)
            print('OFFLINE FIXTURE DEMO — simulated embeddings/generation; no Docker or API calls.')
        else:
            ai=LocalAI(); store=ChromaStore(ai.embedding_model)
            if args.command in ('ingest','demo'):
                rows=chunks(json.loads(args.extraction.read_text()), args.manifest) if args.command=='ingest' and args.extraction else fixtures()
                count=store.ingest(rows,ai.embed([r['text'] for r in rows]))
                print(f'Chroma contains {count} approved chunks; embedding model: {ai.embedding_model}.')
            if args.command=='ingest': return
        for q in ([args.question] if args.command=='ask' else [SUPPORTED,UNSUPPORTED]):
            print(f'\nQuestion: {q}\n{answer(q,ai,store)}')
    except DemoError as exc:
        print(f'Error: {exc}',file=sys.stderr); sys.exit(1)
    except Exception:
        # SDK exceptions may contain request/response bodies. Never print them.
        print('Error: FAQ dependency or document validation failed. Check Docker, local models, or hosted OCR access. No sensitive diagnostic body was logged.',file=sys.stderr)
        sys.exit(1)

if __name__=='__main__': main()
