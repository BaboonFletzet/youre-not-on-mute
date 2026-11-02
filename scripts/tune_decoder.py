"""Choose rescoring weights on development hypotheses only; never optimize on eval."""
import argparse, itertools, json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from services.inference.decoding import DecodeOptions, rank_candidates, LocalLanguageScorer
from services.inference.evaluation import summarize


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('predictions', type=Path); p.add_argument('--output', type=Path, required=True)
    p.add_argument('--lm', action='store_true')
    args = p.parse_args()
    rows = [json.loads(line) for line in args.predictions.read_text().splitlines()]
    if not rows or any(r['split'] != 'dev' for r in rows): raise ValueError('Only development predictions may tune the decoder.')
    if args.lm and not any(c.get('lm_score') for r in rows for c in r.get('candidates', [])):
        scorer = LocalLanguageScorer(ROOT / '.cache/models/gpt2')
        for row in rows:
            candidates = row.get('candidates', [])
            for c, score in zip(candidates, scorer.score([c['text'] for c in candidates])): c['lm_score'] = score
        args.predictions.with_name(args.predictions.stem + '-lm.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
    trials = []
    for ctc, lm, bonus in itertools.product([0, .05, .1, .2, .3, .5], [0, .05, .1, .2] if args.lm else [0], [-.2, 0, .2]):
        options = DecodeOptions(beam_size=rows[0]['options']['beam_size'], ctc_weight=ctc, lm_weight=lm, length_bonus=bonus)
        predictions = [{**r, 'text': rank_candidates(r['candidates'], options)[0]['text'] if r.get('candidates') else ''} for r in rows]
        summary = summarize(predictions)
        trials.append({'options': options.asdict(), **summary})
    trials.sort(key=lambda x: (x['wer'], x['options']['lm_weight'], x['options']['ctc_weight'], abs(x['options']['length_bonus'])))
    result = {'options': trials[0]['options'], 'development_metrics': trials[0], 'source': str(args.predictions),
              'evaluation_status': 'not_yet_evaluated', 'trials': trials}
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result['development_metrics'], indent=2))

if __name__ == '__main__': main()
