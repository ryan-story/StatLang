"""PROC NLP — NLP Sentiment / Classification via Hugging Face transformers.

Usage:
    PROC NLP DATA=reviews mode=sentiment;
        TEXT review_text;
    RUN;

    PROC NLP DATA=reviews mode=classify;
        TEXT review_text;
        MODEL model_name='distilbert-base-uncased-finetuned-sst-2-english';
    RUN;
"""
from typing import Any, Dict, List

import pandas as pd

from ..parser.proc_parser import ProcStatement

try:
    from transformers import pipeline as hf_pipeline
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False


class ProcNLP:
    def execute(self, data: pd.DataFrame, proc_info: ProcStatement, **kw) -> Dict[str, Any]:
        results: Dict[str, Any] = {'output_text': [], 'output_data': None}
        if not HAS_TRANSFORMERS:
            results['output_text'].append("ERROR: transformers not installed")
            return results

        text_col = str(proc_info.options.get('text', ''))
        mode = str(proc_info.options.get('mode', 'sentiment')).lower()
        model_name = str(proc_info.options.get('model_name', ''))

        if not text_col:
            # Try to find a text column
            str_cols = data.select_dtypes(include='object').columns.tolist()
            if str_cols:
                text_col = str_cols[0]
            else:
                results['output_text'].append("ERROR: TEXT column required")
                return results

        if text_col not in data.columns:
            results['output_text'].append(f"ERROR: Column '{text_col}' not found")
            return results

        texts = data[text_col].dropna().astype(str).tolist()
        if not texts:
            results['output_text'].append("ERROR: No text data found")
            return results

        try:
            if mode == 'sentiment':
                pipe = hf_pipeline('sentiment-analysis', model=model_name or None)
                preds = pipe(texts, truncation=True, max_length=512)
                labels = [p['label'] for p in preds]
                scores = [p['score'] for p in preds]
                out = data.copy()
                out['sentiment_label'] = pd.Series(labels, index=data[text_col].dropna().index)
                out['sentiment_score'] = pd.Series(scores, index=data[text_col].dropna().index)
                results['output_text'].append("PROC NLP - Sentiment Analysis")
                results['output_text'].append(f"Model: {model_name or 'default'}")
                results['output_text'].append(f"Processed {len(texts)} texts")
                results['output_data'] = out

            elif mode in ('classify', 'classification'):
                pipe = hf_pipeline('text-classification', model=model_name or None)
                preds = pipe(texts, truncation=True, max_length=512)
                out = data.copy()
                out['predicted_label'] = pd.Series([p['label'] for p in preds], index=data[text_col].dropna().index)
                out['predicted_score'] = pd.Series([p['score'] for p in preds], index=data[text_col].dropna().index)
                results['output_text'].append("PROC NLP - Text Classification")
                results['output_text'].append(f"Model: {model_name or 'default'}")
                results['output_text'].append(f"Processed {len(texts)} texts")
                results['output_data'] = out

            elif mode in ('ner', 'entity'):
                pipe = hf_pipeline('ner', model=model_name or None, aggregation_strategy='simple')
                all_entities: List[Dict] = []
                for i, text in enumerate(texts):
                    ents = pipe(text)
                    for ent in ents:
                        all_entities.append({
                            'row_index': i,
                            'entity_group': ent.get('entity_group', ent.get('entity', '')),
                            'word': ent.get('word', ''),
                            'score': ent.get('score', 0),
                        })
                out = pd.DataFrame(all_entities) if all_entities else pd.DataFrame()
                results['output_text'].append("PROC NLP - Named Entity Recognition")
                results['output_text'].append(f"Found {len(all_entities)} entities")
                results['output_data'] = out

            elif mode in ('summarize', 'summarization'):
                pipe = hf_pipeline('summarization', model=model_name or None)
                summaries = pipe(texts, max_length=130, min_length=30, truncation=True)
                out = data.copy()
                out['summary'] = pd.Series([s['summary_text'] for s in summaries],
                                           index=data[text_col].dropna().index)
                results['output_text'].append("PROC NLP - Summarization")
                results['output_data'] = out

            else:
                results['output_text'].append(f"ERROR: Unknown mode '{mode}'")

        except Exception as e:
            results['output_text'].append(f"ERROR: {e}")
        return results
