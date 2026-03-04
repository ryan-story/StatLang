"""PROC LLM — Large Language Model generation via Hugging Face.

Usage:
    PROC LLM mode=generate;
        PROMPT 'Explain the central limit theorem';
        MODEL model_name='gpt2';
    RUN;
"""
from typing import Any, Dict

import pandas as pd

from ..parser.proc_parser import ProcStatement

try:
    from transformers import pipeline as hf_pipeline
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False


class ProcLLM:
    def execute(self, data: pd.DataFrame, proc_info: ProcStatement, **kw) -> Dict[str, Any]:
        results: Dict[str, Any] = {'output_text': [], 'output_data': None}
        if not HAS_TRANSFORMERS:
            results['output_text'].append("ERROR: transformers not installed")
            return results

        mode = str(proc_info.options.get('mode', 'generate')).lower()
        prompt_text = str(proc_info.options.get('prompt', ''))
        model_name = str(proc_info.options.get('model_name', 'gpt2'))
        max_length = int(proc_info.options.get('maxlength', proc_info.options.get('max_length', 200)))
        temperature = float(proc_info.options.get('temperature', 0.7))
        num_return = int(proc_info.options.get('num_return', 1))

        if not prompt_text:
            results['output_text'].append("ERROR: PROMPT required")
            return results

        try:
            if mode == 'generate':
                generator = hf_pipeline('text-generation', model=model_name)
                outputs = generator(
                    prompt_text,
                    max_length=max_length,
                    temperature=temperature,
                    num_return_sequences=num_return,
                    truncation=True,
                )
                results['output_text'].append("PROC LLM - Text Generation")
                results['output_text'].append("=" * 50)
                results['output_text'].append(f"Model: {model_name}")
                results['output_text'].append(f"Prompt: {prompt_text[:80]}...")
                results['output_text'].append("")
                rows = []
                for i, out in enumerate(outputs):
                    text = out['generated_text']
                    results['output_text'].append(f"--- Generation {i+1} ---")
                    results['output_text'].append(text)
                    results['output_text'].append("")
                    rows.append({'generation': i + 1, 'text': text})
                results['output_data'] = pd.DataFrame(rows)

            elif mode == 'fill':
                filler = hf_pipeline('fill-mask', model=model_name)
                outputs = filler(prompt_text)
                results['output_text'].append("PROC LLM - Fill-Mask")
                results['output_text'].append("=" * 50)
                rows = []
                for pred in outputs:
                    results['output_text'].append(f"  {pred['sequence']}  (score: {pred['score']:.4f})")
                    rows.append({'sequence': pred['sequence'], 'score': pred['score'], 'token': pred['token_str']})
                results['output_data'] = pd.DataFrame(rows)

            elif mode in ('qa', 'question_answering'):
                # Require context in data
                context = str(proc_info.options.get('context', ''))
                if not context and not data.empty:
                    str_cols = data.select_dtypes(include='object').columns
                    if len(str_cols) > 0:
                        context = ' '.join(data[str_cols[0]].dropna().astype(str).tolist())
                qa = hf_pipeline('question-answering', model=model_name or 'distilbert-base-cased-distilled-squad')
                answer = qa(question=prompt_text, context=context)
                results['output_text'].append("PROC LLM - Question Answering")
                results['output_text'].append(f"Answer: {answer['answer']}")
                results['output_text'].append(f"Score: {answer['score']:.4f}")
                results['output_data'] = pd.DataFrame([answer])

            else:
                results['output_text'].append(f"ERROR: Unknown mode '{mode}'")

        except Exception as e:
            results['output_text'].append(f"ERROR: {e}")
        return results
