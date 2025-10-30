"""
PROC LANGUAGE Implementation for Open-SAS

This module implements SAS PROC LANGUAGE functionality for LLM integration
using Hugging Face transformers for open-source language model access.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from ..parser.proc_parser import ProcStatement

# Import transformers with fallback
try:
    from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


class ProcLanguage:
    """Implementation of SAS PROC LANGUAGE procedure."""
    
    def __init__(self):
        self.default_model = "distilgpt2"  # Lightweight, fast model
        self.generator = None
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the Hugging Face model."""
        if not TRANSFORMERS_AVAILABLE:
            return
        
        try:
            # Use a lightweight, fast model that works well for text generation
            self.generator = pipeline(
                "text-generation",
                model=self.default_model,
                max_length=100,
                do_sample=True,
                temperature=0.7,
                pad_token_id=50256  # GPT-2 pad token
            )
        except Exception as e:
            print(f"Warning: Could not initialize model {self.default_model}: {e}")
            self.generator = None
    
    def execute(self, data: pd.DataFrame, proc_info: ProcStatement, dataset_manager=None) -> Dict[str, Any]:
        """
        Execute PROC LANGUAGE on the given data.
        
        Args:
            data: Input DataFrame
            proc_info: Parsed PROC statement information
            
        Returns:
            Dictionary containing results and output data
        """
        results = {
            'output_text': [],
            'output_data': None
        }
        
        # Get PROMPT specification
        prompt = proc_info.options.get('prompt', '')
        if not prompt:
            results['output_text'].append("ERROR: PROMPT specification required for PROC LANGUAGE.")
            return results
        
        # Get model specification
        model = proc_info.options.get('model', self.default_model)
        
        # Get mode specification
        mode = proc_info.options.get('mode', 'generate').lower()
        if mode not in ['generate', 'qna', 'summarize', 'analyze']:
            mode = 'generate'
        
        # Get VAR specification for data analysis
        var_vars = proc_info.options.get('var', [])
        
        results['output_text'].append("PROC LANGUAGE - LLM Analysis")
        results['output_text'].append("=" * 50)
        results['output_text'].append(f"Model: {model}")
        results['output_text'].append(f"Mode: {mode.upper()}")
        results['output_text'].append("")
        
        # Check if transformers is available
        if not TRANSFORMERS_AVAILABLE:
            results['output_text'].append("ERROR: Hugging Face transformers not available.")
            results['output_text'].append("Please install: pip install transformers torch")
            return results
        
        # Check if model is initialized
        if self.generator is None:
            results['output_text'].append("ERROR: Language model not initialized.")
            results['output_text'].append("Please check your internet connection for model download.")
            return results
        
        # Execute based on mode
        if mode == 'generate':
            llm_results = self._generate_text(prompt, model)
        elif mode == 'qna':
            context = proc_info.options.get('context', '')
            llm_results = self._question_answer(prompt, context, model)
        elif mode == 'summarize':
            if var_vars:
                llm_results = self._summarize_data(data, var_vars, prompt, model)
            else:
                llm_results = self._summarize_text(prompt, model)
        else:  # analyze
            if var_vars:
                llm_results = self._analyze_data(data, var_vars, prompt, model)
            else:
                llm_results = self._analyze_text(prompt, model)
        
        # Format output
        results['output_text'].extend(llm_results['output'])
        results['output_data'] = llm_results.get('data', None)
        
        return results
    
    def _generate_text(self, prompt: str, model: str) -> Dict[str, Any]:
        """Generate text using the Hugging Face model."""
        try:
            # Generate text using the pipeline
            result = self.generator(
                prompt,
                max_length=min(len(prompt.split()) + 50, 150),  # Add up to 50 words
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True,
                pad_token_id=50256
            )
            
            generated_text = result[0]['generated_text']
            
            # Remove the original prompt from the generated text
            if generated_text.startswith(prompt):
                generated_text = generated_text[len(prompt):].strip()
            
            output = []
            output.append("Text Generation")
            output.append("-" * 20)
            output.append(f"Prompt: {prompt}")
            output.append("")
            output.append("Generated Text:")
            output.append(generated_text)
            output.append("")
            
            return {
                'output': output,
                'data': None
            }
            
        except Exception as e:
            return {
                'output': [f"ERROR: Text generation failed: {str(e)}"],
                'data': None
            }
    
    def _question_answer(self, question: str, context: str, model: str) -> Dict[str, Any]:
        """Answer questions using the model."""
        try:
            # For Q&A, we'll use text generation with a structured prompt
            if context:
                qa_prompt = f"Context: {context}\n\nQuestion: {question}\n\nAnswer:"
            else:
                qa_prompt = f"Question: {question}\n\nAnswer:"
            
            result = self.generator(
                qa_prompt,
                max_length=min(len(qa_prompt.split()) + 30, 100),
                num_return_sequences=1,
                temperature=0.3,  # Lower temperature for more focused answers
                do_sample=True,
                pad_token_id=50256
            )
            
            generated_text = result[0]['generated_text']
            
            # Extract just the answer part
            if "Answer:" in generated_text:
                answer = generated_text.split("Answer:")[-1].strip()
            else:
                answer = generated_text[len(qa_prompt):].strip()
            
            output = []
            output.append("Question & Answer")
            output.append("-" * 20)
            output.append(f"Question: {question}")
            if context:
                output.append(f"Context: {context}")
            output.append("")
            output.append(f"Answer: {answer}")
            output.append("")
            
            return {
                'output': output,
                'data': None
            }
            
        except Exception as e:
            return {
                'output': [f"ERROR: Q&A failed: {str(e)}"],
                'data': None
            }
    
    def _summarize_text(self, text: str, model: str) -> Dict[str, Any]:
        """Summarize text using the model."""
        try:
            summary_prompt = f"Summarize the following text:\n\n{text}\n\nSummary:"
            
            result = self.generator(
                summary_prompt,
                max_length=min(len(summary_prompt.split()) + 20, 80),
                num_return_sequences=1,
                temperature=0.3,
                do_sample=True,
                pad_token_id=50256
            )
            
            generated_text = result[0]['generated_text']
            
            # Extract the summary
            if "Summary:" in generated_text:
                summary = generated_text.split("Summary:")[-1].strip()
            else:
                summary = generated_text[len(summary_prompt):].strip()
            
            output = []
            output.append("Text Summarization")
            output.append("-" * 20)
            output.append(f"Original text: {text[:100]}{'...' if len(text) > 100 else ''}")
            output.append("")
            output.append(f"Summary: {summary}")
            output.append("")
            
            return {
                'output': output,
                'data': None
            }
            
        except Exception as e:
            return {
                'output': [f"ERROR: Summarization failed: {str(e)}"],
                'data': None
            }
    
    def _analyze_text(self, text: str, model: str) -> Dict[str, Any]:
        """Analyze text using the model."""
        try:
            analysis_prompt = f"Analyze the following text and provide insights:\n\n{text}\n\nAnalysis:"
            
            result = self.generator(
                analysis_prompt,
                max_length=min(len(analysis_prompt.split()) + 40, 120),
                num_return_sequences=1,
                temperature=0.5,
                do_sample=True,
                pad_token_id=50256
            )
            
            generated_text = result[0]['generated_text']
            
            # Extract the analysis
            if "Analysis:" in generated_text:
                analysis = generated_text.split("Analysis:")[-1].strip()
            else:
                analysis = generated_text[len(analysis_prompt):].strip()
            
            output = []
            output.append("Text Analysis")
            output.append("-" * 20)
            output.append(f"Text: {text[:100]}{'...' if len(text) > 100 else ''}")
            output.append("")
            output.append(f"Analysis: {analysis}")
            output.append("")
            
            return {
                'output': output,
                'data': None
            }
            
        except Exception as e:
            return {
                'output': [f"ERROR: Analysis failed: {str(e)}"],
                'data': None
            }
    
    def _summarize_data(self, data: pd.DataFrame, var_vars: List[str], prompt: str, model: str) -> Dict[str, Any]:
        """Summarize data using the model."""
        try:
            # Create a text description of the data
            data_desc = f"Dataset with {len(data)} rows and {len(data.columns)} columns."
            if var_vars:
                data_desc += f" Variables: {', '.join(var_vars)}."
            
            summary_prompt = f"{prompt}\n\nData: {data_desc}\n\nSummary:"
            
            result = self.generator(
                summary_prompt,
                max_length=min(len(summary_prompt.split()) + 30, 100),
                num_return_sequences=1,
                temperature=0.3,
                do_sample=True,
                pad_token_id=50256
            )
            
            generated_text = result[0]['generated_text']
            
            # Extract the summary
            if "Summary:" in generated_text:
                summary = generated_text.split("Summary:")[-1].strip()
            else:
                summary = generated_text[len(summary_prompt):].strip()
            
            output = []
            output.append("Data Summarization")
            output.append("-" * 20)
            output.append(f"Dataset: {data_desc}")
            output.append(f"Prompt: {prompt}")
            output.append("")
            output.append(f"Summary: {summary}")
            output.append("")
            
            return {
                'output': output,
                'data': None
            }
            
        except Exception as e:
            return {
                'output': [f"ERROR: Data summarization failed: {str(e)}"],
                'data': None
            }
    
    def _analyze_data(self, data: pd.DataFrame, var_vars: List[str], prompt: str, model: str) -> Dict[str, Any]:
        """Analyze data using the model."""
        try:
            # Create a text description of the data
            data_desc = f"Dataset with {len(data)} rows and {len(data.columns)} columns."
            if var_vars:
                data_desc += f" Variables: {', '.join(var_vars)}."
            
            analysis_prompt = f"{prompt}\n\nData: {data_desc}\n\nAnalysis:"
            
            result = self.generator(
                analysis_prompt,
                max_length=min(len(analysis_prompt.split()) + 50, 150),
                num_return_sequences=1,
                temperature=0.5,
                do_sample=True,
                pad_token_id=50256
            )
            
            generated_text = result[0]['generated_text']
            
            # Extract the analysis
            if "Analysis:" in generated_text:
                analysis = generated_text.split("Analysis:")[-1].strip()
            else:
                analysis = generated_text[len(analysis_prompt):].strip()
            
            output = []
            output.append("Data Analysis")
            output.append("-" * 20)
            output.append(f"Dataset: {data_desc}")
            output.append(f"Prompt: {prompt}")
            output.append("")
            output.append(f"Analysis: {analysis}")
            output.append("")
            
            return {
                'output': output,
                'data': None
            }
            
        except Exception as e:
            return {
                'output': [f"ERROR: Data analysis failed: {str(e)}"],
                'data': None
            }