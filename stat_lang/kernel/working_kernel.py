#!/usr/bin/env python3
"""
Working StatLang Jupyter Kernel Implementation
"""

import io
import traceback
from contextlib import redirect_stderr, redirect_stdout

from ipykernel.kernelbase import Kernel

from stat_lang import SASInterpreter


class WorkingStatLangKernel(Kernel):
    """Working Jupyter kernel for StatLang."""
    
    implementation = 'StatLang'
    implementation_version = '0.1.2'
    language = 'statlang'
    language_version = '9.4'
    language_info = {
        'name': 'statlang',
        'mimetype': 'text/x-statlang',
        'file_extension': '.statlang',
        'pygments_lexer': 'sas',
        'codemirror_mode': 'sas',
    }
    banner = "StatLang Kernel - Python-based statistical scripting language"
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.interpreter = SASInterpreter()
        self.output_buffer = io.StringIO()
        self.error_buffer = io.StringIO()
        self.datasets_before_execution = set()
    
    def do_execute(self, code, silent, store_history=True, user_expressions=None, allow_stdin=False, *, cell_meta=None, cell_id=None):
        """Execute StatLang code in the kernel."""
        
        # Skip empty cells
        if not code.strip():
            return {
                'status': 'ok',
                'execution_count': self.execution_count,
                'payload': [],
                'user_expressions': {},
            }
        
        # Clear buffers
        self.output_buffer = io.StringIO()
        self.error_buffer = io.StringIO()
        
        # Record datasets before execution
        self.datasets_before_execution = set(self.interpreter.data_sets.keys())
        
        try:
            # Execute code and capture output
            with redirect_stdout(self.output_buffer), redirect_stderr(self.error_buffer):
                self.interpreter.run_code(code)
            
            # Get output and errors
            output = self.output_buffer.getvalue()
            errors = self.error_buffer.getvalue()
            
            # Send output to notebook
            if output and not silent:
                self.send_response(self.iopub_socket, 'stream', {
                    'name': 'stdout',
                    'text': output
                })
            
            # Send errors to notebook
            if errors and not silent:
                self.send_response(self.iopub_socket, 'stream', {
                    'name': 'stderr',
                    'text': errors
                })
            
            # Get datasets created in this execution
            datasets = self._get_new_datasets_info()
            if datasets and not silent:
                self._send_datasets_display(datasets)
            
            # Increment execution count
            self.execution_count += 1
            
            return {
                'status': 'ok',
                'execution_count': self.execution_count,
                'payload': [],
                'user_expressions': {},
            }
            
        except Exception as e:
            # Send error to notebook
            if not silent:
                error_msg = f"Error executing code: {str(e)}\n"
                error_msg += traceback.format_exc()
                
                self.send_response(self.iopub_socket, 'error', {
                    'ename': 'SASError',
                    'evalue': str(e),
                    'traceback': [error_msg]
                })
            
            # Increment execution count even on error
            self.execution_count += 1
            
            return {
                'status': 'error',
                'execution_count': self.execution_count,
                'ename': 'SASError',
                'evalue': str(e),
                'traceback': [traceback.format_exc()]
            }
    
    def do_complete(self, code, cursor_pos):
        """Provide code completion for StatLang syntax."""
        # Simple completion for SAS keywords
        sas_keywords = [
            'data', 'set', 'merge', 'where', 'if', 'then', 'else', 'do', 'end',
            'proc', 'run', 'quit', 'var', 'by', 'class', 'tables', 'model',
            'output', 'drop', 'keep', 'rename', 'input', 'datalines', 'cards',
            'libname', '%let', '%put', '%macro', '%mend', 'means', 'freq',
            'print', 'sort', 'contents', 'univariate'
        ]
        
        # Get the word being completed
        text_before_cursor = code[:cursor_pos]
        word_start = text_before_cursor.rfind(' ') + 1
        word = text_before_cursor[word_start:].lower()
        
        # Find matching keywords
        matches = [kw for kw in sas_keywords if kw.startswith(word)]
        
        if matches:
            return {
                'matches': matches,
                'cursor_start': word_start,
                'cursor_end': cursor_pos,
                'metadata': {},
                'status': 'ok'
            }
        
        return {
            'matches': [],
            'cursor_start': cursor_pos,
            'cursor_end': cursor_pos,
            'metadata': {},
            'status': 'ok'
        }
    
    def do_inspect(self, code, cursor_pos, detail_level=0, omit_sections=()):
        """Provide code inspection/hover information."""
        # Get the word at cursor position
        text_before_cursor = code[:cursor_pos]
        text_after_cursor = code[cursor_pos:]
        
        # Find word boundaries
        word_start = text_before_cursor.rfind(' ') + 1
        word_end = cursor_pos + len(text_after_cursor.split()[0]) if text_after_cursor.split() else cursor_pos
        word = code[word_start:word_end].strip()
        
        # Provide help for StatLang keywords
        help_text = self._get_sas_help(word)
        
        if help_text:
            return {
                'status': 'ok',
                'data': {
                    'text/plain': help_text
                },
                'metadata': {}
            }
        
        return {
            'status': 'ok',
            'data': {},
            'metadata': {}
        }
    
    def _get_new_datasets_info(self):
        """Get information about datasets created in the current execution."""
        datasets = {}
        current_datasets = set(self.interpreter.data_sets.keys())
        new_datasets = current_datasets - self.datasets_before_execution
        
        for name in new_datasets:
            df = self.interpreter.data_sets[name]
            datasets[name] = {
                'shape': df.shape,
                'columns': df.columns.tolist(),
                'dtypes': df.dtypes.to_dict(),
                'head': df.head().to_dict('records') if not df.empty else [],
                'memory_usage': df.memory_usage(deep=True).sum()
            }
        return datasets
    
    def _send_datasets_display(self, datasets):
        """Send dataset information to notebook for display."""
        for name, info in datasets.items():
            # Create HTML display for dataset
            html = self._create_dataset_html(name, info)
            
            self.send_response(self.iopub_socket, 'display_data', {
                'data': {
                    'text/html': html,
                    'text/plain': f"Dataset: {name} ({info['shape'][0]} obs, {info['shape'][1]} vars)"
                },
                'metadata': {}
            })
    
    def _create_dataset_html(self, name, info):
        """Create HTML display for dataset."""
        shape = info['shape']
        columns = info['columns']
        head = info['head']
        
        html = f"""
        <div class="sas-dataset" style="margin: 10px 0; border: 1px solid #ddd; border-radius: 4px; padding: 10px;">
            <h4 style="margin: 0 0 10px 0; color: #333;">Dataset: {name}</h4>
            <p style="margin: 0 0 10px 0; color: #666; font-size: 0.9em;">
                {shape[0]} observations, {shape[1]} variables
            </p>
        """
        
        if head:
            html += """
            <table style="border-collapse: collapse; width: 100%; font-size: 0.9em;">
                <thead>
                    <tr style="background-color: #f5f5f5;">
            """
            
            for col in columns:
                html += f'<th style="border: 1px solid #ddd; padding: 8px; text-align: left;">{col}</th>'
            
            html += """
                    </tr>
                </thead>
                <tbody>
            """
            
            for row in head:
                html += '<tr>'
                for col in columns:
                    value = row.get(col, '')
                    html += f'<td style="border: 1px solid #ddd; padding: 8px;">{value}</td>'
                html += '</tr>'
            
            html += """
                </tbody>
            </table>
            """
        
        html += "</div>"
        return html
    
    def _get_sas_help(self, word):
        """Get help text for StatLang keywords."""
        help_dict = {
            'data': 'DATA step - creates and manipulates datasets',
            'proc': 'PROC procedure - performs analysis and reporting',
            'set': 'SET statement - reads observations from a dataset',
            'where': 'WHERE statement - subsets observations',
            'if': 'IF statement - conditional processing',
            'run': 'RUN statement - executes the step',
            'var': 'VAR statement - specifies analysis variables',
            'by': 'BY statement - groups observations',
            'means': 'PROC MEANS - descriptive statistics',
            'freq': 'PROC FREQ - frequency tables',
            'print': 'PROC PRINT - displays data',
            'sort': 'PROC SORT - sorts observations',
            'libname': 'LIBNAME statement - assigns library references',
            '%let': '%LET statement - creates macro variables'
        }
        
        return help_dict.get(word.lower(), None)


def main():
    """Main entry point for the kernel."""
    from ipykernel.kernelapp import IPKernelApp
    IPKernelApp.launch_instance(kernel_class=WorkingStatLangKernel)


if __name__ == '__main__':
    main()
