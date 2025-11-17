"""
StatLang Jupyter Kernel Implementation

This module implements a Jupyter kernel for executing StatLang code
in notebook environments.
"""
import io
import logging
import os
import tempfile
import traceback
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Optional

from ipykernel.kernelbase import Kernel

from stat_lang import SASInterpreter, __version__

# Set up logging (WARNING level to suppress INFO messages)
# Use platform-independent temp directory
log_dir = Path(tempfile.gettempdir())
log_file = log_dir / 'statlang_kernel_debug.log'
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(str(log_file))
    ]
)
logger = logging.getLogger('StatLangKernel')


class StatLangKernel(Kernel):
    """Jupyter kernel for StatLang."""
    
    implementation = 'statlang'
    implementation_version = __version__
    language = 'statlang'
    language_version = '9.4'
    language_info = {
        'name': 'statlang',
        'version': '9.4',
        'mimetype': 'text/x-statlang',
        'file_extension': '.statlang',
        'pygments_lexer': 'sas',
        'codemirror_mode': 'sas',
        'nbconvert_exporter': 'python',
        'codemirror_mode_name': 'sas',
    }
    banner = "StatLang Kernel - Python-based statistical scripting language"
    
    def __init__(self, **kwargs):
        import traceback
        logger.info("=" * 80)
        logger.info("StatLangKernel __init__ called - Kernel being created/recreated")
        logger.info(f"CWD: {os.getcwd()}")
        logger.info("Stack trace:")
        for line in traceback.format_stack()[-5:-1]:
            logger.info(line.strip())
        logger.info("=" * 80)
        
        super().__init__(**kwargs)
        self.interpreter: Optional[SASInterpreter] = None
        try:
            self.interpreter = SASInterpreter()
            # Kernel base class initializes execution_count, but ensure it starts at 0
            # First execution will increment it to 1, second to 2, etc.
            if not hasattr(self, 'execution_count') or self.execution_count is None:
                self.execution_count = 0
            self.output_buffer = io.StringIO()
            self.error_buffer = io.StringIO()
            self.datasets_before_execution = set()
            logger.info(f"StatLangKernel initialized successfully with execution_count={self.execution_count}")
        except Exception as e:
            # Log the error but don't fail initialization
            logger.error(f"Failed to initialize interpreter: {e}")
            print(f"Warning: Failed to initialize interpreter: {e}")
            self.interpreter = None
    
    def do_execute(self, code, silent, store_history=True, user_expressions=None, allow_stdin=False, *, cell_meta=None, cell_id=None):
        """Execute StatLang code in the kernel."""
        
        logger.info(f"do_execute called with code: {repr(code[:100])}...")
        logger.info(f"Full code received: {repr(code)}")
        logger.info(f"Code lines: {code.split(chr(10))}")
        logger.info(f"silent={silent}, store_history={store_history}, allow_stdin={allow_stdin}")
        
        # Ensure execution_count is initialized as a number (not None)
        if not hasattr(self, 'execution_count') or self.execution_count is None:
            self.execution_count = 0
        
        logger.info(f"Current execution_count: {self.execution_count}")
        
        # Skip empty cells - still need to increment
        if not code.strip():
            logger.info("Empty cell detected, incrementing and returning ok status")
            # Increment execution count
            self.execution_count += 1
            logger.info(f"execute_reply ok, count incremented to {self.execution_count}")
            return {
                'status': 'ok',
                'execution_count': self.execution_count,
                'payload': [],
                'user_expressions': {},
            }
        
        # Check if interpreter is available
        if self.interpreter is None:
            logger.warning("Interpreter is None, attempting to initialize")
            try:
                self.interpreter = SASInterpreter()
                self.output_buffer = io.StringIO()
                self.error_buffer = io.StringIO()
                logger.info("Interpreter initialized successfully")
            except Exception as e:
                error_msg = f"Failed to initialize interpreter: {e}"
                logger.error(error_msg)
                if not silent:
                    self.send_response(self.iopub_socket, 'stream', {
                        'name': 'stderr',
                        'text': error_msg
                    })
                # Increment execution count even on initialization error
                self.execution_count += 1
                logger.info(f"execute_reply error, count incremented to {self.execution_count}")
                return {
                    'status': 'error',
                    'execution_count': self.execution_count,
                    'ename': 'StatLangError',
                    'evalue': error_msg,
                    'traceback': [error_msg]
                }
        
        # Clear buffers
        self.output_buffer = io.StringIO()
        self.error_buffer = io.StringIO()
        
        # Record datasets before execution
        if self.interpreter is None:
            return {'status': 'error', 'ename': 'InterpreterError', 'evalue': 'Interpreter not initialized'}
        datasets_before = set(self.interpreter.data_sets.keys())
        logger.info(f"Datasets before execution: {datasets_before}")
        
        try:
            logger.info("Starting StatLang code execution")
            logger.info(f"Code being passed to interpreter: {repr(code)}")
            # Execute code and capture output
            with redirect_stdout(self.output_buffer), redirect_stderr(self.error_buffer):
                self.interpreter.run_code(code)
            
            # Get output and errors
            output = self.output_buffer.getvalue()
            errors = self.error_buffer.getvalue()
            logger.info(f"Execution completed. Output length: {len(output)}, Errors length: {len(errors)}")
            
            # Send output to notebook
            if output and not silent:
                logger.info(f"Sending stdout output: {repr(output[:100])}...")
                self.send_response(self.iopub_socket, 'stream', {
                    'name': 'stdout',
                    'text': output
                })
            
            # Send errors to notebook
            if errors and not silent:
                logger.info(f"Sending stderr output: {repr(errors[:100])}...")
                self.send_response(self.iopub_socket, 'stream', {
                    'name': 'stderr',
                    'text': errors
                })
            
            # Get datasets created in this execution
            datasets = self._get_new_datasets_info(datasets_before)
            if datasets and not silent:
                # Check if any PROC suppressed dataset display
                suppress_display = getattr(self.interpreter, '_suppress_dataset_display', False)
                if not suppress_display:
                    logger.info(f"Sending dataset display for: {list(datasets.keys())}")
                    self._send_datasets_display(datasets)
                else:
                    # Reset the flag for next execution
                    if self.interpreter is not None:
                        self.interpreter._suppress_dataset_display = False
            
            logger.info("Returning successful execution result")
            # Increment execution count AFTER successful execution
            self.execution_count += 1
            logger.info(f"execute_reply ok, count incremented to {self.execution_count}")
            return {
                'status': 'ok',
                'execution_count': self.execution_count,
                'payload': [],
                'user_expressions': {},
            }
            
        except Exception as e:
            logger.error(f"Exception during execution: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            
            # Send error to notebook
            if not silent:
                error_content = {
                    'ename': 'StatLangError',
                    'evalue': str(e),
                    'traceback': [traceback.format_exc()]
                }
                self.send_response(self.iopub_socket, 'error', error_content)
            
            logger.info("Returning error execution result")
            # Increment execution count even on error
            self.execution_count += 1
            logger.info(f"execute_reply error, count incremented to {self.execution_count}")
            return {
                'status': 'error',
                'execution_count': self.execution_count,
                'ename': 'StatLangError',
                'evalue': str(e),
                'traceback': [traceback.format_exc()]
            }
    
    def do_complete(self, code, cursor_pos):
        """Provide code completion for StatLang syntax."""
        # Simple completion for StatLang keywords
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
    
    def _get_datasets_info(self):
        """Get information about datasets created in the interpreter."""
        if self.interpreter is None:
            return {}
        datasets = {}
        for name, df in self.interpreter.data_sets.items():
            datasets[name] = {
                'shape': df.shape,
                'columns': df.columns.tolist(),
                'dtypes': df.dtypes.to_dict(),
                'head': df.head().to_dict('records') if not df.empty else [],
                'memory_usage': df.memory_usage(deep=True).sum()
            }
        return datasets
    
    def _get_new_datasets_info(self, datasets_before):
        """Get information about datasets created in the current execution."""
        if self.interpreter is None:
            return {}
        datasets = {}
        current_datasets = set(self.interpreter.data_sets.keys())
        new_datasets = current_datasets - datasets_before
        
        logger.info(f"Datasets before execution: {datasets_before}")
        logger.info(f"Current datasets: {current_datasets}")
        logger.info(f"New datasets: {new_datasets}")
        
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
            
            # Use standard HTML display instead of custom renderer
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
        <div class="statlang-dataset" style="margin: 10px 0; border: 1px solid #ddd; border-radius: 4px; padding: 10px;">
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
            'means': 'MEANS - descriptive statistics',
            'freq': 'FREQ - frequency tables',
            'print': 'PRINT - displays data',
            'sort': 'SORT - sorts observations',
            'libname': 'LIBNAME - assigns library references',
            '%let': '%LET - creates macro variables'
        }
        
        return help_dict.get(word.lower(), None)


# Note: The main() function is now in __main__.py to avoid circular imports
