"""
Command Line Interface for StatLang

This module provides a CLI for running .statlang files from the command line.
"""

import argparse
import os
import sys

from .interpreter import SASInterpreter


def main() -> None:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="StatLang: Python-based statistical scripting language",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  statlang program.statlang       # Run a .statlang file
  statlang -i                     # Interactive mode
  statlang --version              # Show version
        """
    )
    
    parser.add_argument(
        'file',
        nargs='?',
        help='Path to .statlang file to execute'
    )
    
    parser.add_argument(
        '-i', '--interactive',
        action='store_true',
        help='Run in interactive mode'
    )
    
    parser.add_argument(
        '--version',
        action='version',
        version='StatLang 0.1.2'
    )
    
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Enable verbose output'
    )
    
    args = parser.parse_args()
    
    # Create interpreter
    interpreter = SASInterpreter()
    
    if args.interactive:
        run_interactive(interpreter)
    elif args.file:
        run_file(interpreter, args.file, args.verbose)
    else:
        parser.print_help()


def run_file(interpreter: SASInterpreter, file_path: str, verbose: bool = False) -> None:
    """Run a .statlang file."""
    if not os.path.exists(file_path):
        print(f"Error: File '{file_path}' not found.", file=sys.stderr)
        sys.exit(1)
    
    if not file_path.endswith('.statlang'):
        print(f"Warning: File '{file_path}' does not have .statlang extension.", file=sys.stderr)
    
    try:
        if verbose:
            print(f"Executing: {file_path}")
        
        interpreter.run_file(file_path)
        
    except Exception as e:
        print(f"Error executing {file_path}: {e}", file=sys.stderr)
        sys.exit(1)


def run_interactive(interpreter: SASInterpreter) -> None:
    """Run in interactive mode."""
    print("StatLang Interactive Mode")
    print("Type 'quit' or 'exit' to exit, 'help' for help")
    print("=" * 50)
    
    while True:
        try:
            # Read input
            line = input("statlang> ").strip()
            
            if not line:
                continue
                
            if line.lower() in ['quit', 'exit']:
                break
                
            if line.lower() == 'help':
                print_help()
                continue
            
            if line.lower() == 'clear':
                interpreter.clear_workspace()
                print("Workspace cleared.")
                continue
            
            if line.lower().startswith('list'):
                datasets = interpreter.list_data_sets()
                if datasets:
                    print("Available datasets:")
                    for ds in datasets:
                        df = interpreter.get_data_set(ds)
                        if df is not None:
                            print(f"  {ds}: {len(df)} observations, {len(df.columns)} variables")
                else:
                    print("No datasets available.")
                continue
            
            # Execute the StatLang code
            interpreter.run_code(line)
            
        except KeyboardInterrupt:
            print("\nUse 'quit' or 'exit' to exit.")
        except EOFError:
            break
        except Exception as e:
            print(f"Error: {e}")
    
    print("Goodbye!")


def print_help() -> None:
    """Print help information."""
    help_text = """
StatLang Interactive Commands:
  help          - Show this help
  clear         - Clear workspace (remove all datasets)
  list          - List available datasets
  quit/exit     - Exit interactive mode

Language Statements:
  DATA steps    - Create and manipulate datasets
  PROC procedures - Analyze data (MEANS, FREQ, PRINT, etc.)
  %LET          - Set macro variables
  %PUT          - Display messages

Examples:
  data work.test; set sashelp.class; run;
  proc means data=work.test; var age height weight; run;
  %let cutoff=15;
  proc print data=work.test; where age > &cutoff; run;
    """
    print(help_text)


if __name__ == '__main__':
    main()
