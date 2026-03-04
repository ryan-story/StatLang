"""
Pipeline Runner for StatLang

Provides a programmatic entry point for running .statlang files
end-to-end and collecting results.
"""

from typing import Any, Dict, List, Optional


def run_pipeline(
    file_path: str,
    variables: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """
    Run a .statlang file end-to-end and return a result summary.

    Args:
        file_path: Path to the .statlang file
        variables: Optional macro variables to set before execution

    Returns:
        Dictionary with keys:
            - datasets: dict of name -> DataFrame
            - models: list of model names in the model store
            - last_dataset: name of the last created dataset
            - errors: list of any error messages captured
    """
    from .interpreter import SASInterpreter

    interp = SASInterpreter()

    # Set any pre-defined variables
    if variables:
        for k, v in variables.items():
            interp.macro_processor.set_variable(k, v)

    errors: List[str] = []
    try:
        interp.run_file(file_path)
    except Exception as e:
        errors.append(str(e))

    last_ds = interp.macro_processor.get_variable('SYSLAST') or '_NULL_'

    return {
        'datasets': dict(interp.data_sets),
        'models': interp.model_store.list_models(),
        'last_dataset': last_ds,
        'errors': errors,
    }
