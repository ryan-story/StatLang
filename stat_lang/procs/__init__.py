"""
PROC Procedures Module for StatLang

This module contains implementations of SAS PROC procedures
using Python libraries as the backend.
"""

from .proc_anova import ProcANOVA
from .proc_append import ProcAppend
from .proc_cluster import ProcCluster
from .proc_contents import ProcContents
from .proc_corr import ProcCorr
from .proc_cvision import ProcCVision
from .proc_datasets import ProcDatasets
from .proc_discrim import ProcDiscrim
from .proc_dnn import ProcDNN
from .proc_export import ProcExport
from .proc_factor import ProcFactor
from .proc_freq import ProcFreq
from .proc_genmod import ProcGenmod
from .proc_glm import ProcGLM
from .proc_import import ProcImport
from .proc_language import ProcLanguage
from .proc_lifereg import ProcLifereg
from .proc_llm import ProcLLM
from .proc_logit import ProcLogit
from .proc_means import ProcMeans
from .proc_mixed import ProcMixed
from .proc_ml import ProcBoost, ProcForest, ProcTree
from .proc_nlp import ProcNLP
from .proc_npar1way import ProcNpar1way
from .proc_phreg import ProcPhreg
from .proc_princomp import ProcPrincomp
from .proc_print import ProcPrint
from .proc_reg import ProcReg
from .proc_rl import ProcRL
from .proc_robustreg import ProcRobustreg
from .proc_sort import ProcSort
from .proc_sql import ProcSQL
from .proc_surveyselect import ProcSurveySelect
from .proc_timeseries import ProcTimeseries
from .proc_transpose import ProcTranspose
from .proc_ttest import ProcTtest
from .proc_univariate import ProcUnivariate

__all__ = [
    "ProcMeans",
    "ProcFreq",
    "ProcPrint",
    "ProcSort",
    "ProcContents",
    "ProcUnivariate",
    "ProcCorr",
    "ProcFactor",
    "ProcCluster",
    "ProcNpar1way",
    "ProcTtest",
    "ProcLogit",
    "ProcTimeseries",
    "ProcTree",
    "ProcForest",
    "ProcBoost",
    "ProcLanguage",
    "ProcSQL",
    "ProcSurveySelect",
    "ProcReg",
    # New SAS procs
    "ProcGLM",
    "ProcANOVA",
    "ProcDiscrim",
    "ProcPrincomp",
    "ProcRobustreg",
    "ProcLifereg",
    "ProcPhreg",
    "ProcGenmod",
    "ProcMixed",
    "ProcTranspose",
    "ProcAppend",
    "ProcDatasets",
    "ProcExport",
    "ProcImport",
    # Deep learning procs
    "ProcDNN",
    "ProcNLP",
    "ProcCVision",
    "ProcRL",
    "ProcLLM",
]
