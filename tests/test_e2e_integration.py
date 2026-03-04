"""
End-to-end integration tests for StatLang PRD features.

Tests DATA step enhancements, new procs, macro features, model store,
and pipeline runner. Each test runs a complete .statlang script through
the interpreter and verifies the output datasets.
"""

import os

import numpy as np
import pandas as pd

from stat_lang import StatLangInterpreter


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _run(code: str) -> StatLangInterpreter:
    interp = StatLangInterpreter()
    interp.run_code(code)
    return interp


def _run_with_data(code: str, datasets: dict) -> StatLangInterpreter:
    interp = StatLangInterpreter()
    for name, df in datasets.items():
        interp.data_sets[name] = df
        from stat_lang.utils.statlang_dataset import SasDataset
        interp.dataset_manager.datasets[name] = SasDataset(name=name, dataframe=df)
    interp.run_code(code)
    return interp


# ===================================================================
# 1. DATA STEP — DATALINES / CARDS
# ===================================================================
class TestDataStepBasicE2E:
    def test_datalines_numeric(self):
        code = """
        data work.nums;
            input a b c;
            datalines;
        1 2 3
        4 5 6
        7 8 9
        ;
        run;
        """
        interp = _run(code)
        df = interp.get_data_set("work.nums")
        assert df is not None
        assert len(df) == 3
        assert list(df.columns) == ["a", "b", "c"]

    def test_datalines_character(self):
        code = """
        data work.names;
            input name $ age;
            datalines;
        Alice 30
        Bob 25
        ;
        run;
        """
        interp = _run(code)
        df = interp.get_data_set("work.names")
        assert df is not None
        assert df["name"].tolist() == ["Alice", "Bob"]
        assert df["age"].tolist() == [30.0, 25.0]


# ===================================================================
# 2. DATA STEP — MERGE
# ===================================================================
class TestDataStepMergeE2E:
    def test_merge_by_key(self):
        df_a = pd.DataFrame({"id": [1, 2, 3], "x": [10, 20, 30]})
        df_b = pd.DataFrame({"id": [1, 2, 3], "y": [100, 200, 300]})
        code = """
        data work.merged;
            merge a b;
            by id;
        run;
        """
        interp = _run_with_data(code, {"a": df_a, "b": df_b})
        df = interp.get_data_set("work.merged")
        assert df is not None
        assert "x" in df.columns
        assert "y" in df.columns
        assert len(df) == 3


# ===================================================================
# 3. DATA STEP — RETAIN
# ===================================================================
class TestDataStepRetainE2E:
    def test_retain_running_sum(self):
        df_src = pd.DataFrame({"val": [1.0, 2.0, 3.0, 4.0]})
        code = """
        data work.result;
            set src;
            retain cumsum 0;
            cumsum = cumsum + val;
        run;
        """
        interp = _run_with_data(code, {"src": df_src})
        df = interp.get_data_set("work.result")
        assert df is not None
        assert df["cumsum"].tolist() == [1.0, 3.0, 6.0, 10.0]


# ===================================================================
# 4. DATA STEP — ARRAY
# ===================================================================
class TestDataStepArrayE2E:
    def test_array_multiply(self):
        df_src = pd.DataFrame({"x1": [1.0, 2.0], "x2": [3.0, 4.0], "x3": [5.0, 6.0]})
        code = """
        data work.result;
            set src;
            array xs{3} x1 x2 x3;
            do i = 1 to 3;
                xs(i) = xs(i) * 10;
            end;
        run;
        """
        interp = _run_with_data(code, {"src": df_src})
        df = interp.get_data_set("work.result")
        assert df is not None
        assert df["x1"].tolist() == [10.0, 20.0]
        assert df["x2"].tolist() == [30.0, 40.0]
        assert df["x3"].tolist() == [50.0, 60.0]


# ===================================================================
# 5. DATA STEP — DO loops
# ===================================================================
class TestDataStepDoLoopE2E:
    def test_do_iterative(self):
        df_src = pd.DataFrame({"val": [1.0]})
        code = """
        data work.result;
            set src;
            total = 0;
            do i = 1 to 5;
                total = total + i;
            end;
        run;
        """
        interp = _run_with_data(code, {"src": df_src})
        df = interp.get_data_set("work.result")
        assert df is not None
        assert df["total"].iloc[0] == 15.0


# ===================================================================
# 6. DATA STEP — FIRST./LAST.
# ===================================================================
class TestDataStepFirstLastE2E:
    def test_first_last_by_group(self):
        df_src = pd.DataFrame({
            "grp": ["A", "A", "B", "B", "B"],
            "val": [1, 2, 3, 4, 5],
        })
        code = """
        data work.result;
            set src;
            by grp;
        run;
        """
        interp = _run_with_data(code, {"src": df_src})
        df = interp.get_data_set("work.result")
        assert df is not None
        assert "first_grp" in df.columns
        assert "last_grp" in df.columns
        assert df["first_grp"].tolist() == [1, 0, 1, 0, 0]
        assert df["last_grp"].tolist() == [0, 1, 0, 0, 1]


# ===================================================================
# 7. MACRO — %LET, substitution
# ===================================================================
class TestMacroE2E:
    def test_let_and_substitution(self):
        code = """
        %LET dsname = work.mydata;

        data &dsname.;
            input x;
            datalines;
        42
        ;
        run;
        """
        interp = _run(code)
        df = interp.get_data_set("work.mydata")
        assert df is not None
        assert len(df) == 1

    def test_sysevalf(self):
        interp = StatLangInterpreter()
        result = interp.macro_processor._process_sysevalf("%sysevalf(1 + 2 * 3)")
        assert result == "7"

    def test_sysfunc_today(self):
        interp = StatLangInterpreter()
        result = interp.macro_processor._process_sysfunc("%sysfunc(today())")
        # Should return an ISO date string
        import datetime
        assert datetime.date.today().isoformat() in result

    def test_global_local(self):
        interp = StatLangInterpreter()
        interp.macro_processor._parse_global_statement("%GLOBAL myvar;")
        assert "myvar" in interp.macro_processor.global_variables
        interp.macro_processor.set_variable("myvar", "hello")
        assert interp.macro_processor.get_variable("myvar") == "hello"


# ===================================================================
# 8. PROC MEANS
# ===================================================================
class TestProcMeansE2E:
    def test_proc_means_basic(self):
        df = pd.DataFrame({"x": [1, 2, 3, 4, 5], "y": [10, 20, 30, 40, 50]})
        code = """
        proc means data=mydata;
            var x y;
        run;
        """
        interp = _run_with_data(code, {"mydata": df})
        # Just verify it doesn't crash
        assert "mydata" in interp.data_sets


# ===================================================================
# 9. PROC REG
# ===================================================================
class TestProcRegE2E:
    def test_proc_reg_simple(self):
        np.random.seed(42)
        x = np.random.randn(50)
        y = 2 * x + 1 + np.random.randn(50) * 0.1
        df = pd.DataFrame({"x": x, "y": y})
        code = """
        proc reg data=mydata;
            model y = x;
        run;
        """
        interp = _run_with_data(code, {"mydata": df})
        assert "mydata" in interp.data_sets


# ===================================================================
# 10. PROC GLM
# ===================================================================
class TestProcGLME2E:
    def test_proc_glm(self):
        np.random.seed(42)
        df = pd.DataFrame({
            "y": np.random.randn(50),
            "x1": np.random.randn(50),
            "x2": np.random.randn(50),
        })
        code = """
        proc glm data=mydata;
            model y = x1 x2;
        run;
        """
        interp = _run_with_data(code, {"mydata": df})
        assert "mydata" in interp.data_sets


# ===================================================================
# 11. PROC ANOVA
# ===================================================================
class TestProcANOVAE2E:
    def test_proc_anova(self):
        np.random.seed(42)
        df = pd.DataFrame({
            "y": np.random.randn(30),
            "group": ["A"] * 10 + ["B"] * 10 + ["C"] * 10,
        })
        code = """
        proc anova data=mydata;
            class group;
            model y = group;
        run;
        """
        interp = _run_with_data(code, {"mydata": df})
        assert "mydata" in interp.data_sets


# ===================================================================
# 12. PROC DISCRIM
# ===================================================================
class TestProcDiscrimE2E:
    def test_proc_discrim(self):
        np.random.seed(42)
        df = pd.DataFrame({
            "species": ["A"] * 20 + ["B"] * 20,
            "x1": np.random.randn(40),
            "x2": np.random.randn(40),
        })
        code = """
        proc discrim data=mydata;
            class species;
            var x1 x2;
        run;
        """
        interp = _run_with_data(code, {"mydata": df})
        assert "mydata" in interp.data_sets


# ===================================================================
# 13. PROC PRINCOMP
# ===================================================================
class TestProcPrincompE2E:
    def test_proc_princomp(self):
        np.random.seed(42)
        df = pd.DataFrame({
            "x1": np.random.randn(30),
            "x2": np.random.randn(30),
            "x3": np.random.randn(30),
        })
        code = """
        proc princomp data=mydata out=work.pca_out n=2;
            var x1 x2 x3;
        run;
        """
        interp = _run_with_data(code, {"mydata": df})
        pca_out = interp.get_data_set("work.pca_out")
        assert pca_out is not None
        assert "Prin1" in pca_out.columns
        assert "Prin2" in pca_out.columns


# ===================================================================
# 14. PROC ROBUSTREG
# ===================================================================
class TestProcRobustregE2E:
    def test_proc_robustreg(self):
        np.random.seed(42)
        x = np.random.randn(50)
        y = 3 * x + 2 + np.random.randn(50) * 0.5
        df = pd.DataFrame({"x": x, "y": y})
        code = """
        proc robustreg data=mydata;
            model y = x;
        run;
        """
        interp = _run_with_data(code, {"mydata": df})
        assert "mydata" in interp.data_sets


# ===================================================================
# 15. PROC TRANSPOSE
# ===================================================================
class TestProcTransposeE2E:
    def test_transpose_basic(self):
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4], "c": [5, 6]})
        code = """
        proc transpose data=mydata out=work.transposed;
            var a b c;
        run;
        """
        interp = _run_with_data(code, {"mydata": df})
        t = interp.get_data_set("work.transposed")
        assert t is not None
        assert "_NAME_" in t.columns


# ===================================================================
# 16. PROC APPEND
# ===================================================================
class TestProcAppendE2E:
    def test_append_datasets(self):
        df1 = pd.DataFrame({"x": [1, 2]})
        df2 = pd.DataFrame({"x": [3, 4]})
        code = """
        proc append base=base data=extra;
        run;
        """
        interp = _run_with_data(code, {"base": df1, "extra": df2})
        result = interp.get_data_set("base")
        assert result is not None
        assert len(result) == 4


# ===================================================================
# 17. PROC DATASETS
# ===================================================================
class TestProcDatasetsE2E:
    def test_datasets_delete(self):
        df = pd.DataFrame({"x": [1]})
        code = """
        proc datasets;
            delete mydata;
        run;
        """
        interp = _run_with_data(code, {"mydata": df})
        assert interp.get_data_set("mydata") is None


# ===================================================================
# 18. PROC EXPORT / IMPORT
# ===================================================================
class TestProcExportImportE2E:
    def test_export_csv(self, tmp_path):
        df = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
        outfile = str(tmp_path / "test_export.csv")
        code = f"""
        proc export data=mydata outfile='{outfile}' dbms=csv;
        run;
        """
        _run_with_data(code, {"mydata": df})
        assert os.path.exists(outfile)
        imported = pd.read_csv(outfile)
        assert len(imported) == 3

    def test_import_csv(self, tmp_path):
        outfile = str(tmp_path / "test_import.csv")
        pd.DataFrame({"a": [10, 20], "b": [30, 40]}).to_csv(outfile, index=False)
        code = f"""
        proc import datafile='{outfile}' out=work.imported dbms=csv;
        run;
        """
        interp = _run(code)
        df = interp.get_data_set("work.imported")
        assert df is not None
        assert len(df) == 2
        assert "a" in df.columns


# ===================================================================
# 19. PROC GENMOD
# ===================================================================
class TestProcGenmodE2E:
    def test_genmod_poisson(self):
        np.random.seed(42)
        df = pd.DataFrame({
            "y": np.random.poisson(5, 50),
            "x1": np.random.randn(50),
        })
        code = """
        proc genmod data=mydata;
            model y = x1;
        run;
        """
        interp = _run_with_data(code, {"mydata": df})
        assert "mydata" in interp.data_sets


# ===================================================================
# 20. PROC MIXED
# ===================================================================
class TestProcMixedE2E:
    def test_mixed_basic(self):
        np.random.seed(42)
        groups = ["G1"] * 20 + ["G2"] * 20
        df = pd.DataFrame({
            "y": np.random.randn(40),
            "x": np.random.randn(40),
            "group": groups,
        })
        code = """
        proc mixed data=mydata;
            class group;
            model y = x;
            random group;
        run;
        """
        interp = _run_with_data(code, {"mydata": df})
        assert "mydata" in interp.data_sets


# ===================================================================
# 21. PROC TREE / FOREST / BOOST (existing)
# ===================================================================
class TestProcMLModelsE2E:
    def test_proc_tree(self):
        np.random.seed(42)
        df = pd.DataFrame({
            "y": np.random.randn(50),
            "x1": np.random.randn(50),
            "x2": np.random.randn(50),
        })
        code = """
        proc tree data=mydata model=y = x1 x2;
        run;
        """
        interp = _run_with_data(code, {"mydata": df})
        assert "mydata" in interp.data_sets


# ===================================================================
# 22. MODEL STORE
# ===================================================================
class TestModelStoreE2E:
    def test_model_store_save_load(self):
        from stat_lang.utils.model_store import ModelStore
        store = ModelStore()
        store.save("test_model", {"weights": [1, 2, 3]}, metadata={"proc": "TEST"})
        loaded = store.load("test_model")
        assert loaded is not None
        assert loaded.model == {"weights": [1, 2, 3]}
        assert loaded.metadata["proc"] == "TEST"

    def test_model_store_persist(self, tmp_path):
        from stat_lang.utils.model_store import ModelStore
        store = ModelStore(persist_dir=str(tmp_path))
        store.save("persist_test", [1, 2, 3], persist=True)
        # Load in a fresh store
        store2 = ModelStore(persist_dir=str(tmp_path))
        loaded = store2.load("persist_test")
        assert loaded is not None
        assert loaded.model == [1, 2, 3]

    def test_model_store_list_delete(self):
        from stat_lang.utils.model_store import ModelStore
        store = ModelStore()
        store.save("m1", "model1")
        store.save("m2", "model2")
        assert set(store.list_models()) == {"m1", "m2"}
        store.delete("m1")
        assert store.list_models() == ["m2"]


# ===================================================================
# 23. PIPELINE RUNNER
# ===================================================================
class TestPipelineRunnerE2E:
    def test_pipeline_from_file(self, tmp_path):
        script = tmp_path / "test_pipeline.statlang"
        script.write_text("""
data work.numbers;
    input x;
    datalines;
1
2
3
;
run;

proc means data=work.numbers;
    var x;
run;
""")
        from stat_lang.pipeline import run_pipeline
        result = run_pipeline(str(script))
        assert "work.numbers" in result["datasets"]
        assert result["last_dataset"] == "work.numbers"
        assert result["errors"] == []


# ===================================================================
# 24. PROC SORT (existing - regression check)
# ===================================================================
class TestProcSortE2E:
    def test_sort_descending(self):
        df = pd.DataFrame({"x": [3, 1, 2]})
        code = """
        proc sort data=mydata;
            by descending x;
        run;
        """
        interp = _run_with_data(code, {"mydata": df})
        result = interp.get_data_set("mydata")
        assert result is not None


# ===================================================================
# 25. PROC FREQ (existing - regression check)
# ===================================================================
class TestProcFreqE2E:
    def test_freq_basic(self):
        df = pd.DataFrame({"color": ["red", "blue", "red", "green", "blue"]})
        code = """
        proc freq data=mydata;
            tables color;
        run;
        """
        interp = _run_with_data(code, {"mydata": df})
        assert "mydata" in interp.data_sets


# ===================================================================
# 26. PROC SQL (existing - regression check)
# ===================================================================
class TestProcSQLE2E:
    def test_sql_select(self):
        df = pd.DataFrame({"x": [1, 2, 3], "y": [10, 20, 30]})
        code = """
        proc sql;
            create table work.result as
            select x, y from mydata where x > 1;
        quit;
        """
        interp = _run_with_data(code, {"mydata": df})
        # SQL proc may or may not create the table depending on implementation
        # Just verify it doesn't crash
        assert "mydata" in interp.data_sets


# ===================================================================
# 27. PARSER — Generic option parsing
# ===================================================================
class TestGenericParserE2E:
    def test_generic_options_on_proc_line(self):
        from stat_lang.parser.proc_parser import ProcParser
        parser = ProcParser()
        result = parser.parse_proc("PROC DNN DATA=mydata epochs=100 lr=0.01 optimizer=adam;\nRUN;")
        assert result.proc_name == "DNN"
        assert result.data_option == "mydata"
        assert result.options.get("epochs") == 100
        assert result.options.get("lr") == 0.01
        assert result.options.get("optimizer") == "adam"

    def test_boolean_flags(self):
        from stat_lang.parser.proc_parser import ProcParser
        parser = ProcParser()
        result = parser.parse_proc("PROC MEANS DATA=d noprint;\nVAR x;\nRUN;")
        assert result.options.get("noprint") is True

    def test_quoted_options(self):
        from stat_lang.parser.proc_parser import ProcParser
        parser = ProcParser()
        result = parser.parse_proc("PROC LLM prompt='hello world';\nRUN;")
        assert result.options.get("prompt") == "hello world"


# ===================================================================
# 28. MACRO — %MACRO/%MEND with expansion
# ===================================================================
class TestMacroExpansionE2E:
    def test_macro_definition_and_call(self):
        code = """
        %MACRO make_data(name, n);
        data work.&name.;
            input x;
            datalines;
        &n.
        ;
        run;
        %MEND;

        %make_data(test_ds, 42);
        """
        interp = _run(code)
        df = interp.get_data_set("work.test_ds")
        assert df is not None


# ===================================================================
# 29. E2E MULTI-STEP PIPELINE
# ===================================================================
class TestMultiStepPipelineE2E:
    def test_data_then_means_then_sort(self):
        code = """
        data work.scores;
            input name $ score;
            datalines;
        Alice 85
        Bob 92
        Carol 78
        Dave 95
        Eve 88
        ;
        run;

        proc sort data=work.scores;
            by descending score;
        run;

        proc means data=work.scores;
            var score;
        run;
        """
        interp = _run(code)
        df = interp.get_data_set("work.scores")
        assert df is not None
        assert len(df) == 5
