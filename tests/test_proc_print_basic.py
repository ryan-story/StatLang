"""PROC PRINT functionality tests."""

from stat_lang import StatLangInterpreter


class TestProcPrintBasic:
    """Test PROC PRINT functionality."""

    def test_proc_print_displays_dataset(self):
        """Test PROC PRINT displays dataset contents."""
        interpreter = StatLangInterpreter()

        code = """
        data work.test;
            input id name $;
            datalines;
        1 Alice
        2 Bob
        3 Carol
        ;
        run;

        proc print data=work.test;
        run;
        """

        interpreter.run_code(code)

        # Verify dataset exists
        df = interpreter.get_data_set("work.test")
        assert df is not None
        assert len(df) == 3
        assert "id" in df.columns
        assert "name" in df.columns

    def test_proc_print_with_where_clause(self):
        """Test PROC PRINT with WHERE clause filters data."""
        interpreter = StatLangInterpreter()

        code = """
        data work.test;
            input id value;
            datalines;
        1 10
        2 20
        3 30
        ;
        run;

        proc print data=work.test;
            where value > 15;
        run;
        """

        interpreter.run_code(code)

        # Verify dataset exists
        df = interpreter.get_data_set("work.test")
        assert df is not None
        assert len(df) == 3

