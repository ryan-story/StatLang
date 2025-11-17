"""PROC SORT functionality tests."""

from stat_lang import StatLangInterpreter


class TestProcSortBasic:
    """Test PROC SORT functionality."""

    def test_proc_sort_ascending(self):
        """Test PROC SORT sorts data in ascending order."""
        interpreter = StatLangInterpreter()

        code = """
        data work.test;
            input value;
            datalines;
        30
        10
        20
        ;
        run;

        proc sort data=work.test;
            by value;
        run;
        """

        interpreter.run_code(code)

        # Verify dataset is sorted
        df = interpreter.get_data_set("work.test")
        assert df is not None
        assert len(df) == 3

        # Check that values are sorted
        values = df["value"].tolist()
        assert values == sorted(values)

    def test_proc_sort_descending(self):
        """Test PROC SORT with DESCENDING option."""
        interpreter = StatLangInterpreter()

        code = """
        data work.test;
            input value;
            datalines;
        10
        30
        20
        ;
        run;

        proc sort data=work.test;
            by descending value;
        run;
        """

        interpreter.run_code(code)

        # Verify dataset is sorted descending
        df = interpreter.get_data_set("work.test")
        assert df is not None

        # Check that values are sorted descending
        values = df["value"].tolist()
        assert values == sorted(values, reverse=True)

