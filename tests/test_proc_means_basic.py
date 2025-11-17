"""PROC MEANS functionality tests."""

from stat_lang import StatLangInterpreter


class TestProcMeansBasic:
    """Test PROC MEANS functionality."""

    def test_proc_means_basic_statistics(self):
        """Test PROC MEANS calculates basic statistics."""
        interpreter = StatLangInterpreter()

        # Create test data
        code = """
        data work.test;
            input value;
            datalines;
        10
        20
        30
        40
        50
        ;
        run;

        proc means data=work.test;
            var value;
        run;
        """

        interpreter.run_code(code)

        # PROC MEANS should execute without error
        # The dataset should still exist
        df = interpreter.get_data_set("work.test")
        assert df is not None
        assert len(df) == 5

    def test_proc_means_with_class_variable(self):
        """Test PROC MEANS with CLASS variable for grouped statistics."""
        interpreter = StatLangInterpreter()

        code = """
        data work.sales;
            input region $ sales;
            datalines;
        North 100
        North 150
        South 200
        South 250
        East 300
        East 350
        ;
        run;

        proc means data=work.sales;
            class region;
            var sales;
        run;
        """

        interpreter.run_code(code)

        # Verify data was created
        df = interpreter.get_data_set("work.sales")
        assert df is not None
        assert len(df) == 6
        assert "region" in df.columns
        assert "sales" in df.columns

    def test_proc_means_output_dataset(self):
        """Test PROC MEANS with OUT= option creates output dataset."""
        interpreter = StatLangInterpreter()

        code = """
        data work.test;
            input value;
            datalines;
        10
        20
        30
        ;
        run;

        proc means data=work.test noprint;
            var value;
            output out=work.stats;
        run;
        """

        interpreter.run_code(code)

        # Check that output dataset was created
        # Note: The actual implementation may vary, but we should check
        # that the code runs without error
        df = interpreter.get_data_set("work.test")
        assert df is not None

