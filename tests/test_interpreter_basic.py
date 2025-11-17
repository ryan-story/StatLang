"""Basic interpreter functionality tests."""

from stat_lang import StatLangInterpreter


class TestInterpreterBasic:
    """Test basic interpreter functionality."""

    def test_interpreter_initialization(self):
        """Test that interpreter can be instantiated."""
        interpreter = StatLangInterpreter()
        assert interpreter is not None
        assert interpreter.data_sets == {}
        assert interpreter.list_data_sets() == []

    def test_interpreter_run_simple_data_step(self):
        """Test running a simple DATA step with DATALINES."""
        interpreter = StatLangInterpreter()

        code = """
        data work.test;
            input x y;
            datalines;
        1 10
        2 20
        3 30
        ;
        run;
        """

        interpreter.run_code(code)

        # Check that dataset was created
        datasets = interpreter.list_data_sets()
        assert "work.test" in datasets

        # Check dataset contents
        df = interpreter.get_data_set("work.test")
        assert df is not None
        assert len(df) == 3
        assert "x" in df.columns
        assert "y" in df.columns
        assert df["x"].tolist() == [1.0, 2.0, 3.0]
        assert df["y"].tolist() == [10.0, 20.0, 30.0]

    def test_interpreter_get_data_set_nonexistent(self):
        """Test getting a non-existent dataset returns None."""
        interpreter = StatLangInterpreter()
        result = interpreter.get_data_set("nonexistent")
        assert result is None

    def test_interpreter_clear_workspace(self):
        """Test clearing the workspace removes all datasets."""
        interpreter = StatLangInterpreter()

        # Create a dataset
        code = """
        data work.test;
            input x;
            datalines;
        1
        2
        ;
        run;
        """
        interpreter.run_code(code)
        assert len(interpreter.list_data_sets()) > 0

        # Clear workspace
        interpreter.clear_workspace()
        assert len(interpreter.list_data_sets()) == 0
        assert interpreter.get_data_set("work.test") is None

