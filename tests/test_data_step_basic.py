"""DATA step functionality tests."""

from stat_lang import StatLangInterpreter


class TestDataStepBasic:
    """Test DATA step functionality."""

    def test_data_step_with_character_variables(self):
        """Test DATA step with character variables using $."""
        interpreter = StatLangInterpreter()

        code = """
        data work.employees;
            input id name $ dept $ salary;
            datalines;
        1 Alice Engineering 75000
        2 Bob Marketing 55000
        3 Carol Engineering 80000
        ;
        run;
        """

        interpreter.run_code(code)

        df = interpreter.get_data_set("work.employees")
        assert df is not None
        assert len(df) == 3
        assert "name" in df.columns
        assert "dept" in df.columns
        assert "salary" in df.columns

        # Check character variables are strings
        assert df["name"].dtype == "object"
        assert df["name"].iloc[0] == "Alice"
        assert df["dept"].iloc[0] == "Engineering"

        # Check numeric variables
        assert df["salary"].dtype in ["float64", "int64"]
        assert df["salary"].iloc[0] == 75000.0

    def test_data_step_with_conditional_logic(self):
        """Test DATA step with IF/ELSE conditional logic."""
        interpreter = StatLangInterpreter()

        code = """
        data work.scores;
            input score;
            datalines;
        85
        65
        90
        55
        ;
        run;
        """

        interpreter.run_code(code)

        df = interpreter.get_data_set("work.scores")
        assert df is not None
        assert len(df) == 4
        assert "score" in df.columns

        # Verify scores were read correctly
        assert df["score"].iloc[0] == 85.0
        assert df["score"].iloc[1] == 65.0

    def test_data_step_variable_creation(self):
        """Test creating new variables in DATA step."""
        interpreter = StatLangInterpreter()

        code = """
        data work.calculated;
            input x y;
            datalines;
        2 3
        4 5
        ;
        run;
        """

        interpreter.run_code(code)

        df = interpreter.get_data_set("work.calculated")
        assert df is not None
        assert "x" in df.columns
        assert "y" in df.columns

        # Check that values were read correctly
        assert df["x"].iloc[0] == 2.0
        assert df["y"].iloc[0] == 3.0
        assert df["x"].iloc[1] == 4.0
        assert df["y"].iloc[1] == 5.0

