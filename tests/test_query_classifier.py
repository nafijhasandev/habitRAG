"""Tests for the query classifier module."""

from core.query_classifier import classify_query, get_prompt_template


class TestClassifyQuery:
    """Test query classification into factual/analytical/comparison/summary."""

    def test_comparison_keywords(self):
        q_type, _ = classify_query("Compare habit stacking vs temptation bundling")
        assert q_type == "comparison"

    def test_comparison_difference(self):
        q_type, _ = classify_query("What is the difference between goals and systems?")
        assert q_type == "comparison"

    def test_comparison_contrast(self):
        q_type, _ = classify_query("Contrast identity-based habits with outcome-based habits")
        assert q_type == "comparison"

    def test_summary_keywords(self):
        q_type, _ = classify_query("Summarize chapter 3")
        assert q_type == "summary"

    def test_summary_overview(self):
        q_type, _ = classify_query("Give me an overview of the book")
        assert q_type == "summary"

    def test_analytical_why(self):
        q_type, _ = classify_query("Why do habits plateau after initial progress?")
        assert q_type == "analytical"

    def test_analytical_how_does(self):
        q_type, _ = classify_query("How does environment design help build habits?")
        assert q_type == "analytical"

    def test_analytical_explain(self):
        q_type, _ = classify_query("Explain the plateau of latent potential")
        assert q_type == "analytical"

    def test_factual_default(self):
        q_type, _ = classify_query("What is the 2-minute rule?")
        assert q_type == "factual"

    def test_factual_simple(self):
        q_type, _ = classify_query("List the four laws")
        assert q_type == "factual"

    def test_returns_reason(self):
        _, reason = classify_query("Compare goals and systems")
        assert "compare" in reason.lower()

    def test_empty_query(self):
        q_type, _ = classify_query("")
        assert q_type == "factual"


class TestGetPromptTemplate:
    """Test prompt template retrieval."""

    def test_valid_types(self):
        for q_type in ("factual", "analytical", "comparison", "summary"):
            template = get_prompt_template(q_type)
            assert isinstance(template, str)
            assert len(template) > 50
            assert "Atomic Habits" in template
            assert "Markdown bold" in template

    def test_invalid_type_raises(self):
        try:
            get_prompt_template("invalid")
            assert False, "Should have raised ValueError"
        except ValueError:
            pass
