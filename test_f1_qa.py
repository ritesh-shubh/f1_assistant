"""
Tests for the F1 Q&A pattern matching engine.
"""

import pytest
import re
from f1_qa import load_data, query_data


@pytest.fixture(scope="module")
def dfs():
    return load_data()


# ── Winning margin ──────────────────────────────────────────────────────

class TestWinningMargin:
    def test_biggest_winning_margin(self, dfs):
        result = query_data("who had the biggest winning margin in f1", dfs)
        assert result is not None
        assert "biggest winning margin" in result.lower() or "gap" in result.lower()
        assert "Grand Prix" in result

    def test_largest_margin_of_victory(self, dfs):
        result = query_data("largest margin of victory", dfs)
        assert result is not None
        assert "margin" in result.lower() or "gap" in result.lower()


# ── Consecutive wins at specific GP ─────────────────────────────────────

class TestConsecutiveGPWins:
    def test_monaco_twice_in_a_row(self, dfs):
        result = query_data("has anyone won monaco grand prix twice in a row?", dfs)
        assert result is not None
        assert "Monaco" in result

    def test_consecutive_race_wins(self, dfs):
        result = query_data("has anyone won more than 5 races in a row", dfs)
        assert result is not None
        assert "consecutive" in result.lower() or "in a row" in result.lower()


# ── Team win percentage ─────────────────────────────────────────────────

class TestTeamWinPercentage:
    def test_highest_win_pct(self, dfs):
        result = query_data("which team has the highest win% in an year?", dfs)
        assert result is not None
        assert "%" in result or "win rate" in result.lower()


# ── Multiple pitstops ───────────────────────────────────────────────────

class TestMultiplePitstops:
    def test_pitted_more_than_3(self, dfs):
        result = query_data("has anyone pitted more than 3 times in a single race?", dfs)
        assert result is not None
        assert "pitstop" in result.lower() or "stop" in result.lower()


# ── Low grid position wins ──────────────────────────────────────────────

class TestLowGridWins:
    def test_won_from_15th(self, dfs):
        result = query_data("who has won a race starting from 15th or lower?", dfs)
        assert result is not None
        assert "P15" in result or "starting" in result.lower()


# ── Last to first (including typo) ──────────────────────────────────────

class TestLastToFirst:
    def test_last_to_first(self, dfs):
        result = query_data("has anyone successfully completed last to first drive in f1", dfs)
        assert result is not None
        assert "lowest grid position" in result.lower() or "starting from" in result.lower()

    def test_last_to_first_typo(self, dfs):
        """The misspelling 'lsat' should still trigger the last-to-first pattern."""
        result = query_data("has anyone succesfully completed lsat to first drive in f1", dfs)
        assert result is not None
        assert "lowest grid position" in result.lower() or "starting from" in result.lower()
        # Must NOT return the "first ever F1 race" answer
        assert "first ever" not in result.lower()


# ── Driver wins at specific GP ──────────────────────────────────────────

class TestDriverGPWins:
    def test_senna_monaco_wins(self, dfs):
        result = query_data("how many monaco grand prix did senna win", dfs)
        assert result is not None
        assert "Monaco" in result
        assert "Senna" in result
        # Should return GP-specific wins, not total wins
        assert "in total" not in result.lower()


# ── Last win for a driver ───────────────────────────────────────────────

class TestLastWin:
    def test_last_senna_win(self, dfs):
        result = query_data("when was the last senna win", dfs)
        assert result is not None
        assert "Senna" in result
        assert "1993" in result
