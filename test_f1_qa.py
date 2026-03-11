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


# ── Team / driver with exactly N wins ───────────────────────────────────

class TestExactWinCount:
    def test_team_only_one_win(self, dfs):
        result = query_data("which team has only one win", dfs)
        assert result is not None
        assert "exactly 1 race win" in result.lower()

    def test_driver_only_one_win(self, dfs):
        result = query_data("which driver has only one win", dfs)
        assert result is not None
        assert "exactly 1 race win" in result.lower()

    def test_team_won_only_once(self, dfs):
        result = query_data("which team won only once", dfs)
        assert result is not None
        assert "exactly 1 race win" in result.lower()

    def test_team_exactly_two_wins(self, dfs):
        result = query_data("which team has exactly 2 wins", dfs)
        assert result is not None
        assert "exactly 2 race wins" in result.lower()


# ── Never won championship but won GP ──────────────────────────────────

class TestNeverWonChampionship:
    def test_never_champ_but_won_monaco(self, dfs):
        result = query_data("who has never won a championship but has won the monaco grand prix", dfs)
        assert result is not None
        assert "Monaco" in result
        assert "never won a World Championship" in result

    def test_never_champ_but_won_race(self, dfs):
        result = query_data("who has never won a championship but won a race", dfs)
        assert result is not None
        assert "never won a World Championship" in result


# ── Most wins / poles / podiums without a win ──────────────────────────

class TestWithoutAWin:
    def test_most_races_without_win(self, dfs):
        result = query_data("who has the most races without a win", dfs)
        assert result is not None
        assert "without ever winning" in result.lower()
        assert "Andrea de Cesaris" in result

    def test_most_poles_without_win(self, dfs):
        result = query_data("most poles without a win", dfs)
        assert result is not None
        assert "without ever winning" in result.lower()

    def test_most_podiums_without_win(self, dfs):
        result = query_data("most podiums without ever winning a race", dfs)
        assert result is not None
        assert "without ever winning" in result.lower()


# ── Most wins without a championship ────────────────────────────────────

class TestWinsWithoutChampionship:
    def test_most_wins_without_championship(self, dfs):
        result = query_data("who has the most wins without winning the championship", dfs)
        assert result is not None
        assert "Stirling Moss" in result
        assert "without" in result.lower()

    def test_most_wins_but_no_championship(self, dfs):
        result = query_data("most wins but no championship", dfs)
        assert result is not None
        assert "Stirling Moss" in result


# ── Championship without winning a race ────────────────────────────────

class TestChampionshipWithoutRaceWin:
    def test_championship_without_race_win(self, dfs):
        result = query_data("has anyone won the championship without winning a race", dfs)
        assert result is not None
        assert "champion" in result.lower() or "Champion" in result


# ── Won on debut ────────────────────────────────────────────────────────

class TestDebutWin:
    def test_won_on_debut(self, dfs):
        result = query_data("who won on their debut", dfs)
        assert result is not None
        assert "debut" in result.lower()

    def test_won_first_race(self, dfs):
        result = query_data("which driver won their first race", dfs)
        assert result is not None
        assert "debut" in result.lower()


# ── Won for multiple teams ─────────────────────────────────────────────

class TestMultipleTeams:
    def test_won_for_multiple_teams(self, dfs):
        result = query_data("which drivers have won for multiple teams", dfs)
        assert result is not None
        assert "teams" in result.lower()

    def test_how_many_teams_driver_won_for(self, dfs):
        result = query_data("how many teams has hamilton won for", dfs)
        assert result is not None
        assert "Hamilton" in result
        assert "different team" in result.lower()


# ── Won both X and Y GPs ───────────────────────────────────────────────

class TestWonBothGPs:
    def test_won_both_monaco_and_british(self, dfs):
        result = query_data("who has won both the monaco and british grand prix", dfs)
        assert result is not None
        assert "Monaco" in result
        assert "Great Britain" in result


# ── Only driver to win GP ──────────────────────────────────────────────

class TestOnlyWinner:
    def test_only_driver_to_win_miami(self, dfs):
        result = query_data("who is the only driver to win the miami grand prix", dfs)
        assert result is not None
        assert "only driver" in result.lower() or "only" in result.lower()
        assert "Miami" in result


# ── All winners in a season ────────────────────────────────────────────

class TestSeasonWinners:
    def test_winners_in_2020(self, dfs):
        result = query_data("which drivers won a race in 2020", dfs)
        assert result is not None
        assert "2020" in result
        assert "Hamilton" in result

    def test_all_winners_2019(self, dfs):
        result = query_data("all winners in 2019", dfs)
        assert result is not None
        assert "2019" in result


# ── Top N lists ────────────────────────────────────────────────────────

class TestTopNLists:
    def test_top_10_podiums(self, dfs):
        result = query_data("top 10 driver in terms of most podiums", dfs)
        assert result is not None
        assert "Top 10" in result
        assert "Hamilton" in result
        assert "\n" in result  # multi-line list

    def test_top_10_poles(self, dfs):
        result = query_data("top 10 drivers in terms of total pole positions", dfs)
        assert result is not None
        assert "Top 10" in result
        assert "\n" in result

    def test_top_10_wins(self, dfs):
        result = query_data("top 10 drivers by wins", dfs)
        assert result is not None
        assert "Top 10" in result
        assert "\n" in result

    def test_top_5_fastest_laps(self, dfs):
        result = query_data("top 5 drivers with most fastest laps", dfs)
        assert result is not None
        assert "Top 5" in result


# ── Finished outside points ────────────────────────────────────────────

class TestOutsidePoints:
    def test_hamilton_outside_points(self, dfs):
        result = query_data("how many races did hamilton finish outside points", dfs)
        assert result is not None
        assert "Hamilton" in result
        assert "outside the points" in result.lower()
        m = re.search(r'(\d+) times', result)
        assert m is not None, "Expected 'N times' in result"
        assert int(m.group(1)) > 0


# ── Least points with race threshold ───────────────────────────────────

class TestLeastPoints:
    def test_least_points_100_races(self, dfs):
        result = query_data("who has the least amount points scored having raced for more than 100 races", dfs)
        assert result is not None
        assert "least" in result.lower() or "pts" in result.lower()
        assert "races" in result.lower()


# ── Points over year range ─────────────────────────────────────────────

class TestPointsRange:
    def test_hamilton_points_2016_to_2020(self, dfs):
        result = query_data("how many points have lewis hamilton scored from 2016 to 2020", dfs)
        assert result is not None
        assert "Hamilton" in result
        assert "2016" in result
        assert "2020" in result
        assert "total points" in result.lower()


# ── Race-by-race head-to-head ──────────────────────────────────────────

class TestRaceHeadToHead:
    def test_rosberg_vs_hamilton_races(self, dfs):
        result = query_data("how many races did nico rosberg finish ahead of lewis hamilton", dfs)
        assert result is not None
        assert "Hamilton" in result
        assert "Rosberg" in result
        assert "finished ahead" in result.lower()
        # Should be race-level, not season-level
        assert "season" not in result.lower()


# ── Championship comparison ────────────────────────────────────────────

class TestChampionshipComparison:
    def test_more_championships_lewis_vs_max(self, dfs):
        result = query_data("how many more championships does lewis have more than max", dfs)
        assert result is not None
        assert "Hamilton" in result
        assert "Verstappen" in result
        assert "titles" in result.lower() or "championship" in result.lower()


# ── Q1/Q2/Q3 qualifying exits ──────────────────────────────────────────

class TestQualifyingExits:
    def test_most_q3_exits(self, dfs):
        result = query_data("who had the most Q3 exits", dfs)
        assert result is not None
        assert "Q3" in result
        # Hamilton has the most Q3 appearances in the dataset
        lines = result.strip().split('\n')
        assert "Hamilton" in lines[1]  # first entry after header

    def test_most_q1_exits(self, dfs):
        result = query_data("who had the most Q1 exits", dfs)
        assert result is not None
        assert "Q1" in result


# ── Last place finishes ────────────────────────────────────────────────

class TestLastPlaceFinishes:
    def test_most_last_finishes(self, dfs):
        result = query_data("who has the most last finishes", dfs)
        assert result is not None
        assert "last-place finishes" in result.lower()
        assert "\n" in result  # multi-line list

    def test_finished_last_most_times(self, dfs):
        result = query_data("who finished last the most times", dfs)
        assert result is not None
        assert "last-place" in result.lower()


# ── Teammate points gap ───────────────────────────────────────────────

class TestTeammateGap:
    def test_biggest_teammate_points_gap(self, dfs):
        result = query_data("which driver had the biggest points gap with their teammate", dfs)
        assert result is not None
        assert "teammate" in result.lower() or "vs" in result.lower()
        assert "pts" in result.lower()


# ── Championship deficit / comeback ────────────────────────────────────

class TestChampionshipDeficit:
    def test_largest_deficit_chased(self, dfs):
        result = query_data("what is the largest points deficit successfully chased by a driver in drivers championship", dfs)
        assert result is not None
        assert "deficit" in result.lower()
        assert "overcame" in result.lower()

    def test_biggest_comeback(self, dfs):
        result = query_data("biggest championship comeback", dfs)
        assert result is not None
        assert "deficit" in result.lower()


# ── Longest running / longest race ─────────────────────────────────────

class TestLongestRace:
    def test_longest_running_race(self, dfs):
        result = query_data("what has been the longest running race", dfs)
        assert result is not None
        assert "editions" in result.lower()
        assert "Great Britain" in result

    def test_longest_race_duration(self, dfs):
        result = query_data("which is the longest race, duration of race wise", dfs)
        assert result is not None
        assert "duration" in result.lower()
        assert "Canada" in result  # 2011 Canada GP is the longest

    def test_longest_race_ever(self, dfs):
        result = query_data("longest race ever", dfs)
        assert result is not None
        assert "duration" in result.lower()
