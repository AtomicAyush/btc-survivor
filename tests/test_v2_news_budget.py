"""The REST budget of PLAN_V2 section 7.2 on a fake clock. Every host group stays inside its rolling limits and
spacing however hard its channels push, a 429 or 403 pauses the whole group as the plan says, and a failed poll
backs off from its cadence to a cap with no jitter."""

import asyncio

import pytest

from btc_survivor.v2.config import load_news_config
from btc_survivor.v2.news import budget, http
from btc_survivor.v2.news.budget import BLS, FED, FINNHUB, SEC, Budget, HostRules, Limit
from helpers import ROOT

CONFIG = load_news_config(ROOT / "configs" / "v2" / "news.toml")
RULES = budget.rules_from_config(CONFIG)


class FakeClock:
    """Integer milliseconds that move only when a test or a sleeping caller moves them."""

    def __init__(self, now=1_000_000):
        self.now = now

    def __call__(self):
        return self.now

    async def sleep(self, seconds):
        self.now += round(seconds * 1000)
        await asyncio.sleep(0)


def fake_budget(rules=RULES):
    clock = FakeClock()
    return Budget(rules, clock, clock.sleep), clock


def greedy_calls(group, minutes):
    """The call times of one channel that asks again the moment it may, for some minutes of fake time."""
    plan, clock = fake_budget()
    end = clock.now + minutes * 60_000
    times = []
    while clock.now < end:
        wait = plan.try_acquire(group)
        if wait:
            clock.now += wait
        else:
            times.append(clock.now)
    return times


def most_in_any_window(times, window_ms):
    """The largest number of calls in any half open window of window_ms, which must start at some call."""
    return max(sum(1 for t in times if start <= t < start + window_ms) for start in times)


def test_the_rules_are_those_of_the_news_config():
    assert RULES[FINNHUB] == HostRules((Limit(30, 60_000),), 1100, 60_000, frozenset({429}))
    assert RULES[SEC] == HostRules((Limit(2, 1000), Limit(60, 60_000)), 0, 900_000, frozenset({403, 429}))
    assert RULES[FED] == RULES[BLS] == HostRules((Limit(10, 60_000),), 0, 900_000, frozenset({403, 429}))
    assert set(RULES) == set(budget.GROUPS)


def test_finnhub_makes_at_most_30_calls_in_any_rolling_minute_at_least_1_1_seconds_apart():
    times = greedy_calls(FINNHUB, 10)
    assert most_in_any_window(times, 60_000) == 30
    assert min(b - a for a, b in zip(times, times[1:])) == 1100
    assert len(times) == 300


def test_sec_makes_at_most_2_calls_a_second_and_60_a_minute():
    times = greedy_calls(SEC, 5)
    assert most_in_any_window(times, 1000) == 2
    assert most_in_any_window(times, 60_000) == 60
    assert len(times) == 300


@pytest.mark.parametrize("group", [FED, BLS])
def test_fed_and_bls_make_at_most_10_calls_a_minute(group):
    times = greedy_calls(group, 5)
    assert most_in_any_window(times, 60_000) == 10
    assert len(times) == 50


def test_both_sec_hosts_share_one_budget_and_fed_and_bls_have_their_own():
    assert {http.host_group("https://www.sec.gov/cgi-bin/browse-edgar?action=getcurrent"),
            http.host_group("https://data.sec.gov/submissions/CIK0000320193.json")} == {SEC}
    plan, clock = fake_budget()
    assert plan.try_acquire(SEC) == 0 and plan.try_acquire(SEC) == 0
    assert plan.try_acquire(SEC) == 1000
    for _ in range(10):
        assert plan.try_acquire(FED) == 0
    assert plan.wait_ms(FED) == 60_000
    assert plan.wait_ms(BLS) == 0 and plan.wait_ms(FINNHUB) == 0


def test_a_refused_call_records_nothing():
    plan, clock = fake_budget()
    assert plan.try_acquire(FINNHUB) == 0
    clock.now += 500
    for _ in range(5):
        assert plan.try_acquire(FINNHUB) == 600
    clock.now += 600
    assert plan.try_acquire(FINNHUB) == 0


def test_many_channels_pushing_together_stay_inside_the_limits():
    plan, clock = fake_budget()
    times = []

    async def channel():
        for _ in range(20):
            await plan.acquire(FINNHUB)
            times.append(clock.now)

    async def main():
        await asyncio.gather(*(channel() for _ in range(8)))

    asyncio.run(main())
    times.sort()
    assert len(times) == 160
    assert most_in_any_window(times, 60_000) <= 30
    assert min(b - a for a, b in zip(times, times[1:])) >= 1100


def test_acquire_returns_the_budget_wait():
    plan, clock = fake_budget()
    assert asyncio.run(plan.acquire(FINNHUB)) == 0
    assert asyncio.run(plan.acquire(FINNHUB)) == 1100
    clock.now += 5000
    assert asyncio.run(plan.acquire(FINNHUB)) == 0


def test_a_finnhub_429_pauses_every_finnhub_channel_for_60_seconds_and_a_403_does_not():
    plan, clock = fake_budget()
    assert plan.report(FINNHUB, 403) is False and plan.paused_ms(FINNHUB) == 0
    assert plan.report(FINNHUB, 200) is False and plan.report(FINNHUB, None) is False
    assert plan.report(FINNHUB, 429) is True
    assert plan.paused_ms(FINNHUB) == 60_000 and plan.try_acquire(FINNHUB) == 60_000
    assert asyncio.run(plan.acquire(FINNHUB)) == 60_000
    assert plan.paused_ms(FINNHUB) == 0


@pytest.mark.parametrize("status", [403, 429])
def test_an_sec_block_pauses_every_sec_channel_for_900_seconds(status):
    plan, clock = fake_budget()
    assert plan.report(SEC, status) is True
    assert plan.wait_ms(SEC) == 900_000
    clock.now += 899_999
    assert plan.wait_ms(SEC) == 1
    clock.now += 1
    assert plan.try_acquire(SEC) == 0


@pytest.mark.parametrize("group, other", [(FED, BLS), (BLS, FED)])
@pytest.mark.parametrize("status", [403, 429])
def test_a_fed_or_bls_block_pauses_that_host_alone(group, other, status):
    plan, clock = fake_budget()
    assert plan.report(group, status) is True
    assert plan.wait_ms(group) == 900_000
    assert plan.wait_ms(other) == 0 and plan.wait_ms(SEC) == 0 and plan.wait_ms(FINNHUB) == 0


def test_a_pause_runs_from_the_latest_report():
    plan, clock = fake_budget()
    plan.report(FED, 403)
    clock.now += 100_000
    plan.report(FED, 429)
    assert plan.paused_ms(FED) == 900_000
    plan.report(FINNHUB, 429)
    clock.now += 30_000
    plan.report(FINNHUB, 429)
    assert plan.paused_ms(FINNHUB) == 60_000


@pytest.mark.parametrize("cadence, delays", [
    (60, [60, 120, 240, 480, 900, 900, 900]),
    (15, [15, 30, 60, 120, 240, 480, 900]),
    (30, [30, 60, 120, 240, 480, 900, 900]),
    (120, [120, 240, 480, 900, 900, 900, 900]),
    (600, [600, 900, 900, 900, 900, 900, 900]),
    (900, [900] * 7),
    (3600, [3600] * 7),
    (7200, [7200] * 7),
])
def test_a_failed_poll_backs_off_from_its_cadence_to_the_cap_with_no_jitter(cadence, delays):
    cap = CONFIG.news.failure_backoff_cap_seconds
    assert cap == 900
    assert [budget.failure_delay_s(cadence, failures, cap) for failures in range(7)] == delays
    assert [budget.failure_delay_s(cadence, failures, cap) for failures in range(7)] == delays


def test_the_backoff_survives_any_number_of_failures():
    assert budget.failure_delay_s(60, 10_000, 900) == 900
    assert budget.failure_delay_s(60, -1, 900) == 60
