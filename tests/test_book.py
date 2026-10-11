import pytest

from btc_survivor.feed.book import LocalBook
from btc_survivor.types import ASK, BID


def make_book(depth=20):
    book = LocalBook(depth)
    book.apply_snapshot(
        bids=[(99.9, 1.0), (99.8, 2.0), (99.7, 5.0), (99.0, 0.0)],
        asks=[(100.1, 1.0), (100.2, 2.0), (100.3, 5.0)],
    )
    return book


def test_snapshot_exposes_best_prices_mid_and_spread():
    book = make_book()
    assert book.valid
    assert book.best_bid == 99.9 and book.best_ask == 100.1
    assert book.mid == pytest.approx(100.0)
    assert book.spread_bps == pytest.approx(20.0)
    assert book.top(BID) == [(99.9, 1.0), (99.8, 2.0), (99.7, 5.0)]
    assert book.top(ASK) == [(100.1, 1.0), (100.2, 2.0), (100.3, 5.0)]
    assert book.level_count(BID) == 3  # the zero size level was skipped


def test_changes_insert_update_and_delete_levels():
    book = make_book()
    book.apply_changes([(BID, 99.95, 0.5), (ASK, 100.1, 3.0), (ASK, 100.2, 0.0)])
    assert book.best_bid == 99.95
    assert book.top(ASK, 2) == [(100.1, 3.0), (100.3, 5.0)]
    book.apply_changes([(BID, 99.95, 0.0), (BID, 99.9, 0.0)])
    assert book.best_bid == 99.8


def test_deleting_an_unknown_level_is_ignored():
    book = make_book()
    book.apply_changes([(ASK, 123.45, 0.0), (BID, 1.0, 0.0)])
    assert book.level_count(ASK) == 3 and book.level_count(BID) == 3


def test_full_depth_is_kept_so_deeper_levels_move_into_the_top():
    book = LocalBook(depth=2)
    book.apply_snapshot(bids=[(99.0 - i, 1.0) for i in range(5)], asks=[(101.0 + i, 1.0) for i in range(5)])
    assert book.top(ASK) == [(101.0, 1.0), (102.0, 1.0)]
    book.apply_changes([(ASK, 101.0, 0.0), (ASK, 102.0, 0.0)])
    assert book.top(ASK) == [(103.0, 1.0), (104.0, 1.0)]


def test_crossed_snapshot_is_invalid_and_crossing_change_is_detected():
    book = LocalBook()
    book.apply_snapshot(bids=[(100.2, 1.0)], asks=[(100.1, 1.0)])
    assert book.is_crossed() and not book.valid
    book = make_book()
    book.apply_changes([(BID, 100.1, 1.0)])
    assert book.is_crossed() and not book.valid
    book.apply_changes([(BID, 100.1, 0.0)])
    assert not book.valid  # stays invalid until a fresh snapshot


def test_empty_book_has_no_prices():
    book = LocalBook()
    assert book.best_bid is None and book.best_ask is None and book.mid is None and book.spread_bps is None
    assert not book.valid


def test_walk_buy_averages_across_levels():
    walk = make_book().walk("buy", 2.5)
    assert walk.avg_price == pytest.approx((100.1 * 1.0 + 100.2 * 1.5) / 2.5)
    assert walk.levels == 2 and walk.unfilled_qty == 0 and walk.filled_qty == 2.5


def test_walk_sell_consumes_bids_from_the_best_down():
    walk = make_book().walk("sell", 4.0)
    assert walk.avg_price == pytest.approx((99.9 * 1.0 + 99.8 * 2.0 + 99.7 * 1.0) / 4.0)
    assert walk.levels == 3


def test_walk_exactly_one_level():
    walk = make_book().walk("buy", 1.0)
    assert walk.avg_price == 100.1 and walk.levels == 1


def test_walk_beyond_visible_depth_reports_unfilled():
    book = make_book(depth=2)
    walk = book.walk("buy", 4.0)  # only 3.0 visible in the top 2 asks, the 100.3 level is hidden
    assert walk.unfilled_qty == pytest.approx(1.0)
    assert walk.filled_qty == pytest.approx(3.0)
    assert walk.avg_price == pytest.approx((100.1 + 2 * 100.2) / 3.0)


def test_walk_on_empty_side():
    book = LocalBook()
    book.apply_snapshot(bids=[(99.0, 1.0)], asks=[])
    walk = book.walk("buy", 1.0)
    assert walk.avg_price is None and walk.unfilled_qty == 1.0 and walk.levels == 0


def test_walk_tolerates_float_residue_at_exact_depth():
    book = LocalBook()
    book.apply_snapshot(bids=[(99.0, 1.0)], asks=[(100.0, 0.01), (100.5, 0.12), (101.0, 5.0)])
    walk = book.walk("buy", 0.13)  # 0.13 - 0.01 - 0.12 leaves a residue of about 1e-17
    assert walk.unfilled_qty == 0.0
    assert walk.levels == 2
    shallow = LocalBook()
    shallow.apply_snapshot(bids=[(99.0, 1.0)], asks=[(100.0, 0.1), (100.5, 0.2)])
    assert shallow.walk("buy", 0.1 + 0.2).unfilled_qty == 0.0
