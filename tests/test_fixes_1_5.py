"""Tests for Fixes #1-5: grid orders, TP matching, entry timeout, broker sync, POC-TP sanity."""
import sys
import os
import time
from datetime import datetime, timezone, timedelta
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.dirname(__file__) + "/..")

from strategy import Strategy, StrategyParams, Signal, BUY, SELL, GridLevel

MSK = timezone(timedelta(hours=3))


class MockOrders:
    """Mock OrderManager for testing."""
    def __init__(self):
        self.placed = []
        self.cancelled = []
        self._next_id = 1000

    def place_market(self, side, quantity, tag=""):
        oid = str(self._next_id)
        self._next_id += 1
        self.placed.append(("MARKET", side, quantity, tag, oid))
        return MagicMock(order_id=oid)

    def place_limit(self, side, quantity, price, tag=""):
        oid = str(self._next_id)
        self._next_id += 1
        self.placed.append(("LIMIT", side, quantity, price, tag, oid))
        return MagicMock(order_id=oid)

    def cancel(self, order_id):
        self.cancelled.append(order_id)

    def record_fill(self, fill):
        pass


def test_fix1_grid_no_cancel():
    """Fix #1: Placing grid-2 should NOT cancel grid-1."""
    from main import Robot
    robot = Robot(paper=True)
    robot.orders = MockOrders()
    robot.strategy.direction = 1
    robot.strategy.entry_price = 72000
    robot.strategy.entry_time = datetime.now(MSK)
    robot.strategy.poc = 72100
    robot.strategy.vah = 72200
    robot.strategy.val = 71800

    # Place grid-1
    sig1 = Signal(action="GRID", direction=BUY, price=71969, quantity=1, level=1, tag="Grid-1")
    robot._place_grid(sig1)

    # Place grid-2
    sig2 = Signal(action="GRID", direction=BUY, price=71938, quantity=1, level=2, tag="Grid-2")
    robot._place_grid(sig2)

    # Both should be tracked
    assert len(robot._grid_order_ids) == 2, f"Expected 2 grid orders, got {len(robot._grid_order_ids)}"

    # No cancellations
    assert len(robot.orders.cancelled) == 0, f"Grid orders were cancelled: {robot.orders.cancelled}"

    print("✅ Fix #1 PASSED: Multiple grid orders, no cancellation")


def test_fix2_tp_dict_tracking():
    """Fix #2: TP orders tracked in dict, matching by order_id."""
    from main import Robot
    robot = Robot(paper=True)
    robot.orders = MockOrders()
    robot.strategy.direction = 1
    robot.strategy.entry_price = 72000
    robot.strategy.entry_time = datetime.now(MSK)

    # Manually add a filled grid level
    robot.strategy.grid_levels = [
        GridLevel(level=1, price=71969, side=BUY, status="FILLED", tp_price=72000),
    ]

    # Place TP-1
    sig1 = Signal(action="TP", direction=SELL, price=72000, quantity=1, level=1, tag="TP-1")
    robot._place_tp(sig1)

    # Place TP-2 (different level)
    robot.strategy.grid_levels.append(
        GridLevel(level=2, price=71938, side=BUY, status="FILLED", tp_price=71969)
    )
    sig2 = Signal(action="TP", direction=SELL, price=71969, quantity=1, level=2, tag="TP-2")
    robot._place_tp(sig2)

    assert len(robot._tp_order_ids) == 2, f"Expected 2 TP orders, got {len(robot._tp_order_ids)}"
    # Values should be levels
    levels = list(robot._tp_order_ids.values())
    assert 1 in levels and 2 in levels, f"Levels not tracked: {levels}"

    print("✅ Fix #2 PASSED: TP orders tracked in dict with levels")


def test_fix3_entry_timeout():
    """Fix #3: Entry pending should timeout after 30 seconds."""
    from main import Robot
    from feed import Quote

    robot = Robot(paper=True)
    robot.orders = MockOrders()
    robot._running = True
    robot._mode = "running"

    # Set entry pending with old timestamp
    robot._entry_pending = True
    robot._entry_pending_since = datetime.now(MSK) - timedelta(seconds=35)
    robot.strategy.val = 71800
    robot.strategy.vah = 72200

    quote = Quote(bid=71900, ask=71901, last=71900, timestamp=datetime.now(MSK))
    robot._on_quote(quote)

    # Should have reset entry pending
    assert not robot._entry_pending, "Entry pending should have been reset after timeout"
    print("✅ Fix #3 PASSED: Entry pending timeout resets flag")


def test_fix4_periodic_sync():
    """Fix #4: _sync_broker called periodically."""
    from main import Robot
    from feed import Quote

    robot = Robot(paper=True)
    robot.orders = MockOrders()
    robot._running = True
    robot._mode = "running"
    robot.strategy.val = 71800
    robot.strategy.vah = 72200
    robot.strategy.current_price = 72000

    # Set last sync to 2 minutes ago
    robot._last_broker_sync = datetime.now(MSK) - timedelta(minutes=2)

    sync_count = [0]
    original_sync = robot._sync_broker
    def mock_sync():
        sync_count[0] += 1
        original_sync()
    robot._sync_broker = mock_sync

    quote = Quote(bid=72000, ask=72001, last=72000, timestamp=datetime.now(MSK))
    robot._on_quote(quote)

    assert sync_count[0] >= 1, f"Expected broker sync, got {sync_count[0]} calls"
    print("✅ Fix #4 PASSED: Periodic broker sync triggered")


def test_fix5_poc_tp_sanity():
    """Fix #5: POC-TP should not be placed if it would be at a loss."""
    s = Strategy(StrategyParams())

    # LONG entry @ 72000, POC @ 71800 (below entry = loss TP)
    s.poc = 71800
    signals = s.on_entry_fill(1, 72000)
    poc_signals = [sig for sig in signals if sig.action == "PLACE_POC_TP"]
    assert len(poc_signals) == 0, f"POC-TP should not be placed: LONG entry 72000, POC 71800 (loss)"

    s.on_close_all()

    # LONG entry @ 72000, POC @ 72200 (above entry = profit TP)
    s.poc = 72200
    signals = s.on_entry_fill(1, 72000)
    poc_signals = [sig for sig in signals if sig.action == "PLACE_POC_TP"]
    assert len(poc_signals) == 1, f"POC-TP should be placed: LONG entry 72000, POC 72200 (profit)"

    s.on_close_all()

    # SHORT entry @ 72000, POC @ 72200 (above entry = loss TP)
    s.poc = 72200
    signals = s.on_entry_fill(-1, 72000)
    poc_signals = [sig for sig in signals if sig.action == "PLACE_POC_TP"]
    assert len(poc_signals) == 0, f"POC-TP should not be placed: SHORT entry 72000, POC 72200 (loss)"

    s.on_close_all()

    # SHORT entry @ 72000, POC @ 71800 (below entry = profit TP)
    s.poc = 71800
    signals = s.on_entry_fill(-1, 72000)
    poc_signals = [sig for sig in signals if sig.action == "PLACE_POC_TP"]
    assert len(poc_signals) == 1, f"POC-TP should be placed: SHORT entry 72000, POC 71800 (profit)"

    print("✅ Fix #5 PASSED: POC-TP sanity check works for all 4 cases")


if __name__ == "__main__":
    test_fix1_grid_no_cancel()
    test_fix2_tp_dict_tracking()
    test_fix3_entry_timeout()
    test_fix4_periodic_sync()
    test_fix5_poc_tp_sanity()
    print("\n🎉 All 5 fixes passed!")
