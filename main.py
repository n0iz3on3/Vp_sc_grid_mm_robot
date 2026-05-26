"""Main robot — orchestrates feed, strategy, orders, state, risk."""
import logging
import os
import signal
import sys
import threading
import time
from datetime import datetime, timezone, timedelta

from FinamPy import FinamPy

import config
from feed import Feed, Quote, Bar, OrderEvent, TradeEvent
from vp import VolumeProfile
from strategy import Strategy, StrategyParams, BUY, SELL
from orders import OrderManager, PlacedOrder, FillInfo
from state import StateManager, TrackedOrder
from risk import RiskManager

log = logging.getLogger("robot")

MSK = timezone(timedelta(hours=3))


class Robot:
    def __init__(self, paper: bool = False):
        self.fp: FinamPy | None = None
        self.feed = Feed()
        self.vp = VolumeProfile(lookback=33, bin_size=50, va_percent=0.70)
        # VP params synced from strategy params on start
        self.strategy = Strategy(StrategyParams())
        self.orders: OrderManager | None = None
        self.state = StateManager("/tmp/robot-state.json")
        self.risk = RiskManager()

        self._paper = paper
        self._running = False
        self._mode = "stopped"  # running, paused, stopped

        # Pending entry lock
        self._entry_pending = False
        self._entry_pending_since: datetime | None = None
        self._entry_pending_timeout = timedelta(seconds=30)

        # Tracked order IDs (for cancelling)
        self._grid_order_id: str | None = None
        self._tp_order_ids: list[str] = []  # Multiple TPs possible
        self._poc_tp_order_id: str | None = None

        # Skip ticks
        self._skip_ticks = 0

        # Last entry for recovery
        self._last_entry_price = 0.0
        self._last_direction = 0

        # Price log
        self._last_price_log: datetime = datetime.now(MSK) - timedelta(minutes=1)

    # === LIFECYCLE ===

    def start(self):
        """Connect to Finam, restore state, subscribe to data."""
        log.info(f"Starting robot{' [PAPER MODE]' if self._paper else ''}...")
        self.fp = FinamPy(config.FINAM_TOKEN)
        self.orders = OrderManager(self.fp)
        self.feed.connect()

        # Restore state — with validation
        s = self.state.load()
        if s.direction != 0 and s.entry_price > 0:
            self.strategy.direction = s.direction
            self.strategy.entry_price = s.entry_price
            if s.grid_levels:
                self.strategy.restore_grid(s.grid_levels)
                log.info(f"Grid restored: {len(s.grid_levels)} levels")
            if s.entry_time:
                try:
                    self.strategy.entry_time = datetime.fromisoformat(s.entry_time)
                except:
                    pass
            self._last_entry_price = s.last_entry_price
            self._last_direction = s.last_direction
            log.info(f"State restored: dir={s.direction} entry={s.entry_price:.0f} levels={len(s.grid_levels)}")
        elif s.direction != 0 and s.entry_price <= 0:
            log.warning("State has direction but entry_price=0 — resetting (corrupt state)")
            self.state.clear()

        # Warmup VP from gRPC bars
        self._warmup_vp()

        # Wire callbacks
        self.feed.on_quote = self._on_quote
        self.feed.on_bar = self._on_bar
        self.feed.on_order = self._on_order_event
        self.feed.on_trade = self._on_trade_event
        self.feed._on_stale = self._on_stale_streams

        # Subscribe to everything
        self.feed.subscribe_all()

        # Sync with broker
        self._sync_broker()

        self._running = True
        self._mode = "running"
        self.state.state.mode = "running"
        self.state.save()
        log.info(f"Robot started{' [PAPER]' if self._paper else ''}. VP: VAL={self.vp.val:.0f} VAH={self.vp.vah:.0f} POC={self.vp.poc:.0f}")

    def stop(self, close_position: bool = False):
        """Stop robot, optionally close position."""
        log.info("Stopping robot...")
        self._running = False
        self._mode = "stopped"

        if close_position and self.strategy.has_position and self.orders:
            self._close_all("Manual stop")

        # Cancel all tracked orders
        if self.orders:
            self._cancel_all_tracked()

        self._reset_tracked()

        self.feed.disconnect()
        if self.fp:
            self.fp.close_channel()

        self.state.state.mode = "stopped"
        self.state.save()
        log.info("Robot stopped")

    def pause(self):
        """Pause trading (cancel orders, keep position)."""
        self._mode = "paused"
        if self.orders:
            self._cancel_all_tracked()
        self._reset_tracked()
        log.info("Robot paused")

    def resume(self):
        """Resume from pause."""
        self._mode = "running"
        self._sync_broker()
        log.info("Robot resumed")

    # === CALLBACKS ===

    def _on_stale_streams(self):
        """Called by feed watchdog when streams go stale. Reconnect."""
        log.warning("Reconnecting stale streams...")
        self.feed.disconnect()
        time.sleep(2)
        self.feed.connect()
        self._warmup_vp()
        self.feed.subscribe_all()
        log.info(f"Reconnected. VP: VAL={self.vp.val:.0f} VAH={self.vp.vah:.0f} POC={self.vp.poc:.0f}")

    def _on_quote(self, q: Quote):
        """Quote callback — update current price, check exits and fills."""
        if not self._running or self._mode != "running":
            return
        if self._skip_ticks > 0:
            self._skip_ticks -= 1
            return

        self.strategy.current_price = q.last

        # Periodic price log
        if (datetime.now(MSK) - self._last_price_log).seconds >= 30:
            self._last_price_log = datetime.now(MSK)
            log.info(f"Price: {q.last:.0f} | VP: VAL={self.strategy.val:.0f} VAH={self.strategy.vah:.0f} POC={self.strategy.poc:.0f}")

        if not self.strategy.has_position:
            # Check entry on quote
            if (self._mode == "running" and not self._entry_pending
                    and self.strategy.val > 0 and q.last > 0):
                sig = self.strategy.check_entry(q.last)
                if sig:
                    self._execute_entry(sig)
            return

        # Check risk
        pnl = self.strategy.calc_unrealized_pnl(q.last)
        ok, msg = self.risk.check_pnl(pnl)
        if not ok:
            log.warning(f"Risk stop: {msg}")
            self._close_all(msg)
            return

        # Check exit
        sig = self.strategy.check_exit()
        if sig:
            self._close_all(sig.tag)
            return

        # Paper mode: simulate fills
        if self._paper:
            self._paper_process_fills(q.last)

    def _on_bar(self, b: Bar):
        """Bar callback — update VP, check entry signals."""
        if not self._running or self._mode != "running":
            return

        self.vp.add_bar(b.close, b.volume)
        result = self.vp.calculate()
        if result:
            self.strategy.poc = result.poc
            self.strategy.vah = result.vah
            self.strategy.val = result.val

    def _on_order_event(self, evt: OrderEvent):
        """Order status update from gRPC push."""
        if not self._running:
            return
        log.info(f"Order event: id={evt.order_id} status={evt.status} exec_qty={evt.executed_quantity}")

    def _on_trade_event(self, evt: TradeEvent):
        """Trade (fill) from gRPC push."""
        if not self._running:
            return

        log.info(f"Trade: {'BUY' if evt.side==1 else 'SELL'} {evt.quantity:.0f} @ {evt.price:.0f} order={evt.order_id}")

        fill = FillInfo(
            order_id=evt.order_id,
            client_order_id="",
            side=evt.side,
            price=evt.price,
            quantity=evt.quantity,
            trade_id=evt.trade_id,
            timestamp=evt.timestamp,
        )
        if self.orders:
            self.orders.record_fill(fill)

        # Match fill to tracked orders (non-paper mode)
        if not self._paper:
            self._process_real_fill(evt)

    # === EXECUTION ===

    def _execute_entry(self, sig):
        """Execute entry signal."""
        if self._paper:
            if sig.price <= 0:
                log.warning(f"[PAPER] Entry rejected: price={sig.price:.0f}")
                self._skip_ticks = 60
                return
            log.info(f"[PAPER] ENTRY {'LONG' if sig.direction == 1 else 'SHORT'} @ {sig.price:.0f} — {sig.tag}")
            self._process_entry_fill(sig.direction, sig.price)
            return

        if not self.orders:
            return

        # Entry guard: check broker first
        pos = self._get_broker_position()
        if pos and pos[0] != 0 and pos[1] > 0:
            log.info(f"Entry guard: broker has {pos[1]} lots dir={pos[0]} — syncing")
            self._restore_position(pos[0], pos[1], pos[2])
            return

        side = BUY if sig.direction == 1 else SELL
        tag = f"ENTRY-{'LONG' if sig.direction == 1 else 'SHORT'}"
        po = self.orders.place_market(side, 1, tag)

        if po:
            self._entry_pending = True
            self._entry_pending_since = datetime.now(MSK)
            self._skip_ticks = 30
        else:
            self._skip_ticks = 60

    def _process_entry_fill(self, direction: int, price: float):
        """Process confirmed entry fill (from paper or real trade)."""
        signals = self.strategy.on_entry_fill(direction, price)
        self._entry_pending = False
        self._last_entry_price = price
        self._last_direction = direction

        for sig in signals:
            if sig.action == "GRID":
                self._place_grid(sig)
            elif sig.action == "PLACE_POC_TP":
                self._place_poc_tp(sig)

        self._save_state()

    def _place_grid(self, sig):
        """Place grid limit order."""
        if self._paper:
            log.info(f"[PAPER] GRID-{sig.level} {'BUY' if sig.direction==BUY else 'SELL'} @ {sig.price:.0f}")
            return
        if not self.orders:
            return
        # Cancel existing grid if any
        if self._grid_order_id:
            self.orders.cancel(self._grid_order_id)
            self._grid_order_id = None

        po = self.orders.place_limit(sig.direction, sig.quantity, sig.price, f"GRID-{sig.level}")
        if po:
            self._grid_order_id = po.order_id
        else:
            self._grid_order_id = None

    def _place_tp(self, sig):
        """Place TP limit order."""
        if self._paper:
            log.info(f"[PAPER] TP-{sig.level} {'BUY' if sig.direction==BUY else 'SELL'} @ {sig.price:.0f}")
            return
        if not self.orders:
            return

        po = self.orders.place_limit(sig.direction, sig.quantity, sig.price, f"TP-{sig.level}")
        if po:
            self._tp_order_ids.append(po.order_id)
        # Don't cancel other TPs — each grid level has its own TP

    def _place_poc_tp(self, sig):
        """Place POC-TP limit order (for entry lot)."""
        if self._paper:
            log.info(f"[PAPER] POC-TP {'BUY' if sig.direction==BUY else 'SELL'} @ {sig.price:.0f}")
            return
        if not self.orders:
            return
        if self._poc_tp_order_id:
            self.orders.cancel(self._poc_tp_order_id)
            self._poc_tp_order_id = None

        po = self.orders.place_limit(sig.direction, sig.quantity, sig.price, "POC-TP")
        if po:
            self._poc_tp_order_id = po.order_id

    def _close_all(self, reason: str):
        """Close all positions and cancel all orders."""
        pnl = 0
        if self.strategy.has_position:
            if self.strategy.entry_price > 0:
                pnl = self.strategy.calc_unrealized_pnl(self.strategy.current_price)
                log.info(f"{'[PAPER] ' if self._paper else ''}CLOSE ALL: {reason} | PnL={pnl:.0f} | lots={self.strategy.total_lots}")
            else:
                log.info(f"{'[PAPER] ' if self._paper else ''}CLOSE ALL: {reason} | entry_price=0 (invalid)")

        if not self._paper:
            if self.orders:
                self._cancel_all_tracked()
                if self.strategy.has_position:
                    close_side = SELL if self.strategy.direction == 1 else BUY
                    self.orders.place_market(close_side, self.strategy.total_lots, f"CLOSE-{reason}")

        # Stats
        if self.strategy.has_position and self.strategy.entry_price > 0:
            self.state.state.round_trips += 1
            self.state.state.realized_pnl += pnl
            dir_str = "LONG" if self.strategy.direction == 1 else "SHORT"
            log.info(f"Round trip #{self.state.state.round_trips}: {dir_str} entry={self.strategy.entry_price:.0f} PnL={pnl:.0f} | Total: RT={self.state.state.round_trips} PnL={self.state.state.realized_pnl:.0f}")

        self.strategy.on_close_all()
        self._entry_pending = False
        self._reset_tracked()
        self._save_state()

    # === PAPER FILL PROCESSING ===

    def _paper_process_fills(self, price: float):
        """Paper mode: process all pending grid/TP fills."""
        if not self._paper:
            return

        # Process grid fills (may be multiple)
        changed = True
        while changed:
            changed = False
            grid = self.strategy.check_paper_grid_fill(price)
            if grid:
                log.info(f"[PAPER] GRID-{grid.level} filled @ {grid.price:.0f}")
                signals = self.strategy.on_grid_fill(grid.price)
                changed = True
                for sig in signals:
                    if sig.action == "GRID":
                        self._place_grid(sig)
                    elif sig.action == "TP":
                        self._place_tp(sig)

        # Process TP fills (may be multiple)
        changed = True
        while changed:
            changed = False
            for g in self.strategy.grid_levels:
                if g.status != "FILLED" or g.tp_price <= 0:
                    continue
                hit = False
                if self.strategy.direction == 1 and price >= g.tp_price:
                    hit = True
                elif self.strategy.direction == -1 and price <= g.tp_price:
                    hit = True
                if hit:
                    log.info(f"[PAPER] TP-{g.level} filled @ {g.tp_price:.0f}")
                    self.strategy.on_tp_fill(g.tp_price)
                    changed = True
                    break  # Re-iterate after modification

        if changed or grid:
            self._save_state()

    # === REAL FILL PROCESSING ===

    def _process_real_fill(self, evt: TradeEvent):
        """Match fill to tracked orders (non-paper)."""
        oid = evt.order_id

        # Entry fill
        if self._entry_pending and not self.strategy.has_position:
            self._process_entry_fill(
                BUY if evt.side == 1 else SELL,
                evt.price,
            )
            return

        # Grid fill
        if oid == self._grid_order_id:
            log.info(f"Grid fill @ {evt.price:.0f}")
            signals = self.strategy.on_grid_fill(evt.price)
            self._grid_order_id = None
            for sig in signals:
                if sig.action == "GRID":
                    self._place_grid(sig)
                elif sig.action == "TP":
                    self._place_tp(sig)
            self._save_state()
            return

        # TP fill
        if oid in self._tp_order_ids:
            log.info(f"TP fill @ {evt.price:.0f}")
            self.strategy.on_tp_fill(evt.price)
            self._tp_order_ids.remove(oid)
            self._save_state()
            return

        # POC-TP fill
        if oid == self._poc_tp_order_id:
            log.info(f"POC-TP fill @ {evt.price:.0f} → close all")
            self._poc_tp_order_id = None
            self._close_all("POC-TP filled")
            return

    # === BROKER SYNC ===

    def _get_broker_position(self) -> tuple | None:
        """Get broker position via gRPC. Returns (dir, lots, avg) or None."""
        if not self.fp:
            return None
        try:
            from FinamPy.grpc.accounts_service_pb2 import GetAccountRequest
            account = self.fp.call_function(
                self.fp.accounts_stub.GetAccount,
                GetAccountRequest(account_id=config.FINAM_ACCOUNT_ID),
            )
            if not account:
                return None
            for pos in account.positions:
                if config.SYMBOL in pos.symbol or config.TICKER in pos.symbol:
                    qty = float(pos.quantity.value) if pos.quantity else 0
                    avg = float(pos.average_price.value) if pos.average_price else 0
                    if qty > 0:
                        return (1, int(qty), avg)
                    elif qty < 0:
                        return (-1, int(abs(qty)), avg)
            return None
        except Exception as e:
            log.error(f"Broker position error: {e}")
            return None

    def _sync_broker(self):
        """Sync robot state with broker position."""
        pos = self._get_broker_position()
        if pos and pos[1] > 0:
            if not self.strategy.has_position:
                self._restore_position(pos[0], pos[1], pos[2])
            elif self.strategy.direction != pos[0]:
                log.warning(f"Dir mismatch! Robot={self.strategy.direction} Broker={pos[0]}")
                self._restore_position(pos[0], pos[1], pos[2])

    def _restore_position(self, direction: int, lots: int, avg_price: float):
        """Restore position from broker data."""
        entry = avg_price if avg_price > 0 else (self._last_entry_price if self._last_direction == direction else 0)
        if entry <= 0:
            entry = self.strategy.current_price if self.strategy.current_price > 0 else 0
        if entry <= 0:
            log.warning("Cannot restore: no valid entry price")
            return

        self.strategy.direction = direction
        self.strategy.entry_price = entry
        self.strategy.filled_levels = lots - 1
        self.strategy.entry_time = datetime.now(MSK)
        self._last_entry_price = entry
        self._last_direction = direction
        self._save_state()
        log.info(f"Restored: dir={direction} entry={entry:.0f} lots={lots}")

    # === HELPERS ===

    def _cancel_all_tracked(self):
        """Cancel all tracked orders."""
        if not self.orders:
            return
        if self._grid_order_id:
            self.orders.cancel(self._grid_order_id)
        for oid in self._tp_order_ids:
            self.orders.cancel(oid)
        if self._poc_tp_order_id:
            self.orders.cancel(self._poc_tp_order_id)

    def _reset_tracked(self):
        """Reset all tracked order IDs."""
        self._grid_order_id = None
        self._tp_order_ids = []
        self._poc_tp_order_id = None

    def _warmup_vp(self):
        """Load historical bars for VP warmup."""
        if not self.fp:
            return
        try:
            from google.protobuf.timestamp_pb2 import Timestamp
            from google.type.interval_pb2 import Interval
            import FinamPy.grpc.marketdata_service_pb2 as md_pb2

            finam_tf, _, _ = self.fp.timeframe_to_finam_timeframe(config.TIMEFRAME)
            now = datetime.now(timezone.utc)
            start = now - timedelta(hours=3)

            resp = self.fp.call_function(
                self.fp.marketdata_stub.Bars,
                md_pb2.BarsRequest(
                    symbol=config.SYMBOL,
                    timeframe=finam_tf,
                    interval=Interval(
                        start_time=Timestamp(seconds=int(start.timestamp())),
                        end_time=Timestamp(seconds=int(now.timestamp())),
                    ),
                ),
            )
            if resp and resp.bars:
                for bar in resp.bars[-config.WARMUP_BARS:]:
                    self.vp.add_bar(float(bar.close.value), float(bar.volume.value))
                result = self.vp.calculate()
                if result:
                    self.strategy.poc = result.poc
                    self.strategy.vah = result.vah
                    self.strategy.val = result.val
                    log.info(f"Warmup: {len(list(resp.bars))} bars, VAL={result.val:.0f} VAH={result.vah:.0f} POC={result.poc:.0f}")
        except Exception as e:
            log.error(f"Warmup error: {e}")

    def _save_state(self):
        """Persist current state."""
        s = self.state.state
        s.direction = self.strategy.direction
        s.entry_price = self.strategy.entry_price
        s.filled_levels = self.strategy.filled_levels
        s.entry_time = self.strategy.entry_time.isoformat() if self.strategy.entry_time else ""
        s.grid_levels = self.strategy.serialize_grid()
        s.last_entry_price = self._last_entry_price
        s.last_direction = self._last_direction
        self.state.save()

    def get_status(self) -> dict:
        """Get current status dict."""
        pnl = self.strategy.calc_unrealized_pnl(self.strategy.current_price) if self.strategy.has_position else 0
        return {
            "mode": self._mode,
            "paper": self._paper,
            "direction": self.strategy.direction,
            "dir_str": "LONG" if self.strategy.direction == 1 else "SHORT" if self.strategy.direction == -1 else "FLAT",
            "entry_price": self.strategy.entry_price,
            "total_lots": self.strategy.total_lots,
            "filled_levels": self.strategy.filled_levels,
            "grid_levels": len(self.strategy.grid_levels),
            "pnl": round(pnl, 1),
            "round_trips": self.state.state.round_trips,
            "realized_pnl": round(self.state.state.realized_pnl, 1),
            "poc": round(self.strategy.poc, 0),
            "vah": round(self.strategy.vah, 0),
            "val": round(self.strategy.val, 0),
            "current_price": self.strategy.current_price,
            "hold_minutes": self.strategy.hold_minutes,
            "connected": self.feed.connected,
        }


# === ENTRY POINT ===

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--paper", action="store_true", help="Paper trading (no real orders)")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    robot = Robot(paper=args.paper)

    def shutdown(sig, frame):
        log.info("Shutdown signal received")
        robot.stop(close_position=True)
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    # Start API server in background thread
    import uvicorn
    from api import app, set_robot
    set_robot(robot)
    api_thread = threading.Thread(
        target=lambda: uvicorn.run(app, host="0.0.0.0", port=5070, log_level="warning"),
        daemon=True,
    )
    api_thread.start()
    log.info("API server started on port 5070")

    robot.start()

    # Block until interrupted
    try:
        while robot._running:
            time.sleep(1)
    except KeyboardInterrupt:
        pass

    robot.stop(close_position=True)


if __name__ == "__main__":
    main()
