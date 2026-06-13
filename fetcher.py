import aiohttp
import asyncio
import json
import logging
import threading
from datetime import datetime, timezone
from pathlib import Path

DATA_URL = "https://data-api.polymarket.com"
DEFAULT_ADDRESS = "0x272531256c25F583450ED9a5D316cAAeDa0be7d1"
CACHE_DIR = Path(__file__).resolve().parent / ".cache"

logger = logging.getLogger(__name__)


def _position_key(row: dict) -> tuple:
    return row.get("conditionId"), row.get("asset"), row.get("outcomeIndex")


def _closed_positions_cache_path(address: str) -> Path:
    return CACHE_DIR / f"closed_positions_{address.lower()}.json"


def _load_closed_positions_cache(address: str) -> list:
    path = _closed_positions_cache_path(address)
    if not path.exists():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows = payload.get("positions", []) if isinstance(payload, dict) else payload
        return rows if isinstance(rows, list) else []
    except Exception as e:
        logger.warning(f"Could not read closed-positions cache {path}: {e}")
        return []


def _save_closed_positions_cache(address: str, positions: list) -> None:
    path = _closed_positions_cache_path(address)
    temp_path = path.with_suffix(".tmp")
    payload = {
        "version": 1,
        "address": address.lower(),
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "positions": positions,
    }
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        temp_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        temp_path.replace(path)
    except Exception as e:
        logger.warning(f"Could not write closed-positions cache {path}: {e}")


async def _fetch_page(session: aiohttp.ClientSession, address: str, offset: int, limit: int = 1000) -> list:
    url = f"{DATA_URL}/activity"
    params = {"user": address, "type": "TRADE", "limit": limit, "offset": offset}
    try:
        async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=30)) as resp:
            if resp.status != 200:
                logger.warning(f"fetch offset={offset}: status {resp.status}")
                return []
            data = await resp.json()
            return data if isinstance(data, list) else data.get("data", [])
    except Exception as e:
        logger.error(f"fetch offset={offset}: {e}")
        return []


async def _get_all_async(address: str) -> list:
    all_trades: list = []
    offset = 0
    limit = 1000
    async with aiohttp.ClientSession() as session:
        while True:
            page = await _fetch_page(session, address, offset, limit)
            if not page:
                break
            all_trades.extend(page)
            logger.info(f"Fetched {len(all_trades)} trades (offset={offset})")
            if len(page) < limit or offset + limit > 3000:
                break
            offset += limit
            await asyncio.sleep(0.25)
    return all_trades


async def _get_rebates_async(address: str) -> list:
    all_rebates: list = []
    offset = 0
    limit = 1000
    async with aiohttp.ClientSession() as session:
        while True:
            url = f"{DATA_URL}/activity"
            params = {"user": address, "type": "MAKER_REBATE", "limit": limit, "offset": offset}
            try:
                async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=30)) as resp:
                    data = await resp.json()
                    page = data if isinstance(data, list) else data.get("data", [])
            except Exception as e:
                logger.error(f"get_rebates offset={offset}: {e}")
                break
            if not page:
                break
            all_rebates.extend(page)
            if len(page) < limit or offset + limit > 3000:
                break
            offset += limit
            await asyncio.sleep(0.25)
    return all_rebates


async def _fetch_closed_page(
    session: aiohttp.ClientSession,
    address: str,
    offset: int,
    limit: int,
) -> list:
    url = f"{DATA_URL}/closed-positions"
    params = {
        "user": address,
        "limit": limit,
        "offset": offset,
        "sortBy": "TIMESTAMP",
        "sortDirection": "DESC",
    }
    for attempt in range(3):
        try:
            async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=30)) as resp:
                if resp.status != 200:
                    raise RuntimeError(f"status {resp.status}")
                data = await resp.json()
                return data if isinstance(data, list) else data.get("data", [])
        except Exception as e:
            if attempt == 2:
                raise RuntimeError(f"closed-positions offset={offset}: {e}") from e
            logger.warning(f"closed-positions offset={offset}, retry={attempt + 1}: {e}")
            await asyncio.sleep(attempt + 1)
    return []


async def _get_closed_positions_async(address: str, known_keys: set[tuple] | None = None) -> list:
    """Fetch newest positions until a page intersects with the local cache."""
    positions: list = []
    known_keys = known_keys or set()
    offset = 0
    limit = 50
    async with aiohttp.ClientSession() as session:
        while offset <= 100000:
            page = await _fetch_closed_page(session, address, offset, limit)
            positions.extend(page)
            page_keys = {_position_key(row) for row in page}
            if len(page) < limit or (known_keys and page_keys & known_keys):
                break
            offset += limit
            await asyncio.sleep(0.1)

    unique = {_position_key(row): row for row in positions}
    return sorted(unique.values(), key=lambda row: row.get("timestamp", 0), reverse=True)


def _run_async(coro) -> list:
    result: list = []
    exc_box: list = []

    def _run():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result.extend(loop.run_until_complete(coro))
        except Exception as e:
            exc_box.append(e)
        finally:
            loop.close()

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    t.join()
    if exc_box:
        raise exc_box[0]
    return result


def get_closed_positions(address: str = DEFAULT_ADDRESS) -> list:
    cached = _load_closed_positions_cache(address)
    known_keys = {_position_key(row) for row in cached}
    try:
        fresh = _run_async(_get_closed_positions_async(address, known_keys))
    except Exception:
        if cached:
            logger.exception("Closed-positions API failed; using local cache")
            return cached
        raise

    merged = {_position_key(row): row for row in cached}
    merged.update({_position_key(row): row for row in fresh})
    positions = sorted(merged.values(), key=lambda row: row.get("timestamp", 0), reverse=True)
    _save_closed_positions_cache(address, positions)
    logger.info(f"Closed positions: {len(fresh)} fetched, {len(positions)} cached total")
    return positions


def get_rebates(address: str = DEFAULT_ADDRESS) -> list:
    result: list = []
    exc_box: list = []

    def _run():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result.extend(loop.run_until_complete(_get_rebates_async(address)))
        except Exception as e:
            exc_box.append(e)
        finally:
            loop.close()

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    t.join()
    if exc_box:
        raise exc_box[0]
    return result


def get_all_trades(address: str = DEFAULT_ADDRESS) -> list:
    result: list = []
    exc_box: list = []

    def _run():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result.extend(loop.run_until_complete(_get_all_async(address)))
        except Exception as e:
            exc_box.append(e)
        finally:
            loop.close()

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    t.join()
    if exc_box:
        raise exc_box[0]
    return result
