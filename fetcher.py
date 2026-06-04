import aiohttp
import asyncio
import logging
import threading

DATA_URL = "https://data-api.polymarket.com"
DEFAULT_ADDRESS = "0x272531256c25F583450ED9a5D316cAAeDa0be7d1"

logger = logging.getLogger(__name__)


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
