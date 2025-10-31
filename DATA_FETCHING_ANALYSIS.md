# Data Fetching Strategy Analysis

## Current Implementation

### How It Works Now

#### 1. **AlpacaDataClient** (`src/data/ingestion/market_data.py`)

**Caching Mechanism**:
- ✅ **Has caching** - Uses Parquet files stored in `./data_cache/`
- ✅ **Cache key format**: `{symbols}_{start_date}_{end_date}_{timeframe}.parquet`
- ✅ **Cache check**: Loads from cache if exact match exists
- ✅ **Cache save**: Saves downloaded data to cache

**Example**:
```python
# Request 1: AAPL from 2024-01-01 to 2024-12-31
# Cache file: AAPL_20240101_20241231_1Day.parquet
# Result: Fetches from API, saves to cache

# Request 2: Same request (same dates)
# Cache file: AAPL_20240101_20241231_1Day.parquet
# Result: Loads from cache ✅
```

#### 2. **Handler Implementation** (`deployment/runpod/handler.py`)

**Current Flow**:
```python
# In analyze_handler():
days = days_map.get(period, 365)  # e.g., 365 days
end = datetime.now()                # ← CURRENT TIME (changes every request!)
start = end - timedelta(days=days) # ← Changes every request!

df = data_client.download_latest(
    symbols=symbol,
    days=days,
    timeframe="1Day"
)
```

**Problem**:
- ❌ **`download_latest()` calculates `end = datetime.now()` every time**
- ❌ **Cache key includes end date** → Different end date = different cache key
- ❌ **Cache NEVER matches** (except if requests happen in same second)
- ❌ **Every request fetches full dataset** from API

**Example**:
```python
# Request 1 at 10:00 AM: AAPL last 365 days
# Cache: AAPL_20240101_20241231_1Day.parquet (end=2024-12-31 10:00:00)

# Request 2 at 10:01 AM: AAPL last 365 days  
# Cache: AAPL_20240102_20241231_1Day.parquet (end=2024-12-31 10:01:00)
# ❌ Cache miss! Fetches entire dataset again
```

#### 3. **Serverless Context** (RunPod)

**Additional Issues**:
- ❌ **No shared storage** - Each worker has its own filesystem
- ❌ **Cache doesn't persist** - Between requests on different workers
- ❌ **Cold starts** - New workers have empty cache
- ❌ **yfinance fallback** - No caching implemented

---

## Current Behavior Summary

### What Happens Now:

1. **First Request** (or cache miss):
   - Fetches full dataset from API
   - Saves to cache file
   - Returns data

2. **Subsequent Requests**:
   - **Same worker + same second**: ✅ Cache hit (rare)
   - **Different worker**: ❌ Cache miss (common in serverless)
   - **Different second**: ❌ Cache miss (almost always)

3. **Result**:
   - **Most requests fetch full dataset** from API
   - **No incremental updates**
   - **No diff/appending**
   - **Inefficient** for serverless

---

## Problems Identified

### ❌ Problem 1: Cache Key Mismatch
**Issue**: `download_latest()` uses `datetime.now()` as end date
- Cache key includes end date
- End date changes every second
- Cache almost never matches

**Impact**: Cache is ineffective

### ❌ Problem 2: No Incremental Updates
**Issue**: No logic to check if cached data exists and fetch only missing days
- Always fetches full period
- Wastes API calls
- Slow for large timeframes

**Impact**: Inefficient and expensive

### ❌ Problem 3: Serverless Limitations
**Issue**: RunPod serverless has ephemeral filesystem
- Cache doesn't persist between requests
- Each worker has separate cache
- Cold starts = empty cache

**Impact**: Cache doesn't work in production

### ❌ Problem 4: No Diff Strategy
**Issue**: No mechanism to append new data to existing cache
- If cache exists for "2024-01-01 to 2024-12-30"
- Request for "2024-01-01 to 2024-12-31" fetches ENTIRE dataset again
- Should fetch only missing day (2024-12-31)

**Impact**: Wasted API calls and bandwidth

---

## Recommended Solutions

### Option 1: Smart Caching with Incremental Updates (Recommended)

**Strategy**:
1. **Check cache** for existing data
2. **Identify gaps** - What date range is missing?
3. **Fetch only missing data** - Don't refetch entire dataset
4. **Merge** - Append new data to cached data
5. **Update cache** - Save merged dataset

**Implementation**:
```python
def download_latest_smart(
    self,
    symbols: Union[str, List[str]],
    days: int = 365,
    timeframe: str = "1Day",
    use_cache: bool = True,
) -> pd.DataFrame:
    """Download latest N days with smart caching and incremental updates"""
    
    end = datetime.now()
    start = end - timedelta(days=days)
    
    # Try to load from cache
    if use_cache:
        # Look for cache files that overlap with requested range
        cached_df = self._load_overlapping_cache(symbols, start, end, timeframe)
        
        if cached_df is not None:
            # Check what's missing
            cached_start = cached_df.index.min()
            cached_end = cached_df.index.max()
            
            # Calculate missing ranges
            missing_before = []  # Data before cached range
            missing_after = []   # Data after cached range
            
            if start < cached_start:
                # Need data before cache
                missing_before = self.download_bars(
                    symbols, start, cached_start - timedelta(days=1), timeframe, use_cache=False
                )
            
            if end > cached_end:
                # Need data after cache (incremental update!)
                missing_after = self.download_bars(
                    symbols, cached_end + timedelta(days=1), end, timeframe, use_cache=False
                )
            
            # Merge all data
            dfs_to_merge = []
            if missing_before is not None and not missing_before.empty:
                dfs_to_merge.append(missing_before)
            dfs_to_merge.append(cached_df)
            if missing_after is not None and not missing_after.empty:
                dfs_to_merge.append(missing_after)
            
            if len(dfs_to_merge) > 1:
                result = pd.concat(dfs_to_merge).sort_index()
                # Update cache with merged data
                self._save_to_cache(result, symbols, start, end, timeframe)
                return result
            else:
                return cached_df
    
    # No cache or cache miss - fetch full dataset
    return self.download_bars(symbols, start, end, timeframe, use_cache=True)
```

### Option 2: Fixed Date Range Caching

**Strategy**:
- Cache by **symbol + timeframe** (not date range)
- Store "latest data up to X date"
- Always fetch from "cache end date" to "now"
- Append and update cache

**Implementation**:
```python
def _get_latest_cache(symbol, timeframe):
    """Get cache file with latest data for symbol"""
    pattern = f"{symbol}_*_{timeframe}.parquet"
    cache_files = list(self.cache_dir.glob(pattern))
    if cache_files:
        # Return file with latest end date
        return max(cache_files, key=lambda f: f.stat().st_mtime)
    return None

def download_latest_smart(self, symbol, days, timeframe):
    # Check for existing cache
    cache_file = self._get_latest_cache(symbol, timeframe)
    
    if cache_file:
        cached_df = pd.read_parquet(cache_file)
        cached_end = cached_df.index.max()
        now = datetime.now()
        
        if now > cached_end:
            # Fetch only new data
            new_data = self.download_bars(symbol, cached_end + timedelta(days=1), now, timeframe)
            # Merge
            result = pd.concat([cached_df, new_data]).sort_index()
            # Update cache
            self._save_to_cache(result, symbol, result.index.min(), result.index.max(), timeframe)
            return result
        else:
            # Cache is up to date
            return cached_df
    
    # No cache - fetch full dataset
    return self.download_bars(symbol, start, end, timeframe)
```

### Option 3: External Cache (Redis/Database)

**Strategy**:
- Use Redis or database for persistent cache
- Shared across all workers
- Can implement TTL and incremental updates
- Better for serverless

**Implementation**:
```python
import redis

class RedisCache:
    def __init__(self, redis_url):
        self.redis = redis.from_url(redis_url)
    
    def get(self, symbol, start, end):
        key = f"data:{symbol}:{start}:{end}"
        cached = self.redis.get(key)
        if cached:
            return pd.read_msgpack(cached)
        return None
    
    def set(self, symbol, start, end, df):
        key = f"data:{symbol}:{start}:{end}"
        self.redis.setex(key, 86400, df.to_msgpack())  # 24h TTL
```

---

## Recommendation

### For Serverless (RunPod):

**Best approach**: **Option 3 (External Cache)** + **Option 1 (Incremental Updates)**

**Why**:
1. ✅ **Persistent** - Works across workers
2. ✅ **Efficient** - Only fetches missing data
3. ✅ **Scalable** - Shared cache reduces API calls
4. ✅ **Cost-effective** - Fewer API requests

**Implementation Priority**:
1. **Phase 1**: Fix cache key issue (use fixed end date or ignore seconds)
2. **Phase 2**: Add incremental update logic
3. **Phase 3**: Add Redis/external cache for serverless

### For Local Development:

**Best approach**: **Option 1 (Smart Caching)**

**Why**:
- File-based cache works fine locally
- Incremental updates save API calls
- No infrastructure needed

---

## Current State Summary

| Aspect | Current | Ideal |
|--------|---------|------|
| **Caching** | ✅ Exists but ineffective | ✅ Smart incremental caching |
| **Cache Key** | ❌ Includes timestamp (changes every second) | ✅ Fixed dates or symbol-based |
| **Incremental Updates** | ❌ None | ✅ Fetch only missing days |
| **Serverless Support** | ❌ Ephemeral filesystem | ✅ External cache (Redis) |
| **API Efficiency** | ❌ Fetches full dataset every time | ✅ Fetches only what's needed |

---

## Next Steps

1. **Immediate Fix**: Change cache key to use date-only (not timestamp)
2. **Short-term**: Implement incremental update logic
3. **Long-term**: Add Redis cache for serverless deployment

Would you like me to implement any of these improvements?
