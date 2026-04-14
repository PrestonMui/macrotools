"""
Storage module for caching and credentials management.

Handles:
- Data caching (parquet files with TTL, with .meta.json for DataFrame attrs)
- User-facing save/load (parquet by default, pickle for .pkl/.pickle extensions)
- Credentials storage (email, API keys, etc.)
"""

from typing import Optional, Dict
import pandas as pd
import json
import os
import time
from pathlib import Path


# ============================================================================
# Parquet + Metadata Helpers
# ============================================================================

def _save_attrs_meta(meta_path: Path, attrs: dict) -> None:
    """Save DataFrame.attrs to a companion JSON file."""
    if not attrs:
        return
    serializable = {}
    for key, value in attrs.items():
        try:
            json.dumps(value, default=str)
            serializable[key] = value
        except (TypeError, ValueError):
            serializable[key] = str(value)
    with open(meta_path, 'w') as f:
        json.dump(serializable, f, indent=2, default=str)


def _load_attrs_meta(meta_path: Path) -> dict:
    """Load DataFrame.attrs from a companion JSON file."""
    if not meta_path.exists():
        return {}
    try:
        with open(meta_path, 'r') as f:
            return json.load(f)
    except Exception:
        return {}


def _save_cache_file(data: pd.DataFrame, file_path: Path) -> None:
    """Save a DataFrame as feather + companion .meta.json for attrs."""
    index_name = data.index.name
    data.reset_index().to_feather(file_path)
    meta_path = file_path.with_suffix('.meta.json')
    attrs = dict(data.attrs)
    if index_name:
        attrs['__index_name__'] = index_name
    _save_attrs_meta(meta_path, attrs)


def _load_cache_file(file_path: Path, columns=None) -> pd.DataFrame:
    """Load a DataFrame from feather and restore attrs from companion .meta.json."""
    meta_path = file_path.with_suffix('.meta.json')
    meta = _load_attrs_meta(meta_path)
    index_name = meta.pop('__index_name__', None)
    # Always include the index column in the read so we can restore it
    read_columns = columns
    if columns is not None and index_name and index_name not in columns:
        read_columns = [index_name] + list(columns)
    data = pd.read_feather(file_path, columns=read_columns)
    if index_name and index_name in data.columns:
        data = data.set_index(index_name)
    data.attrs = meta
    return data


# ============================================================================
# User-Facing Save Helper
# ============================================================================

def _save_dataframe(data: pd.DataFrame, file_path: str) -> None:
    """
    Save a DataFrame to a file, inferring format from extension.

    .pkl / .pickle  -> pickle
    .parquet        -> parquet
    anything else   -> feather (default)
    """
    path = Path(file_path)
    if path.suffix.lower() in ('.pkl', '.pickle'):
        data.to_pickle(path)
    elif path.suffix.lower() == '.parquet':
        data.to_parquet(path)
    else:
        data.reset_index().to_feather(path)


# ============================================================================
# Cache Management
# ============================================================================

def _get_cache_dir():
    """Get cache directory from environment or default."""
    cache_dir = os.getenv('MACRODATA_CACHE_DIR')
    if cache_dir:
        return Path(cache_dir)
    return Path.home() / '.macrodata_cache'


def _get_cache_file_path(source: str, freq: str) -> Path:
    """Generate cache file path for given parameters."""
    cache_dir = _get_cache_dir()
    cache_dir.mkdir(parents=True, exist_ok=True)
    ext = '.parquet' if freq == 'all' else '.feather'
    return cache_dir / f'{source}_{freq}{ext}'


def _get_cache_age_days(cache_file: Path) -> Optional[float]:
    """Get age of cache file in days. Returns None if file doesn't exist."""
    if not cache_file.exists():
        return None
    mtime = cache_file.stat().st_mtime
    age_seconds = time.time() - mtime
    return age_seconds / 86400


def _should_refresh_cache(cache_file: Path, ttl_days: int = 7) -> bool:
    """Check if cache should be refreshed (older than ttl_days)."""
    age = _get_cache_age_days(cache_file)
    if age is None:
        return True  # No cache exists
    return age >= ttl_days


def _load_cached_data(source: str, freq: str, columns=None) -> Optional[pd.DataFrame]:
    """Load data from cache if it exists and is valid."""
    cache_file = _get_cache_file_path(source, freq)
    if cache_file.exists():
        try:
            if freq == 'all':
                data = pd.read_parquet(cache_file, columns=columns)
            else:
                data = _load_cache_file(cache_file, columns=columns)
            age = _get_cache_age_days(cache_file)
            print(f"Loaded {source} from cache ({age:.1f} days old)")
            return data
        except Exception as e:
            print(f"Warning: Could not load cache for {source}: {e}")
            return None
    return None


def _save_cached_data(data: pd.DataFrame, source: str, freq: str) -> None:
    """Save data to cache."""
    cache_file = _get_cache_file_path(source, freq)
    try:
        if freq == 'all':
            data.to_parquet(cache_file)
        else:
            _save_cache_file(data, cache_file)
    except Exception as e:
        print(f"Warning: Could not save cache for {source}: {e}")


def get_cache_age(source: str, freq: str = 'M') -> Optional[float]:
    """
    Get age of cached data in days.

    Parameters:
        source : str; Data source (e.g., 'ce', 'nipa-pce')
        freq : str; Frequency key (e.g., 'M', 'Q', 'A', 'all', 'default')

    Returns:
        float or None; Age in days if cached, None if not cached
    """
    cache_file = _get_cache_file_path(source, freq)
    return _get_cache_age_days(cache_file)


def clear_macrodata_cache(source: Optional[str] = None) -> None:
    """
    Clear cached data files.

    Parameters:
    -----------
    source : str, optional
        If provided, only clear cache for this source.
        If None, clear all cached files.
    """
    cache_dir = _get_cache_dir()

    if not cache_dir.exists():
        print("Cache directory does not exist.")
        return

    if source:
        # Clear specific source (parquet + meta + legacy pkl)
        for pattern in [f'{source}_*.feather', f'{source}_*.meta.json', f'{source}_*.parquet', f'{source}_*.pkl']:
            for cache_file in cache_dir.glob(pattern):
                try:
                    cache_file.unlink()
                    print(f"Cleared cache: {cache_file.name}")
                except Exception as e:
                    print(f"Error deleting {cache_file.name}: {e}")
    else:
        # Clear all caches (feather + meta + legacy parquet/pkl)
        for pattern in ['*.feather', '*.meta.json', '*.parquet', '*.pkl']:
            for cache_file in cache_dir.glob(pattern):
                try:
                    cache_file.unlink()
                    print(f"Cleared cache: {cache_file.name}")
                except Exception as e:
                    print(f"Error deleting {cache_file.name}: {e}")
        print("All cached data cleared.")


# ============================================================================
# Credentials Management
# ============================================================================

def _get_credentials_dir():
    """Get credentials directory in user's home."""
    return Path.home() / '.macrodata_credentials'


def _get_credentials_file_path():
    """Get path to credentials JSON file."""
    cred_dir = _get_credentials_dir()
    cred_dir.mkdir(parents=True, exist_ok=True)
    return cred_dir / 'credentials.json'


def _load_credentials() -> Dict:
    """Load credentials from file."""
    cred_file = _get_credentials_file_path()
    if cred_file.exists():
        try:
            with open(cred_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Warning: Could not load credentials: {e}")
            return {}
    return {}


def _save_credentials(credentials: Dict) -> None:
    """Save credentials to file."""
    cred_file = _get_credentials_file_path()
    try:
        with open(cred_file, 'w') as f:
            json.dump(credentials, f, indent=2)
        try:
            os.chmod(cred_file, 0o600)
        except OSError:
            pass  # Windows doesn't support Unix file permissions
    except Exception as e:
        print(f"Warning: Could not save credentials: {e}")


_CREDENTIAL_REGISTRY = {
    "email": {
        "json_key": "email",
        "env_var": "MACROTOOLS_EMAIL",
        "prompt_message": "Please enter your email address for BLS data pulls: ",
        "store_success_message": "Email stored in {path}",
        "help_text": None,
    },
    "fred_api_key": {
        "json_key": "fred_api_key",
        "env_var": "FRED_API_KEY",
        "prompt_message": "Please enter your FRED API key: ",
        "store_success_message": "FRED API key stored in {path}",
        "help_text": "FRED API key is required. Get one free at: https://fred.stlouisfed.org/docs/api/api_key.html",
    },
    "bls_api_key": {
        "json_key": "bls_api_key",
        "env_var": "BLS_API_KEY",
        "prompt_message": "Please enter your BLS API key: ",
        "store_success_message": "BLS API key stored in {path}",
        "help_text": "BLS API key is required. Register free at: https://data.bls.gov/registrationEngine/",
    },
}


def _validate_credential_name(name: str) -> dict:
    """Validate credential name and return its registry entry."""
    if name not in _CREDENTIAL_REGISTRY:
        raise ValueError(
            f"Unknown credential: '{name}'. "
            f"Valid names: {', '.join(sorted(_CREDENTIAL_REGISTRY))}"
        )
    return _CREDENTIAL_REGISTRY[name]


def store_credential(name: str, value: str) -> None:
    """
    Store a credential by name.

    Parameters
    ----------
    name : str
        Credential name. One of: 'email', 'fred_api_key', 'bls_api_key'.
    value : str
        The credential value to store.
    """
    config = _validate_credential_name(name)
    credentials = _load_credentials()
    credentials[config["json_key"]] = value
    _save_credentials(credentials)
    print(config["store_success_message"].format(path=_get_credentials_file_path()))


def get_stored_credential(name: str) -> Optional[str]:
    """
    Get a stored credential by name.

    Parameters
    ----------
    name : str
        Credential name. One of: 'email', 'fred_api_key', 'bls_api_key'.

    Returns
    -------
    str or None
        The stored value, or None if not set.
    """
    config = _validate_credential_name(name)
    credentials = _load_credentials()
    return credentials.get(config["json_key"])


def _resolve_credential(name: str, value: Optional[str] = None) -> str:
    """
    Resolve a credential using the priority chain:
    argument > stored > environment variable > interactive prompt.

    Parameters
    ----------
    name : str
        Credential name (must be a key in _CREDENTIAL_REGISTRY).
    value : str, optional
        Explicitly provided value (highest priority).

    Returns
    -------
    str
        The resolved credential value.
    """
    if value:
        return value

    stored = get_stored_credential(name)
    if stored:
        return stored

    config = _CREDENTIAL_REGISTRY[name]
    env_val = os.getenv(config["env_var"])
    if env_val:
        return env_val

    # Interactive prompt
    if config["help_text"]:
        print(config["help_text"])
    value = input(config["prompt_message"]).strip()
    if value:
        cred_path = _get_credentials_file_path()
        store_choice = input(f"Credentials are stored as plain text in {cred_path}. Store for future use? (y/n): ").strip().lower()
        if store_choice == 'y':
            store_credential(name, value)

    return value
