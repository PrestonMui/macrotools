"""
Storage module for caching and credentials management.

Handles:
- Data caching (pickle files with TTL)
- Credentials storage (email, API keys, etc.)
"""

from typing import Optional, Dict
import pandas as pd
import json
import os
import time
from pathlib import Path


# ============================================================================
# Cache Management
# ============================================================================

def _get_cache_dir():
    """Get cache directory from environment or default."""
    cache_dir = os.getenv('MACRODATA_CACHE_DIR')
    if cache_dir:
        return Path(cache_dir)
    return Path.home() / '.macrodata_cache'


def _get_cache_file_path(source: str, pivot: bool) -> Path:
    """Generate cache file path for given parameters."""
    cache_dir = _get_cache_dir()
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f'{source}_{pivot}.pkl'


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


def _load_cached_data(source: str, pivot: bool) -> Optional[pd.DataFrame]:
    """Load data from cache if it exists and is valid."""
    cache_file = _get_cache_file_path(source, pivot)
    if cache_file.exists():
        try:
            data = pd.read_pickle(cache_file)
            age = _get_cache_age_days(cache_file)
            print(f"Loaded {source} from cache ({age:.1f} days old)")
            return data
        except Exception as e:
            print(f"Warning: Could not load cache for {source}: {e}")
            return None
    return None


def _save_cached_data(data: pd.DataFrame, source: str, pivot: bool) -> None:
    """Save data to cache."""
    cache_file = _get_cache_file_path(source, pivot)
    try:
        data.to_pickle(cache_file)
    except Exception as e:
        print(f"Warning: Could not save cache for {source}: {e}")


def get_cache_age(source: str, pivot: bool = True) -> Optional[float]:
    """
    Get age of cached data in days.

    Parameters:
        source : str; Data source (e.g., 'ce', 'nipa-pce')
        pivot : bool; Whether data is pivoted

    Returns:
        float or None; Age in days if cached, None if not cached
    """
    cache_file = _get_cache_file_path(source, pivot)
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
        # Clear specific source
        for cache_file in cache_dir.glob(f'{source}_*.pkl'):
            try:
                cache_file.unlink()
                print(f"Cleared cache: {cache_file.name}")
            except Exception as e:
                print(f"Error deleting {cache_file.name}: {e}")
    else:
        # Clear all caches
        for cache_file in cache_dir.glob('*.pkl'):
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
    except Exception as e:
        print(f"Warning: Could not save credentials: {e}")


def store_email(email: str) -> None:
    """
    Store an email address for BLS data pulls.

    Parameters:
    -----------
    email : str
        Email address to store
    """
    credentials = _load_credentials()
    credentials['email'] = email
    _save_credentials(credentials)
    print(f"Email stored: {email}")


def get_stored_email() -> Optional[str]:
    """
    Get the stored email address.

    Returns:
    --------
    str or None
        The stored email address, or None if not set
    """
    credentials = _load_credentials()
    return credentials.get('email')


def _get_email_for_bls(email: Optional[str] = None) -> str:
    """
    Get email to use for BLS requests.

    Priority:
    1. If email is provided as argument, use it
    2. If email is stored, use stored email
    3. Otherwise, prompt user for email

    Parameters:
    -----------
    email : str, optional
        Email address provided by user

    Returns:
    --------
    str
        Email address to use
    """
    if email:
        return email

    stored_email = get_stored_email()
    if stored_email:
        return stored_email

    # Prompt user for email
    email = input("No email provided or stored. Please enter your email address for BLS data pulls: ").strip()
    if email:
        # Ask if user wants to store it
        store_choice = input(f"Would you like to store this email for future use? (y/n): ").strip().lower()
        if store_choice == 'y':
            store_email(email)

    return email


def store_fred_api_key(api_key: str) -> None:
    """
    Store a FRED API key for data pulls.

    Parameters:
    -----------
    api_key : str
        FRED API key to store
    """
    credentials = _load_credentials()
    credentials['fred_api_key'] = api_key
    _save_credentials(credentials)
    print("FRED API key stored successfully")


def get_stored_fred_api_key() -> Optional[str]:
    """
    Get the stored FRED API key.

    Returns:
    --------
    str or None
        The stored API key, or None if not set
    """
    credentials = _load_credentials()
    return credentials.get('fred_api_key')


def _get_fred_api_key(api_key: Optional[str] = None) -> str:
    """
    Get FRED API key to use for requests.

    Priority:
    1. If api_key is provided as argument, use it
    2. If api_key is stored, use stored api_key
    3. Check environment variable FRED_API_KEY
    4. Otherwise, prompt user for api_key

    Parameters:
    -----------
    api_key : str, optional
        API key provided by user

    Returns:
    --------
    str
        API key to use
    """
    if api_key:
        return api_key

    stored_key = get_stored_fred_api_key()
    if stored_key:
        return stored_key

    # Check environment variable
    env_key = os.getenv('FRED_API_KEY')
    if env_key:
        return env_key

    # Prompt user for API key
    print("FRED API key is required. Get one free at: https://fred.stlouisfed.org/docs/api/api_key.html")
    api_key = input("Please enter your FRED API key: ").strip()
    if api_key:
        store_choice = input("Would you like to store this API key for future use? (y/n): ").strip().lower()
        if store_choice == 'y':
            store_fred_api_key(api_key)

    return api_key
