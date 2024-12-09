import requests
from pathlib import Path

def download_or_cached(url):
    cache_path = Path("cache")
    cache_path.mkdir(parents=True, exist_ok=True)
    local_file = cache_path / url.rpartition('/')[-1]

    if local_file.is_file():
        return local_file
    
    # NOTE the stream=True parameter below
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(local_file, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192): 
                # If you have chunk encoded response uncomment if
                # and set chunk_size parameter to None.
                #if chunk: 
                f.write(chunk)
    return local_file