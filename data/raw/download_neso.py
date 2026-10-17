
import csv
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

BASE_URL = "https://api.neso.energy"
DATASET_ID = "historic-demand-data"

# Save files in your existing project structure.
OUTPUT_DIR = Path("energy-project/data/raw/demand")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

USER_AGENT = "energy-crisis-research/1.0"


def get_json(url):
    """Fetch JSON from the NESO API."""
    request = urllib.request.Request(
        url,
        headers={"User-Agent": USER_AGENT}
    )

    with urllib.request.urlopen(request, timeout=60) as response:
        return json.load(response)


def safe_filename(name):
    """Remove characters unsuitable for filenames."""
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", name)
    return name.strip("._") or "neso_resource"


def download_file(url, destination):
    """Download a resource unless a non-empty file already exists."""
    if destination.exists() and destination.stat().st_size > 0:
        print(f"Already downloaded: {destination.name}")
        return

    request = urllib.request.Request(
        url,
        headers={"User-Agent": USER_AGENT}
    )

    try:
        with urllib.request.urlopen(request, timeout=180) as response:
            content_type = response.headers.get("Content-Type", "")

            # Read the response before accepting the file.
            data = response.read()

        if not data:
            raise ValueError("The downloaded file is empty")

        # Avoid saving an API error page as a CSV.
        if "json" in content_type.lower():
            try:
                payload = json.loads(data)
                if isinstance(payload, dict) and payload.get("success") is False:
                    raise ValueError(f"NESO API error: {payload}")
            except json.JSONDecodeError:
                pass

        # Basic CSV validation.
        text_sample = data[:4096].decode("utf-8-sig", errors="replace")
        if "<html" in text_sample.lower() or "<!doctype html" in text_sample.lower():
            raise ValueError("The URL returned HTML, not a CSV file")

        destination.write_bytes(data)
        print(f"Downloaded: {destination} ({len(data):,} bytes)")

    except Exception:
        # Remove a partial/invalid file if one was written.
        if destination.exists():
            destination.unlink()
        raise


def main():
    # Discover the dataset and its resources.
    metadata_url = (
        f"{BASE_URL}/api/3/action/package_show?"
        + urllib.parse.urlencode({"id": DATASET_ID})
    )

    metadata = get_json(metadata_url)

    if not metadata.get("success"):
        raise RuntimeError("NESO did not return dataset metadata")

    resources = metadata["result"].get("resources", [])

    if not resources:
        raise RuntimeError(
            "No resources found. Check the dataset ID or NESO API."
        )

    print(f"Found {len(resources)} resources.")

    downloaded = 0

    for resource in resources:
        name = resource.get("name") or resource.get("id", "resource")
        url = resource.get("url")
        fmt = (resource.get("format") or "").upper()
        resource_id = resource.get("id")

        # Select CSV resources, or resources explicitly identified as CSV.
        is_csv = (
            fmt == "CSV"
            or (url and ".csv" in url.lower().split("?")[0])
        )

        if not is_csv:
            print(f"Skipping non-CSV resource: {name} ({fmt or 'unknown format'})")
            continue

        if not url:
            print(f"Skipping resource without a URL: {name}")
            continue

        # Add a stable resource ID to the filename where available.
        filename = safe_filename(name)
        if not filename.lower().endswith(".csv"):
            filename += ".csv"

        if resource_id:
            filename = (
                f"{safe_filename(str(resource_id))}_{filename}"
            )

        destination = OUTPUT_DIR / filename

        try:
            download_file(url, destination)
            downloaded += 1
        except (urllib.error.URLError, TimeoutError, ValueError) as error:
            print(f"Failed to download {name}: {error}")

        # Avoid hammering the public API.
        time.sleep(1)

    print(f"\nFinished. Successfully downloaded or retained {downloaded} CSV resources.")
    print(f"Output directory: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
