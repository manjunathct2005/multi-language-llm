import os
import urllib.request
import py7zr

def download_and_extract_models():
    model_dir = "models"
    archive_path = "models.7z"

    # Step 1: Check if already exists
    if os.path.exists(model_dir) and os.listdir(model_dir):
        print("✅ Models already downloaded and extracted.")
        return

    # Step 2: Download
    url = "https://euc1p.proxy.ucloud.link/publink/download/show?code=XZ4j3B5ZsaPM45tUBakFhtWhDNYBEpAAoHcV"
    print("⬇️ Downloading models from:", url)

    try:
        urllib.request.urlretrieve(url, archive_path)
    except Exception as e:
        print(f"❌ Download failed: {e}")
        return

    if not os.path.exists(archive_path) or os.path.getsize(archive_path) < 100000:
        print("❌ Downloaded file is too small or missing. Check the link.")
        return

    # Step 3: Extract
    try:
        print("📦 Extracting models...")
        with py7zr.SevenZipFile(archive_path, mode='r') as archive:
            archive.extractall(path=model_dir)
        print("✅ Models extracted to:", model_dir)
    except Exception as e:
        print(f"❌ Extraction failed: {e}")
        return
    finally:
        if os.path.exists(archive_path):
            os.remove(archive_path)

if __name__ == "__main__":
    download_and_extract_models()
