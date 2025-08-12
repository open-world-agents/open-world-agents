import os
import io
import hashlib
import pathlib
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

# --- CONFIGURATION ---
CLIENT_SECRET_FILE = "/mnt/raid12/workspace/jyjung/confidential/client_secret_146245970336-um8dpgb0utkqlmbt4gsuktg2am1bpk8s.apps.googleusercontent.com.json"
SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]
SHARED_DRIVE_ID = "0AHLjQqrHnATRUk9PVA"  # Shared Drive ID
ROOT_FOLDER_ID = "0AHLjQqrHnATRUk9PVA"  # Folder inside the Shared Drive
DOWNLOAD_EXTENSIONS = [".log", ".mkv", ".mcap"]
LOCAL_DOWNLOAD_DIR = "/mnt/raid12/datasets/owa_game_dataset"


# --- AUTHENTICATION ---
def authenticate():
    token_file = os.path.join(os.path.dirname(CLIENT_SECRET_FILE), "token.json")

    if os.path.exists(token_file):
        creds = Credentials.from_authorized_user_file(token_file, SCOPES)
    else:
        flow = InstalledAppFlow.from_client_secrets_file(CLIENT_SECRET_FILE, SCOPES)
        creds = flow.run_local_server(port=0)
        with open(token_file, "w") as token:
            token.write(creds.to_json())
    return build("drive", "v3", credentials=creds)


# --- LOCAL FILE HASHING ---
def compute_md5(file_path, chunk_size=8192):
    md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        while chunk := f.read(chunk_size):
            md5.update(chunk)
    return md5.hexdigest()


# --- RECURSIVE FILE FINDER ---
def list_files(service, folder_id, path="", shared_drive_id=None):
    all_files = []
    query = f"'{folder_id}' in parents and trashed = false"
    page_token = None

    print(f"Querying folder ID: {folder_id}, path: {path}")

    while True:
        response = (
            service.files()
            .list(
                q=query,
                spaces="drive",
                corpora="drive",
                driveId=shared_drive_id,
                includeItemsFromAllDrives=True,
                supportsAllDrives=True,
                fields="nextPageToken, files(id, name, mimeType, md5Checksum)",
                pageToken=page_token,
            )
            .execute()
        )

        for file in response.get("files", []):
            file_path = os.path.join(path, file["name"])
            if file["mimeType"] == "application/vnd.google-apps.folder":
                all_files += list_files(service, file["id"], file_path, shared_drive_id)
            else:
                ext = pathlib.Path(file["name"]).suffix.lower()
                if ext in DOWNLOAD_EXTENSIONS:
                    all_files.append(
                        {"id": file["id"], "name": file["name"], "path": file_path, "md5": file.get("md5Checksum")}
                    )

        page_token = response.get("nextPageToken", None)
        if not page_token:
            break

    return all_files


# --- FILE DOWNLOADER ---
def download_file(service, file_id, local_path):
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    request = service.files().get_media(fileId=file_id)
    with io.FileIO(local_path, "wb") as fh:
        downloader = MediaIoBaseDownload(fh, request)
        done = False
        while not done:
            status, done = downloader.next_chunk()
            if status:
                print(f"Downloading {local_path} - {int(status.progress() * 100)}%")


# --- MAIN ---
def main():
    service = authenticate()
    print("Scanning Shared Drive folder structure...")
    matched_files = list_files(service, ROOT_FOLDER_ID, shared_drive_id=SHARED_DRIVE_ID)
    print(f"Found {len(matched_files)} matching files.")

    for file in matched_files:
        local_path = os.path.join(LOCAL_DOWNLOAD_DIR, file["path"])

        # Check local existence and MD5
        if os.path.exists(local_path):
            if file["md5"] is not None:
                local_md5 = compute_md5(local_path)
                if local_md5 == file["md5"]:
                    print(f"✔ Skipping (hash match): {file['path']}")
                    continue
                else:
                    print(f"↻ Re-downloading (hash mismatch): {file['path']}")
            else:
                print(f"⚠ Skipping (no MD5 from Drive): {file['path']}")
                continue
        else:
            print(f"→ Downloading: {file['path']}")

        download_file(service, file["id"], local_path)


if __name__ == "__main__":
    main()
