# How to Create a Release and Host Large Model Assets on Internal GitLab

This guide explains how to create a Release for `edsan-doc-trimmer` on your hospital's internal, on-premises GitLab instance and attach the model archive (`edsan-doc-trimmer-v1.1.0.zip`, containing `model.onnx` for CPU and `model.safetensors` for CUDA acceleration) without bloating git history.

---

## ⚡ Quick Diagnostic: "I Can't Find How to Create a Release!"

If you are looking at your hospital's GitLab and cannot find the "New Release" button, check these 4 points:

### 1. Direct URL (Bypasses all navigation menus)
Paste this directly into your browser:
```
https://<your-gitlab-host>/<group-or-user>/edsan-doc-trimmer/-/releases/new
```
For example: `https://gitlab.hospital.fr/redsan/edsan-doc-trimmer/-/releases/new`.
If this URL opens the release form, you're ready! If it says 404 or 403, check items 3 and 4 below.

### 2. Where the menu is located in the left sidebar:
* **GitLab 16+ (Current)**: Left sidebar -> **Deploy** (rocket icon) -> **Releases** -> click blue **New release** button (top right).
* **GitLab 14 – 15**: Left sidebar -> **Deployments** (or **Project overview**) -> **Releases** -> click **New release**.
* **Shortcut via Tags**: Left sidebar -> **Code** -> **Tags** -> find tag `v1.1.0` -> click the speech bubble icon / **Add release notes**.

### 3. Check your Project Permissions (Must be Developer or Maintainer)
* In GitLab, **Reporter** and **Guest** roles **cannot** create releases or tags.
* To check: Look under **Manage** -> **Members**. If your role is "Reporter", contact your hospital project owner to grant you **Developer** or **Maintainer** rights.

### 4. Check if the "Releases" feature is disabled in Settings
If the "Releases" menu item is completely invisible in the sidebar:
1. Go to **Settings** -> **General** (left sidebar bottom).
2. Expand **Visibility, project features, permissions**.
3. Scroll to **Releases** and verify the toggle is **ON** (green).
4. Click **Save changes**. The menu will immediately appear under **Deploy > Releases**.

---

## 1. Core Principles: Why GitLab Works This Way

### Why Does GitLab Require a Tag First?
* In Git, a **Tag** is an immutable reference pointing to a specific commit hash (e.g. `v1.1.0`).
* In GitLab, a **Release** is not an independent file folder; it is **metadata, release notes, and attached download links bound to a Git Tag**.
* Therefore, you cannot have a Release without a Tag. You can either:
  1. Select an existing Git Tag (e.g. create beforehand via `git tag v1.1.0 && git push origin v1.1.0`), OR
  2. Type `v1.1.0` in the "Tag name" field of the New Release form and select **Create tag from: main**.

### Why Must 442 MB Binaries NEVER Be Committed to Git?
* **Git history is forever**: If you run `git add model.onnx` or `git add edsan-doc-trimmer-v1.1.0.zip` and commit, that 442 MB blob is baked permanently into `.git/objects`.
* **Permanent clone penalty**: Every colleague who clones the repository (`git clone`) will be forced to download all 442 MB across the hospital network, even if they only want to edit documentation or R code.
* **Server-side rejection**: Many enterprise hospital GitLab installations set a push limit (e.g., 50 MB or 100 MB per commit). Pushing a 442 MB file directly will fail with `remote: fatal: pack exceeds maximum allowed size`.
* **Why web UI drag-and-drop fails**: GitLab's web markdown description has a default file upload limit of **10 MB**. Dragging a 442 MB zip into the release notes will fail with HTTP 413 ("Request Entity Too Large").
* **The Solution**: Upload the binary to the **GitLab Generic Package Registry** (designed for assets up to 5 GB), and link it as a **Release Asset Link**.


---

## 3. How to Attach the 442 MB Binary Asset

Choose the method best suited for your hospital infrastructure:

### Method A: GitLab Generic Package Registry (Recommended Best Practice)
GitLab includes a built-in Generic Package Registry designed specifically for storing release binaries and model weights of any size.

#### Step 1: Find your Project ID
1. Go to your project's main page in GitLab.
2. Directly below the project title, find the **Project ID** (e.g. `Project ID: 1428`).

#### Step 2: Generate a Personal Access Token (or Project Access Token)
1. In GitLab top right, click your avatar -> **Preferences** -> **Access Tokens**.
2. Name: `model-uploader`.
3. Scopes: Check `api` and `write_repository`.
4. Click **Create personal access token** and copy the token (`glpat-xxxxxxxxxxxx`).

#### Step 3: Upload the Zip File via cURL
Run this command from your terminal (PowerShell, Bash, or Command Prompt) where `edsan-doc-trimmer-v1.1.0.zip` is located:

```bash
# Set your variables
GITLAB_URL="https://gitlab.hospital.fr"       # Your internal GitLab base URL
PROJECT_ID="1428"                              # Your Project ID from Step 1
TOKEN="glpat-xxxxxxxxxxxx"                     # Your Access Token from Step 2
ZIP_FILE="artifacts/edsan-doc-trimmer-v1.1.0.zip"

# Upload to Generic Package Registry
curl --header "PRIVATE-TOKEN: ${TOKEN}" \
     --upload-file "${ZIP_FILE}" \
     "${GITLAB_URL}/api/v4/projects/${PROJECT_ID}/packages/generic/edsan-doc-trimmer/1.1.0/edsan-doc-trimmer-v1.1.0.zip"
```

*In Windows PowerShell:*
```powershell
$headers = @{ "PRIVATE-TOKEN" = "glpat-xxxxxxxxxxxx" }
Invoke-RestMethod -Uri "https://gitlab.hospital.fr/api/v4/projects/1428/packages/generic/edsan-doc-trimmer/1.1.0/edsan-doc-trimmer-v1.1.0.zip" `
                  -Method Put `
                  -Headers $headers `
                  -InFile "artifacts\edsan-doc-trimmer-v1.1.0.zip"
```

Once uploaded, the permanent download URL for anyone in the hospital with read access is:
```
https://gitlab.hospital.fr/api/v4/projects/1428/packages/generic/edsan-doc-trimmer/1.1.0/edsan-doc-trimmer-v1.1.0.zip
```

#### Step 4: Link the Package in the Release
1. In GitLab, navigate to **Deploy** -> **Releases** -> **New release**.
2. **Tag name**: Select or type `v1.1.0`.
3. **Create from**: Select `main`.
4. **Release title**: `edsan-doc-trimmer v1.1.0: Production DrBERT Document Trimmer`.
5. Under **Release assets** -> **Release asset links**:
   * **URL**: `https://gitlab.hospital.fr/api/v4/projects/1428/packages/generic/edsan-doc-trimmer/1.1.0/edsan-doc-trimmer-v1.1.0.zip`
   * **Link title**: `edsan-doc-trimmer-v1.1.0.zip (DrBERT ONNX CPU + PyTorch CUDA Safetensors)`
   * **Type**: `Package`
6. Click **Create release**.

---

### Method B: GitLab Web UI Direct Drag-and-Drop (If Instance Allows Large Uploads)
If your hospital's GitLab omnibus configuration has `max_attachment_size` configured to >= 1 GB:

1. Navigate to **Deploy** -> **Releases** -> **New release**.
2. In the **Release notes** Markdown text box, drag and drop `artifacts/edsan-doc-trimmer-v1.1.0.zip` directly onto the text box.
3. Wait for the upload bar to complete. GitLab will output a markdown link:
   ```markdown
   [Download edsan-doc-trimmer-v1.1.0.zip](/uploads/1a2b3c.../edsan-doc-trimmer-v1.1.0.zip)
   ```
4. Copy that URL and paste it under **Release assets** -> **Release asset links** -> **URL**.
5. Click **Create release**.

*(Note: If the upload fails with a 413 "Request Entity Too Large" error, your GitLab instance has an upload size limit. Use Method A or Method C instead).*

---

### Method C: Hospital HDW Shared Network Storage (Standard for Air-Gapped Platforms)
On hospital computing servers, multiple data scientists and scripts usually share files via an internal NFS mount or Windows network share:

1. Copy `edsan-doc-trimmer-v1.1.0.zip` directly to the shared models directory and unpack:
   ```bash
   # On Linux HDW server:
   mkdir -p /data/shared/models/edsan-doc-trimmer/v1.1.0
   unzip artifacts/edsan-doc-trimmer-v1.1.0.zip -d /data/shared/models/edsan-doc-trimmer/v1.1.0/
   chmod -R a+rX /data/shared/models/edsan-doc-trimmer/
   ```
   *On Windows server (PowerShell):*
   ```powershell
   New-Item -ItemType Directory -Force -Path "D:\shared\models\edsan-doc-trimmer\v1.1.0"
   Expand-Archive -Path "artifacts\edsan-doc-trimmer-v1.1.0.zip" -DestinationPath "D:\shared\models\edsan-doc-trimmer\v1.1.0" -Force
   ```
2. In the GitLab Release description, document the exact path:
   ```markdown
   ### Shared Server Deployment (HDW / On-Premises)
   The production model is pre-installed on the shared cluster at:
   `/data/shared/models/edsan-doc-trimmer/v1.1.0`

   To use it in your environment:
   ```bash
   export EDSAN_TRIMMER_PATH="/data/shared/models/edsan-doc-trimmer/v1.1.0"
   ```
   Or in R:
   ```r
   Sys.setenv(EDSAN_TRIMMER_PATH = "/data/shared/models/edsan-doc-trimmer/v1.1.0")
   ```
3. Create the Release on GitLab referencing this location.

---

## 4. How Users and Colleagues Download & Install

Once the Release is published:

### For R Users (`redsan`)
```r
library(redsan)

# Option A: Install from the downloaded zip file into persistent user cache:
edsan_install_trimmer("/path/to/downloaded/edsan-doc-trimmer-v1.1.0.zip")

# Option B: Point directly to shared HDW server model:
Sys.setenv(EDSAN_TRIMMER_PATH = "/data/shared/models/edsan-doc-trimmer/v1.1.0")

# If Python is in a custom virtual environment:
Sys.setenv(REDSAN_PYTHON_PATH = "/path/to/python")

# Verify:
trimmed <- trim_doceds_onnx("Consultation du 12/03/2024. Patient vu pour controle.")
```

### For Python Users
```bash
# Option A: Extract inside the cloned repo
# Linux / macOS:
unzip edsan-doc-trimmer-v1.1.0.zip -d artifacts/active_learning/onnx_export
# Windows PowerShell:
Expand-Archive -Path edsan-doc-trimmer-v1.1.0.zip -DestinationPath artifacts/active_learning/onnx_export -Force

# Option B: Extract to a custom path and export environment variable
export EDSAN_TRIMMER_PATH="/path/to/extracted_model"
# Windows PowerShell:
# $env:EDSAN_TRIMMER_PATH = "C:\path\to\extracted_model"

# Run batch worker or document trimmer
python scripts/trim_document.py
python scripts/trim_batch_service.py -i documents.json -o trimmed.json
```
