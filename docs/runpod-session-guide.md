# Runpod Session Guide — video_to_data (CARI4D, Track 1)

How to get a working A100 pod again, from any Mac or Windows computer.

---

## What persists and what doesn't

| Thing | Where it lives | Survives a pod ending? |
|---|---|---|
| Docker image | `ghcr.io/melanieww/v2d-runpod:v1.1` (GitHub Container Registry) | ✅ |
| Template | Runpod → My Templates | ✅ |
| Secrets (`HF_TOKEN`, Kaggle) | Runpod → Secrets | ✅ |
| Registry credential `ghcr` | Runpod → Credentials | ✅ |
| Network volume `hostile_purple_louse` (250 GB, **EU-RO-1**) | Mounted at **`/vol`** | ✅ |
| The pod itself | — | ❌ Deploy a new one each session |
| Container disk (`/`, `/root`, apt installs, `~/.bashrc`) | — | ❌ Wiped |

**Rule:** anything you want to keep goes in `/vol`. Code changes go to GitHub.

Pods with a network volume are normally **terminated** rather than stopped, so you deploy a fresh pod each session. That's expected: the image, template and volume make it a ~5 minute routine.

Key names:

- Fork: `MelanieWW/video_to_data` (`origin`), upstream `Beardear/video_to_data`
- Image: `ghcr.io/melanieww/v2d-runpod:v1.1`
- Image code (baked in, read-only workflow): `/workspace/v2d_*`
- Volume layout: `/vol/video_to_data`, `/vol/data`, `/vol/weights`, `/vol/outputs`, `/vol/submission`, `/vol/cache`, `/vol/secrets`

---

## Part A — One-time setup on a NEW computer

Do this once per computer. Skip it on a computer that already works.

### A1. Create an SSH key

**Mac (Terminal):**
```bash
ssh-keygen -t ed25519            # press Enter to accept ~/.ssh/id_ed25519; set a passphrase
cat ~/.ssh/id_ed25519.pub | pbcopy
```

**Windows (CMD):**
```
ssh-keygen -t ed25519
type %USERPROFILE%\.ssh\id_ed25519.pub | clip
```

Never share or upload the file **without** `.pub`; that's the private key.

### A2. Add the public key to the Runpod TEMPLATE

Runpod's account-level SSH keys did **not** reach this image's pods, so the key must be in the template itself.

Each computer gets its **own variable**. Leave the existing ones untouched.

| Variable | Computer |
|---|---|
| `PUBLIC_KEY` | MacBook (already set) |
| `PUBLIC_KEY_2` | second computer (e.g. Windows PC) |

1. Runpod → **My Templates** → edit the template → Environment variables → **+ Add Environment variable**.
2. Key: `PUBLIC_KEY_2`. Value: the new computer's full `.pub` line (`ssh-ed25519 AAAAC3... user@host`).
3. Save. Only pods deployed **after** saving get the key.

The image's `start.sh` (v1.1+) reads `PUBLIC_KEY`, `PUBLIC_KEY_2` and `PUBLIC_KEY_3`, so up to three computers need only template variables. A fourth would need a `start.sh` change and an image rebuild (Part E).

### A3. Add the public key to GitHub

Install GitHub CLI if needed:

- Mac: `brew install gh`
- Windows: `winget install --id GitHub.cli`

Then:
```bash
gh auth login                                   # GitHub.com → HTTPS → Yes → browser
gh auth refresh -h github.com -s admin:public_key,workflow
gh ssh-key add ~/.ssh/id_ed25519.pub --title "<computer name>"
```
On Windows use `%USERPROFILE%\.ssh\id_ed25519.pub` as the path.

### A4. Load the key into the SSH agent

This is needed for `-A`, so the pod can push to GitHub using your key.

**Mac:**
```bash
ssh-add --apple-use-keychain ~/.ssh/id_ed25519
ssh-add -l          # should list the key
```

**Windows:** first, once, in an **admin PowerShell**:
```
Get-Service ssh-agent | Set-Service -StartupType Automatic
Start-Service ssh-agent
```
Then in CMD:
```
ssh-add %USERPROFILE%\.ssh\id_ed25519
ssh-add -l
```

### A5. Test

```bash
ssh -T git@github.com      # → "Hi MelanieWW! You've successfully authenticated..."
```

### A6. (Optional) Local copy of the repo, for rebuilding the image

```bash
git clone https://github.com/MelanieWW/video_to_data.git
cd video_to_data
git remote add upstream https://github.com/Beardear/video_to_data.git
gh repo set-default MelanieWW/video_to_data
```

---

## Part B — Template settings (reference)

The template should hold all of this permanently, so no overrides are needed at deploy time.

| Field | Value |
|---|---|
| Container image | `ghcr.io/melanieww/v2d-runpod:v1.1` |
| Registry credential | `ghcr` (Docker type; username `MelanieWW`, classic token with `read:packages`) |
| Container disk | 50 GB |
| Volume mount path | `/vol` |
| HTTP ports | `8080` (output browser) |
| TCP ports | `22` (SSH) |

**Start command:** leave **empty**. Since `v1.1`, the image's `start.sh` installs the SSH keys itself.

**Environment variables:**

| Key | Value |
|---|---|
| `PUBLIC_KEY` | MacBook's `.pub` line |
| `PUBLIC_KEY_2` | second computer's `.pub` line *(add when set up)* |
| `PUBLIC_KEY_3` | third computer's `.pub` line *(optional)* |
| `HF_TOKEN` | `{{ RUNPOD_SECRET_HF_TOKEN }}` |
| `KAGGLE_USERNAME` | `{{ RUNPOD_SECRET_KAGGLE_USERNAME }}` *(once the secret exists)* |
| `KAGGLE_KEY` | `{{ RUNPOD_SECRET_KAGGLE_KEY }}` *(once the secret exists)* |
| `HF_HOME` | `/vol/cache/hf` |
| `TORCH_HOME` | `/vol/cache/torch` |
| `HF_HUB_ENABLE_HF_TRANSFER` | `1` |

Keep the `{{ ... }}` placeholders literally; Runpod fills in the secret values at start.

---

## Part C — Every session

### C1. Deploy

1. Go to https://console.runpod.io/deploy → **GPU** tab.
2. Set **Network volume** (next to Search GPUs) to `hostile_purple_louse`. This limits GPUs to EU-RO-1. **Don't skip this**: without it you can get a GPU in another region with no volume attached.
3. Pick **A100 SXM** (80 GB) or **A100 PCIe** (80 GB). Never a 40 GB card. Avoid H100 until `sm_90` support is confirmed.
4. **Change Template** → **My Templates** → yours.
5. Check the bottom: **Persistent storage** shows `hostile_purple_louse` at `/vol`.
6. Click **Deploy On-Demand**. If both A100s are out of capacity, wait and retry, or use "Deploy when available".

### C2. Connect

Pod → **Connect** → copy the **direct TCP** command (`root@<IP> -p <PORT>`), and add `-A`:

**Mac:**
```bash
ssh -A root@<IP> -p <PORT> -i ~/.ssh/id_ed25519
```

**Windows (CMD):**
```
ssh -A root@<IP> -p <PORT> -i %USERPROFILE%\.ssh\id_ed25519
```

Answer `yes` to the fingerprint question.

### C3. Verify (30 seconds)

```bash
nvidia-smi                                  # A100 80GB
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_capability())"   # 2.5.1+cu124 True (8, 0)
df -h /vol                                  # ~250 GB network volume, NOT "overlay 50G"
echo ${#HF_TOKEN}                           # non-zero (37)
ssh -T git@github.com                       # Hi MelanieWW!
```

**If `df -h /vol` shows `overlay 50G`, stop.** The volume isn't attached. Terminate and redeploy with the network-volume filter set (C1 step 2).

### C4. Restore the session environment

The container disk is wiped each time, so run the bootstrap script:
```bash
bash /vol/bootstrap.sh
source ~/.bashrc
```

Create `/vol/bootstrap.sh` **once**. It persists on the volume:
```bash
cat > /vol/bootstrap.sh <<'EOF'
#!/bin/bash
# Re-applies per-pod settings lost when the container disk is wiped.
mkdir -p /vol/{data,weights,outputs/logs,submission,cache,secrets}

git config --global user.name  "Moran"
git config --global user.email "<your GitHub email>"
git config --global --add safe.directory /vol/video_to_data

# Kaggle credentials, if saved on the volume
if [ -f /vol/secrets/kaggle.json ]; then
  mkdir -p ~/.kaggle && ln -sf /vol/secrets/kaggle.json ~/.kaggle/kaggle.json
  chmod 600 /vol/secrets/kaggle.json
fi

grep -q "cd /vol/video_to_data" ~/.bashrc || cat >> ~/.bashrc <<'RC'
cd /vol/video_to_data 2>/dev/null
alias sam2py=/opt/venvs/sam2/bin/python
RC

# Extra apt packages beyond the image: add here
# apt-get update && apt-get install -y <pkgs>
echo "bootstrap done"
EOF
chmod +x /vol/bootstrap.sh
```

### C5. Work

- Use **tmux** for anything long, so it survives disconnects: `tmux new -s run`. Detach with `Ctrl+B` then `D`; reattach with `tmux attach -t run`.
- Write outputs and logs under `/vol/outputs`.
- Browse outputs in a web browser: Pod → Connect → **HTTP 8080**.

### C6. Before ending the session

```bash
cd /vol/video_to_data
git status
git add -A && git commit -m "<message>" && git push origin main
ls /vol/outputs                             # results are on the volume, not the container disk
```

Then, in Runpod: Pod → **Stop / Terminate**.

**The A100 bills every second it runs. Don't leave it idle.** The volume keeps billing a small storage fee regardless.

---

## Part D — Troubleshooting

| Symptom | Fix |
|---|---|
| Asks `root@...'s password:` | The pod doesn't have your key. Check this computer's `.pub` line is in `PUBLIC_KEY`, `PUBLIC_KEY_2` or `PUBLIC_KEY_3`, the template image is `v1.1` or later, and the **Start command** is empty. Then terminate and redeploy. Logs → **Container** shows `DIAG keylen=... key2len=... key3len=...`; a zero means that variable didn't reach the pod. |
| Asks `Enter passphrase for key` | Normal. That's your local key's passphrase. Run `ssh-add` (A4) to stop the prompts. |
| `REMOTE HOST IDENTIFICATION HAS CHANGED` | `ssh-keygen -R "[<IP>]:<PORT>"`, then reconnect. |
| `df -h /vol` shows `overlay 50G` | Volume not attached (wrong region). Redeploy with the network-volume filter. |
| `git@github.com: Permission denied (publickey)` on the pod | On your computer: `ssh-add -l` must list the key (A4), the key must be on GitHub (A3), and you must connect with `-A`. |
| Pod stuck pulling the image, or "unauthorized" | Check the `ghcr` credential (token has `read:packages`) and that the image name is all lowercase. |
| GPU "Out of capacity" | Try the other A100 type in EU-RO-1, wait, or use "Deploy when available". The volume can't move regions without copying. |
| `HF_TOKEN` length 0 | The template's `HF_TOKEN` variable is missing or the secret name doesn't match. |

---

## Part E — Rebuilding the image

> ⚠️ **Only rebuild when the image itself must change.** A rebuild takes about 50 minutes. Every template that uses the image (yours and your teammate's) must then be updated to the new tag. Most day-to-day work never needs one.

**Rebuild when you change:**
- `runpod-image/Dockerfile` or `runpod-image/start.sh`
- `reconstruction/modules/v2d_cari4d/docker/Dockerfile`
- code that the image bakes into `/workspace/v2d_*` (anything under `reconstruction/modules/` that the Dockerfile `COPY`s), **if** you run it from `/workspace` and need the change there
- system packages or Python dependencies that must be present on every pod
- support for a 4th computer's SSH key

**Don't rebuild for:**
- code you run from `/vol/video_to_data`. Commit and push to GitHub instead.
- data, weights or outputs. These live on `/vol`.
- template settings: environment variables, secrets, ports, disk size, `PUBLIC_KEY` / `_2` / `_3`
- a one-off package for one session. Add it to `/vol/bootstrap.sh` instead.
- new Runpod or GitHub credentials

From a local clone (A6), after editing `runpod-image/` or the CARI4D Dockerfile:
```bash
git add -A && git commit -m "<change>" && git push origin main
gh workflow run build-runpod-image.yml -f tag=v1.2      # bump the tag every rebuild
gh run watch
```
Then update the template's container image to the new tag. The build takes about 50 minutes on GitHub.

The pull-down default tag in the workflow form (line 9 of the workflow file) is only a suggestion; the tag you type is what gets built.

**Image history:**
- `v1.0`: first build. Needed a start-command override for SSH keys.
- `v1.1`: `start.sh` writes `PUBLIC_KEY`/`_2`/`_3` to `/root/.ssh`. No override needed. **Current.**

---

## Part F — Sharing with a teammate (their own Runpod account, image stays private)

The image `ghcr.io/melanieww/v2d-runpod` stays **private**. MelanieWW grants the teammate read access on GitHub. The teammate then uses **their own** token, Runpod credential, template, secrets and network volume.

Nothing from MelanieWW's Runpod account is shared: not the template, the `ghcr` credential, the secrets, or the volume `hostile_purple_louse`.

### F1. MelanieWW: grant access to the image

1. Go to https://github.com/MelanieWW?tab=packages → **v2d-runpod** → **Package settings** (right sidebar).
2. Under **Manage access**, click **Invite** (or **Add**).
3. Search for the teammate's **GitHub username**, select it, and set the role to **Read**.
4. Leave the package visibility as **Private**.

Access covers every tag (`v1.0`, `v1.1`, ...). To revoke it later, return to the same page and remove the teammate.

**Optional: let the teammate push code.** Go to https://github.com/MelanieWW/video_to_data → **Settings** → **Collaborators** → **Add people** → their username. Otherwise they work in their own fork and open pull requests.

### F2. Teammate: accept the invitation

If GitHub sends an invitation (by email or in notifications at https://github.com/notifications), accept it. Then confirm that https://github.com/MelanieWW/video_to_data/pkgs/container/v2d-runpod opens while signed in.

### F3. Teammate: create a GitHub token for Runpod

GitHub's container registry only accepts **classic** tokens. Fine-grained tokens don't work here.

1. Sign in to GitHub as the teammate and go to https://github.com/settings/tokens.
2. Click **Generate new token** → **Generate new token (classic)**.
3. **Note:** `runpod-ghcr-pull`. **Expiration:** your choice. A no-expiry token is convenient, but a fixed date is safer; set a calendar reminder to renew it.
4. **Scopes:** check **only** `read:packages`.
5. Click **Generate token** and copy it right away. GitHub shows it only once.

### F4. Teammate: add the token to Runpod

In the **teammate's own** Runpod account:

1. Go to https://console.runpod.io/user/credentials?tab=registry-credentials → **Create new registry credential**.
2. Choose type **Docker**.
3. Fill in:

   | Field | Value |
   |---|---|
   | Name | `ghcr` |
   | Username | the teammate's **own** GitHub username |
   | Password | the token from F3 |
   | Registry / server URL *(only if shown)* | `ghcr.io` |

4. Save.

### F5. Teammate: everything else in their own account

| Item | Where | Notes |
|---|---|---|
| SSH key | their computer | Part A1. Put it in **their** template's `PUBLIC_KEY`. |
| SSH key on GitHub | GitHub | Part A3, if they push code |
| `HF_TOKEN` secret | Runpod → Secrets | Their own Hugging Face token. They must accept the `nvidia/video_to_data_challenge` terms with that HF account. |
| Network volume | Runpod → Storage | Their own, at least 200 GB, in a datacenter with A100 80GB capacity |
| Template | Runpod → My Templates | Copy every setting from **Part B**. Image `ghcr.io/melanieww/v2d-runpod:v1.1`, registry credential **their** `ghcr`, **their** `PUBLIC_KEY`, empty start command, same other variables. |

Then they follow **Part C** each session, with their own volume selected in the Network volume filter.

### F6. Teammate: verify the image pull

Deploy as in Part C. In the pod's **Logs** → **System**, the pull should end with:
```
Status: Downloaded newer image for ghcr.io/melanieww/v2d-runpod:v1.1
```
If it shows `unauthorized` or `denied`, check, in order:

1. F1: they're listed under Manage access.
2. F2: the invitation was accepted.
3. F3: the token is **classic** and has `read:packages`.
4. F4: the credential's username is **their** GitHub username, and the template uses that credential.

### F7. When MelanieWW rebuilds the image

Tell the teammate the new tag (e.g. `v1.2`). They update the container image in **their** template. No new access or token is needed.

### Sharing data between accounts

Network volumes can't be shared across Runpod accounts.

- **Code:** goes through GitHub (`MelanieWW/video_to_data`).
- **Large outputs:** upload to a private Hugging Face dataset repo, e.g. `hf upload <user>/v2d-outputs /vol/outputs --repo-type dataset --private`, and download on the other side with `hf download`.
- **Submission files:** whoever submits needs the final outputs on their own volume.
