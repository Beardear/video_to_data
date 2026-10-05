# Runpod Guide — video_to_data (CARI4D, Track 1)

How to run the project on a Runpod A100, from any Mac or Windows computer.

**Where to start:**

| You are… | Go to |
|---|---|
| A **new teammate** | Ask the owners to do **Part 1**, then follow **Part 2** once |
| Setting up **another computer** of your own | **Part 3** |
| Starting a normal **work session** | **Part 4** |
| Stuck | **Part 5** |
| An **image maintainer**, changing or releasing the image | **Part 6** |

---

## Quick reference

| Item | Value |
|---|---|
| Team working repo | `Beardear/video_to_data`, branch `main`. Business changes and PRs go here. |
| Deployment source | The same repository, stable branch `docker-image`. |
| Image namespace | `Beardear`; multiple maintainers may have package Write/Admin access. |
| Team image (private, after a release) | `ghcr.io/beardear/v2d-runpod:<version>`; use the exact tag/digest from a successful build. |
| Existing image before the first team release | `ghcr.io/melanieww/v2d-runpod:v1.1`; its existing administrators still manage access. |
| GPU | A100 **80 GB** (SXM or PCIe). Never 40 GB. Avoid H100 for now. |
| Code baked into the image | `/workspace/v2d_*` |
| Your network volume | mounted at **`/vol`**: `video_to_data`, `runpod-config`, `data`, `weights`, `outputs`, `submission`, `cache`, `secrets` |

**What persists:** the image, your template, secrets, registry credential and network volume (`/vol`).
**What doesn't:** the pod itself and its container disk (`/`, `/root`, `~/.bashrc`). Each session deploys a fresh pod.
**Rule:** keep files in `/vol`; push code to GitHub.

Everyone uses **their own Runpod account**. Templates, secrets and volumes are never shared between accounts. Only the image (Part 1) and the GitHub repo are shared.

### Team branch contract

Imported from `MelanieWW/video_to_data` at `4b212439ca75e459fbd919aa6ab2c8ca3b171f04`,
retaining its directory layout and `/workspace` image-code / `/vol` volume contract.

- Start business/algorithm branches from `main` and submit their PRs to `main`.
- Sync `main` into `docker-image`. Do **not** merge `docker-image` back into `main`.
  Business fixes discovered during deployment must land on `main` first.
- Keep team deployment additions under `runpod-image/`, `scripts/runpod/`, this
  guide, and `.github/workflows/build-runpod-image.yml` on `docker-image` only.
  Upstream's existing module `docker/` directories remain part of the main code.
- Branch pushes run lightweight checks. A `runpod-vX.Y.Z` tag on this branch
  explicitly requests an image build and publication. The workflow stays off `main`.
- Record the business commit, image build commit, and image digest for experiments.
  A successful image build is not a GPU inference validation.

GitHub Actions publishes with `GITHUB_TOKEN`. The namespace follows the repository
owner, not the person pushing the tag. Read permits pulling; Write permits
publishing; Admin manages package permissions. Multiple teammates can maintain
the image without sharing a personal token.

---

## Part 1 — Owners: give a teammate access

Done once per teammate. Repository and package permissions are separate and may
be managed by different administrators.

1. **Image administrator:** after the first team release, go to https://github.com/Beardear?tab=packages → **v2d-runpod** → **Package settings**. Keep visibility **Private** and grant teammates **Read**, through the linked repository or explicit package roles. Grant selected maintainers Write/Admin as needed. For the existing MelanieWW image, use that package's settings instead.
2. **Repo (Beardear owner):** https://github.com/Beardear/video_to_data → **Settings** → **Collaborators** → **Add people** → their username.

To revoke, each owner removes them from their page.

---

## Part 2 — New teammate: first-time setup

Do these in order, once. Commands are given for **Mac (Terminal)** and **Windows (CMD)** where they differ.

### 2.1 Accept the invitations

Accept the repository invitation (email, or https://github.com/notifications).
Check that the selected image's package page opens while signed in. Package roles
can take effect directly; they do not necessarily send another invitation.

### 2.2 Create an SSH key on your computer

| Mac | Windows |
|---|---|
| `ssh-keygen -t ed25519` | `ssh-keygen -t ed25519` |
| `cat ~/.ssh/id_ed25519.pub \| pbcopy` | `type %USERPROFILE%\.ssh\id_ed25519.pub \| clip` |

Accept the default path and set a passphrase. The second command copies your **public** key (the `.pub` file) to the clipboard. Never share the file without `.pub`.

### 2.3 Connect the key to GitHub and the SSH agent

Install GitHub CLI (Mac: `brew install gh`; Windows: `winget install --id GitHub.cli`), then:

```bash
gh auth login                                    # GitHub.com → HTTPS → Yes → browser
gh auth refresh -h github.com -s admin:public_key
gh ssh-key add ~/.ssh/id_ed25519.pub --title "<computer name>"
```
On Windows, use `%USERPROFILE%\.ssh\id_ed25519.pub` as the path.

Load the key into the agent. This lets the pod use your key for `git push`:

- **Mac:** `ssh-add --apple-use-keychain ~/.ssh/id_ed25519`
- **Windows:** once, in an **admin PowerShell**, run `Get-Service ssh-agent | Set-Service -StartupType Automatic; Start-Service ssh-agent`. Then in CMD: `ssh-add %USERPROFILE%\.ssh\id_ed25519`

Test it: `ssh -T git@github.com` should print `Hi <username>!`.

### 2.4 Create a GitHub token for pulling the image

1. Go to https://github.com/settings/tokens → **Generate new token (classic)**. Fine-grained tokens don't work with the container registry.
2. Name it `runpod-ghcr-pull`, check **only** `read:packages`, and generate.
3. Copy it right away; GitHub shows it only once.

### 2.5 Set up your Runpod account

**a. Registry credential:** go to https://console.runpod.io/user/credentials?tab=registry-credentials → **Create** → type **Docker**:

| Field | Value |
|---|---|
| Name | `ghcr` |
| Username | **your** GitHub username |
| Password | the token from 2.4 |
| Registry URL *(if shown)* | `ghcr.io` |

**b. Secrets:** go to https://console.runpod.io/user/secrets. Create `HF_TOKEN` with your Hugging Face **Read** token from https://huggingface.co/settings/tokens. With that HF account, also accept the terms on the `nvidia/video_to_data_challenge` dataset page. Add `KAGGLE_USERNAME` and `KAGGLE_KEY` later, when you're ready to submit.

**c. Network volume:** go to **Storage** → **New Network Volume**. Make it at least 200 GB, in a datacenter that currently lists A100 80GB. Billing runs continuously, even with no pod.

**d. Template:** go to **My Templates** → **New Template**:

| Field | Value |
|---|---|
| Type | Pod |
| Container image | A published tag/digest; the existing `ghcr.io/melanieww/v2d-runpod:v1.1` remains usable with access until a team release is available |
| Registry credential | `ghcr` |
| Container disk | 50 GB |
| Volume mount path | `/vol` |
| HTTP ports | `8080` |
| TCP ports | `22` |
| Start command | *(leave empty)* |

Environment variables:

| Key | Value |
|---|---|
| `PUBLIC_KEY` | your public key line from 2.2 (`ssh-ed25519 AAAA... you@host`) |
| `HF_TOKEN` | `{{ RUNPOD_SECRET_HF_TOKEN }}` (type it literally, or pick the secret with the 🔑 icon) |
| `HF_HOME` | `/vol/cache/hf` |
| `TORCH_HOME` | `/vol/cache/torch` |
| `HF_XET_HIGH_PERFORMANCE` | `1` *(optional: faster Hugging Face downloads)* |
| `GIT_USER_NAME` | your name |
| `GIT_USER_EMAIL` | your GitHub email |
| `KAGGLE_USERNAME` / `KAGGLE_KEY` | `{{ RUNPOD_SECRET_KAGGLE_USERNAME }}` / `{{ RUNPOD_SECRET_KAGGLE_KEY }}` *(add later)* |

### 2.6 First pod: initialize your volume

1. Deploy and connect as in **Part 4** (steps 4.1–4.2). Run the verification commands in 4.3, but skip its bootstrap command until the checkout below exists.
2. Run once:
   ```bash
   GIT_LFS_SKIP_SMUDGE=1 git clone --branch main git@github.com:Beardear/video_to_data.git /vol/video_to_data
   git -C /vol/video_to_data fetch origin docker-image
   GIT_LFS_SKIP_SMUDGE=1 git -C /vol/video_to_data worktree add --detach /vol/runpod-config origin/docker-image
   tmux new -s init
   bash /vol/runpod-config/scripts/runpod/init_volume.sh
   ```
   This sets up the folders and downloads the Track 1 data (~0.5 GB) and the CARI4D weights into `/vol/weights/cari4d`. Detach with `Ctrl+B`, then `D`. It's done when it prints `volume initialized`. If it's interrupted, rerun it; the download resumes.

`/vol/video_to_data` is the business checkout. `/vol/runpod-config` is a deployment
worktree sharing its Git objects, so scripts remain available when you switch
business branches. LFS smudging is skipped initially; fetch needed reconstruction
assets and the submission kit explicitly. The dataset revision is pinned to
`5f68335f3acc802033d1e80728c1633197521de8`.

---

## Part 3 — Adding another computer

Each computer has its own key. Never copy private keys between machines.

1. On the new computer, do **2.2** and **2.3**.
2. In your template, add the new public key as **`PUBLIC_KEY_2`**, or `PUBLIC_KEY_3` for a third computer. Don't change the existing `PUBLIC_KEY`.
3. Deploy a new pod. Template changes only apply to pods deployed after saving.

The image supports up to three keys. `GIT_USER_NAME` and `GIT_USER_EMAIL` stay single: one person per template.

---

## Part 4 — Every session

### 4.1 Deploy

1. Go to https://console.runpod.io/deploy → **GPU** tab.
2. Set **Network volume** (next to Search GPUs) to **your** volume. This limits the GPU list to its datacenter. Without it, you can get a GPU with no volume attached.
3. Pick an **A100 80GB** (SXM or PCIe). Then **Change Template** → **My Templates** → yours.
4. Confirm **Persistent storage** shows your volume at `/vol`, then click **Deploy On-Demand**.

### 4.2 Connect

Go to Pod → **Connect** and copy the **direct TCP** command (`root@<IP> -p <PORT>`). Add `-A`:

```bash
ssh -A root@<IP> -p <PORT> -i ~/.ssh/id_ed25519                 # Mac
ssh -A root@<IP> -p <PORT> -i %USERPROFILE%\.ssh\id_ed25519      # Windows CMD
```

### 4.3 Verify and bootstrap

```bash
nvidia-smi                 # A100 80GB
df -h /vol                 # your network volume, NOT "overlay 50G"
echo ${#HF_TOKEN}          # non-zero
ssh -T git@github.com      # Hi <username>!
bash /vol/runpod-config/scripts/runpod/bootstrap.sh && source ~/.bashrc
```

If `df -h /vol` shows `overlay 50G`, stop: the volume isn't attached. Terminate and redeploy with step 4.1.2.

### 4.4 Work

- Run long jobs in **tmux**: `tmux new -s run`. Detach with `Ctrl+B`, `D`; reattach with `tmux attach -t run`.
- The toolkit's Docker wrappers (`python -m v2d.*.docker.*`) don't work inside a pod. Run the library module directly with the same flags, e.g. `python -m v2d.cari4d.lib.run_inference --video_path ... --output_dir ...`. This runs the code baked into `/workspace/v2d_*`.
- Write outputs and logs to `/vol/outputs`. To browse them in a web browser, go to Pod → Connect → **HTTP 8080**.

Pulling or editing `/vol/video_to_data` does not change the baked Python packages.
Verify the imported package path when testing a change. A development source
override must be configured explicitly and preserve FoundationPose/SAM 3D Body
paths; otherwise release a new image containing the reviewed main-code change.
Keep the producing business commit URL with submission artifacts.

Port 8080 is an output-file browser, not a SAM2 annotation UI. The SAM2 venv exists,
but toolkit annotation/mask entry points still need integration for full episodes.
For noninteractive SSH jobs, source `/etc/rp_environment` explicitly before Python;
interactive sessions load it through `.bashrc`.

### 4.5 End the session

```bash
cd /vol/video_to_data
git status
# Stage reviewed business changes on your feature branch, then commit and push.
```

Then go to Runpod → Pod → **Terminate**. **The A100 bills every second it runs.**

---

## Part 5 — Troubleshooting

| Symptom | Fix |
|---|---|
| Asks `root@...'s password:` | This computer's key isn't in your template (`PUBLIC_KEY`/`_2`/`_3`), or the image tag is older than `v1.1`, or the start command isn't empty. Fix the template and redeploy. Logs → **Container** shows `DIAG keylen=...`; a zero means that key is missing. |
| Asks `Enter passphrase for key` | Normal; that's your local key's passphrase. Run `ssh-add` (2.3) to stop the prompts. |
| `REMOTE HOST IDENTIFICATION HAS CHANGED` | `ssh-keygen -R "[<IP>]:<PORT>"`, then reconnect. |
| `df -h /vol` shows `overlay 50G` | Wrong region. Redeploy with the Network volume filter (4.1). |
| `Permission denied (publickey)` from GitHub on the pod | On your computer, `ssh-add -l` must list your key, the key must be on GitHub (2.3), and you must connect with `-A`. |
| Image pull `unauthorized` / `denied` | Check in order: Part 1 access was granted, the invitation was accepted (2.1), the token is **classic** with `read:packages` (2.4), and the credential uses **your** username (2.5a). |
| GPU "Out of capacity" | Try the other A100 type, wait, or use "Deploy when available". Your volume can't change regions. |
| `HF_TOKEN` length 0, or HF 403 | Check the `HF_TOKEN` template variable and secret name, and that the dataset terms were accepted. |
| Wrong Git author | Set `GIT_USER_NAME`/`GIT_USER_EMAIL` in the template. For now: `git config --global user.name "..."` and `user.email "..."`. |

**Sharing data between teammates:** volumes can't be shared across accounts. Share code through GitHub. Share large outputs through a private Hugging Face dataset: `hf upload <user>/v2d-outputs /vol/outputs --repo-type dataset --private`, then `hf download` on the other side.

---

## Part 6 — Maintainers: releasing the image

Rebuild when the image itself must change. The imported build took about 50
minutes; future builds may differ. Update templates after publication and GPU
validation of the new image.

**Rebuild for:** changes to `runpod-image/` (`Dockerfile`, `start.sh`); changes to `reconstruction/modules/v2d_cari4d/docker/Dockerfile`; changes to code baked into `/workspace/v2d_*` that you run from there; dependencies needed on every pod; a fourth SSH key.

**Don't rebuild for:** code explicitly configured to run from `/vol/video_to_data`, data, weights, outputs, template settings, credentials, or session packages (maintain those in `scripts/runpod/bootstrap.sh` on `docker-image`).

**Steps.** From a clean dedicated worktree of `Beardear/video_to_data`:
```bash
git fetch origin
git switch docker-image
git merge --ff-only origin/docker-image
git merge origin/main
# Review deployment changes and commit them with git commit -s.
python3 scripts/runpod/check_branch.py
python3 -m unittest discover -s scripts/runpod/tests -v
git push origin docker-image
# Release only when ready. Choose a new tag; never overwrite a released tag.
git tag -a runpod-v1.2.0 -m "RunPod image v1.2.0"
git push origin runpod-v1.2.0
```
The branch push runs deployment checks. The tag push additionally builds and
publishes `ghcr.io/beardear/v2d-runpod:v1.2.0` on a Linux x86-64 GitHub runner.
Watch `Beardear/video_to_data` → Actions → build-runpod-image. Its summary records
the published digest; image labels record the build commit and main-code base.
The workflow does not rent a GPU or download inference weights.

The first team image exists only after a successful release. New GHCR packages
are private by default; verify visibility and access before team use. Existing
MelanieWW tags are not moved or overwritten by this workflow.

To update the deployment worktree for a later session:
```bash
git -C /vol/video_to_data fetch origin docker-image
git -C /vol/runpod-config checkout --detach origin/docker-image
```
Keep this worktree clean; develop deployment changes in a dedicated branch worktree.

After GPU validation, update your template's image digest/tag and share it with
teammates. Switching from MelanieWW's package to Beardear's package requires
access to the new package even when the source code is already accessible.

**Imported image history (MelanieWW namespace):**
- `v1.0`: first build; needed a start-command override for SSH.
- `v1.1`: `start.sh` installs `PUBLIC_KEY`/`_2`/`_3`; no override needed.
- Team releases are separate. Importing these files does not publish a team image
  or establish that GPU validation has passed.
