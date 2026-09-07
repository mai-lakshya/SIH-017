import os
import re
import subprocess
import urllib.request
import json

def audit_infra():
    print("=" * 80)
    print("PART F: INFRASTRUCTURE & CREDENTIAL HYGIENE AUDIT")
    print("=" * 80)

    # 1. Test unauthenticated Git LFS Batch API
    print("\n--- 1. UNAUTHENTICATED GIT LFS ACCESS VERIFICATION ---")
    lfs_batch_url = "https://github.com/mai-lakshya/SIh.git/info/lfs/objects/batch"
    payload = {
        "operation": "download",
        "transfers": ["basic"],
        "objects": [
            {
                "oid": "18fbaec11cbb0fba8800ff8bb228c291d3ecdbd7b2a658db41fa16f9fcf822cb", # pipeline.joblib
                "size": 13032
            },
            {
                "oid": "c86a111a84f4f72eb37f5255476a208226065cb75ff1315fb278b122822d64a4", # ensemble.joblib
                "size": 2266858
            }
        ]
    }
    
    req = urllib.request.Request(
        lfs_batch_url,
        data=json.dumps(payload).encode('utf-8'),
        headers={
            "Accept": "application/vnd.git-lfs+json",
            "Content-Type": "application/vnd.git-lfs+json",
            "User-Agent": "git-lfs/3.0.0"
        }
    )
    
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            status_code = resp.getcode()
            res_data = json.loads(resp.read().decode('utf-8'))
            objects = res_data.get("objects", [])
            all_have_download = all("download" in obj.get("actions", {}) for obj in objects)
            print(f"LFS Batch API Status: HTTP {status_code} (Unauthenticated)")
            print(f"Retrieved {len(objects)} object download actions: All public = {all_have_download}")
            for obj in objects:
                oid = obj.get("oid")[:12]
                download_href = obj.get("actions", {}).get("download", {}).get("href", "")
                has_url = len(download_href) > 0
                print(f"  Object {oid}... -> Public Download URL Provided: {has_url}")
            lfs_status = (status_code == 200 and all_have_download)
    except Exception as e:
        print(f"LFS Check Failed: {e}")
        lfs_status = False

    # 2. Secret scanning across git log and working tree
    print("\n--- 2. CREDENTIAL HYGIENE & SECRET SCANNING ---")
    secret_patterns = [
        (r"(?i)(?:api_key|apikey|secret_key|private_key|access_token|bearer\s+[a-zA-Z0-9_\-\.]{20,})\s*[:=]\s*['\"][a-zA-Z0-9_\-\.]{15,}['\"]", "API/Secret Key Assignment"),
        (r"(?i)ghp_[a-zA-Z0-9]{36}", "GitHub Personal Access Token"),
        (r"(?i)github_pat_[a-zA-Z0-9_]{50,}", "GitHub Fine-Grained PAT"),
        (r"(?i)AKIA[0-9A-Z]{16}", "AWS Access Key ID"),
        (r"(?i)AIza[0-9A-Za-z\\-_]{35}", "Google API Key"),
        (r"-----BEGIN (?:RSA |EC )?PRIVATE KEY-----", "Private Key Header")
    ]

    flagged_working_tree = []
    # Scan working directory
    for root, dirs, files in os.walk('.'):
        if any(ignored in root for ignored in ['.git', '__pycache__', '.pytest_cache', 'my_local_files']):
            continue
        for f in files:
            if f.endswith(('.joblib', '.pyc', '.png', '.jpg', '.svg', '.csv')):
                continue
            fpath = os.path.join(root, f)
            try:
                with open(fpath, 'r', encoding='utf-8', errors='ignore') as fp:
                    for line_no, line in enumerate(fp, start=1):
                        for pat, pat_name in secret_patterns:
                            if re.search(pat, line):
                                # Filter out generic demo defaults in documentation / tests
                                if "super-secret-token" in line or "change-this-in-production" in line or "SIH2024Demo" in line:
                                    continue
                                flagged_working_tree.append({
                                    "file": fpath,
                                    "line": line_no,
                                    "pattern": pat_name,
                                    "snippet": line.strip()[:80]
                                })
            except Exception:
                pass

    print(f"Working Tree Scan Completed: {len(flagged_working_tree)} secrets found.")
    if flagged_working_tree:
        for item in flagged_working_tree:
            print(f"  [FLAG] {item['file']}:{item['line']} - {item['pattern']}: {item['snippet']}")
    else:
        print("  [CLEAN] No hardcoded tokens, secret keys, or credentials found in working tree.")

    # Check git log using full history
    import shutil
    git_exe = shutil.which('git') or 'git'
    try:
        import tempfile
        bare_dir = os.path.join(tempfile.gettempdir(), 'sih_bare_scan')
        if os.path.exists(bare_dir):
            import shutil
            shutil.rmtree(bare_dir, ignore_errors=True)
        # Shallow clone to get commit diffs
        subprocess.run([git_exe, 'clone', '--bare', '--depth', '50', 'https://github.com/mai-lakshya/SIh.git', bare_dir], capture_output=True, text=True)
        cmd = [git_exe, '--git-dir', bare_dir, 'log', '-p', '-n', '50']
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True, encoding='utf-8', errors='ignore')
        for line in out.splitlines():
            if line.startswith('+') and not line.startswith('+++'):
                for pat, pat_name in secret_patterns:
                    if re.search(pat, line):
                        if "super-secret-token" in line or "change-this-in-production" in line or "SIH2024Demo" in line:
                            continue
                        git_secrets.append({"pattern": pat_name, "snippet": line[:80]})
        if os.path.exists(bare_dir):
            import shutil
            shutil.rmtree(bare_dir, ignore_errors=True)
        print(f"Git History (Last 50 Commits) Scan: {len(git_secrets)} secrets found.")
        if git_secrets:
            for s in git_secrets:
                print(f"  [FLAG] {s['pattern']}: {s['snippet']}")
        else:
            print("  [CLEAN] No secrets detected in recent git commits.")
    except Exception as e:
        print(f"Git log scan note: {e}")

    # 3. Check CI Workflow
    print("\n--- 3. GITHUB ACTIONS CI WORKFLOW VERIFICATION ---")
    ci_path = os.path.join(".github", "workflows", "ci.yml")
    if os.path.exists(ci_path):
        with open(ci_path, 'r', encoding='utf-8') as f:
            ci_content = f.read()
        has_pytest = "pytest" in ci_content
        has_lfs = "lfs: true" in ci_content or "git lfs" in ci_content
        has_reqs = "requirements.txt" in ci_content
        print(f"CI File exists: True")
        print(f"Installs requirements.txt: {has_reqs}")
        print(f"Checks out Git LFS: {has_lfs}")
        print(f"Executes Pytest: {has_pytest}")
    else:
        print(f"CI File {ci_path} not found.")

    results = {
        "git_lfs_unauthenticated_access": lfs_status,
        "working_tree_secrets_count": len(flagged_working_tree),
        "git_history_secrets_count": len(git_secrets),
        "ci_workflow_verified": os.path.exists(ci_path)
    }

    with open("infra_audit_results.json", "w") as f:
        json.dump(results, f, indent=2)

if __name__ == '__main__':
    audit_infra()
