# Regenerating the Warp API reference

These files mirror the Warp docs as of **Warp 1.15.0**. After a Warp upgrade,
refresh them so the function lists, signatures, and the version stamp at the top
of each file match the installed version.

## How

There is no public machine-readable index for these pages, so regeneration is a
fetch-and-format step, not a fully automated scrape. Use an AI assistant (e.g.
Claude Code) and ask it to re-fetch each source URL and rewrite the matching
file in place, keeping the existing format (section headings, `name — one-line
description` rows, source URL + version stamp at the top).

Source URL for each file:

| File | Source URL |
|------|------------|
| `builtins.md`   | https://nvidia.github.io/warp/stable/language_reference/builtins.html |
| `warp.md`       | https://nvidia.github.io/warp/stable/api_reference/warp.html |
| `sparse.md`     | https://nvidia.github.io/warp/stable/api_reference/warp_sparse.html |
| `utils.md`      | https://nvidia.github.io/warp/stable/api_reference/warp_utils.html |
| `fem_linalg.md` | https://nvidia.github.io/warp/stable/api_reference/warp_fem_linalg.html |

## Check the installed version

Run `reference/warp_api/warp_version.py` to print the installed Warp version and
the `stable` docs version, and to flag the files whose stamp is out of date:

```sh
python reference/warp_api/warp_version.py
```
