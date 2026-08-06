# Regenerating the Warp API reference

These files mirror the Warp docs as of **Warp 1.16.0**. After a Warp upgrade,
refresh them so the function lists, signatures, and the version stamp at the top
of each file match the installed version.

## How

Do **not** rewrite the files from scratch — diff them against the docs and patch
what changed. Each page is a Sphinx `autosummary` table, so the name and its
one-line description are machine-extractable, and each release only moves a
handful of rows.

Source URL for each file:

| File | Source URL |
|------|------------|
| `builtins.md`   | https://nvidia.github.io/warp/stable/language_reference/builtins.html |
| `warp.md`       | https://nvidia.github.io/warp/stable/api_reference/warp.html |
| `sparse.md`     | https://nvidia.github.io/warp/stable/api_reference/warp_sparse.html |
| `utils.md`      | https://nvidia.github.io/warp/stable/api_reference/warp_utils.html |
| `fem_linalg.md` | https://nvidia.github.io/warp/stable/api_reference/warp_fem_linalg.html |

Rows in those pages match:

```
<tr class="row-\w+"><td><p><a class="reference internal"[^>]*>
<code[^>]*><span class="pre">(NAME)</span></code></a></p></td>\s*<td><p>(DESCRIPTION)</p></td>
```

and the enclosing `<section id="...">` gives the heading each row belongs under.
Compare that name set against the backtick-quoted leading token of every `- ` row
in the matching `.md`. Expect false positives in both directions: these files
deliberately collapse families onto one row (`volume_lookup_f/i/v/index`,
`atomic_add/sub/min/max`, `bvh_query_aabb` `/_tiled`), so grep the file for the
stem before concluding a name is missing.

Full per-symbol signatures live on the generated pages, under
`<dt class="sig sig-object py">`:

- `api_reference/_generated/warp.<Name>.html` — Python-scope classes and functions
  (this is where class *methods* such as `HashGrid.build` or `Volume.rebuild` are,
  and they are **not** in the autosummary tables, so a changed method signature
  will not show up in the name diff — check the classes these files mention by hand).
- `language_reference/_generated/warp.<name>.html` — kernel-scope builtins, with
  every dtype overload.

There is also a Sphinx inventory at `https://nvidia.github.io/warp/stable/objects.inv`
(zlib-compressed after four header lines) listing every documented object with its
page. It is the quickest way to confirm a name exists before hunting for its page.

Finally, read the release's `CHANGELOG.md`
(`https://raw.githubusercontent.com/NVIDIA/warp/v<VERSION>/CHANGELOG.md`) — the
`Deprecated` entries and behavioural fixes are not visible in a name diff at all,
and they are usually the part worth writing down.

## Check the installed version

Run `reference/warp_api/warp_version.py` to print the installed Warp version and
the `stable` docs version, and to flag the files whose stamp is out of date:

```sh
python reference/warp_api/warp_version.py
```

Note the stamps track the **docs** version these files were transcribed from,
which can run ahead of the installed `warp-lang` during an upgrade.
