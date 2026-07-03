"""Generate API reference pages for every triwarp submodule."""

from pathlib import Path

import mkdocs_gen_files

nav = mkdocs_gen_files.Nav()

root = Path(__file__).parent.parent
src = root / "triwarp"

for path in sorted(src.glob("*.py")):
    if path.name == "__init__.py":
        continue

    module_name = path.stem
    doc_path = Path("api", f"{module_name}.md")
    nav[("API Reference", module_name)] = doc_path.as_posix()

    with mkdocs_gen_files.open(doc_path, "w") as fd:
        print(f"::: triwarp.{module_name}", file=fd)

    mkdocs_gen_files.set_edit_path(doc_path, path.relative_to(root))

with mkdocs_gen_files.open("SUMMARY.md", "w") as nav_file:
    nav_file.write("* [Home](index.md)\n")
    nav_file.writelines(nav.build_literate_nav())
