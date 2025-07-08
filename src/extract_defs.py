#!/usr/bin/env python3
"""
Extract top-level classes & functions from a package,
emit Graphviz DOT to stdout.

Usage: python extract_defs.py dynamic_tasker > structure.dot
"""

import ast, pathlib, sys

def walk_module(path: pathlib.Path, dotted: str, graph):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            fn = f"{dotted}.{node.name}"
            graph.append(f'    "{dotted}" -> "{fn}";')
        elif isinstance(node, ast.ClassDef):
            cls = f"{dotted}.{node.name}"
            graph.append(f'    "{dotted}" -> "{cls}";')

def main(pkg_root):
    base = pathlib.Path(pkg_root)
    print(base)
    edges = ["digraph G {"]
    for py in base.rglob("*.py"):
        mod = py.relative_to(base).with_suffix("")     # areas/sub.py → areas/sub
        dotted = f"{pkg_root}.{'.'.join(mod.parts)}"   # dynamic_tasker.areas.sub
        walk_module(py, dotted, edges)
    edges.append("}")
    print("\n".join(edges))

if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("python extract_defs.py <package_name>")
    main(sys.argv[1])
