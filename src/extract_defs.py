#!/usr/bin/env python3
import os
import re
import argparse

def find_module_functions(root_dir):
    """
    Walk root_dir and return a dict mapping module names to lists of top-level functions.
    Module name is the .py path (relative), with os.sep → '.' and without '.py'.
    """
    modules = {}
    func_re = re.compile(r'^def\s+([A-Za-z_]\w*)\s*\(')
    for dirpath, _, filenames in os.walk(root_dir):
        for fname in filenames:
            if not fname.endswith('.py'):
                continue
            fullpath = os.path.join(dirpath, fname)
            # Compute module name, e.g. imagery.py → imagery, or sub/thing.py → sub.thing
            rel = os.path.relpath(fullpath, root_dir)
            mod = os.path.splitext(rel)[0].replace(os.sep, '.')
            funcs = []
            with open(fullpath, 'r', encoding='utf-8') as f:
                for line in f:
                    # only match top-level defs (no indent)
                    if line.startswith("def "):
                        m = func_re.match(line)
                        if m:
                            funcs.append(m.group(1))
            if funcs:
                modules[mod] = funcs
    return modules

def emit_puml(modules, stream):
    """
    Write a PlantUML diagram to `stream` given a dict of modules→function lists.
    """
    stream.write("@startuml\n")
    stream.write("skinparam class {\n")
    stream.write("  BackgroundColor<<module>> LightSkyBlue\n")
    stream.write("  BorderColor<<module>> SteelBlue\n")
    stream.write("}\n\n")
    for mod, funcs in sorted(modules.items()):
        stream.write(f'class {mod} <<module>> {{\n')
        for fn in funcs:
            stream.write(f'    + {fn}()\n')
        stream.write("}\n\n")
    stream.write("@enduml\n")

def main():
    parser = argparse.ArgumentParser(
        description="Generate a PlantUML diagram of module-level functions in a folder."
    )
    parser.add_argument("folder", help="Root directory to scan for .py files")
    parser.add_argument("-o", "--output", help="Where to write the .puml (default: diagram.puml)",
                        default="diagram.puml")
    args = parser.parse_args()

    modules = find_module_functions(args.folder)
    with open(args.output, "w", encoding='utf-8') as out:
        emit_puml(modules, out)

    print(f"Wrote PlantUML to {args.output}")

if __name__ == "__main__":
    main()
