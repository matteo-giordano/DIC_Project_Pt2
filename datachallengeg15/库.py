import os
import re
import json

# 简单Python标准库名列表（可根据需要扩展）
# 注意：这不是完整列表，复杂场景建议用更专业工具
standard_libs = {
    'abc', 'argparse', 'array', 'asyncio', 'base64', 'binascii', 'bisect', 'cmath', 'collections',
    'copy', 'csv', 'datetime', 'functools', 'glob', 'hashlib', 'heapq', 'inspect', 'io',
    'itertools', 'json', 'logging', 'math', 'operator', 'os', 'pathlib', 'pickle', 'platform',
    'random', 're', 'shutil', 'socket', 'sqlite3', 'string', 'struct', 'subprocess', 'sys',
    'tempfile', 'threading', 'time', 'types', 'typing', 'unittest', 'uuid', 'warnings', 'weakref',
    'xml', 'zipfile',
}

def extract_imports_from_py(file_path):
    imports = set()
    import_pattern = re.compile(r'^\s*(?:import|from)\s+([\w\.]+)')
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            match = import_pattern.match(line)
            if match:
                # 只取最外层包名，比如 pandas.core -> pandas
                lib = match.group(1).split('.')[0]
                imports.add(lib)
    return imports

def extract_imports_from_ipynb(file_path):
    imports = set()
    with open(file_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
        for cell in nb.get('cells', []):
            if cell.get('cell_type') == 'code':
                source = cell.get('source', [])
                for line in source:
                    line = line.strip()
                    if line.startswith('import ') or line.startswith('from '):
                        # 使用正则提取库名
                        match = re.match(r'(?:import|from)\s+([\w\.]+)', line)
                        if match:
                            lib = match.group(1).split('.')[0]
                            imports.add(lib)
    return imports

def main():
    imports = set()
    for root, _, files in os.walk('.'):
        for file in files:
            if file.endswith('.py'):
                imports.update(extract_imports_from_py(os.path.join(root, file)))
            elif file.endswith('.ipynb'):
                imports.update(extract_imports_from_ipynb(os.path.join(root, file)))
    # 去掉标准库
    imports = {lib for lib in imports if lib not in standard_libs}
    # 排序后写入requirements.txt
    with open('requirements.txt', 'w', encoding='utf-8') as f:
        for lib in sorted(imports):
            f.write(lib + '\n')
    print(f"Found {len(imports)} third-party libraries. requirements.txt generated.")

if __name__ == '__main__':
    main()
