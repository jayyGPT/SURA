from pathlib import Path

helper = Path('tools/tmp_notation_math_pass.py')
source = helper.read_text(encoding='utf-8')
old = "**Status:** applicable.\\n\\n### [ ] J13."
new = "**Status:** applicable.\\n\\n### [x] J13."
if source.count(old) != 1:
    raise RuntimeError(f'expected one J12/J13 tracker pattern in helper, found {source.count(old)}')
source = source.replace(old, new, 1)
exec(compile(source, str(helper), 'exec'))
