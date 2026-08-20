from pathlib import Path

path = Path('paper/main.tex')
text = path.read_text(encoding='utf-8')
old = r'''\begin{align}
m_{N,n} &= \|\mathbf{m}_n\|, &
m_{V,n} &= \mathbf{m}_n^T\hat{\mathbf{a}}_n, \nonumber\\
m_{H,n} &= \sqrt{\max(m_{N,n}^2-m_{V,n}^2,0)}, &
\delta_n &= \operatorname{atan2}(m_{V,n},m_{H,n}).
\label{eq:mag_features}
\end{align}'''
new = r'''\begin{align}
m_{N,n} &= \|\mathbf{m}_n\|,
& m_{V,n} &= \mathbf{m}_n^T\hat{\mathbf{a}}_n,
\nonumber\\
m_{H,n} &= \sqrt{\max(m_{N,n}^2-m_{V,n}^2,0)},
\nonumber\\
\delta_n &= \operatorname{atan2}(m_{V,n},m_{H,n}).
\label{eq:mag_features}
\end{align}'''
if text.count(old) != 1:
    raise RuntimeError(f'expected one magnetic feature equation, found {text.count(old)}')
path.write_text(text.replace(old, new, 1), encoding='utf-8')
print('Split magnetic feature equation into column-safe lines.')
