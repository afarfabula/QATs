import re,sys,os
def parse(p):
    accs=[]; kl=[]
    tr=re.compile(r'^Test: \[distributed-summary\].*Acc@1: ([\d.]+)\s+Acc@5: ([\d.]+)')
    kr=re.compile(r'^Train: \d+ \[.*?TeacherAttnKL: ([0-9.eE+-]+) \(([0-9.eE+-]+)\)')
    for line in open(p, errors='replace'):
        m=tr.match(line)
        if m: accs.append((float(m.group(1)), float(m.group(2))))
        k=kr.match(line)
        if k: kl.append(float(k.group(2)))
    return accs, kl
for p in sys.argv[1:]:
    accs, kl = parse(p)
    name=os.path.basename(p).replace('.log','')
    a=[f'{v[0]:.2f}' for v in accs[:2]]
    print(f'{name:42s} ep1={a[0] if a else "-":>7s} ep2={a[1] if len(a)>1 else "-":>7s}  lastKL={kl[-1] if kl else 0:8.1f}')
