import json
import math
from pathlib import Path
import torch
from reference import fake_quant_ref, rope_ref, repack_ref, scale_bit_reference

torch.manual_seed(17)
checks = []

# Independent E8M0 scale construction around exact powers of two.
vals=[]
for exponent in range(-127, 128):
    v=torch.tensor(math.ldexp(1.0, exponent), dtype=torch.float32)
    vals.extend([torch.nextafter(v, torch.tensor(0.0)), v,
                 torch.nextafter(v, torch.tensor(float('inf')))])
v=torch.stack(vals)
m,e=torch.frexp(v.double())
p=torch.where(m == 0.5, e-1, e).clamp(-127,127)
expected=torch.ldexp(torch.ones_like(v),p)
assert torch.equal(scale_bit_reference(v),expected)
checks.append({'name':'E8M0 scale boundary cases','cases':len(vals)})

x=torch.randn(5,8,576)
x[...,32:64]*=20
whole=fake_quant_ref(x)
parts=torch.cat((fake_quant_ref(x[...,:512]),fake_quant_ref(x[...,512:])),dim=-1)
assert torch.equal(whole,parts)
checks.append({'name':'512 + 64 split equals 576 contiguous group-32 quantization'})

changed=x.clone(); changed[0,0,0:32]*=32
quant_changed=fake_quant_ref(changed)
assert torch.equal(whole[0,0,32:],quant_changed[0,0,32:])
assert torch.equal(whole[1:],quant_changed[1:])
checks.append({'name':'No cross-group, cross-token or cross-head scale mixing'})

rope=torch.full((2,8,64), 0.0002); rope[...,0]=64
proper=fake_quant_ref(rope)
wrong=torch.stack((fake_quant_ref(rope[...,0::2]),fake_quant_ref(rope[...,1::2])),dim=-1).flatten(-2)
assert not torch.equal(proper,wrong)
checks.append({'name':'Interleaved PE grouping differs from separately quantized even/odd vectors'})

T,H=7,8
nope=torch.randn(H,T,512).transpose(0,1).bfloat16()
pe=torch.randn(T,H,256).bfloat16()[...,192:]
pos=torch.arange(T,dtype=torch.int64)
a=torch.randn(T,32)
cache=torch.cat((a.cos(),a.sin()),-1).bfloat16()
rot=rope_ref(pe,pos,cache)
assert not torch.equal(fake_quant_ref(rot), rope_ref(fake_quant_ref(pe),pos,cache))
checks.append({'name':'Quantization and RoPE do not commute'})

qscale=torch.tensor([0.25],dtype=torch.float32)
req=torch.tensor([1,0,2,0,1,2,-1],dtype=torch.int32)
phase=torch.tensor([True,False,False])
out=torch.full((T,H,576),3.0).to(torch.float8_e4m3fn)
source_n=nope.clone(); source_p=pe.clone()
y=repack_ref(nope,pe,out,qscale,req,phase,positions=pos,cache=cache)
for t in (1,3,6):
    assert torch.equal(y[t].view(torch.uint8),out[t].view(torch.uint8))
assert torch.equal(source_n,nope) and torch.equal(source_p,pe)
checks.append({'name':'Prefill/invalid request rows preserved; high-precision sources unchanged'})

# For already-rotated PE, do not apply RoPE twice.
y2=repack_ref(nope,rot,out,qscale,req,phase)
assert torch.equal(y.view(torch.uint8),y2.view(torch.uint8))
checks.append({'name':'Raw-PE + RoPE path equals pre-rotated PE path when inputs match'})
report={'status':'passed','scope':'CPU mathematical reference only; NOT Triton/GPU/vLLM',
        'torch':torch.__version__,'checks':checks,
        'gpu_available':torch.cuda.is_available(),'gpu_kernel_test':'not executed'}
Path(__file__).with_name('validation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
print(json.dumps(report,ensure_ascii=False,indent=2))
