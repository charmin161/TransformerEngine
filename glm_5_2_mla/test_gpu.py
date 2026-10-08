"""Run in the target vLLM CUDA environment. Not executed in the authoring environment."""
import torch
from reference import repack_ref, rope_ref

if not torch.cuda.is_available():
    raise SystemExit('CUDA is required; no GPU kernel has been tested by this run.')
from mxfp8_q_prepack import repack_mxfp8_decode_q_

torch.manual_seed(9)
T,H=7,8
# Noncontiguous high-precision sources (matching transposed BMM and PE slice).
nope_cpu=torch.randn(H,T,512).bfloat16().transpose(0,1)
pe_cpu=torch.randn(T,H,256).bfloat16()[...,192:]
pos_cpu=torch.arange(T,dtype=torch.int64)
a=torch.randn(T,32)
cache_cpu=torch.cat((a.cos(),a.sin()),-1).bfloat16()
ids_cpu=torch.tensor([1,0,2,0,1,2,-1],dtype=torch.int32)
phase_cpu=torch.tensor([True,False,False])
scale_cpu=torch.tensor([0.25],dtype=torch.float32)
nope=torch.empty_strided(nope_cpu.shape,nope_cpu.stride(),dtype=nope_cpu.dtype,device='cuda').copy_(nope_cpu)
pe=torch.empty_strided(pe_cpu.shape,pe_cpu.stride(),dtype=pe_cpu.dtype,device='cuda').copy_(pe_cpu)
pos,cache=pos_cpu.cuda(),cache_cpu.cuda()
ids,phase,scale=ids_cpu.cuda(),phase_cpu.cuda(),scale_cpu.cuda()
# Deliberately strided output as well.
for rotate in (False,True):
    source_pe_cpu=pe_cpu if rotate else rope_ref(pe_cpu,pos_cpu,cache_cpu).bfloat16()
    source_pe=pe if rotate else source_pe_cpu.cuda()
    for apply in (False,True):
        out_cpu=torch.full((T,H,576),3.0).to(torch.float8_e4m3fn)
        backing=torch.empty((T,H,1152),dtype=torch.float8_e4m3fn,device='cuda')
        out=backing[...,::2]
        out.copy_(out_cpu)
        expected=repack_ref(nope_cpu,source_pe_cpu,out_cpu,scale_cpu,ids_cpu,phase_cpu,
                            positions=pos_cpu if rotate else None,
                            cache=cache_cpu if rotate else None,
                            apply_mxfp8=apply)
        repack_mxfp8_decode_q_(nope,source_pe,out,scale,ids,phase,
                              positions=pos if rotate else None,
                              cos_sin_cache=cache if rotate else None,
                              apply_mxfp8=apply)
        torch.cuda.synchronize()
        actual=out.cpu()
        assert torch.equal(actual.view(torch.uint8),expected.view(torch.uint8)), (rotate,apply)
        print('PASS', {'rotate_pe':rotate,'apply_mxfp8':apply,'shape':tuple(out.shape)})
print('Standalone GPU/reference checks passed. Model and PIECEWISE replay still need end-to-end verification.')
