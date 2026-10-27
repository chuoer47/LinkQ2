"""Numerical regressions for strided stores and fused prefix materialization."""
import pytest
import torch

from qslab.kernels.kv4 import materialize_kv, paged_attention_decode, store_kv_quant

pytestmark = pytest.mark.gpu


def unpack(words):
    shifts = torch.arange(8, device='cuda') * 4
    return (((words.int().unsqueeze(-1) >> shifts) & 15) - 8).flatten(-2).float()


@pytest.mark.parametrize('group', [32, 64, 128])
@torch.inference_mode()
def test_strided_store_and_fused_materialize(group):
    torch.manual_seed(19)
    n, h, d = 67, 8, 128
    fused = torch.randn(n, h * 3, d, device='cuda', dtype=torch.float16)
    _, k, v = fused.chunk(3, dim=1)
    assert not k.is_contiguous() and not v.is_contiguous()
    slots = torch.randperm(n, device='cuda').int()
    scale = torch.full((h, d), 0.3, device='cuda', dtype=torch.float16)

    def pool():
        return ((torch.zeros(n,h,d//8,device='cuda',dtype=torch.uint32),scale),
                (torch.zeros(n,h,d//8,device='cuda',dtype=torch.uint32),
                 torch.zeros(n,h,d//group,device='cuda',dtype=torch.float16)))
    kc, vc = pool()
    refk, refv = pool()
    store_kv_quant(k, v, kc, vc, slots, v_group=group)
    store_kv_quant(k.contiguous(), v.contiguous(), refk, refv, slots, v_group=group)
    assert all(torch.equal(x.view(torch.int32), y.view(torch.int32)) for x,y in [(kc[0],refk[0]),(vc[0],refv[0])])
    assert torch.equal(vc[1],refv[1])
    # Duplicate and shuffled slots exercise a gather rather than a linear copy.
    selected = torch.tensor([66,3,18,3,0,65],device='cuda')
    km, vm = materialize_kv(kc,vc,selected,v_group=group)
    kr = unpack(kc[0].view(torch.int32)[selected]) * scale.float()
    vr = (unpack(vc[0].view(torch.int32)[selected]).reshape(-1,h,d//group,group)
          * vc[1][selected].float().unsqueeze(-1)).reshape(-1,h,d)
    assert torch.equal(km,kr.half())
    assert torch.equal(vm,vr.half())


@pytest.mark.parametrize('m', [1,16])
@torch.inference_mode()
def test_long_split_decode_matches_fp32_reference(m):
    torch.manual_seed(21)
    hq,h,d,page,n = 32,8,128,256,2304
    k=torch.randn(n,h,d,device='cuda',dtype=torch.float16)*0.5
    v=torch.randn_like(k)
    scale=(k.float().abs().amax(0)/7).clamp_min(1e-4).half()
    kc=(torch.zeros(n,h,d//8,device='cuda',dtype=torch.uint32),scale)
    vc=(torch.zeros_like(kc[0]),torch.zeros(n,h,2,device='cuda',dtype=torch.float16))
    slots=torch.arange(n,device='cuda',dtype=torch.int32)
    store_kv_quant(k,v,kc,vc,slots)
    table=torch.arange(8,-1,-1,device='cuda',dtype=torch.int32)[None]
    lens=torch.tensor([2049],device='cuda',dtype=torch.int32)
    q=torch.randn(m,hq,d,device='cuda',dtype=torch.float16)*0.5
    actual=paged_attention_decode(q,kc,vc,table,lens,block_n=page,num_kv_heads=h,verify_m=m)
    positions=torch.arange(2049+m-1,device='cuda')
    physical=table[0,positions//page].long()*page+positions%page
    # FP32 dequantization is independent of the materialization kernel.
    kr=unpack(kc[0].view(torch.int32)[physical])*scale.float()
    vr=(unpack(vc[0].view(torch.int32)[physical]).reshape(-1,h,2,64)
        *vc[1][physical].float().unsqueeze(-1)).reshape(-1,h,d)
    heads=torch.arange(hq,device='cuda')//4
    for row in range(m):
        length=2049+row
        scores=torch.einsum('hd,thd->ht',q[row].float(),kr[:length,heads])*d**-0.5
        ref=torch.einsum('ht,thd->hd',scores.softmax(-1),vr[:length,heads])
        torch.testing.assert_close(actual[row].float(),ref,atol=0.002,rtol=0.002)
