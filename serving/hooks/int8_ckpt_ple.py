import os,re,pathlib
S=pathlib.Path(os.environ['SGLANG_SRT'])/'mem_cache'
APP='''

# int8_ckpt_ple_siblings: carry Qwen4-Exp PLE side states through the int8 checkpoint pool
def _i8_attach(self, sibs):
    self.siblings = []
    for sib in sibs:
        _, src, ax = sib.hicache_transfer_spec()
        sh = list(src.shape); sh[ax] = self.num_slots + 1
        self.siblings.append((src, torch.empty(sh, dtype=src.dtype, device=src.device), ax))
    logger.info("int8 ckpt pool carries %d PLE side state(s)", len(self.siblings))
_i8_st0 = MambaCheckpointPool.store_from_active
def _i8_st(self, pool, a, k):
    _i8_st0(self, pool, a, k)
    for src, buf, ax in getattr(self, "siblings", []):
        buf.index_copy_(ax, k.reshape(-1).to(buf.device).long(), src.index_select(ax, a.reshape(-1).to(src.device).long()))
_i8_ld0 = MambaCheckpointPool.load_to_active
def _i8_ld(self, pool, k, a):
    _i8_ld0(self, pool, k, a)
    for src, buf, ax in getattr(self, "siblings", []):
        src.index_copy_(ax, a.reshape(-1).to(src.device).long(), buf.index_select(ax, k.reshape(-1).to(buf.device).long()))
MambaCheckpointPool.attach_slot_siblings = _i8_attach
MambaCheckpointPool.store_from_active = _i8_st
MambaCheckpointPool.load_to_active = _i8_ld
'''
p=S/'memory_pool.py'; s=p.read_text()
if 'attach_slot_siblings' not in s:
    s2=re.sub(r'raise ValueError\(\s*"--enable-int8-mamba-checkpoint is incompatible with Qwen4-Exp "\s*"PLE side states"\s*\)','self.mamba_ckpt_pool.attach_slot_siblings(list(self.mamba_pool._slot_siblings))',s)
    assert s2!=s,'anchor memory_pool'; p.write_text(s2)
q=S/'mamba_checkpoint_pool.py'; t=q.read_text()
if '_i8_attach' not in t: q.write_text(t+APP)
print('INT8CKPT-PLE ALL-OK')
