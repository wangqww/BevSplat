"""CPU state/clock checks and one opt-in actual E2 continuation batch."""
import __future__,argparse,ast,copy,hashlib,importlib.util,json,math,os,random,sys,tempfile,time,types,unittest,warnings
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from torch import nn,optim
from torch.optim.lr_scheduler import OneCycleLR
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));torch.set_num_threads(2)
def module(name,path):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
bug=module('continue_bug_helpers',ROOT/'tests/test_vigor_model_bugfix.py')
SCOPE=dict(globals());bug.load('vigor_main/train.py',['_digest_tensors','_restore_continuation_clock','_save_training_state'],SCOPE)
restore=SCOPE['_restore_continuation_clock'];snapshot=SCOPE['_save_training_state'];digest=SCOPE['_digest_tensors']
def toy():
 n=nn.Sequential(nn.Linear(2,2),nn.Linear(2,2),nn.Linear(2,2));o=optim.AdamW([{'params':m.parameters(),'lr':lr} for m,lr in zip(n,[1.25e-6,1.25e-5,1.25e-4])],weight_decay=.005,eps=1e-8);s=OneCycleLR(o,max_lr=[1.25e-6,1.25e-5,1.25e-4],total_steps=78915,pct_start=.01,anneal_strategy='cos',cycle_momentum=False);return n,o,s
def set_rng(r):
 random.setstate(r['python_rng']);v=r['numpy_rng'];np.random.set_state((v['kind'],v['keys'].numpy().astype(np.uint32),v['position'],v['has_gauss'],v['cached_gaussian']));torch.set_rng_state(r['torch_rng'])
class ContinuationCpuTests(unittest.TestCase):
 def test_cli_optin_pair_and_exclusions(self):
  from vigor_main import config
  parse=bug.load('vigor_main/train.py',['parse_args'],dict(globals(),config=config))['parse_args'];d=parse([]);self.assertEqual(d.continue_weights,'');self.assertEqual(d.completed_steps,0)
  with tempfile.NamedTemporaryFile() as f:
   base=['--continue_weights',f.name,'--completed_steps','5261','--resume','1','--seed','20261022'];a=parse(base);self.assertEqual(a.epochs,15)
   for more in [['--test','1'],['--init_ckpt',f.name],['--pilot_steps','1'],['--max_batches','1'],['--zero_init_gaussian_offsets','1'],['--resume','0'],['--resume','15'],['--completed_steps','0']]:
    with self.assertRaises(SystemExit):parse(base+more)
  for argv in [['--completed_steps','1'],['--continue_weights','/no/such/file','--resume','1','--seed','1','--completed_steps','5261']]:
   with self.assertRaises(SystemExit):parse(argv)
 def test_clock_matches_three_group_independent_reference(self):
  n,o,s=toy();before=digest(n.state_dict().items());rng=torch.get_rng_state();a=restore(n,o,s,5261);self.assertEqual(a['scheduler_steps'],5261);self.assertFalse(o.state);self.assertEqual(before,digest(n.state_dict().items()));self.assertTrue(torch.equal(rng,torch.get_rng_state()))
  ref,ro,rs=toy()
  for _ in range(5261):ro.step();rs.step()
  self.assertEqual(s.get_last_lr(),rs.get_last_lr());self.assertEqual(s.state_dict(),rs.state_dict())
  for end in [5262,10522,78915]:
   while s.last_epoch<end:o.step();s.step();ro.step();rs.step()
   self.assertEqual(s.get_last_lr(),rs.get_last_lr());self.assertEqual(s.last_epoch,end)
  self.assertEqual(14*5261,73654);self.assertEqual(14*42087,589218);self.assertEqual(5261+73654,78915)
 def test_safe_sidecar_roundtrip_and_next_update(self):
  random.seed(12);np.random.seed(12);torch.manual_seed(12);n,o,s=toy();restore(n,o,s,5261)
  for p in n.parameters():p.grad=torch.ones_like(p)
  o.step();s.step()
  with tempfile.TemporaryDirectory() as d:
   snapshot(n,o,s,types.SimpleNamespace(continue_weights='bound',completed_steps=5261),5262,2,Path(d));r=torch.load(Path(d)/'training_state_latest.pth',weights_only=True,map_location='cpu');self.assertEqual(r['global_step'],5262);self.assertEqual(r['next_epoch'],2);self.assertFalse((Path(d)/'training_state_latest.pth.tmp').exists())
   expected=(random.random(),np.random.random(),torch.rand(3));m,mo,ms=toy();m.load_state_dict(r['model']);ms.load_state_dict(r['scheduler']);mo.load_state_dict(r['optimizer']);set_rng(r);actual=(random.random(),np.random.random(),torch.rand(3));self.assertEqual(expected[:2],actual[:2]);self.assertTrue(torch.equal(expected[2],actual[2]));self.assertEqual(s.state_dict(),ms.state_dict())
   for model,opt,sch in [(n,o,s),(m,mo,ms)]:
    opt.zero_grad()
    for p in model.parameters():p.grad=torch.ones_like(p)
    opt.step();sch.step()
   self.assertEqual(digest(n.state_dict().items()),digest(m.state_dict().items()));self.assertEqual(s.state_dict(),ms.state_dict());self.assertEqual(o.param_groups[0]['lr'],mo.param_groups[0]['lr'])
@unittest.skipUnless(os.environ.get('BEVSPLAT_E2_CONTINUE_SMOKE')=='1','opt-in actual continued batch8')
class ContinuedCudaSmoke(unittest.TestCase):
 def test_actual_update(self):
  self.assertEqual(os.environ.get('CUDA_VISIBLE_DEVICES'),'0');self.assertTrue(torch.cuda.is_available());np.random.seed(20261022);random.seed(20261022);torch.manual_seed(20261022);torch.cuda.manual_seed_all(20261022);torch.cuda.set_device(0);torch.cuda.reset_peak_memory_stats();torch.backends.cudnn.benchmark=False;start=time.monotonic()
  driver=module('continue_driver_smoke',ROOT/'.codex-implement/runs/vigor-e2-continue15-20261023/run_experiment.py');hub=torch.hub.load
  def offline(repo,model,*a,**kw):
   if kw.get('source')!='local':self.assertEqual(repo,'facebookresearch/dino:main');repo=str(Path(torch.hub.get_dir())/'facebookresearch_dino_main');kw['source']='local'
   return hub(repo,model,*a,**kw)
  with patch.object(torch.hub,'load',offline):
   import vigor_main.train as train
   args=train.parse_args(driver.COMMANDS['train'][driver.COMMANDS['train'].index('--cuda'):]);net=train.BevSplatVIGOR(args,'cuda:0').cuda();train._load_initial(net,args.continue_weights)
  out=driver.RUN/'smoke';out.mkdir(exist_ok=False);scope=dict(vars(train),net=net,args=args,save_path=out,device=torch.device('cuda:0'));fn=next(n for n in ast.parse((ROOT/'vigor_main/train.py').read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='_train');prefix=[]
  for n in fn.body:
   if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='t0' for t in n.targets):break
   prefix.append(n)
  exec(compile(ast.fix_missing_locations(ast.Module(body=prefix,type_ignores=[])),'vigor_main/train.py','exec',flags=__future__.annotations.compiler_flag),scope)
  opt,sch=scope['optimizer'],scope['scheduler'];self.assertFalse(opt.state);self.assertEqual(scope['steps'],5261);self.assertEqual(scope['audit']['initial_full_state'],driver.INPUTS['e2_full_state']);self.assertEqual(len(scope['train_loader']),5261);driver.verify_runtime(json.loads((out/'runtime.json').read_text()));train._set_training_mode(net,args)
  before={label:train._digest_tensors(m.named_parameters()) for label,m in [('gaussian',net.gaussian_encoder),('sat',net.dpt_sat),('grd',net.dpt_grd),('dino',net.dino_feat)]};bns=[(n,m,int(m.num_batches_tracked)) for n,m in net.gaussian_encoder.named_modules() if isinstance(m,nn.modules.batchnorm._BatchNorm)];captures=[]
  def hook(module,inputs,output):
   for t in [output.means,output.covariances,output.opacities]:t.retain_grad()
   captures.append(output)
  handle=net.gaussian_encoder.register_forward_hook(hook);iterator=iter(scope['train_loader']);batch=next(iterator);self.assertEqual(batch[0].shape[0],8);scope.update(batch=batch,loop_idx=0,epoch=1,epoch_samples=0,last_batch=0)
  epochloop=next(n for n in fn.body if isinstance(n,ast.For) and isinstance(n.target,ast.Name) and n.target.id=='epoch');inner=next(n for n in epochloop.body if isinstance(n,ast.For));body=[]
  for node in inner.body:
   body.append(node)
   if isinstance(node,ast.AugAssign) and isinstance(node.target,ast.Name) and node.target.id=='steps':break
  # Strip only max_batches break guard, impossible for validated formal argv.
  body=body[1:]
  exec(compile(ast.fix_missing_locations(ast.Module(body=body,type_ignores=[])),'vigor_main/train.py','exec',flags=__future__.annotations.compiler_flag),scope)
  self.assertEqual(scope['steps'],5262);self.assertEqual(sch.last_epoch,5262);self.assertEqual(sch.total_steps,78915);self.assertTrue(opt.state);self.assertTrue(all(int(st['step'])==1 for st in opt.state.values()));rows=net.gaussian_encoder.to_gaussians[-1].weight.grad.abs().sum(1)
  self.assertTrue(torch.isfinite(rows).all());self.assertTrue((rows[[3,14,25]]>0).all())
  for tensor in [captures[0].means,captures[0].covariances,captures[0].opacities]:self.assertIsNotNone(tensor.grad);self.assertTrue(torch.isfinite(tensor.grad).all());self.assertGreater(float(tensor.grad.abs().sum()),0)
  for label,m in [('gaussian',net.gaussian_encoder),('sat',net.dpt_sat),('grd',net.dpt_grd)]:self.assertNotEqual(before[label],train._digest_tensors(m.named_parameters()))
  self.assertEqual(before['dino'],train._digest_tensors(net.dino_feat.named_parameters()));self.assertTrue(all(p.grad is None for p in net.dino_feat.parameters()))
  for name,m,count in bns:self.assertTrue(m.training);self.assertEqual(int(m.num_batches_tracked)-count,0 if '.layer4.' in name else 1)
  handle.remove();torch.cuda.synchronize();free,total=torch.cuda.mem_get_info();self.assertGreaterEqual(free,1024**3);del iterator
  print('VIGOR_CONTINUE_CUDA_SMOKE='+json.dumps({'checks_passed':True,'initial_full_state':scope['audit']['initial_full_state'],'global_step':scope['steps'],'scheduler_steps':sch.last_epoch,'scheduler_total_steps':sch.total_steps,'adam_new_step':1,'opacity_row_grad_l1':rows[[3,14,25]].tolist(),'peak_reserved_MiB':torch.cuda.max_memory_reserved()/2**20,'free_MiB':free/2**20,'elapsed_seconds':time.monotonic()-start,'scope':'actual unchanged update body; native connectivity not finite difference certification'}),flush=True)
if __name__=='__main__':unittest.main()
