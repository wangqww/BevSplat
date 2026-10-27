"""Focused opt-in VIGOR bugfix regression tests; no GPU in ordinary unittest."""
import argparse,ast,copy,hashlib,importlib.util,json,math,os,random,sys,tempfile,time,types,unittest
from pathlib import Path
from dataclasses import dataclass
from unittest.mock import patch
import __future__
import numpy as np
import torch
from torch import nn,optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import OneCycleLR
from einops import rearrange
import torchvision.transforms.functional as TF
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));torch.set_num_threads(2)
def load(rel,names,scope=None):
 scope=globals().copy() if scope is None else scope
 tree=ast.parse((ROOT/rel).read_text());tree.body=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in names]
 exec(compile(ast.fix_missing_locations(tree),rel,'exec',flags=__future__.annotations.compiler_flag),scope);return scope
def dpt_scope():return load('models/dpt_single.py',['L2_norm','ResidualConvUnit','FeatureFusionBlock','DPT'],dict(globals(),interpolate=F.interpolate))
def loss_scope():return load('vigor_main/losses.py',['_translation_origin','corr_for_translation'])
def driver_module():
 p=ROOT/'.codex-implement/runs/vigor-model-bugfix-epoch1-20261022/run_experiment.py'
 if not p.is_file():raise unittest.SkipTest('private historical experiment controller is not part of this release')
 sp=importlib.util.spec_from_file_location('bugfix_driver',p);m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m);return m
class VigorBugfixCpuTests(unittest.TestCase):
 def test_dpt_schema_rng_and_legacy(self):
  D=dpt_scope()['DPT'];torch.manual_seed(31);a=D([2]*4,hidden_dim=2);rng=torch.get_rng_state();torch.manual_seed(31);b=D([2]*4,hidden_dim=2,confidence_mode='corrected');self.assertTrue(torch.equal(rng,torch.get_rng_state()));self.assertEqual(list(a.state_dict()),list(b.state_dict()))
  for k,v in a.state_dict().items():self.assertTrue(torch.equal(v,b.state_dict()[k]))
  feats=[torch.ones(1,2,1,1) for _ in range(4)];captured=[];handle=a.conf.register_forward_hook(lambda m,x,y:captured.append(y));_,c=a([x.clone() for x in feats]);handle.remove();self.assertTrue(torch.equal(c,torch.sigmoid(-captured[0])))
  with self.assertRaises(ValueError):D([2]*4,hidden_dim=2,confidence_mode='bad')
 def test_confidence_range_and_negative_gradient(self):
  d=dpt_scope()['DPT']([2]*4,hidden_dim=2,confidence_mode='corrected')
  with torch.no_grad():
   for p in d.parameters():p.zero_()
   d.out_conv[-1].bias.fill_(1.)
  values=[]
  for w in [-100.,0.,100.]:
   with torch.no_grad():d.conf[1].weight.fill_(w)
   _,c=d([torch.ones(1,2,1,1) for _ in range(4)]);values.append(float(c.mean()))
   if w==0:c.sum().backward();self.assertLess(float(d.conf[1].weight.grad.max()),0)
  self.assertEqual(values,[1.,.5,0.])
 def test_identity_signed_translations_and_rounding(self):
  fn=loss_scope()['corr_for_translation'];torch.manual_seed(27)
  for A,K in [(128,51),(130,51),(129,51),(128,50)]:
   sat=torch.randn(1,4,A,A);conf=torch.ones(1,1,A,A);start=int(round((A-K)/2))
   for u,v in [(0,0),(2,-1),(-2,1)]:
    grd=sat[:,:,start+v:start+v+K,start+u:start+u+K];gc=torch.ones(1,1,K,K);args=types.SimpleNamespace(level='1',ConfGrd=1,ConfSat=0,use_uncertainty=0,translation_origin='crop');pu,pv,c=fn({1:sat},{1:conf},{1:grd},{1:gc},args);self.assertEqual((float(pu),float(pv)),(4*u,4*v));self.assertEqual(pu.dtype,torch.float32)
    args.translation_origin='legacy';lu,lv,_=fn({1:sat},{1:conf},{1:grd},{1:gc},args);L=A-K+1;self.assertEqual((float(lu),float(lv)),(4*(start+u-L/2),4*(start+v-L/2)))
  with self.assertRaises(ValueError):loss_scope()['_translation_origin'](78,'bad')
 def test_opacity_route_and_per_branch_task_gradients(self):
  class BackboneDino(nn.Module):
   d_out=512
   def forward(self,x):return x.mean(1,keepdim=True).repeat(1,512,1,1)
  scope=dict(globals(),BackboneDino=BackboneDino);load('gaussian/build_gaussians.py',['quaternion_to_matrix','build_covariance'],scope);load('gaussian/encoder_pano.py',['Gaussians','equirectangular_to_xyz','GaussianEncoder'],scope);load('vis_gaussian_pano.py',['compute_equal_aabb_with_margin','render_projections'],scope)
  captured=[]
  def renderer(*a,**kw):
   captured.append((a[10],a[12]));x=((a[10]*a[11]).sum()+.01*a[7].square().sum()+.01*a[8].sum()).reshape(1,1,1,1);return types.SimpleNamespace(color=x,feature=x,confidence=x,depth=x)
  scope['render_cuda_orthographic']=renderer;torch.manual_seed(42);enc=scope['GaussianEncoder']();img=torch.rand(1,3,2,4);feat=torch.ones(1,32,2,4);conf=torch.ones(1,1,2,4)*.35;depth=torch.ones(1,1,2,4)*10
  for corrected in [False,True]:
   enc.zero_grad();g=enc(img,feat,conf,depth);g.opacities.retain_grad();scope['render_projections'](g,(2,2),use_predicted_opacity=corrected)[1].sum().backward();a,c=captured[-1];self.assertTrue(torch.equal(c,g.confidence));self.assertTrue(torch.equal(a,g.opacities if corrected else g.confidence));grad=enc.to_gaussians[-1].weight.grad;opacity=grad[[3,14,25]].abs().sum(1)
   if corrected:self.assertTrue((opacity>0).all());self.assertIsNotNone(g.opacities.grad)
   else:self.assertEqual(float(opacity.sum()),0);self.assertIsNone(g.opacities.grad)
   for rows in [[0,1,2,11,12,13,22,23,24],[4,5,6,7,15,16,17,18,26,27,28,29],[8,9,10,19,20,21,30,31,32]]:self.assertGreater(float(grad[rows].abs().sum()),0)
 def test_cli_defaults_and_mode_validation(self):
  from vigor_main import config
  parse=load('vigor_main/train.py',['parse_args'],dict(globals(),config=config))['parse_args'];a=parse([]);self.assertEqual((a.model_behavior,a.translation_origin),('legacy','legacy'));b=parse(['--model_behavior','corrected','--translation_origin','crop']);self.assertEqual((b.model_behavior,b.translation_origin),('corrected','crop'))
  with self.assertRaises(SystemExit):parse(['--model_behavior','bad'])
  cmd=driver_module().COMMANDS['train'];self.assertEqual(parse(cmd[cmd.index('--cuda'):]).pilot_steps,5261)
 def test_model_behavior_propagation_and_grid_invariant(self):
  class Dino(nn.Module):
   feat_dim=[2]*4
   def forward(self,x):return [torch.ones(x.shape[0],2,32 if x.shape[-1]==512 else 20,32 if x.shape[-1]==512 else 40)]*4
  class Dpt(nn.Module):
   def __init__(self,dims,confidence_mode):super().__init__();self.mode=confidence_mode
   def forward(self,fs):x=fs[0];shape=(x.shape[0],32,x.shape[-2]*4,x.shape[-1]*4);return torch.ones(shape),torch.ones(shape[0],1,*shape[-2:])
  class Encoder(nn.Module):
   def __init__(self,**kw):super().__init__()
   def forward(self,*a):return types.SimpleNamespace(batch=a[0].shape[0])
  calls=[];size=[128]
  def render(g,res,**kw):calls.append(kw);B=g.batch;A=size[0];return torch.zeros(B,3,A,A),torch.ones(B,32,A,A),torch.ones(B,1,A,A),torch.zeros(B,A,A)
  from vigor_main.config import BEV_CROP_RATIO,BEV_HEIGHT_M,BEV_RES,BEV_WIDTH_M,GRD_H,GRD_W,SAT_H,SAT_W
  scope=dict(globals(),DINO=Dino,DPT=Dpt,GaussianEncoder=Encoder,VGGUnet=lambda *a:nn.Identity(),render_projections=render,**{k:v for k,v in locals().items() if k.startswith(('BEV_','GRD_','SAT_'))});C=load('vigor_main/model.py',['BevSplatVIGOR'],scope)['BevSplatVIGOR']
  for mode in [None,'legacy','corrected']:
   a=types.SimpleNamespace(share=0,grd_res=80,level='1',channels='32_16_4',N_iters=1,area='same',rotation_range=0,translation_origin='crop')
   if mode is not None:a.model_behavior=mode
   net=C(a,'cpu');self.assertEqual(net.dpt_sat.mode,mode or 'legacy');self.assertEqual(net.dpt_grd.mode,mode or 'legacy');sat=torch.zeros(1,3,512,512);grd=torch.zeros(1,3,320,640);depth=torch.ones(1,1,320,640);zero=torch.zeros(1);net(sat,grd,depth,grd,torch.ones(1),zero);self.assertEqual(calls[-1]['use_predicted_opacity'],mode=='corrected')
   size[0]=127
   with self.assertRaises(ValueError):net(sat,grd,depth,grd,torch.ones(1),zero)
   size[0]=128
 def test_driver_exact_scope_and_invalid_audits(self):
  m=driver_module();self.assertEqual(list(m.COMMANDS),['smoke','baseline_val','train','old0_val']);cmd=m.COMMANDS['train'];self.assertEqual(cmd[cmd.index('--pilot_steps')+1],'5261');self.assertEqual(cmd[cmd.index('--init_ckpt')+1],'');self.assertEqual(cmd[cmd.index('--model_behavior')+1],'corrected');self.assertEqual(m.CAPS,{'smoke':300,'baseline_val':1800,'train':10800,'old0_val':1800});a=copy.deepcopy(m.INPUTS['baseline_audit']);self.assertTrue(m.verify_audit(a,a));b=copy.deepcopy(a);b['optimizer_steps']=5262
  with self.assertRaises(RuntimeError):m.verify_audit(b,a)
  b=copy.deepcopy(a);b['initial_full_state']='bad'
  with self.assertRaises(RuntimeError):m.verify_audit(b,a)
  for phase in ['baseline_val','old0_val']:self.assertIn('legacy',m.COMMANDS[phase]);self.assertIn('crop',m.COMMANDS[phase]);self.assertNotIn('--zero_init_gaussian_offsets',m.COMMANDS[phase])
 def test_driver_timeout_reaps_only_its_child(self):
  m=driver_module()
  with tempfile.TemporaryDirectory() as d:
   log=Path(d)/'timeout.log'
   with self.assertRaises(__import__('subprocess').TimeoutExpired):m.run_command([sys.executable,'-c','import time; print("started",flush=True); time.sleep(5)'],log,.1,dict(os.environ))
   self.assertIsNone(m.ACTIVE);self.assertTrue(log.exists())

 def test_driver_cleanup_early_leader_exit_and_term_resistant_descendants(self):
  import subprocess
  m=driver_module();stop=m.stop_child
  worker='import os,signal,sys,time;from pathlib import Path;signal.signal(signal.SIGTERM,signal.SIG_IGN);Path(sys.argv[1]).write_text(str(os.getpid()));time.sleep(60)'
  outsider=subprocess.Popen([sys.executable,'-c','import time;time.sleep(60)'],start_new_session=True)
  try:
   with tempfile.TemporaryDirectory() as directory:
    for early in [True,False]:
     pidfile=Path(directory)/('early.pid' if early else 'timeout.pid')
     parent='import subprocess,sys,time;from pathlib import Path;subprocess.Popen([sys.executable,"-c",'+repr(worker)+',sys.argv[1]]);p=Path(sys.argv[1]);\nwhile not p.exists():time.sleep(.01)\n'+('' if early else 'time.sleep(60)')
     expected=RuntimeError if early else subprocess.TimeoutExpired
     with patch.object(m,'stop_child',side_effect=lambda child:stop(child,grace=.2)):
      with self.assertRaises(expected):m.run_command([sys.executable,'-c',parent,str(pidfile)],Path(directory)/(str(early)+'.log'),1.,dict(os.environ))
     self.assertIsNone(m.ACTIVE);self.assertTrue(pidfile.exists());identity=m.process_identity(int(pidfile.read_text()));self.assertTrue(identity is None or identity['state'] in ('Z','X'));self.assertIsNone(outsider.poll())
  finally:
   outsider.terminate();outsider.wait(timeout=5)

 def test_separate_smoke_receipt_binding(self):
  m=driver_module()
  with tempfile.TemporaryDirectory() as d:
   m.RUN=Path(d);result={'checks_passed':True,'batch_size':8,'scheduler_total_steps':78915,'initial_full_state':m.INPUTS['baseline_audit']['initial_full_state'],'opacity_row_grad_l1':[1.,2.,3.]};(m.RUN/'smoke.log').write_text('VIGOR_BUGFIX_CUDA_SMOKE='+json.dumps(result));(m.RUN/'smoke.exit').write_text('0');(m.RUN/'launch-inputs.json').write_text('fixture')
   receipt={'session':'bevsplat-vigor-bugfix-smoke-20261022','owned_processes_terminated':True,'launch_inputs_sha256':m.sha(m.RUN/'launch-inputs.json'),'source_hashes':m.INPUTS['source_hashes'],'log_sha256':m.sha(m.RUN/'smoke.log'),'result':result};(m.RUN/'smoke-receipt.json').write_text(json.dumps(receipt));self.assertEqual(m.load_smoke_receipt()['result'],result);(m.RUN/'smoke.log').write_text('changed')
   with self.assertRaises(RuntimeError):m.load_smoke_receipt()

@unittest.skipUnless(os.environ.get('BEVSPLAT_VIGOR_BUGFIX_CUDA_SMOKE')=='1','opt-in actual corrected batch8')
class VigorBugfixCudaSmoke(unittest.TestCase):
 def test_corrected_update(self):
  self.assertEqual(os.environ.get('CUDA_VISIBLE_DEVICES'),'0');self.assertTrue(torch.cuda.is_available());np.random.seed(20261022);random.seed(20261022);torch.manual_seed(20261022);torch.cuda.manual_seed_all(20261022);torch.cuda.set_device(0);torch.cuda.reset_peak_memory_stats();start=time.monotonic()
  hub=torch.hub.load
  def offline(repo,model,*a,**kw):
   if kw.get('source')!='local':self.assertEqual(repo,'facebookresearch/dino:main');repo=str(Path(torch.hub.get_dir())/'facebookresearch_dino_main');kw['source']='local'
   return hub(repo,model,*a,**kw)
  with patch.object(torch.hub,'load',offline):
   from vigor_main import config
   from vigor_main.train import parse_args,_audit_initial,_audit_finish,_TraceSampler,_digest_tensors,_gaussian_bn_audit,_set_training_mode
   from vigor_main.model import BevSplatVIGOR
   from vigor_main.data import load_train_and_val
   from vigor_main.losses import batch_wise_cross_corr,Weakly_supervised_loss_w_GPS_error
   args=parse_args(['--seed','20261022','--batch_size','8','--cuda','0','--epochs','15','--gaussian_lr_scale','0.1','--geometry_prior_coe','0.1','--zero_init_gaussian_offsets','1','--model_behavior','corrected','--translation_origin','crop']);self.assertEqual(args.init_ckpt,'');self.assertEqual(args.resume,0);net=BevSplatVIGOR(args,'cuda:0').cuda();net.gaussian_encoder.zero_initialize_position_offsets()
  out=ROOT/'.codex-implement/runs/vigor-model-bugfix-epoch1-20261022/smoke-audit';out.mkdir(exist_ok=False);scope=dict(globals(),net=net,args=args,save_path=out,device=torch.device('cuda:0'),config=config,load_train_and_val=load_train_and_val,_audit_initial=_audit_initial,_audit_finish=_audit_finish,_TraceSampler=_TraceSampler,_digest_tensors=_digest_tensors,_gaussian_bn_audit=_gaussian_bn_audit)
  fn=next(n for n in ast.parse((ROOT/'vigor_main/train.py').read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='_train');prefix=[]
  for n in fn.body:
   if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='t0' for t in n.targets):break
   prefix.append(n)
  exec(compile(ast.fix_missing_locations(ast.Module(body=prefix,type_ignores=[])),'vigor_main/train.py','exec',flags=__future__.annotations.compiler_flag),scope)
  optimizer=scope['optimizer'];scheduler=scope['scheduler'];self.assertEqual(len(scope['train_loader']),5261);self.assertEqual(scheduler.total_steps,78915);self.assertEqual(scope['audit']['initial_full_state'],'f3db7173051fed39bf455f88e0e9669eb00a132fcf240b9ed0a2e476a6db3f51');dino_before=_digest_tensors(net.dino_feat.named_parameters());captures=[];used=set();bns=[m for m in net.gaussian_encoder.modules() if isinstance(m,nn.BatchNorm2d)];counts={id(m):int(m.num_batches_tracked) for m in bns};handles=[m.register_forward_hook(lambda layer,inputs,output:used.add(id(layer))) for m in bns]
  def hook(module,inputs,output):
   for t in [output.means,output.covariances,output.opacities]:
    if t.requires_grad:t.retain_grad()
   captures.append(output)
  handles.append(net.gaussian_encoder.register_forward_hook(hook));iterator=iter(scope['train_loader']);batch=next(iterator);self.assertEqual(len(batch),8);grd,sat,depth,grd_ori,gtu,gtv,rot,mpp=(v.cuda() for v in batch);self.assertEqual(sat.shape[0],8);_set_training_mode(net,args);optimizer.zero_grad();outputs=net(sat,grd,depth,grd_ori,mpp,rot,gtu,gtv);corr=batch_wise_cross_corr(*outputs[:4],args,outputs[4]);weak,gps=Weakly_supervised_loss_w_GPS_error(corr,gtu,gtv,args,mpp,args.GPS_error);loss=weak+args.GPS_error_coe*gps+args.geometry_prior_coe*net.gaussian_encoder.geometry_prior();self.assertTrue(torch.isfinite(loss));loss.backward();head=net.gaussian_encoder.to_gaussians[-1];rows=head.weight.grad.abs().sum(1);self.assertTrue(torch.isfinite(rows).all());self.assertTrue((rows[[3,14,25]]>0).all());self.assertIsNotNone(captures[0].opacities.grad);self.assertGreater(float(captures[0].opacities.grad.abs().sum()),0)
  for group in [[0,1,2,11,12,13,22,23,24],[4,5,6,7,15,16,17,18,26,27,28,29],[8,9,10,19,20,21,30,31,32]]:self.assertGreater(float(rows[group].sum()),0)
  for t in [captures[0].means,captures[0].covariances,captures[0].opacities]:self.assertIsNotNone(t.grad);self.assertTrue(torch.isfinite(t.grad).all());self.assertGreater(float(t.grad.abs().sum()),0)
  selected=[]
  for module in [net.dpt_sat,net.dpt_grd]:
   active=[p for p in module.parameters() if p.grad is not None and float(p.grad.abs().sum())>0];self.assertTrue(active);selected.append((active[0],active[0].detach().clone()))
  used_bns=[m for m in bns if id(m) in used];self.assertTrue(used_bns);self.assertTrue(all(m.training and int(m.num_batches_tracked)==counts[id(m)]+1 for m in used_bns));affine=[p for m in used_bns for p in m.parameters(recurse=False) if p.grad is not None and float(p.grad.abs().sum())>0];self.assertTrue(affine);selected.append((affine[0],affine[0].detach().clone()))
  for g in optimizer.param_groups:
   for p in g['params']:
    if p.grad is not None:self.assertTrue(torch.isfinite(p.grad).all())
  before=head.weight.detach().clone();optimizer.step();scheduler.step();self.assertFalse(torch.equal(before,head.weight));self.assertTrue(all(not torch.equal(p,v) for p,v in selected));self.assertEqual(scheduler.last_epoch,1);self.assertAlmostEqual(scheduler.get_last_lr()[0]/scheduler.get_last_lr()[1],.1);self.assertTrue(all(p.grad is None for p in net.dino_feat.parameters()));self.assertEqual(dino_before,_digest_tensors(net.dino_feat.named_parameters()));counters=[int(m.num_batches_tracked) for m in bns];net.eval()
  with torch.no_grad():net(sat,grd,depth,grd_ori,mpp,rot,gtu,gtv)
  self.assertEqual(counters,[int(m.num_batches_tracked) for m in bns]);_set_training_mode(net,args);self.assertTrue(all(m.training for m in bns));torch.cuda.synchronize();free,total=torch.cuda.mem_get_info();self.assertGreaterEqual(free,1024**3)
  for h in handles:h.remove()
  del iterator
  print('VIGOR_BUGFIX_CUDA_SMOKE='+json.dumps({'checks_passed':True,'batch_size':8,'actual_training_batch':True,'elapsed_seconds':time.monotonic()-start,'loss':float(loss),'opacity_row_grad_l1':rows[[3,14,25]].tolist(),'head_row_grad_l1':rows.tolist(),'initial_full_state':scope['audit']['initial_full_state'],'scheduler_total_steps':scheduler.total_steps,'scheduler_steps':scheduler.last_epoch,'dino_unchanged':True,'bn_and_dpt_updated':True,'peak_reserved_MiB':torch.cuda.max_memory_reserved()/2**20,'free_MiB':free/2**20,'scope':'native gradient connectivity/update, not finite-difference derivative certification'}),flush=True)
if __name__=='__main__':unittest.main()
