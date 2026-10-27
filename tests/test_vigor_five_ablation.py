"""CPU controls and one disposable actualbatch8 update for an explicitly selected arm."""
import argparse,ast,copy,hashlib,importlib.util,json,math,os,random,sys,tempfile,time,types,unittest
from pathlib import Path
from unittest.mock import patch
import __future__
import numpy as np
import torch
from torch import nn,optim
from torch.optim.lr_scheduler import OneCycleLR
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));torch.set_num_threads(2)
def module(name,path):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
bug=module('old_bug_tests',ROOT/'tests/test_vigor_model_bugfix.py');joint=module('old_joint_tests',ROOT/'tests/test_vigor_joint_optimization.py');suite_path=ROOT/'.codex-implement/runs/vigor-five-ablation-20261023-repair02/run_experiment.py';suite=module('five_suite',suite_path) if suite_path.is_file() else None;ARM=os.environ.get('BEVSPLAT_FIVE_ARM','')
class FiveAblationCpuTests(unittest.TestCase):
 @unittest.skipIf(suite is None, 'private five-arm controller is not part of this release')
 def test_cli_scopes_and_validation(self):
  from vigor_main import config
  parse=bug.load('vigor_main/train.py',['parse_args'],dict(vars(bug),config=config))['parse_args'];default=parse([]);self.assertEqual(default.confidence_mode,'inherit');self.assertIsNone(default.gaussian_pretrained_lr_scale)
  for arm,cmd in suite.COMMANDS.items():
   args=parse(cmd[cmd.index('--cuda'):]);self.assertEqual({k:getattr(args,k) for k in suite.expected_flags(arm)},suite.expected_flags(arm));self.assertEqual(args.pilot_steps,5261);self.assertEqual(args.init_ckpt,'')
  for key,value in [('gaussian_offset_max','0'),('gaussian_scale_max','nan'),('gaussian_opacity_max','1.1'),('gaussian_pretrained_lr_scale','-1')]:
   with self.assertRaises(SystemExit):parse(['--'+key,value])
  self.assertEqual(list(suite.COMMANDS),['e1','e2','e3','e4','e5'])
 def test_independent_confidence_keeps_corrected_opacity(self):
  class Dino(nn.Module):
   feat_dim=[2]*4
  class Dpt(nn.Module):
   def __init__(self,dims,confidence_mode):super().__init__();self.mode=confidence_mode
  scope=dict(vars(bug),DINO=Dino,DPT=Dpt,GaussianEncoder=lambda **kw:nn.Identity(),VGGUnet=lambda *a:nn.Identity())
  C=bug.load('vigor_main/model.py',['BevSplatVIGOR'],scope)['BevSplatVIGOR'];args=types.SimpleNamespace(model_behavior='corrected',confidence_mode='legacy',share=0,grd_res=80,level='1',channels='32_16_4',N_iters=1,area='same');net=C(args,'cpu');self.assertEqual(net.dpt_sat.mode,'legacy');self.assertEqual(net.dpt_grd.mode,'legacy');self.assertEqual(net.model_behavior,'corrected')
 def test_ranges_preserve_state_rng_prior_and_smooth_opacity(self):
  class Backbone(nn.Module):
   d_out=4
   def forward(self,x):return x.new_ones(x.shape[0],4,*x.shape[-2:])
  scope=dict(vars(bug),BackboneDino=Backbone);bug.load('gaussian/build_gaussians.py',['quaternion_to_matrix','build_covariance'],scope);C=bug.load('gaussian/encoder_pano.py',['Gaussians','equirectangular_to_xyz','GaussianEncoder'],scope)['GaussianEncoder'];torch.manual_seed(29);enc=C();state=copy.deepcopy(enc.state_dict());rng=torch.get_rng_state();img=torch.rand(1,3,2,4);feat=torch.ones(1,32,2,4);conf=torch.ones(1,1,2,4);depth=torch.ones(1,1,2,4)
  base=enc(img,feat,conf,depth);prior=enc.geometry_prior().clone();enc.configure_ranges(3,.3,.95);self.assertEqual(set(enc.state_dict()),set(state));self.assertTrue(all(torch.equal(v,enc.state_dict()[k]) for k,v in state.items()));limited=enc(img,feat,conf,depth);self.assertTrue(torch.equal(limited.opacities,.95*base.opacities));self.assertTrue(torch.equal(prior,enc.geometry_prior()));enc.configure_ranges(3,.2,1);scaled=enc(img,feat,conf,depth);self.assertTrue(torch.allclose(scaled.covariances,base.covariances*(4/9),atol=1e-6));enc.configure_ranges(1,.3,1);enc(img,feat,conf,depth);self.assertTrue(torch.allclose(enc.geometry_prior(),prior/9));before=torch.get_rng_state();enc.configure_ranges();self.assertTrue(torch.equal(before,torch.get_rng_state()))
  for values in [(0,.3,1),(3,float('nan'),1),(3,.3,1.01)]:
   with self.assertRaises(ValueError):enc.configure_ranges(*values)
 def test_lr_group_pretrained_only_with_correct_schedule(self):
  for share in [0,1]:
   net=joint.TinyNet(share);g=nn.Module();g.backbone=nn.Module();g.backbone.resnet_backbone=nn.Module();g.backbone.resnet_backbone.model=nn.Linear(2,2);g.backbone.resnet_backbone.projections=nn.Linear(2,2);g.head=nn.Linear(2,2);net.gaussian_encoder=g;args=joint.args_for(share);args.gaussian_lr_scale=.1;args.gaussian_pretrained_lr_scale=.1
   with tempfile.TemporaryDirectory() as d:scope=joint.optimizer_scope('vigor_main/train.py','_train',net,args,Path(d),joint.small_config())
   opt=scope['optimizer'];groups=opt.param_groups;ids=[id(p) for group in groups for p in group['params']];self.assertEqual(len(ids),len(set(ids)));self.assertEqual(set(id(p) for p in groups[0]['params']),set(id(p) for p in g.backbone.resnet_backbone.model.parameters()));self.assertTrue(set(id(p) for p in g.backbone.resnet_backbone.projections.parameters())<=set(id(p) for p in groups[1]['params']));expected=set(id(p) for p in list(g.parameters())+list(net.dpt_sat.parameters())+([] if share else list(net.dpt_grd.parameters())));self.assertEqual(set(ids),expected)
   for step in range(3):
    lr=scope['scheduler'].get_last_lr();self.assertAlmostEqual(lr[0]/lr[1],.1);self.assertAlmostEqual(lr[1]/lr[2],.1);opt.step();scope['scheduler'].step()
 def test_fp32_small_lr_does_not_guarantee_individual_bit_change(self):
  large=nn.Parameter(torch.full((8,),8.,dtype=torch.float32));zero=nn.Parameter(torch.zeros(8));before=large.detach().clone();prior=zero.detach().clone();opt=optim.AdamW([large,zero],lr=5e-8,weight_decay=.005,eps=1e-8);large.grad=torch.ones_like(large);zero.grad=torch.ones_like(zero);opt.step();self.assertTrue(torch.equal(large,before));self.assertFalse(torch.equal(zero,prior));self.assertGreater(float(large.grad.abs().sum()),0)
@unittest.skipUnless(suite is not None and ARM in suite.NAMES,'opt-in private actual batch8 arm')
class FiveAblationCudaSmoke(unittest.TestCase):
 def test_corrected_update(self):
  self.assertEqual(os.environ.get('CUDA_VISIBLE_DEVICES'),'0');self.assertTrue(torch.cuda.is_available());np.random.seed(20261022);random.seed(20261022);torch.manual_seed(20261022);torch.cuda.manual_seed_all(20261022);torch.cuda.set_device(0);torch.cuda.reset_peak_memory_stats();torch.backends.cudnn.benchmark=False;start=time.monotonic()
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
   args=parse_args(suite.COMMANDS[ARM][suite.COMMANDS[ARM].index('--cuda'):]);self.assertEqual(args.init_ckpt,'');self.assertEqual(args.resume,0);net=BevSplatVIGOR(args,'cuda:0').cuda();net.gaussian_encoder.zero_initialize_position_offsets()
  out=suite.RUN/('smoke_'+ARM);out.mkdir(exist_ok=False);scope=dict(globals(),net=net,args=args,save_path=out,device=torch.device('cuda:0'),config=config,load_train_and_val=load_train_and_val,_audit_initial=_audit_initial,_audit_finish=_audit_finish,_TraceSampler=_TraceSampler,_digest_tensors=_digest_tensors,_gaussian_bn_audit=_gaussian_bn_audit)
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
  used_bns=[m for m in bns if id(m) in used];self.assertTrue(used_bns);self.assertTrue(all(m.training and int(m.num_batches_tracked)==counts[id(m)]+1 for m in used_bns));affine=[p for m in used_bns for p in m.parameters(recurse=False) if p.grad is not None and float(p.grad.abs().sum())>0];self.assertTrue(affine);bn_before=[p.detach().clone() for p in affine]
  for g in optimizer.param_groups:
   for p in g['params']:
    if p.grad is not None:self.assertTrue(torch.isfinite(p.grad).all())
  before=head.weight.detach().clone();optimizer.step();scheduler.step();dpt_deltas=[float((p-v).abs().max()) for p,v in selected];bn_deltas=[float((p-v).abs().max()) for p,v in zip(affine,bn_before)];update_evidence={'dpt_representative_max_abs_deltas':dpt_deltas,'bn_affine_max_abs_deltas':bn_deltas,'bn_changed_tensors':sum(d>0 for d in bn_deltas),'bn_active_affine_tensors':len(bn_deltas),'initial_optimizer_lrs':[g['initial_lr'] for g in optimizer.param_groups]};(out/'step_update.json').write_text(json.dumps(update_evidence,indent=2)+'\n');self.assertFalse(torch.equal(before,head.weight));self.assertTrue(all(d>0 for d in dpt_deltas));self.assertGreater(update_evidence['bn_changed_tensors'],0);self.assertEqual(scheduler.last_epoch,1);self.assertAlmostEqual(scheduler.get_last_lr()[0]/scheduler.get_last_lr()[1],.1);self.assertTrue(all(p.grad is None for p in net.dino_feat.parameters()));self.assertEqual(dino_before,_digest_tensors(net.dino_feat.named_parameters()));counters=[int(m.num_batches_tracked) for m in bns];net.eval()
  with torch.no_grad():net(sat,grd,depth,grd_ori,mpp,rot,gtu,gtv)
  self.assertEqual(counters,[int(m.num_batches_tracked) for m in bns]);_set_training_mode(net,args);self.assertTrue(all(m.training for m in bns));torch.cuda.synchronize();free,total=torch.cuda.mem_get_info();self.assertGreaterEqual(free,1024**3)
  for h in handles:h.remove()
  del iterator
  print('VIGOR_FIVE_CUDA_SMOKE='+json.dumps({'arm':ARM,'update_evidence':update_evidence,'checks_passed':True,'batch_size':8,'actual_training_batch':True,'elapsed_seconds':time.monotonic()-start,'loss':float(loss),'opacity_row_grad_l1':rows[[3,14,25]].tolist(),'head_row_grad_l1':rows.tolist(),'initial_full_state':scope['audit']['initial_full_state'],'scheduler_total_steps':scheduler.total_steps,'scheduler_steps':scheduler.last_epoch,'dino_unchanged':True,'bn_and_dpt_updated':True,'peak_reserved_MiB':torch.cuda.max_memory_reserved()/2**20,'free_MiB':free/2**20,'scope':'native gradient connectivity/update, not finite-difference derivative certification'}),flush=True)
if __name__=='__main__':unittest.main()
