import os, json, time
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from PIL import Image
from skimage.metrics import structural_similarity as ssim_metric

DEVICE="cuda"; SCENE_DIR="/home/logus/env/iscte/co/data/nerf_synthetic/lego"
N_SAMPLES=64; LR=5e-4; SEED=0; NEAR,FAR=2.0,6.0
BATCH=4096; N_ITERATIONS=40000; EVAL_EVERY=2000; N_TEST_VIEWS=3
# (resolution, optimizer) — SGD and Adam at the same LR for a clean controlled comparison.
# Adam@200x200 is reused from the earlier bench_800 run (identical config: best 23.08 dB).
CONFIGS=[(200,"sgd"),(400,"adam"),(400,"sgd")]
print(f"SEPARABILITY CHECK | small MLP 128x4 | batch {BATCH} | {N_ITERATIONS} iters | LR {LR}", flush=True)

def load_split(scene_dir, split, max_views=None):
    with open(os.path.join(scene_dir,f"transforms_{split}.json")) as f: meta=json.load(f)
    frames=meta["frames"][:max_views] if max_views else meta["frames"]
    imgs,poses=[],[]
    for fr in frames:
        rgba=np.asarray(Image.open(os.path.join(scene_dir,fr["file_path"]+".png")),dtype=np.float32)/255.0
        imgs.append(rgba[...,:3]*rgba[...,3:4]+(1.0-rgba[...,3:4]))
        poses.append(np.asarray(fr["transform_matrix"],dtype=np.float32))
    images=torch.from_numpy(np.stack(imgs)); poses=torch.from_numpy(np.stack(poses))
    focal=0.5*images.shape[2]/np.tan(0.5*float(meta["camera_angle_x"]))
    return images,poses,float(focal)

def downscale(images,res):
    x=images.permute(0,3,1,2); x=F.interpolate(x,size=(res,res),mode="area")
    return x.permute(0,2,3,1).contiguous()

class PositionalEncoding(nn.Module):
    def __init__(self,num_freqs=10):
        super().__init__()
        self.freqs=2.0**torch.arange(num_freqs).float().to(DEVICE)
        self.out_dim=3+3*2*num_freqs
    def forward(self,x):
        encs=[x]
        for f in self.freqs: encs+=[torch.sin(f*x),torch.cos(f*x)]
        return torch.cat(encs,dim=-1)

class TinyNeRF(nn.Module):
    def __init__(self,enc_dim,width=128,depth=4):
        super().__init__()
        layers=[nn.Linear(enc_dim,width),nn.ReLU()]
        for _ in range(depth-1): layers+=[nn.Linear(width,width),nn.ReLU()]
        self.trunk=nn.Sequential(*layers)
        self.density=nn.Linear(width,1); self.rgb=nn.Linear(width,3)
    def forward(self,x):
        h=self.trunk(x)
        return torch.relu(self.density(h))[...,0], torch.sigmoid(self.rgb(h))

def get_rays(H,W,focal,pose):
    i,j=torch.meshgrid(torch.arange(W,device=DEVICE).float(),
                       torch.arange(H,device=DEVICE).float(),indexing="xy")
    dirs=torch.stack([(i-W*0.5)/focal,-(j-H*0.5)/focal,-torch.ones_like(i)],dim=-1)
    return pose[:3,3].expand(H,W,3), dirs@pose[:3,:3].T

def render_rays(rays_o,rays_d,model,encoding,near=NEAR,far=FAR,N=N_SAMPLES):
    t=torch.linspace(near,far,N,device=DEVICE); delta=(far-near)/N
    t=t+(torch.rand(rays_o.shape[0],N,device=DEVICE)-0.5)*delta
    pts=rays_o[:,None]+rays_d[:,None]*t[:,:,None]
    sigma,c=model(encoding(pts.reshape(-1,3)))
    sigma=sigma.reshape(rays_o.shape[0],N); c=c.reshape(rays_o.shape[0],N,3)
    deltas=torch.cat([t[:,1:]-t[:,:-1],torch.full_like(t[:,:1],1e10)],dim=-1)
    alpha=1.0-torch.exp(-sigma*deltas)
    T=torch.cumprod(torch.cat([torch.ones_like(alpha[:,:1]),1-alpha+1e-10],dim=-1),dim=-1)[:,:-1]
    return ((T*alpha)[...,None]*c).sum(dim=1)

class MyAdam:
    def __init__(self,params,lr=1e-3,b1=0.9,b2=0.999,eps=1e-8):
        self.params=[p for p in params if p.requires_grad]
        self.lr,self.b1,self.b2,self.eps=lr,b1,b2,eps
        self.m=[torch.zeros_like(p) for p in self.params]
        self.v=[torch.zeros_like(p) for p in self.params]; self.t=0
    @torch.no_grad()
    def step(self):
        self.t+=1; bc1,bc2=1-self.b1**self.t,1-self.b2**self.t
        for p,m,v in zip(self.params,self.m,self.v):
            if p.grad is None: continue
            g=p.grad
            m.mul_(self.b1).add_(g,alpha=1-self.b1)
            v.mul_(self.b2).addcmul_(g,g,value=1-self.b2)
            p.addcdiv_(m/bc1,v.div(bc2).sqrt().add_(self.eps),value=-self.lr)
    def zero_grad(self):
        for p in self.params:
            if p.grad is not None: p.grad.detach_(); p.grad.zero_()

class MySGD:
    def __init__(self,params,lr=1e-3):
        self.params=[p for p in params if p.requires_grad]; self.lr=lr
    @torch.no_grad()
    def step(self):
        for p in self.params:
            if p.grad is not None: p.add_(p.grad,alpha=-self.lr)
    def zero_grad(self):
        for p in self.params:
            if p.grad is not None: p.grad.detach_(); p.grad.zero_()

@torch.inference_mode()
def render_full(model,encoding,pose,focal,H,W,chunk=8192):
    rays_o,rays_d=get_rays(H,W,focal,pose)
    rays_o,rays_d=rays_o.reshape(-1,3),rays_d.reshape(-1,3)
    out=[render_rays(rays_o[i:i+chunk],rays_d[i:i+chunk],model,encoding)
         for i in range(0,rays_o.shape[0],chunk)]
    return torch.cat(out).reshape(H,W,3)

def evaluate(model,encoding,test_images,test_poses,focal,H,W):
    ps,ss=[],[]
    for gt_t,pose in zip(test_images,test_poses):
        pred=render_full(model,encoding,pose,focal,H,W).clamp(0,1).cpu().numpy()
        gt=gt_t.cpu().numpy(); mse=float(((pred-gt)**2).mean())
        ps.append(-10*np.log10(max(mse,1e-10)))
        ss.append(float(ssim_metric(pred,gt,channel_axis=-1,data_range=1.0)))
    return float(np.mean(ps)),float(np.mean(ss))

print("Loading Lego...",flush=True)
train_full,train_poses_cpu,native_focal=load_split(SCENE_DIR,"train")
test_full,test_poses_cpu,_=load_split(SCENE_DIR,"test",max_views=N_TEST_VIEWS)
native_W=train_full.shape[2]

results={}
for res,optname in CONFIGS:
    print(f"\n=== {res}x{res}  optimizer={optname.upper()} ===",flush=True)
    torch.cuda.empty_cache()
    train_images=downscale(train_full,res).to(DEVICE)
    test_images=downscale(test_full,res).to(DEVICE)
    train_poses=train_poses_cpu.to(DEVICE); test_poses=test_poses_cpu.to(DEVICE)
    H=W=res; focal=native_focal*res/native_W
    torch.manual_seed(SEED); np.random.seed(SEED)
    encoding=PositionalEncoding(10).to(DEVICE)
    model=TinyNeRF(encoding.out_dim).to(DEVICE)
    opt=MyAdam(list(model.parameters()),lr=LR) if optname=="adam" else MySGD(list(model.parameters()),lr=LR)
    curve=[]
    t0=time.time()
    for it in range(1,N_ITERATIONS+1):
        idx=np.random.randint(len(train_images))
        rays_o,rays_d=get_rays(H,W,focal,train_poses[idx])
        rays_o,rays_d=rays_o.reshape(-1,3),rays_d.reshape(-1,3)
        target=train_images[idx].reshape(-1,3)
        pix=torch.randint(0,H*W,(BATCH,),device=DEVICE)
        pred=render_rays(rays_o[pix],rays_d[pix],model,encoding)
        loss=((pred-target[pix])**2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        if it%EVAL_EVERY==0:
            psnr,ssim=evaluate(model,encoding,test_images,test_poses,focal,H,W)
            curve.append((it,psnr,ssim))
            print(f"  iter {it:6d}  PSNR {psnr:6.2f}  SSIM {ssim:.3f}",flush=True)
    results[(res,optname)]=curve
    del train_images,test_images,model,encoding,opt; torch.cuda.empty_cache()

ADAM_200={"best":23.08,"final":22.97}  # reused from bench_800 (identical config)
print("\n\n========== SEPARABILITY SUMMARY ==========")
# 200x200 — Adam reused from bench_800, SGD measured here
s=results[(200,"sgd")]
s_best=max(p for _,p,_ in s); s_fin=s[-1][1]
print(f"\n200x200:")
print(f"  Adam : best {ADAM_200['best']:6.2f} dB   final {ADAM_200['final']:6.2f} dB   (reused from bench_800)")
print(f"  SGD  : best {s_best:6.2f} dB   final {s_fin:6.2f} dB")
print(f"  delta (Adam - SGD): best {ADAM_200['best']-s_best:+.2f} dB   final {ADAM_200['final']-s_fin:+.2f} dB")
# 400x400 — both measured here
a=results[(400,"adam")]; s=results[(400,"sgd")]
a_best=max(p for _,p,_ in a); s_best=max(p for _,p,_ in s)
a_fin=a[-1][1]; s_fin=s[-1][1]
print(f"\n400x400:")
print(f"  Adam : best {a_best:6.2f} dB   final {a_fin:6.2f} dB")
print(f"  SGD  : best {s_best:6.2f} dB   final {s_fin:6.2f} dB")
print(f"  delta (Adam - SGD): best {a_best-s_best:+.2f} dB   final {a_fin-s_fin:+.2f} dB")
