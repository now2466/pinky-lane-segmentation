import argparse,json,hashlib,torch
from pathlib import Path
from torch.utils.data import DataLoader
from pinky_lane.dataset import LaneDataset
from pinky_lane.metrics import evaluate
from pinky_lane.checkpoints import load_model
p=argparse.ArgumentParser();p.add_argument('--architecture',choices=['pidnet','unet'],required=True);p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--dataset',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--device',default='cuda');args=p.parse_args();torch.set_num_threads(4);device=torch.device(args.device)
if args.architecture=='pidnet':
 from pinky_pidnet.model import load_checkpoint
 model,metadata=load_checkpoint(args.checkpoint,device)
else:model,_,_,metadata=load_model(args.checkpoint,str(device),ignore_top=0)
class ROI(torch.utils.data.Dataset):
 def __init__(self,data):self.data=data
 def __len__(self):return len(self.data)
 def __getitem__(self,i):
  im,l,pol,g=self.data[i];l=l.clone();l[:110]=255;return im,l,pol,g
source=LaneDataset(args.dataset,'test',5);report=evaluate(model,DataLoader(ROI(source),batch_size=8),device,5);report.update(architecture=args.architecture,split='test',sample_count=len(source),checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),manifest_sha256=hashlib.sha256((args.dataset/'manifest.json').read_bytes()).hexdigest(),evaluation_policy='argmax, no postprocessing, y>=110 AND stored label ignore/policy masks')
args.output.parent.mkdir(parents=True,exist_ok=True);assert not args.output.exists();args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n');print(json.dumps(report['fixed_foreground']),flush=True)
