import json, os, sys, shutil, cv2, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from derive_v5 import derive
R = 'relay/v13-derived-149b0251-final1428/'
OUT = sys.argv[1]
idx = json.load(open('rev/index.json'))
for d in ('images', 'masks_v1', 'masks_orig'): os.makedirs(os.path.join(OUT, d), exist_ok=True)
web = []; acc = []
for e in idx:
    fid = e['id']; bgr = cv2.imread(R + e['image']); m = cv2.imread(R + e['mask'], cv2.IMREAD_UNCHANGED)
    src = np.where((m >= 1) & (m <= 4), m, 0).astype(np.uint8)
    d = derive(src, bgr)
    cv2.imwrite(os.path.join(OUT, 'masks_v1', f'{fid}.png'), d)
    shutil.copy(R + e['image'], os.path.join(OUT, 'images', f'{fid}.jpg'))
    shutil.copy(R + e['mask'], os.path.join(OUT, 'masks_orig', f'{fid}.png'))
    b = d[110:]; floor = (b == 0) | (b == 5) | (b == 255)
    unk = float((b[floor] == 255).mean()) if floor.any() else 1.0
    acc.append([(b == 5).mean(), (b == 0).mean(), (b == 255).mean()])
    web.append({'id': fid, 'image': e['image'], 'split': e['split'], 'unknown_frac': round(unk, 3),
                'flag': '빈칸' if unk > 0.15 else '', 'note': f'모름 {unk:.0%}' if unk > 0.15 else '',
                'drivable_px': int((d == 5).sum())})
json.dump(web, open(os.path.join(OUT, 'index.json'), 'w'), ensure_ascii=False, indent=1)
a = np.array(acc).mean(0)
print(len(web), 'frames  below top: D %.2f N %.2f U %.2f' % tuple(a), ' flagged', sum(w['flag'] != '' for w in web))
