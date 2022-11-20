from PIL import Image
import numpy as np
import torch
import tqdm

import deep_danbooru_model

model = deep_danbooru_model.DeepDanbooruModel()
model.load_state_dict(torch.load('model-resnet_custom_v3.pt'))

model.eval()
model.half()
model.cuda()

pic = Image.open("test.jpg").convert("RGB").resize((512, 512))
a = np.expand_dims(np.array(pic, dtype=np.float32), 0) / 255

with torch.no_grad(), torch.autocast("cuda"):
    x = torch.from_numpy(a).cuda()

    # first run
    y = model(x)[0].detach().cpu().numpy()

    # measure performance
    for n in tqdm.tqdm(range(10)):
        model(x)


for i, p in enumerate(y):
    if p >= 0.5:
        print(model.tags[i], p)
