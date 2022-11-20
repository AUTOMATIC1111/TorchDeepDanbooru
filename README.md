This is a pure pytorch implementation of DeepDanbooru neural network: https://github.com/KichangKim/DeepDanbooru/.

There is no code to export weights from an existing model yet, so you are stuck with the only checkpoint that I will upload to releases.

Use example is in test.py: basically just load the checkpoint as usual using `model.load_state_dict()`, and use `model.tags` to get textual represntation of model's `forward` output.
test.py loads checkpoint from `model-resnet_custom_v3.pt`, image from `test.jpg`, and prints tags to stdout.

I used a modified version of onnx-pytorch librtary to write this.
The modification is related to fixing incorrect padding code for conv layers.
Because my modification only works for Conv2d and the original code works with all dimensionalities, I won't be releasing the modifications.


