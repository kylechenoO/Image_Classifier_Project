import sys
import json
import argparse
import torch
import numpy as np
from PIL import Image

## load checkpoint
def load_checkpoint(fp, arch):
    checkpoint = torch.load(fp)
    if checkpoint['arch'] == arch:
        model = checkpoint['model']
        model.classifier = checkpoint['classifier']
        model.load_state_dict(checkpoint['state_dict'])
        model.class_to_idx = checkpoint['class_to_idx']

    return(model)

## process_image
def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    # TODO: Process a PIL image for use in a PyTorch model
    # load Image
    img = Image.open(image)

    # resize
    lenth, width = img.size
    base = 1
    if lenth > width:
        base = 256 / width

    else:
        base = 256 / lenth

    img = img.resize((int(lenth * base), int(width * base)))

    # cut image
    lenth, width = img.size
    lenth /= 2
    width /= 2
    img = img.crop(
    (
        lenth - 112,
        width - 112,
        lenth + 112,
        width + 112
    ))

    # to numpy
    np_image = np.array(img) / np.array([225., 225., 225.])

    # normalized
    normalized = (np_image - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
    normalized = normalized.transpose()
    return(normalized)

## predict
def predict(image, model, topk):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''

    # TODO: Implement the code to predict the class from an image file
    value = torch.from_numpy(process_image(image))
    test_value = torch.autograd.Variable(value.cuda())
    test_value = test_value.float()
    test_value = test_value.unsqueeze(0)
    prediction = model(test_value)
    prediction = torch.exp(prediction)
    pred = prediction.cpu()
    probs, classes = torch.topk(pred, topk)
    probs = probs.data.numpy()[0]
    idx_to_class = {val: key for key, val in model.class_to_idx.items()}
    classes = classes.data.numpy()[0]
    top_classes = [idx_to_class[each] for each in classes]

    return(probs, classes)

## main run part
if __name__ == '__main__':
    ## read par
    parser = argparse.ArgumentParser()
    parser.add_argument('image', help = 'Input Image', default = False)
    parser.add_argument('checkpoint', help = 'CheckPoint File', default = False)
    parser.add_argument('--category_names', help = 'Category Names', default = 'cat_to_name.json')
    parser.add_argument('--topk', help = 'topk', default = 5, type = int)
    parser.add_argument('--gpu', help = 'To use GPU.', action = 'store_true', default = False)
    args = parser.parse_args()

    ## precheck
    if (not args.image) or (not args.checkpoint):
        print('Please run -h to read help info')
        sys.exit(-1)

    ## load json
    with open(args.category_names, 'r') as f:
        cat_to_name = json.load(f)

    ## load checkpoint
    model = load_checkpoint(args.checkpoint, 'vgg19')

    ## predict
    probs, classes = predict(args.image, model, args.topk)
    # print(classes)
    classes = [ cat_to_name[str(c)] for c in classes ]

    ## pirnt result
    for t in zip(classes, probs):
        print(t)
