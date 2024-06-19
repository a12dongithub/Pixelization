import os
import cv2
import subprocess
from glob import glob


import os
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.visualizer import save_images
from util import html
from sklearn.manifold import TSNE
from sklearn.datasets import load_iris,load_digits
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def extract_frame(video_capture, frame_number, output_dir):
    video_capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    success, image = video_capture.read()
    if success:
        frame_filename = os.path.join(output_dir, "frame.png")
        cv2.imwrite(frame_filename, image)
        print(f"Extracted frame {frame_number} to {frame_filename}")
    else:
        print(f"Failed to extract frame {frame_number}")
    return success

def run_command(command):
    print(f"Running command: {command}")
    subprocess.run(command, shell=True, check=True)

def clean_directories(directories):
    for directory in directories:
        files = glob(os.path.join(directory, "*"))
        for file in files:
            os.remove(file)
    print(f"Cleaned directories: {directories}")

def collect_fake_images(visuals, collected_images):
    
        fake_image_tensor = visuals.get('fake_B')
        if fake_image_tensor is not None:
            fake_image_tensor = fake_image_tensor.squeeze(0).cpu().numpy()
            # Normalize the values to the range [0, 255]
            fake_image = ((fake_image_tensor + 1) / 2.0 * 255).astype(np.uint8)
            # Convert to HWC format (Height, Width, Channels)
            fake_image = np.transpose(fake_image, (1, 2, 0))
            collected_images.append(fake_image)
            print(f"Collected a fake image with shape {fake_image.shape}")

def create_video_from_images(images, output_video, fps=30):
    if not images:
        print("No images to create video.")
        return
    
    height, width, layers = images[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

    for image in images:
        video.write(image)
        print("Added frame to video")

    cv2.destroyAllWindows()
    video.release()
    print(f"Created video: {output_video}")

def main():
    input_video = "./tc.mp4"
    frame_dir = "./datasets/TEST_DATA/Input/"
    testA_dir = "./datasets/TEST_DATA/testA/"
    result_image_dir = "./results/YOUR_MODEL_NAME/test_160/images/"
    output_video = "output.mp4"
    model_name = "YOUR_MODEL_NAME"
    fps = 30

    video_capture = cv2.VideoCapture(input_video)
    total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    collected_images = []

    ###################
    opt = TestOptions().parse()  # get test options
    # hard-code some parameters for test
    opt.num_threads = 0   # test code only supports num_threads = 0
    opt.batch_size = 1    # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.
      # create a dataset given opt.dataset_mode and other options
    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers
    # create a website
    web_dir = os.path.join(opt.results_dir, opt.name, '{}_{}'.format(opt.phase, opt.epoch))  # define the website directory
    if opt.load_iter > 0:  # load_iter is 0 by default
        web_dir = '{:s}_iter{:d}'.format(web_dir, opt.load_iter)
    print('creating web directory', web_dir)
    webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.epoch))
    # test with eval mode. This only affects layers like batchnorm and dropout.

    ###################

    for frame_number in range(total_frames):
        print(f"Processing frame {frame_number + 1}/{total_frames}")
        success = extract_frame(video_capture, frame_number, frame_dir)
        if not success:
            print(f"Failed to extract frame {frame_number + 1}")
            break
        
        run_command("python prepare_data.py")
        dataset = create_dataset(opt)
        #run_command(f"python test.py --gpu_ids 0 --batch_size 1 --preprocess none --num_test 4 --epoch 160 --dataroot ./datasets/TEST_DATA/ --name {model_name}")
        #######

        if opt.eval:
            model.eval()
        for i, data in enumerate(dataset):
            if i >= opt.num_test:  # only apply our model to opt.num_test images.
                break
            model.set_input(data)  # unpack data from data loader
            model.test()           # run inference
            visuals = model.get_current_visuals()  # get image results
            collect_fake_images(visuals, collected_images)
                    #######
        
        clean_directories([frame_dir, testA_dir,result_image_dir])

        #if frame_number >= 10:
        #    print("Processed 10 frames, stopping for verification.")
        #    break

    video_capture.release()
    create_video_from_images(collected_images, output_video, fps)

if __name__ == "__main__":
    main()


