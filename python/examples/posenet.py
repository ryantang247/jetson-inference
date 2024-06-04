#!/usr/bin/env python3
#
# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
#

import sys
import argparse
import math

from jetson_inference import poseNet
from jetson_utils import videoSource, videoOutput, Log, cudaFont

# parse the command line
parser = argparse.ArgumentParser(
    description="Run pose estimation DNN on a video/image stream.",
    formatter_class=argparse.RawTextHelpFormatter,
    epilog=poseNet.Usage() + videoSource.Usage() + videoOutput.Usage() + Log.Usage(),
)

parser.add_argument(
    "input", type=str, default="", nargs="?", help="URI of the input stream"
)
parser.add_argument(
    "output", type=str, default="", nargs="?", help="URI of the output stream"
)
parser.add_argument(
    "--network",
    type=str,
    default="resnet18-body",
    help="pre-trained model to load (see below for options)",
)
parser.add_argument(
    "--overlay",
    type=str,
    default="links,keypoints",
    help="pose overlay flags (e.g. --overlay=links,keypoints)\nvalid combinations are:  'links', 'keypoints', 'boxes', 'none'",
)
parser.add_argument(
    "--threshold", type=float, default=0.15, help="minimum detection threshold to use"
)


def angle_between(x1, x2, y1, y2):
    dx = x2 - x1
    dy = y2 - y1
    # Use math.atan2 for robust calculation of angle in all quadrants
    rad = math.atan2(dy, dx)

    # Convert radians to degrees and ensure the angle is within the 0 to 360 range
    angle_in_degrees = math.degrees(rad) % 360

    return angle_in_degrees


def is_sitting_leaned(pose):
    # Extract relevant keypoints

    left_hip_idx = pose.FindKeypoint("left_hip")
    right_hip_idx = pose.FindKeypoint("right_hip")
    left_ear_idx = pose.FindKeypoint("left_ear")
    right_ear_idx = pose.FindKeypoint("right_ear")
    neck_idx = pose.FindKeypoint("neck")

    if (
        left_hip_idx < 0
        or right_hip_idx < 0
        or neck_idx < 0
        or right_ear_idx < 0
        or left_ear_idx < 0
    ):
        return False  # If any keypoint is missing, cannot determine

    left_hip = pose.Keypoints[left_hip_idx]
    right_hip = pose.Keypoints[right_hip_idx]
    left_ear = pose.Keypoints[left_ear_idx]
    right_ear = pose.Keypoints[right_ear_idx]
    neck = pose.Keypoints[neck_idx]

    if left_hip and left_ear:
        slanted_angle = angle_between(left_hip.x, left_ear.x, left_hip.y, left_ear.y)
    elif right_hip and right_ear:
        slanted_angle = angle_between(
            right_hip.x, right_ear.x, right_hip.y, right_ear.y
        )
    else:
        return -1
    font = cudaFont(size=32)
    font.OverlayText(
        img,
        img.width,
        img.height,
        slanted_angle,
        5,
        50,
        font.White,
        font.Gray40,
    )
    if slanted_angle < 70:
        return 2
    elif slanted_angle > 110:
        return 1
    elif 70 < slanted_angle < 110:
        return 0
    else:
        return -1


def is_sitting_slanted(pose):
    # Extract relevant keypoints
    left_shoulder_idx = pose.FindKeypoint("left_shoulder")
    right_shoulder_idx = pose.FindKeypoint("right_shoulder")
    left_hip_idx = pose.FindKeypoint("left_hip")
    right_hip_idx = pose.FindKeypoint("right_hip")
    neck_idx = pose.FindKeypoint("neck")

    if (
        left_shoulder_idx < 0
        or right_shoulder_idx < 0
        or left_hip_idx < 0
        or right_hip_idx < 0
        or neck_idx < 0
    ):
        return False  # If any keypoint is missing, cannot determine

    left_shoulder = pose.Keypoints[left_shoulder_idx]
    right_shoulder = pose.Keypoints[right_shoulder_idx]
    left_hip = pose.Keypoints[left_hip_idx]
    right_hip = pose.Keypoints[right_hip_idx]
    neck = pose.Keypoints[neck_idx]

    # Calculate vertical differences (slant)
    shoulder_slant = abs(left_shoulder.y - right_shoulder.y)
    hip_slant = abs(left_hip.y - right_hip.y)

    # Define a threshold for slant detection
    slant_threshold = (
        10  # This threshold may need to be adjusted based on the scale of your images
    )

    is_shoulder_slanted = shoulder_slant > slant_threshold
    is_hip_slanted = hip_slant > slant_threshold

    # Check vertical alignment of neck, shoulders, and hips
    vertical_alignment = (
        abs(neck.x - ((left_shoulder.x + right_shoulder.x) / 2)) < slant_threshold
        and abs(
            (left_shoulder.x + right_shoulder.x) / 2 - (left_hip.x + right_hip.x) / 2
        )
        < slant_threshold
    )

    # Determine if person is slanted
    is_slanted = is_shoulder_slanted or is_hip_slanted or not vertical_alignment

    return is_slanted


try:
    args = parser.parse_known_args()[0]
except:
    print("")
    parser.print_help()
    sys.exit(0)

# load the pose estimation model
net = poseNet(args.network, sys.argv, args.threshold)

# create video sources & outputs
input = videoSource(args.input, argv=sys.argv)
output = videoOutput(args.output, argv=sys.argv)

# process frames until EOS or the user exits
while True:
    # capture the next image
    img = input.Capture()

    if img is None:  # timeout
        continue

    # perform pose estimation (with overlay)
    poses = net.Process(img, overlay=args.overlay)

    # print the pose results
    print("detected {:d} objects in image".format(len(poses)))
    y_position = 0
    for pose in poses:
        print(pose)
        print(pose.Keypoints)
        isslanted = is_sitting_slanted(pose)
        isleaned = is_sitting_leaned(pose)
        font = cudaFont(size=32)

        if isslanted:
            font.OverlayText(
                img,
                img.width,
                img.height,
                "Bad sitting posture!!!",
                5,
                5 + y_position,
                font.White,
                font.Gray40,
            )

        else:
            font.OverlayText(
                img,
                img.width,
                img.height,
                "Good sitting posture XD",
                5,
                5 + y_position,
                font.White,
                font.Gray40,
            )

        if isleaned == 2:
            font.OverlayText(
                img,
                img.width,
                img.height,
                "Leaning forwards!!",
                5,
                30 + y_position,
                font.White,
                font.Gray40,
            )
        elif isleaned == 1:
            font.OverlayText(
                img,
                img.width,
                img.height,
                "Leaning backwards!!",
                5,
                30,
                font.White,
                font.Gray40,
            )
        elif isleaned == 0:
            font.OverlayText(
                img,
                img.width,
                img.height,
                "Not leaning!",
                5,
                30 + y_position,
                font.White,
                font.Gray40,
            )
        else:
            font.OverlayText(
                img,
                img.width,
                img.height,
                "Cannot detect",
                5,
                30 + y_position,
                font.White,
                font.Gray40,
            )

        y_position += 5
        print("Links", pose.Links)

    # render the image
    output.Render(img)

    # update the title bar
    output.SetStatus(
        "{:s} | Network {:.0f} FPS".format(args.network, net.GetNetworkFPS())
    )

    # print out performance info
    net.PrintProfilerTimes()

    # exit on input/output EOS
    if not input.IsStreaming() or not output.IsStreaming():
        break
