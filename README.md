# DonKa Detector

A tool to map audio input to keypresses for the rhythm game Taiko no Tatsujin

## Overview

**DonKa Detector** aims to make the fun of Taiko no Tatsujin accessible anywhere, anytime. It gives people more options to play the game without needing specialized equipment.

## Features

 - **DonKa Mapping:** Map your audio inputs to left/right Don's and Ka's by following a metronome. Map left/right Don's and Ka's to native keys.
 - **Performance Options:** Similar to how you can reduce graphics options to increase FPS, you can also modify chunk sizes, buffer sizes, and delay for optimal gaming performance.

## Installation

This tool is a work in progress. For now, the only way to use this is via the command line.

### CLI

Make sure Python is installed in your machine. I recommend using your preferred Python package manager. This guide uses Anaconda.

Create and activate the environment using

    conda create -n DonkaDetector python=3.12
    conda activate DonkaDetector

Then install dependencies using

    pip install -r python_requirements.txt

## Setup

### CLI

Define audio inputs using
    
    python audio_func/record_donka.py

and follow the instructions.

Then, validate the inputs using

    python audio_func/validate_input.py

This should give you some information on expected accuracy for specific note rates (in notes/second) and conduct further fine-tuning.

Now, you can run the program using

    python main.py

You can modify paths and keybindings in *config.ini*.

#### Metric Details for Nerds

This application uses *normalize=False* and *K=1* for now. The outputs are in the format *(total # of predicted notes, DonKa/Side errors, DonKa errors)*. The third value corresponds to mistaking Dons for Kas and vice versa, and are more fatal. The second value includes a left/right mistake, which is not critical in real-time gameplay. Roughly, the accuracy is (*Donka Errors*) / (*Total # of predicted notes*). For further details and caveats on the metrics, check the risk analysis notebooks.