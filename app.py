# Copyright (c) OpenMMLab. All rights reserved.
import os
import sys
from functools import partial
from typing import Optional

os.system('python -m mim install "mmcv>=2.0.0"')
os.system('python -m mim install mmengine')
os.system('python -m mim install "mmdet>=3.0.0"')
os.system('python -m mim install -e .')

project_path = os.path.join(os.getcwd(), 'projects', 'just_dance')
os.environ['PATH'] = f"{os.environ['PATH']}:{project_path}"
os.environ[
    'PYTHONPATH'] = f"{os.environ.get('PYTHONPATH', '.')}:{project_path}"
sys.path.append(project_path)

import gradio as gr
from just_dance import VideoProcessor


def process_video(
    teacher_video: Optional[str] = None,
    student_video: Optional[str] = None,
):
    print(teacher_video)
    print(student_video)

    video_processor = VideoProcessor()
    if student_video is None and teacher_video is not None:
        # Pre-process the teacher video when users record the student video
        # using a webcam. This allows users to view the teacher video and
        # follow the dance moves while recording the student video.
        _ = video_processor.get_keypoints_from_video(teacher_video)
        return teacher_video
    elif teacher_video is None and student_video is not None:
        _ = video_processor.get_keypoints_from_video(student_video)
        return student_video
    elif teacher_video is None and student_video is None:
        return None

    return video_processor.run(teacher_video, student_video)


with gr.Blocks() as demo:
    with gr.Tab('Upload-Video'):
        with gr.Row():
            with gr.Column():
                gr.Markdown('Student Video')
                student_video = gr.Video(type='mp4')
                gr.Examples(['projects/just_dance/resources/tom.mp4'],
                            student_video)
            with gr.Column():
                gr.Markdown('Teacher Video')
                teacher_video = gr.Video(type='mp4')
                gr.Examples(
                    ['projects/just_dance/resources/idol_producer.mp4'],
                    teacher_video)

        button = gr.Button('Grading', variant='primary')
        gr.Markdown('## Display')
        out_video = gr.Video()

        button.click(
            partial(process_video), [teacher_video, student_video], out_video)

    with gr.Tab('Webcam-Video'):
        with gr.Row():
            with gr.Column():
                gr.Markdown('Student Video')
                student_video = gr.Video(source='webcam', type='mp4')
            with gr.Column():
                gr.Markdown('Teacher Video')
                teacher_video = gr.Video(type='mp4')
                gr.Examples(
                    ['projects/just_dance/resources/idol_producer.mp4'],
                    teacher_video)
                button_upload = gr.Button('Upload', variant='primary')

        button = gr.Button('Grading', variant='primary')
        gr.Markdown('## Display')
        out_video = gr.Video()

        button_upload.click(
            partial(process_video), [teacher_video, student_video], out_video)
        button.click(
            partial(process_video), [teacher_video, student_video], out_video)

gr.close_all()
demo.queue()
demo.launch()
