import ffmpeg

(
    ffmpeg
    .input('dyn-hetero-movie/*0.210.png', pattern_type='glob', framerate=10)
    .output('movie-hetero-20.avi', crf=20)
    .run()
)