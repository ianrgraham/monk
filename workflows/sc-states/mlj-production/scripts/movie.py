import ffmpeg

(
    ffmpeg
    .input('2test*.png', pattern_type='glob', framerate=2)
    .output('movie-soft-20.avi', crf=20)
    .run()
)