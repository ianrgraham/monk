import ffmpeg

(
    ffmpeg
    .input('dyn-hetero-movie/*0.840.png', pattern_type='glob', framerate=10)
    .output('movie-liq-20.avi')
    .run()
)