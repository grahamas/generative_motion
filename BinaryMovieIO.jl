module BinaryMovieIO

export BinaryMovieReader, BinaryMovieWriter

using FFmpegPipe
using ColorTypes

function _binarize(pixel::ColorTypes.RGB)
    pixel.r > 0.5
end
function _debinarize(b::Bool)
    if b
        RGB(1,1,1)
    else
        RGB(0,0,0)
    end
end

struct BinaryMovieReader
    video_path::String
end
struct BinaryMovieWriter
    video::Base.Process
    BinaryMovieWriter(new_path::String, fps=24) = new(openvideo(new_path, "w", r=fps))
end

function Base.write(video::BinaryMovieWriter, img::Array{Bool,2})
    writeframe(video.video, _debinarize.(img))
end

function Base.iterate(iter::BinaryMovieReader)
    iterate(iter, openvideo(iter.video_path))
end

function Base.iterate(iter::BinaryMovieReader, state::Base.Process)
    if eof(video)
        close(video)
        return nothing
    else
        return (_binarize.(readframe(video)), video)
    end
end

end
