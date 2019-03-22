push!(LOAD_PATH, pwd())

using BinaryMovieIO
using Turing
using Flux

E(y,h,x) = - sum( (Wx * x) .* (Wy * y) .* (Wh * h) ) -  dot(bh, h) - dot(by, y)

p_y(h, x, W_yf, b_y, W_fh, W_fx) = [p_yj_is_1(h, x, W_fy[:,j], b_y[j], W_fh, W_fx) for j in 1:length(b_y)]
p_yj_is_1( h, x, W_jf, b_j, W_fh, W_fx) = σ(sum( (W_fx * x) .* (W_fh * h) .* W_jf) + b_j)
p_h(y, x, W_hf, b_h, W_fy, W_fx) = [p_hj_is_1(y, x, W_fh[:,k], b_h[k], W_fy, W_fx) for j in 1:length(b_h)]
p_yj_is_1( y, x, W_jf, b_j, W_fx, W_fy) = σ(sum( (W_fx * x) .* (W_fy * y) .* W_jf) + b_j)
# x, h, y, f
macro def_unpack(Ns...)
    N_h, N_y, N_x, N_f = map(eval, Ns)

    W_fh_start = W_fx_end + 1
    W_fh_end = W_fx_end + N_h * N_f
    b_h_start = W_fh_end + 1
    b_h_end = W_fh_end + N_h

    W_fy_start = b_h_end + 1
    W_fy_end = b_h_end + N_y * N_f
    b_y_start = W_fy_end + 1
    b_y_end = W_fy_end + N_y

    W_fx_start = 1
    W_fx_end = N_x * N_f

    quote
        function frbm_unpack(frbm_params::AbstractVector)
            W_fh = reshape(frbm_params[$W_fh_start:$W_fh_end], $N_f, $N_h)
            b_h = reshape(frbm_params[$b_h_start:$b_h_end], $N_h)

            W_fy = reshape(frbm_params[$W_fy_start:$W_fy_end], $N_f, $N_y)
            b_y = reshape(frbm_params[$b_y_start:$b_y_end], $N_y)

            W_fx = reshape(frbm_params[$W_fx_start:$W_fx_end], $N_f, $N_x)

            return W_fh, b_h, W_fy, b_y, W_fx
        end
    end
end
# Use macro to define
#def_unpack(Nh,Nx,Ny,Nf)
@def_frbm_unpack(10,10,10,10)

@model visible(y, h, x, frbm_params) = begin
    W_fh, b_h, W_fy, b_y, W_fx = frbm_unpack(frbm_params)

    p = p_y(h, x, W_fy, b_y, W_fh, W_fx)

    for i in 1:length(y)
        y[i] ~ Bernoulli(p[i])
    end
end

@model hidden(h, y, x, frbm_params) = begin
    W_fh, b_h, W_fy, b_y, W_fx = frbm_unpack(frbm_params)

    p = p_h(y, x, W_fh, b_h, W_fy, W_fx)

    for i in 1:length(h)
        h[i] ~ Bernoulli(p[i])
    end
end

function loss

struct FRBM{T}
    W_fh::Array{T,2}
    b_h::Array{T,1}
    W_fy::Array{T,2}
    b_y::Array{T,1}
    W_fx::Array{T,2}
end
function FRBM(inp::Integer, vis::Integer, hid::Integer, fac::Integer,
                initW=glorot_uniform, initb=zeros)
    return FRBM(
        param(initW(fac, hid)),
        param(initb(hid)),
        param(initW(fac, vis)),
        param(initb(vis)),
        param(initW(fac, inp))
    )
end


mutable struct ContrastiveDivergence
    eta::Float64
end
ContrastiveDivergence() = ContrastiveDivergence(0.1)

function apply!(o::Descent)
