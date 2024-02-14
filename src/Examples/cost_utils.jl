function smooth_max(vals...; sharpness = 1.0)
    # For improved numerical stability, we subtract the mean of the values
    c = mean(v * sharpness for v in vals)
    1 / sharpness * (c + log(sum(exp(sharpness * v - c) for v in vals)))
end

function my_norm_sqr(x)
    sum(x .^ 2)
end

function my_norm(x; regularization = 0.0)
    sqrt(my_norm_sqr(x) + regularization)
end
