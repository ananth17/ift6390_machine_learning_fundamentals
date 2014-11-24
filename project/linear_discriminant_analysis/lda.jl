# just do the example from
# http://people.revoledu.com/kardi/tutorial/LDA/Numerical%20Example.html

x = [2.95 6.63; 2.53 7.79; 3.57 5.65; 3.16 5.47; 2.58 4.46; 2.16 6.22; 3.27 3.52]
y = [1 1 1 1 2 2 2]




function num_per_class(y::Matrix{Int})
  # calculate the number of elements per class
  # for priors...
  total = 0
  all_classes = unique(y)
  
  result = Dict{Int, Int}()
  for i in all_classes
    result[i] = 0
  end
  
  for i in y
    result[i] += 1
    total+=1
  end
  
  result
end


function to_class{X}(x::Matrix{X}, y::Matrix{Int})
  # separate the examples
  result = Dict{Int, Matrix{X}}()
  
  # initialize the arrays in the dict
  classes = unique(y)
  for class in classes
    result[class] = Array(X, (0, size(x)[2]))
  end
  
  # add the elements to the corresponding classes
  for i=1:size(x)[1]
    result[y[i]] =  vcat(result[y[i]], x[i, :])
  end
  
  result
end


function per_class_avg{X}(d::Dict{Int, Matrix{X}})
  #
  result = Dict{Int, Matrix{X}}()
  for (index, mat) in d
    result[index] = mapslices(mean, mat, 1)
  end
  result
end


function per_class_cov{X}(d::Dict{Int, Matrix{X}}, num_per_class::Dict{Int, Int})
  #
  result = Dict{Int, Matrix{X}}()
  for (index, mat) in d
    result[index] = (transpose(mat)*mat) / num_per_class[index]
  end
  result
end


function correct_mean{X}(d::Dict{Int, Matrix{X}}, global_mean::Matrix{X})
  #
  result = Dict{Int, Matrix{X}}()
  for (index, mat) in d
    result[index] = deepcopy(mat)
    for i=1:size(mat)[1]
      result[index][i,:] -= global_mean
    end
  end
  result
end


function pooled_covariance{X}(covariances::Dict{Int, Matrix{X}},
                              num_per_class::Dict{Int, Int})
  # 
  total = 0
  for (_, i) in num_per_class
    total+=i
  end
  
  dim = 0
  for (_, mat) in covariances
    dim = size(mat) # just get the size of one
    break           # they are all the same...
  end
  
  # 
  result = zeros(dim)
  for (index, mat) in covariances
    normalized = mat * (num_per_class[index] / total)
    result += normalized
  end
  
  result
end


function get_priors(y)
  # naive prior
  np = num_per_class(y)
  total = 0
  for (_, i) in np
    total += i
  end
  
  result = Dict{Int, FloatingPoint}()
  for (index, num) in np
    result[index] = num/total
  end
  
  result
end


function get_probs(new_x, class_avg, pooled_cov, priors)
  #
  inverse = inv(pooled_cov)
  
  result = Dict{Int, FloatingPoint}()
  for (index, avg) in class_avg
    left = avg * inverse * transpose(new_x)
    right = -0.5 * avg*inverse*transpose(avg)
    prior = log(priors[index])
    result[index] = (left + right + prior)[1]
  end
  
  result
end

avg_all = mapslices(mean, x, 1)  # average over columns, 2 would be rows


classes = to_class(x, y)

class_avg = per_class_avg(classes)

normalized_class_avg = correct_mean(classes, avg_all)


np = num_per_class(y)

covs = per_class_cov(normalized_class_avg, np)

pooled_cov = pooled_covariance(covs, np)

priors = get_priors(y)

get_probs([2.81 5.46], class_avg, pooled_cov, priors)
