### A Pluto.jl notebook ###
# v0.19.27

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
end

# ╔═╡ 120a282a-91c1-11ec-346f-25d56e50d38c
begin
	using Distributions,Random, StatsBase, Clustering, LinearAlgebra
	using PlutoUI
	using StatsPlots
	# using PlutoTeachingTools
end

# ╔═╡ 06d45497-b465-4370-8411-9651e33e70e6
begin
	# using LinearAlgebra
	# using PlutoUI
	using PlutoTeachingTools
	using LaTeXStrings
	using Latexify
	# using Random
	using Statistics
	using LogExpFunctions
	using HypertextLiteral
	using Plots; default(fontfamily="Computer Modern", framestyle=:box) # LaTex-style
	
end

# ╔═╡ 48bf8355-1bfe-41a1-97fc-5e2d0470a6c3
begin
	using Logging
	Logging.disable_logging(Logging.Info); # or e.g. Logging.Info
end;

# ╔═╡ 646dd3d8-6092-4435-aee9-01fa6a281bdc
ChooseDisplayMode()

# ╔═╡ 6f051fad-2c4b-4a9e-9361-e9b62ba189c5
TableOfContents()

# ╔═╡ be9bcfcb-7ec7-4851-bd3f-24d4c29462fe
md"""

# CS5914 Machine Learning Algorithms


#### Unsupervised learning 
###### Clustering 1
\

$(Resource("https://www.st-andrews.ac.uk/assets/university/brand/logos/standard-vertical-black.png", :width=>130, :align=>"right"))

Lei Fang(@lf28 $(Resource("https://raw.githubusercontent.com/edent/SuperTinyIcons/bed6907f8e4f5cb5bb21299b9070f4d7c51098c0/images/svg/github.svg", :width=>10)))

*School of Computer Science*

*University of St Andrews, UK*

"""

# ╔═╡ e61193e8-3f91-455b-9e33-101de4cfd4a3
md"""

## Topics to cover
	
"""

# ╔═╡ 234a8b66-d84a-4f6f-aad2-60db238c1850
aside((md"""$(@bind next1 Button("next")) 
$(@bind init1 Button("init"))
	"""))

# ╔═╡ edf96a48-1e6d-4866-85ba-7e1afd19ca61
begin
	init1
	next_idx = [0];
end;

# ╔═╡ 203b859f-f6ea-4ff8-be1b-dc68a44c10d1
begin
	next1
	topics = ["Un-supervised learning - clustering problem", "Algorithm: K means algorithm", "Algorithm: Gaussian mixture models"]
	@htl "<ul>$([@htl("""<li>$b</li><br>""") for b in topics[1:min(next_idx[1], length(topics))]])</ul>"
end

# ╔═╡ 76d4b64b-471f-4688-9da2-f896d9d0d05c
let
	next1
	next_idx[1] += 1
end;

# ╔═╡ e136edb5-3e98-4355-83e2-55761eb8b15c
md"""
## Why probabilistic approach

A key message I hope to convey here
!!! correct ""
	**Probabilistic models** unify most *interesting* machine learning models 
    * supervised learning
	* and also **unsupervised learning**

In other words: **from probabilistic models' eyes, they are the same**


**Machine learning** are just **probabilistic inferences**:

$$P(y|x)$$

* assume different $P(\cdot)$ and plug in different $x$ and $y$ for different problems/situations
  * e.g. regression: $P$ is Gaussian;
  * and classification: $P$ is Bernoulli or Multinoulli 
* we will come back to this key message at the end of next lecture

""";

# ╔═╡ f8b54f24-c144-4ac5-8083-706436a368e9
md"""
# Clustering problem
"""

# ╔═╡ abd46485-03c9-4077-8ace-34f9b897bb04
md"""

## Supervised learning: classification

> **Classification**: input data ${X}=\{\mathbf{x}^{(1)}, \mathbf{x}^{(2)}, \ldots, \mathbf{x}^{(n)}\}$ and their targets $Y=\{y^{(1)}, y^{(2)}, \ldots, y^{(n)}\}$

  - ``\mathbf{x}^{(i)} \in \mathbb{R}^d``: ``d`` is the dimension of ``\mathbf{x}``
  - ``y^{(i)} \in \{1,2,\ldots,K\}`` are also known

"""

# ╔═╡ 3ea57a8e-8d15-4f41-acfb-e3bd1d65e585
begin
	Random.seed!(123)
	K₁ =3
	n_each = 100
	# D₁ = zeros(n₁, 2)
	# 200 per cluster
	truezs₁ = repeat(1:K₁; inner=n_each)
	trueμs₁ = zeros(2, K₁)
	trueμs₁[:,1] = [-3.0, 2.0]
	trueμs₁[:,2] = [3.0, 2.0]
	trueμs₁[:,3] = [0., -2]
	data₁ = trueμs₁[:,1]' .+ randn(n_each, 2)
	data₁ = vcat(data₁, trueμs₁[:,2]' .+ randn(n_each, 2))
	data₁ = vcat(data₁, trueμs₁[:,3]' .+ randn(n_each, 2))
	plt₁ = plot(ratio=1, framestyle=:origin)
	for k in 1:K₁
		scatter!(data₁[truezs₁ .== k,1], data₁[truezs₁ .==k,2], label="Class"*string(k)) 
	end
	title!(plt₁, "Supervised learning: classification")
end

# ╔═╡ 08c17e06-b9f4-40de-acb1-5481fa53e5e8
md"""
## Unsupervised learning: clustering
"""

# ╔═╡ cc38bdc2-4c1b-4a08-b8e6-1504a11924a5
md"""

> **Clustering**: input features ${X}=\{\mathbf{x}^{(1)}, \mathbf{x}^{(2)}, \ldots, \mathbf{x}^{(n)}\}$ only; **without** targets



* intuitively: **clustering** is to **color**/**group** ``\mathbf{x}^{(i)}`` into $K$ groups

* *unsupervised*:  un--labelled data
"""

# ╔═╡ 77f0d45a-c658-4143-ad14-3a3a20ce2b4b
let
	plt = plot(ratio=1, framestyle=:origin, title="Clustering data")
	# for k in 1:K₁
	scatter!(data₁[:,1], data₁[:,2], label="data") 
	# end

end

# ╔═╡ eb1c7d49-fffc-44ab-9d1b-026d7a39fc15
md"""
## Unsupervised learning: clustering
"""

# ╔═╡ 0ee59618-d28c-4929-bb7c-cd590068c7b7
md"""

> **Clustering**: input features ${X}=\{\mathbf{x}^{(1)}, \mathbf{x}^{(2)}, \ldots, \mathbf{x}^{(n)}\}$ only; **without** targets



* intuitively: **clustering** is to **color**/**group** ``\mathbf{x}^{(i)}`` into $K$ groups

* *unsupervised*:  un--labelled data
"""

# ╔═╡ 4c2cf2db-0042-4074-a740-c6458c552ebe
html"<center><img src='https://miro.medium.com/v2/resize:fit:1400/format:webp/1*rw8IUza1dbffBhiA4i0GNQ.png' width = '500' /></center>"	

# ╔═╡ c63b4e0d-9f64-4248-ac7b-75ceeedddf0c
md"""

# K-means clustering
"""

# ╔═╡ 38e89872-c1ff-4d50-ab99-1ed063f5118b
md"""

## K-means 


**K-means** is a very simple and intuitive clustering algorithm


\


---

**Initialisation**: random guess $K$ centroids $\{\boldsymbol{\mu}_k\}$

**Repeat** the following two steps until converge:

* **assignment step**: assign each $\mathbf{x}^{(i)}$ to the closest centroid $\boldsymbol{\mu}_k$


* **update step**: update $\boldsymbol{\mu}_k$ by averaging the assigned samples
---

"""

# ╔═╡ 2831de52-9f49-494e-8b2b-4cee810cee06
md"""

## K-means (more details)


A very simple and intuitive clustering algorithm

**Initialisation**: random guess $K$ centroids $\{\boldsymbol{\mu}_k\}$

**Repeat** the following two steps until converge:

* **assignment step**: assign each $\mathbf{x}^{(i)}$ to the closest centroid $\boldsymbol{\mu}_k$; for $i= 1\ldots n,\; k= 1\ldots K$:

  $$\large {z}^{(i)} \leftarrow \arg\min_{k} \boxed{d(\mathbf{x}^{(i)}, \boldsymbol{\mu}_k)}_{\text{Euclidean distance}}$$ 
  * where $$d(\mathbf{x}^{(i)}, \boldsymbol{\mu}_k)=\|\mathbf{x}^{(i)} - \boldsymbol{\mu}_k\|^2 = \sum_{j=1}^d (x^i_j - \mu_{kj})^2$$
\

* **update step**: update $\boldsymbol{\mu}_k$ by taking the average

  $$\boldsymbol{\mu}_{k} \leftarrow  \frac{\sum_{i=1}^n I({z}^{(i)}=k)\cdot \mathbf{x}^{(i)}}{\sum_{i=1}^n I({z}^{(i)}=k)}\;$$
  - where ``I`` is the indicator function: return ``1`` if true; ``0`` otherwise  



## Convergence check

We need to do **convergence check** to decide when to stop

$$\text{loss} = \sum_{i=1}^n \|\mathbf{x}^{(i)} - \boldsymbol{\mu}_{z_i}\|^2 = \sum_{i=1}^n\sum_{k=1}^K I(z^{(i)}=k) \cdot \|\mathbf{x}^{(i)} - \boldsymbol\mu_{k}\|^2$$

* the total distances between $\mathbf{x}^{(i)}$ and its assigned cluster centre
* the loss will only decrease over the iterations
  * can be used to test/debug your implementation
"""

# ╔═╡ 7c9b404e-ef6c-4ed1-910e-2f409e579309
md"""

## K-means demon (initialisation)
"""

# ╔═╡ 434f1bf9-b6d2-4029-9ffa-36241c6ce23f
md"""
Initialisation: randomly guess ``K=3`` centres
"""

# ╔═╡ b81503f3-1caf-498a-8dc9-639ea5a5d569
begin
	gr()
	# random initialisation step: randn: zero mean Gaussian
	Random.seed!(123)
	ms = randn(2,K₁)
	plt_demo = plot(data₁[:,1], data₁[:,2], ratio=1, st=:scatter, label= "data",title="K means initialisation; Iteration=0", ms =3, alpha=0.5, framestyle=:origin)
	for k in 1:K₁
		scatter!([ms[1,k]], [ms[2,k]], label="μ"*string(k), markerstrokewidth =3, markershape = :star4, color=k, markersize = 10)
	end
	plt_demo
end

# ╔═╡ 63ce270b-c5ed-45c2-ae96-2146d0487b87
md"""

## K-means demon (step through)
"""

# ╔═╡ 99a717ee-1b29-42c3-9ef7-80201e715e79
aside((md"""$(@bind next2 Button("next")) 
$(@bind init2 Button("init"))
	"""))

# ╔═╡ f6bb31fd-d30c-4e59-86bb-ebac1163609f
begin
	init2
	next_idx2 = [0];
end;

# ╔═╡ 29b0734a-41d4-43e0-938a-3ced8e70b85b
let
	next2
	next_idx2[1] += 1
end;

# ╔═╡ 61fb0322-07f6-4e59-a5d3-dfdd7399d558
md"""
##

The animation:
"""

# ╔═╡ e1ccac4e-1cba-43d2-b0e0-e8c650a79d9a
md"""
## A demonstration of K-means (cont.)

"""

# ╔═╡ 8efbba52-7a34-4c17-bca9-52a417f69d56
md"""

## An alternative K-means formula

We can swap the order of **assignment** and **update** steps

* initialise by randomly guessing on the assignments $z^{(i)}$
* will usually converge to the same results


## An alternative K-means formula

We can swap the order of **assignment** and **update** steps

* initialise by randomly guessing on the assignments $z^{(i)}$
* will usually converge to the same results

Specifically, the algorithm is 

---
**Initialisation**: random guess of $z^{(i)}$ for $i=1,\ldots,n$

$$z^{(i)} \leftarrow \text{rand}(1,\ldots,K)$$
**Repeat** until convergence:

  - **update step**: update ${\mu}_k$ by averaging the assigned samples
$$\boldsymbol{\mu}_{k} \leftarrow  \frac{\sum_{i=1}^n I({z}^{(i)}=k)\cdot \mathbf{x}^{(i)}}{\sum_{i=1}^n I({z}^{(i)}=k)}\;$$
  * **assignment step**: assign each $\mathbf{x}^{(i)}$ to the closest centroid $\boldsymbol{\mu}_k$: for $i= 1\ldots n,\; k= 1\ldots K$:
    

$${z}^{(i)} \leftarrow \arg\min_{k} \|\mathbf{x}^{(i)} - \boldsymbol{\mu}_k\|^2$$
---
"""

# ╔═╡ 3873ef48-34c1-479a-b6f8-2cd05bcdc8bb
md"""

## A quick demon of alternative K-means

"""

# ╔═╡ 5276ff0e-bc4f-4b24-b98c-8f9e7c0d018d
# begin
# 	gr()
# 	pltsₖₘ2 = []
# 	zs0 = rand(1:K₁, size(data₁)[1])
# 	l_ = Inf
# 	anim = @animate for iter in 1:6
# 		p = plot_clusters(data₁, zs0, K₁, l_, iter)
# 		ms0 = update_step(data₁, zs0, K₁)
# 		for k in 1:K₁
# 			scatter!([ms0[1,k]], [ms0[2,k]], markershape = :star4, markersize = 10, c=k, markerstrokewidth =3, color=k, labels= "μ"*string(k))
# 			plot!(-6:0.1:6, -5:0.1:5, (x,y) -> sum(([x, y] - ms0[:, k]).^2), st=:contour, c=k, colorbar = false, levels=[5], lw=2)
# 		end
# 		ls, zs0 = assignment_step(data₁, ms0)
# 		l_ = sum(ls)
# 		push!(pltsₖₘ2, p)
# 	end
# end;

# ╔═╡ dd461997-cf3c-4124-8503-11f5b5a9e530
aside((md"""$(@bind next3 Button("next")) 
$(@bind init3 Button("init"))
	"""))

# ╔═╡ d2365f95-9b61-41fa-83d8-9bf4a8b5ee92
begin
	init3
	next_idx3 = [1];
end;

# ╔═╡ b5dc223b-c5b5-4eda-889e-c39171c44499
let
	next3
	next_idx3[1] += 1
end;

# ╔═╡ a391d4a3-9fe3-4ccf-9a62-c2cb19ea8813
# md"""
# ##

# For convenience, I have created a method `kmeansDemoGif` to produce gifs for us
# (check the code in Appendix)

# `kmeansDemoGif(data, K, iterations; init_step="a" or "u", add_contour=true or false)`

# * 2-d input only
# * produces three types of gifs (more example next)
# """

# ╔═╡ d8e832e1-bc53-400d-a5a9-7cda2c5b8799
md"""

## *Implementation 
"""

# ╔═╡ a414e554-3a8c-472d-af82-07c2f0843627
begin
	function assignment_step(D, μs)
		_, K = size(μs)
		distances = hcat([sum((D .- μs[:,k]').^2, dims=2) for k in 1:K]...)
		min_dis, zs_ = findmin(distances, dims=2)
		# zs_ is a cartesian tuple; retrieve the min k for each obs.
		zs = [c[2] for c in zs_][:]
		return min_dis[:], zs
	end

	function update_step(D, zs, K)
		_, d = size(D)
		μs = zeros(d,K)
		# update
		for k in 1:K
			μₖ = mean(D[zs.==k,:], dims=1)[:]
			μs[:,k] = μₖ
		end
		return μs
	end
end;

# ╔═╡ 33cc44b4-bd32-4112-bf33-6807ae53818c
function kmeans(D, K=3; tol= 1e-6, maxIters= 100, seed= 123)
	Random.seed!(seed)
	# initialise
	n, d = size(D)
	zs = rand(1:K, n)
	μs = D[rand(1:n, K),:]'
	loss = zeros(maxIters)
	i = 1
	while i <= maxIters
		# assigment
		min_dis, zs = assignment_step(D, μs)
		# update
		μs = update_step(D, zs, K)
		
		loss[i] = sum(min_dis)

		if i > 1 && abs(loss[i]-loss[i-1]) < tol
			i = i + 1
			break;
		end
		i = i + 1
	end
	return loss[1:i-1], zs, μs
end;

# ╔═╡ 00d85bc5-c320-41dc-8a85-4ad409b058a8
loss_km₁, zs_km₁, μs_km₁ = kmeans(data₁, K₁);

# ╔═╡ ceaedbd8-6c88-407b-84f6-aedc0a59a22d
plot(loss_km₁, label="loss", xlab="Iterations", ylab="Loss", title="Loss trajectory of K-means", framestyle=:box)

# ╔═╡ 8c803aa8-de06-4194-bb55-67af083cfd74
md"""

## Recap: distance metrics
"""

# ╔═╡ bba94ae7-a6a8-4958-bcf5-9380864a9147
md"""

**Distance metric**: a distance between $\mathbf{x}$ and $\boldsymbol\mu$ is:

$\Large d_{\boldsymbol{\Sigma}}^2 = {(\mathbf{x} - \boldsymbol{\mu})^\top \boldsymbol\Sigma^{-1}(\mathbf{x}-\boldsymbol{\mu})}$

"""

# ╔═╡ 4dc678ce-8f35-44c7-b58c-628345350b49
md"""

**Euclidean distance** is an example
\
\


$$\large\begin{align}d^2(\mathbf{x}, \boldsymbol{\mu}) =\sum_{j=1}^d (x_j - \mu_{j})^2
&= \underbrace{(\mathbf{x} - \boldsymbol\mu)^\top}_{\text{row vector}} \underbrace{(\mathbf{x} - \boldsymbol\mu)}_{\text{column vector}}\\ 
&= \boxed{(\mathbf{x} - \boldsymbol\mu)^\top  \mathbf{I}^{-1}  (\mathbf{x} - \boldsymbol\mu)}\end{align}$$

* *i.e.* ``\boldsymbol{\Sigma} =\mathbf{I}``

"""

# ╔═╡ 9f18a392-8edb-4f4e-a468-6d1d735d05bf
md"""

## Examples of distance metrics
"""

# ╔═╡ 40d1d8ec-3875-49c8-9425-f5cb6765603b
md"""

**Distance metric**: a distance between $\mathbf{x}$ and $\boldsymbol\mu$ is:

$\Large d_{\boldsymbol{\Sigma}}^2 = {(\mathbf{x} - \boldsymbol{\mu})^\top \boldsymbol\Sigma^{-1}(\mathbf{x}-\boldsymbol{\mu})}$

"""

# ╔═╡ 542b0f6f-befa-4dfe-acb7-b7aa6ab9f56c
md"""

## Decision boundary of K-means

K-means imposes linear **decision boundaries**

$$\large\mathbf{x} \in \mathbb{R}^d \;\; \text{s.t.}\; \|\mathbf{x} - \boldsymbol{\mu}_i\|^2 = \|\mathbf{x} - \boldsymbol{\mu}_j\|^2$$

* the boundary: $\mathbf{x}\in \mathbb{R}^d$ with equal distances to $\boldsymbol\mu_i$ and $\boldsymbol\mu_j$

"""

# ╔═╡ 05de404f-d2d7-4feb-ba70-a25d8eba71d4
md"""

## Limitations of K-means


K-means is very similar LDA
* *i.e.* ``\boldsymbol\Sigma_k =\mathbf{I}``
* they share the same limitations

## Limitations of K-means


K-means is very similar LDA
* *i.e.* ``\boldsymbol\Sigma_k =\mathbf{I}``
* they share the same limitations

How do you think K-means would fare on **this dataset** ?

* Euclidean distance (circular distance contour) do not make sense for cluster 3
* ellipse seems better
"""

# ╔═╡ b44fce15-1056-4f42-9ec7-7a70092c355b
md"""

## K-means result
"""

# ╔═╡ adb47976-1d3c-42d9-941c-8a791866eb97
md"""

## K-means result



**K-means** fails miserably


* it fails at discovering the hidden data structure 
"""

# ╔═╡ 5f9ad998-c410-4358-925b-66e5d3b2f9e9
begin
	Random.seed!(123)
	K₂ = 3
	trueμs₂ = zeros(2,K₂)
	trueΣs₂ = zeros(2,2,K₂)
	trueμs₂[:,1], trueΣs₂[:,:,1] = [-2.0, 0.0], 0.5 * Matrix(1.0I, 2,2)
	trueμs₂[:,2], trueΣs₂[:,:,2] = [2.0, 0.0], 0.5 * Matrix(1.0I, 2,2)
	trueμs₂[:,3], trueΣs₂[:,:,3] = [0., 0],  Matrix([0.5 0; 0 2])
	trueπs₂ = [0.2, 0.2, 0.6]
	truemvns₂ = [MvNormal(trueμs₂[:,k], trueΣs₂[:,:,k]) for k in 1:K₂]
	n₂= 500
	truezs₂ = rand(Categorical(trueπs₂), n₂)
	data₂= vcat([rand(truemvns₂[z])' for z in truezs₂]...)
	# data₂, truezs₂ = sampleMixGaussian(n₂, truemvns₂, trueπs₂)
end;

# ╔═╡ c8cb2535-7b65-4d86-9405-71880fdb906e
_, zskm₂, ms₂ = kmeans(data₂, K₂; seed= 442);

# ╔═╡ ca7f12d8-f823-448b-ab1f-0e28c85a3f7f
md"""

## Limitations of K-means (cont.)


"""

# ╔═╡ 8d2778fd-edfe-4a02-92f9-2b9a67f2a2c0
md"""

## K-means results

"""

# ╔═╡ 8107c1d9-58c4-41f0-8870-bfd7084e42b5
md"""

K-means fails again (and even worse for this case)


* K-means in general **cannot** handle overlapping dataset
"""

# ╔═╡ d9eaf592-1522-4f7e-a7db-cf503a152eb9
md"""

## 


### Question: can we find non-trivial structures from data unsupervisedly?

#### _Yes!_ EM algorithm
"""

# ╔═╡ aabdfd01-c9b8-4c45-83c3-29881581e6e3
# begin
# 	Random.seed!(4321)
# 	K₄ = 2
# 	trueπs₄ = [0.6, 0.4]
# 	trueμs₄ = [[-2, -2] [0.5, 0.5] ]
# 	trueΣs₄ = zeros(2, 2, K₄)
# 	trueΣs₄[:,:, 1] = [1 0.9; 0.9 1]
# 	trueΣs₄[:,:, 2] = [0.5 0.; 0. 0.5]
# 	truemvns₄ = [MvNormal(trueμs₄[:,k], trueΣs₄[:,:,k]) for k in 1:K₄]
# 	data₄, truezs₄ = sampleMixGaussian(100, truemvns₄, trueπs₄)
# 	xs₄ = (minimum(data₄[:,1])-1):0.1: (maximum(data₄[:,1])+1)
# 	ys₄ = (minimum(data₄[:,2])-1):0.1: (maximum(data₄[:,2])+1)

# 	dataset4 = data_set(data₄, truezs₄, truemvns₄, trueπs₄)
# end;

# ╔═╡ bbd25bdf-e82d-4f65-bfad-7d8e8e9cca18
struct data_set
	X
	labels
	mvns
	πs
end

# ╔═╡ 4cc1f89c-6e1d-4457-9e84-fd5bdd0b575a
Σs = [Matrix(1.0I, 2,2), [2 0; 0 0.5],  [.5 0; 0 2] , [1 0.9; 0.9 1], [1 -0.9; -0.9 1]];

# ╔═╡ ba06623e-a9e6-4f39-8c4d-d8f86c765a8c
plts_mvns=let
	Random.seed!(123)
	nobs= 250
	plts = []

	for Σ in Σs
		mvn = MvNormal(zeros(2), Σ)
		# data = rand(mvn, nobs)
	  	# scatter(data[1,:], data[2,:])
		plt = plot(-3:0.1:3, -3:0.1:3, (x, y) -> pdf(mvn, [x,y]), st=:contour, c=:jet, clabels=false, ratio=1, lw=2, levels=5, colorbar=false, framestyle=:origin)

		scatter!([0], [0], markershape=:star, color=:red, label=L"\mu")
		# plot!(-3:0.1:3, -3:0.1:3, (x, y) -> pdf(mvn, [x,y]), st=:contour, c=:jet, ratio=1, lw=3, levels=5, colorbar=false, framestyle=:origin)
		# scatter!(data[1,:], data[2,:], c=1, alpha=0.5, ms=2, label="")
		push!(plts, plt)
	end

	# color=:turbo, clabels=true,
	plts
end;

# ╔═╡ c4c51b3c-c0d3-44d7-b5e8-1a65c3e737cc
ThreeColumn(
md"""

$$\mathbf \Sigma=$$
$(latexify_md(Σs[1]))

$(plot(plts_mvns[1], size=(220,220)))
	


"""	,
md"""

$$\mathbf \Sigma=$$
$(latexify_md(Σs[2]))

$(plot(plts_mvns[2], size=(220,220)))
	



"""
	,


md"""

$$\mathbf \Sigma=$$
$(latexify_md(Σs[4]))


$(plot(plts_mvns[4], size=(220,220)))
	


"""
)

# ╔═╡ a0465ae8-c843-4fc0-abaf-0497ada26652
md"""

## Appendix

Utility functions
"""

# ╔═╡ dafd1a68-715b-4f06-a4f2-287c123761f8
begin
	function sampleMixGaussian(n, mvns, πs)
		d = size(mvns[1].Σ)[1]
		samples = zeros(n, d)
		# sample from the multinoulli distribution of cⁱ
		cs = rand(Categorical(πs), n)
		for i in 1:n
			samples[i,:] = rand(mvns[cs[i]])
		end
		return samples, cs
	end
end

# ╔═╡ e0cfcb9b-794b-4731-abf7-5435f67ced42
begin
	Random.seed!(123)
	K₃ = 3
	trueπs₃ = [0.25, 0.5, 0.25]
	trueμs₃ = [[1, 1] [0.0, 0] [-1, -1]]
	trueΣs₃ = zeros(2,2,K₃)
	trueΣs₃ .= [1 -0.9; -0.9 1]
	trueΣs₃[:,:,2] = [1 0.9; 0.9 1]
	truemvns₃ = [MvNormal(trueμs₃[:,k], trueΣs₃[:,:,k]) for k in 1:K₃]
	n₃ = 200* K₃
	data₃, truezs₃ = sampleMixGaussian(200, truemvns₃, trueπs₃)
	data₃test, truezs₃test = sampleMixGaussian(100, truemvns₃, trueπs₃)
	xs₃ = (minimum(data₃[:,1])-1):0.1: (maximum(data₃[:,1])+1)
	ys₃ = (minimum(data₃[:,2])-1):0.1: (maximum(data₃[:,2])+1)
	# dataset3
	dataset3 = data_set(data₃, truezs₃, truemvns₃, trueπs₃)
end;

# ╔═╡ c7fd532d-d72a-439a-9e71-e85392c66f8c
_, zskm₃, ms₃ = kmeans(data₃, K₃) ;

# ╔═╡ 76859d4c-f3e2-4576-b4d6-b637e9c99877
function QDA_fit(data, labels)
	n, d = size(data)
	# sse = zeros(d, d)
	K = length(unique(labels))
	μs = zeros(d, K)
	Σs = zeros(d,d,K)
	ns = zeros(Int, K)
	for k in (unique(labels)|>sort)
		ns[k] = sum(labels .==k)
		datak = data[labels .== k, :]
		μs[:, k] = μk = mean(datak, dims=1)[:]
		error = (datak .- μk')
		Σs[:,:,k] = error'*error/ns[k]
	end
	μs, Σs, ns/n
end

# ╔═╡ 620789b7-59bc-4e17-bcfb-728a329eed0f
qdform(x, S) = dot(x, S, x)

# ╔═╡ 8d0c6fdc-4717-4203-b933-4b37fe60d512
function logLikMixGuassian(x, mvns, πs, logLik=true) 
	l = logsumexp(log.(πs) .+ [logpdf(mvn, x) for mvn in mvns])
	logLik ? l : exp(l)
end

# ╔═╡ d66e373d-8443-4810-9332-305d9781a21a
md"""

Functions used to plot and produce the gifs

"""

# ╔═╡ acfb80f0-f4d0-4870-b401-6e26c1c99e45
function plot_clusters(D, zs, K, loss=nothing, iter=nothing,  framestyle=:origin; title_string=nothing, alpha=0.5)
	if isnothing(title_string)
		title_string = ""
		if !isnothing(iter)
			title_string ="Iteration: "*string(iter)*";"
		end
		if !isnothing(loss)
			title_string *= " L = "*string(round(loss; digits=2))
		end
	end
	plt = plot(title=title_string, ratio=1, framestyle=framestyle)
	for k in 1:K
		scatter!(D[zs .==k,1], D[zs .==k, 2], label="cluster "*string(k), ms=3, alpha=alpha)
	end
	return plt
end

# ╔═╡ f2b7b63a-fd54-4f80-af28-f255d186f6f9
begin
	gr()
	anim1 = Animation() 
	pltsₖₘ = [plt_demo]
	for iter in 1:5
		ls, initzs = assignment_step(data₁, ms)
	    p = plot_clusters(data₁, initzs, K₁, sum(ls), iter; title_string="Assignment step; Iteration = "*L"%$(iter)")
		for k in 1:K₁
			ctr = ms[:, k]
			scatter!([ms[1,k]], [ms[2,k]], markershape = :star4, markerstrokewidth =3, markersize = 10, c= k, labels= "μ"*string(k))
			plot!(-6:0.1:6, -5:0.1:5, (x,y) -> sum(([x, y] - ms[:, k]).^2), st=:contour, c=k, colorbar = false, levels=[5], lw=2)

			
			for x in eachrow(data₁[initzs.==k, :])
				plot!([x[1], ctr[1]],  [x[2], ctr[2]], lc=k, arrow=Plots.Arrow(:close, :both, 1,1),  lw=0.1, st=:path, label="")
			end
		end
		push!(pltsₖₘ, p)
		frame(anim1, p)
		p = plot_clusters(data₁, initzs, K₁, sum(ls), iter; title_string="Update step; Iteration = "*L"%$(iter)")
		ms = update_step(data₁, initzs, K₁)
		for k in 1:K₁
			scatter!([ms[1,k]], [ms[2,k]], markershape = :star4, markerstrokewidth =3, markersize = 10, c= k, labels= "μ"*string(k))
			plot!(-6:0.1:6, -5:0.1:5, (x,y) -> sum(([x, y] - ms[:, k]).^2), st=:contour, c=k, colorbar = false, levels=[5], lw=2)
		end
		push!(pltsₖₘ, p)
		frame(anim1, p)
	end
end;

# ╔═╡ 188b9f9d-78e3-4e78-8eae-795a9b012cab
begin
	next2
	pltsₖₘ[min(next_idx2[1]+1, length(pltsₖₘ))]
end

# ╔═╡ bd670cde-bcbe-4104-8114-34f761f7f850
gif(anim1, fps=2)

# ╔═╡ a2b02ca9-6edc-48bf-b5e5-746d986d8252
plts_km_alt, anim_km_alt =let
	gr()
	pltsₖₘ2 = []
	zs0 = rand(1:K₁, size(data₁)[1])
	anim2 = Animation()
	l_ = Inf
	p = plot_clusters(data₁, zs0, K₁, l_, 0)
	push!(pltsₖₘ2, p)
	frame(anim2, p)
	for iter in 1:6
		ms0 = update_step(data₁, zs0, K₁)
		p = plot_clusters(data₁, zs0, K₁, l_, iter; title_string="Update step; Iteration = "*L"%$(iter)")
		for k in 1:K₁
			scatter!([ms0[1,k]], [ms0[2,k]], markershape = :star4, markersize = 10, c=k, markerstrokewidth =3, color=k, labels= "μ"*string(k))
			plot!(-6:0.1:6, -5:0.1:5, (x,y) -> sum(([x, y] - ms0[:, k]).^2), st=:contour, c=k, colorbar = false, levels=[5], lw=2)
		end
		push!(pltsₖₘ2, p)
		frame(anim2, p)
		
		ls, zs0 = assignment_step(data₁, ms0)
		p = plot_clusters(data₁, zs0, K₁, l_, iter; title_string="Assignment step; Iteration = "*L"%$(iter)")
		for k in 1:K₁
			ctr = ms0[:, k]
			scatter!([ms0[1,k]], [ms0[2,k]], markershape = :star4, markerstrokewidth =3, markersize = 10, c= k, labels= "μ"*string(k))
			plot!(-6:0.1:6, -5:0.1:5, (x,y) -> sum(([x, y] - ms0[:, k]).^2), st=:contour, c=k, colorbar = false, levels=[5], lw=2)
			for x in eachrow(data₁[zs0.==k, :])
				plot!([x[1], ctr[1]],  [x[2], ctr[2]], lc=k, arrow=Plots.Arrow(:close, :both, 1,1),  lw=0.1, st=:path, label="")
			end
		end
		push!(pltsₖₘ2, p)
		frame(anim2, p)
	end

	pltsₖₘ2, anim2
end;

# ╔═╡ 20e84d48-0f5f-403e-a8aa-1cbd11cd3b04
begin
	next3
	plts_km_alt[min(next_idx3[1], length(plts_km_alt))]
end

# ╔═╡ 8d86466e-93c5-4263-b436-6fe684eeccd8
gif(anim_km_alt, fps = 1)

# ╔═╡ e091ce93-9526-4c7f-9f14-7634419bfe57
# plot clustering results: scatter plot + Gaussian contours
function plot_clustering_rst(data, K, zs, mvns, πs= 1/K .* ones(K); title="", add_gaussian_contours= false)
	xs = (minimum(data[:,1])-0.5):0.1: (maximum(data[:,1])+0.5)
	ys = (minimum(data[:,2])-0.5):0.1: (maximum(data[:,2])+0.5)
	_, dim = size(data)
	# if center parameters are given rather than an array of MvNormals
	if mvns isa Matrix{Float64}
		mvns = [MvNormal(mvns[:,k], Matrix(1.0I, dim, dim)) for k in 1:K]
		πs = 1/K .* ones(K)
	end
	if ndims(zs) >1
		zs = [c[2] for c in findmax(zs, dims=2)[2]][:]
	end
	p = plot_clusters(data, zs, K)
	for k in 1:K 
		if add_gaussian_contours
			plot!(xs, ys, (x,y)-> pdf(mvns[k], [x,y]), levels=5,  st=:contour, colorbar = false, ratio=1, color=:jet, linewidth=2) 
		else
			plot!(xs, ys, (x,y)-> qdform([x,y]-mvns[k].μ, inv(mvns[k].Σ)), levels=[2.0],  st=:contour, colorbar = false, ratio=1, color=k, linewidth=3) 
		end
		
		scatter!([mvns[k].μ[1]], [mvns[k].μ[2]], color = k, label = "", markersize = 10, markershape=:star4, markerstrokewidth=3)
	end
	title!(p, title)
	return p
end

# ╔═╡ 9612d7c1-3c86-45bc-806a-a53bbf869d87
begin
	gr()
	plt2_= plot_clustering_rst(data₂,  K₂, truezs₂, truemvns₂, trueπs₂)
	title!(plt2_, "Ground truth dataset", titlefontsize=8)
	plot!(plt2_, legendfontsize=5)
	plt2_
end;

# ╔═╡ eb273a3a-60d6-439e-bb63-6af8130b4ec0
plot(plt2_,  title="Ground truth data", titlefontsize=15, legendfontsize=8)

# ╔═╡ caf21a20-8dd8-4a31-8463-96d3869d3ee9
TwoColumn(plot(plt2_, size=(350,350), titlefontsize=10, legendfontsize=6), let
	gr()
	plt=plot_clustering_rst(data₂, K₂, zskm₂, ms₂; title= "K-means result for dataset 2")

	plot(plt, size=(350,350), titlefontsize=10, legendfontsize=6)
end)

# ╔═╡ 9cbc7ef6-788c-4dfd-a98d-2a9409eb5127
begin
	gr()
	plt₂_ =plot_clustering_rst(data₂, K₂, truezs₂,  truemvns₂, trueπs₂)
	title!(plt₂_, "Ground truth for dataset")
end;

# ╔═╡ 6569c4e1-5d62-42ad-94c0-927dd6b6f504
TwoColumn(md"""
\
\

**How about this dataset ?**


* rotated ellipsoid shaped distances
* **_overlapping_** data
""", begin
	gr()
	plt₃_= plot_clustering_rst(data₃,  K₃, truezs₃, truemvns₃, trueπs₃)
	title!(plt₃_, "Overlapping dataset")
	plot(plt₃_, size=(300,300), titlefontsize=10)
end)

# ╔═╡ f6b06459-0608-4981-bdb4-8f54d0cadfe1
TwoColumn(plot(plt₃_, titlefontsize=10, legendfontsize=6, size=(350,350)), let
	gr()
	plt=plot_clustering_rst(data₃, K₃, zskm₃, ms₃; title= "K-means result for dataset 2")

	plot(plt, titlefontsize=10, legendfontsize=6, size=(350,350))
end)

# ╔═╡ d44526f4-3051-47ee-8b63-f5e694c2e609
function e_step(data, mvns, πs)
	K = length(mvns)
	# logLiks: a n by K matrix of P(dᵢ|μₖ, Σₖ)
	logLiks = hcat([logpdf(mvns[k], data') for k in 1:K]...)
	# broadcast log(P(zᵢ=k)) to each row 
	logPost = log.(πs') .+ logLiks
	# apply log∑exp to each row to find the log of the normalising constant of p(zᵢ|…)
	logsums = logsumexp(logPost, dims=2)
	# normalise in log space then transform back to find the responsibility matrix
	ws = exp.(logPost .- logsums)
	# return the responsibility matrix and the log-likelihood
	return ws, sum(logsums)
end

# ╔═╡ 7b47cda6-d772-468c-a8f3-75e3d77369d8
begin
# decision boundary function of input [x,y] 
function decisionBdry(x,y, mvns, πs)
	z, _ = e_step([x,y]', mvns, πs)
	findmax(z[:])
end

end

# ╔═╡ 5a8cdbe7-6abe-4f07-8bcc-89dd71fc35f7
function kmeansDemoGif(data, K, iters = 10; init_step="a", add_contour=false, seed=123)
	Random.seed!(seed)
	# only support 2-d
	anims = [Animation() for i in 1:3]
	dim =2 
	# initialise by random assignment
	if init_step == "a"
		zs = rand(1:K, size(data)[1])
		l = Inf
	# initialise by randomly setting the centers 
	else
		ms = reshape(repeat(mean(data, dims=1)', K), (dim,K))
		ms .+= randn(dim,K)
		ls, zs = assignment_step(data, ms)
		l = sum(ls)
	end
	xs = (minimum(data[:,1])-0.1):0.1:(maximum(data[:,1])+0.1)
	ys = (minimum(data[:,2])-0.1):0.1:(maximum(data[:,2])+0.1)
	# cs = cgrad(:lighttest, K+1, categorical = true)
	ps = 1/K .* ones(K)
	for iter in 1:iters
		ms = update_step(data, zs, K)
		# animation 1: classification evolution
		p1 = plot_clusters(data, zs, K, l, iter)
		if add_contour
			for k in 1:K 
				plot!(xs, ys, (x,y)-> sum((ms[:, k] - [x,y]).^2), levels=[5],  st=:contour, colorbar = false, ratio=1, color=k, linewidth=3)  
				scatter!([ms[1,k]], [ms[2,k]], color = k, label = "", markersize = 10, markershape=:star4, markerstrokewidth=2)
			end
		end
		frame(anims[1], p1)
		# animation 2: decision boundary
		mvns = [MvNormal(ms[:,k], Matrix(1.0I, dim, dim)) for k in 1:K]
		p2 = plot(xs, ys, (x,y) -> decisionBdry(x,y, mvns, ps)[2],  leg=:none, title="Iteration: "*string(iter)*"; L="*string(round(l; digits=2)), ratio=1, framestyle=:origin, c=1:K, alpha=0.5, st=:heatmap)
		for k in 1:K
			scatter!(data[zs .==k, 1], data[zs .==k, 2], c= k, ms=3, alpha=0.5)
			# plot!(xs, ys, (x,y)-> sum((ms[:, k] - [x,y]).^2), levels=[1.5],  st=:contour, colorbar = false, ratio=1, color=k, linewidth=3)  
		end
		frame(anims[2], p2)

		# animation 3: contour evolution
		# animation 3: contour plot
		# p3 = plot_clusters(data, zs, K, l, iter)
		p3 = plot(xs, ys, (x,y) -> logLikMixGuassian([x,y], mvns, ps), st=:contour, fill=true, colorbar=false, ratio=1, title="Iteration: "*string(iter)*"; L="*string(round(l; digits=2)))
		# for k in 1:K
		# 	scatter!(data[zs .==k, 1], data[zs .==k, 2], c= cs[k], label="")
		# end
		frame(anims[3], p3)
		
		ls,zs = assignment_step(data, ms)
		l = sum(ls)
	end

	return anims
end

# ╔═╡ a0ebfea7-bbe5-4798-8db3-163381bf67c2
begin
	gr()
	kmAnim₁ = kmeansDemoGif(data₁, K₁, 10; init_step="a", add_contour=true);
end;

# ╔═╡ ab0c4765-e9a0-4ba3-8a21-e81b214ab060
TwoColumn(gif(kmAnim₁[1], fps=1), gif(kmAnim₁[2], fps=1))

# ╔═╡ 533efc1a-fd99-4a5d-9ef3-4ff7ce483f24
begin
	gr()
	kmAnim3 = kmeansDemoGif(data₂, K₂, 22; init_step="a", add_contour=true)
end;

# ╔═╡ c46949ad-46c8-4bcb-9e83-1eb646b4ea2a
gif(kmAnim3[2], fps=5)

# ╔═╡ ad943140-9510-463e-b57c-4956ed2cd5d5
begin
	gr()
	kmAnim₃ = kmeansDemoGif(data₃, K₃, 10; init_step="a", add_contour=true)
end;

# ╔═╡ 27755688-f647-48e5-a939-bb0fa70c95d8
function m_step(data, ws)
	_, d = size(data)
	K = size(ws)[2]
	ns = sum(ws, dims=1)
	πs = ns ./ sum(ns)
	# weighted sums ∑ wᵢₖ xᵢ where wᵢₖ = P(zᵢ=k|\cdots)
	ss = data' * ws
	# the weighted ML for μₖ = ∑ wᵢₖ xᵢ/ ∑ wᵢₖ
	μs = ss ./ ns
	Σs = zeros(d, d, K)
	for k in 1:K
		error = (data .- μs[:,k]')
		# weighted sum of squared error
		# use Symmetric to remove floating number numerical error
		Σs[:,:,k] =  Symmetric((error' * (ws[:,k] .* error))/ns[k])
	end
	# this is optional: you can just return μs and Σs
	mvns = [MvNormal(μs[:,k], Σs[:,:,k]) for k in 1:K]
	return mvns, πs[:]
end

# ╔═╡ 8d06ce32-2c8d-4317-8c38-108ec0e7fe23
function em_mix_gaussian(data, K=3; maxIters= 100, tol= 1e-4, init_step="e", seed=123)
	Random.seed!(seed)
	# initialisation
	n,d = size(data)
	if init_step == "e"
		zᵢ = rand(1:K, n)
		μs = zeros(d, K)
		[μs[:,k] = mean(data[zᵢ .== k,:], dims=1)[:] for k in 1:K] 
	elseif init_step == "m"
		μs = data[rand(1:n, K), :]'
	else
		μs = randn(d,K)
		μs .+= mean(data, dims=1)[:] 
	end
	Σs = zeros(d,d,K)
	Σs .= Matrix(1.0I, d,d)
	mvns = [MvNormal(μs[:,k], Σs[:,:,k]) for k in 1:K]
	πs = 1/K .* ones(K)
	zs = zeros(n,K)
	logLiks = Array{Float64,1}()
	i = 1
	for i in 1:maxIters
		# E-step
		zs, logLik = e_step(data, mvns, πs)
		# M-step
		mvns, πs = m_step(data, zs)
		push!(logLiks, logLik)
		# be nice, let it run at least three iters
		if i>2 && abs(logLiks[end] - logLiks[end-1])< tol
			break;
		end
	end
	return logLiks, mvns, πs, zs
end

# ╔═╡ c46e0b36-c3fd-4b7f-8f31-25c3315bb10c
# plot type: cl: classification; db: decision boundary; ct: contour
function mixGaussiansDemoGif(data, K, iters = 10; init_step="e", add_contour=false, seed=123)
	Random.seed!(seed)
	# only support 2-d
	dim = 2 
	anims = [Animation() for i in 1:3]
	if init_step == "e"
		zs_ = rand(1:K, size(data)[1])
		zs = Matrix(I,K,K)[zs_,:]
		l = Inf
	else
		ms = reshape(repeat(mean(data, dims=1)', K), (dim,K))
		ms .+= randn(dim,K)
		mvns = [MvNormal(ms[:,k], Matrix(1.0I,dim,dim)) for k in 1:K]
		zs, l = e_step(data, mvns, 1/K .* ones(K))
		zs_ = [c[2] for c in findmax(zs, dims=2)[2]][:]
	end
	xs = (minimum(data[:,1])-0.1):0.1:(maximum(data[:,1])+0.1)
	ys = (minimum(data[:,2])-0.1):0.1:(maximum(data[:,2])+0.1)
	# cs = cgrad(:lighttest, K+1, categorical = true)

	for iter in 1:iters
		# M step
		mvns, ps  = m_step(data, zs)
		# animation 1: classification evolution 
		p1 = plot_clusters(data, zs_, K, l, iter)
		if add_contour
			for k in 1:K 
				plot!(xs, ys, (x,y)-> qdform([x,y]-mvns[k].μ, inv(mvns[k].Σ)), levels=[2.0],  st=:contour, colorbar = false, ratio=1, color=k, linewidth=3) 
				scatter!([mvns[k].μ[1]], [mvns[k].μ[2]], color = k, label = "", markersize = 10, markershape=:star4, markerstrokewidth=2)
			end
		end
		frame(anims[1], p1)
		# animation 2: decision boundary evolution 
		p2 = heatmap(xs, ys, (x,y) -> decisionBdry(x,y, mvns, ps)[2], c=1:K, leg=:none, title="Iteration: "*string(iter)*"; L="*string(round(l; digits=2)), ratio=1, framestyle=:origin)
		for k in 1:K
			scatter!(data[zs_ .==k, 1], data[zs_ .==k, 2], c= k)
		end
		frame(anims[2], p2)

		# animation 3: contour plot
		# p3 = plot_clusters(data, zs_, K, l, iter)
		p3 = plot(xs, ys, (x,y) -> logLikMixGuassian([x,y], mvns, ps), st=:contour, fill=true, colorbar=false, title="Iteration: "*string(iter)*"; L="*string(round(l; digits=2)), ratio=1, framestyle=:origin)
		# for k in 1:K
		# 	scatter!(data[zs_ .==k, 1], data[zs_ .==k, 2], c= cs[k], label="")
		# end
		frame(anims[3], p3)
		# E step
		zs, l = e_step(data, mvns, ps)
		zs_ = [c[2] for c in findmax(zs, dims=2)[2]][:]
	end
	return anims
end

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
Clustering = "aaaa29a8-35af-508c-8bc3-b662a17a0fe5"
Distributions = "31c24e10-a181-5473-b8eb-7969acd0382f"
HypertextLiteral = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"
LaTeXStrings = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
Latexify = "23fbe1c1-3f47-55db-b15f-69d7ec21a316"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
LogExpFunctions = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
Logging = "56ddb016-857b-54e1-b83d-db4d58db5568"
Plots = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
PlutoTeachingTools = "661c6b06-c737-4d37-b85c-46df65de6f69"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
Statistics = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
StatsBase = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
StatsPlots = "f3b207a7-027a-5e70-b257-86293d7955fd"

[compat]
Clustering = "~0.15.4"
Distributions = "~0.25.100"
HypertextLiteral = "~0.9.4"
LaTeXStrings = "~1.3.0"
Latexify = "~0.16.1"
LogExpFunctions = "~0.3.26"
Plots = "~1.38.17"
PlutoTeachingTools = "~0.2.13"
PlutoUI = "~0.7.52"
StatsBase = "~0.34.0"
StatsPlots = "~0.15.6"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.9.4"
manifest_format = "2.0"
project_hash = "14d6b30cc4457409f52221963f394b8ee3122732"

[[deps.AbstractFFTs]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "d92ad398961a3ed262d8bf04a1a2b8340f915fef"
uuid = "621f4979-c628-5d54-868e-fcf4e3e8185c"
version = "1.5.0"
weakdeps = ["ChainRulesCore", "Test"]

    [deps.AbstractFFTs.extensions]
    AbstractFFTsChainRulesCoreExt = "ChainRulesCore"
    AbstractFFTsTestExt = "Test"

[[deps.AbstractPlutoDingetjes]]
deps = ["Pkg"]
git-tree-sha1 = "91bd53c39b9cbfb5ef4b015e8b582d344532bd0a"
uuid = "6e696c72-6542-2067-7265-42206c756150"
version = "1.2.0"

[[deps.Adapt]]
deps = ["LinearAlgebra", "Requires"]
git-tree-sha1 = "76289dc51920fdc6e0013c872ba9551d54961c24"
uuid = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
version = "3.6.2"
weakdeps = ["StaticArrays"]

    [deps.Adapt.extensions]
    AdaptStaticArraysExt = "StaticArrays"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"
version = "1.1.1"

[[deps.Arpack]]
deps = ["Arpack_jll", "Libdl", "LinearAlgebra", "Logging"]
git-tree-sha1 = "9b9b347613394885fd1c8c7729bfc60528faa436"
uuid = "7d9fca2a-8960-54d3-9f78-7d1dccf2cb97"
version = "0.5.4"

[[deps.Arpack_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "OpenBLAS_jll", "Pkg"]
git-tree-sha1 = "5ba6c757e8feccf03a1554dfaf3e26b3cfc7fd5e"
uuid = "68821587-b530-5797-8361-c406ea357684"
version = "3.5.1+1"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"

[[deps.AxisAlgorithms]]
deps = ["LinearAlgebra", "Random", "SparseArrays", "WoodburyMatrices"]
git-tree-sha1 = "66771c8d21c8ff5e3a93379480a2307ac36863f7"
uuid = "13072b0f-2c55-5437-9ae7-d433b7a33950"
version = "1.0.1"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"

[[deps.BitFlags]]
git-tree-sha1 = "43b1a4a8f797c1cddadf60499a8a077d4af2cd2d"
uuid = "d1d4a3ce-64b1-5f1a-9ba4-7e7e69966f35"
version = "0.1.7"

[[deps.Bzip2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "19a35467a82e236ff51bc17a3a44b69ef35185a2"
uuid = "6e34b625-4abd-537c-b88f-471c36dfa7a0"
version = "1.0.8+0"

[[deps.Cairo_jll]]
deps = ["Artifacts", "Bzip2_jll", "CompilerSupportLibraries_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "JLLWrappers", "LZO_jll", "Libdl", "Pixman_jll", "Pkg", "Xorg_libXext_jll", "Xorg_libXrender_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "4b859a208b2397a7a623a03449e4636bdb17bcf2"
uuid = "83423d85-b0ee-5818-9007-b63ccbeb887a"
version = "1.16.1+1"

[[deps.Calculus]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "f641eb0a4f00c343bbc32346e1217b86f3ce9dad"
uuid = "49dc2e85-a5d0-5ad3-a950-438e2897f1b9"
version = "0.5.1"

[[deps.ChainRulesCore]]
deps = ["Compat", "LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "e30f2f4e20f7f186dc36529910beaedc60cfa644"
uuid = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
version = "1.16.0"

[[deps.Clustering]]
deps = ["Distances", "LinearAlgebra", "NearestNeighbors", "Printf", "Random", "SparseArrays", "Statistics", "StatsBase"]
git-tree-sha1 = "b86ac2c5543660d238957dbde5ac04520ae977a7"
uuid = "aaaa29a8-35af-508c-8bc3-b662a17a0fe5"
version = "0.15.4"

[[deps.CodeTracking]]
deps = ["InteractiveUtils", "UUIDs"]
git-tree-sha1 = "a1296f0fe01a4c3f9bf0dc2934efbf4416f5db31"
uuid = "da1fd8a2-8d9e-5ec2-8556-3022fb5608a2"
version = "1.3.4"

[[deps.CodecZlib]]
deps = ["TranscodingStreams", "Zlib_jll"]
git-tree-sha1 = "02aa26a4cf76381be7f66e020a3eddeb27b0a092"
uuid = "944b1d66-785c-5afd-91f1-9de20f533193"
version = "0.7.2"

[[deps.ColorSchemes]]
deps = ["ColorTypes", "ColorVectorSpace", "Colors", "FixedPointNumbers", "PrecompileTools", "Random"]
git-tree-sha1 = "d9a8f86737b665e15a9641ecbac64deef9ce6724"
uuid = "35d6a980-a343-548e-a6ea-1d62b119f2f4"
version = "3.23.0"

[[deps.ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "eb7f0f8307f71fac7c606984ea5fb2817275d6e4"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.11.4"

[[deps.ColorVectorSpace]]
deps = ["ColorTypes", "FixedPointNumbers", "LinearAlgebra", "Requires", "Statistics", "TensorCore"]
git-tree-sha1 = "a1f44953f2382ebb937d60dafbe2deea4bd23249"
uuid = "c3611d14-8923-5661-9e6a-0046d554d3a4"
version = "0.10.0"
weakdeps = ["SpecialFunctions"]

    [deps.ColorVectorSpace.extensions]
    SpecialFunctionsExt = "SpecialFunctions"

[[deps.Colors]]
deps = ["ColorTypes", "FixedPointNumbers", "Reexport"]
git-tree-sha1 = "fc08e5930ee9a4e03f84bfb5211cb54e7769758a"
uuid = "5ae59095-9a9b-59fe-a467-6f913c188581"
version = "0.12.10"

[[deps.Compat]]
deps = ["UUIDs"]
git-tree-sha1 = "e460f044ca8b99be31d35fe54fc33a5c33dd8ed7"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "4.9.0"
weakdeps = ["Dates", "LinearAlgebra"]

    [deps.Compat.extensions]
    CompatLinearAlgebraExt = "LinearAlgebra"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"
version = "1.0.5+0"

[[deps.ConcurrentUtilities]]
deps = ["Serialization", "Sockets"]
git-tree-sha1 = "5372dbbf8f0bdb8c700db5367132925c0771ef7e"
uuid = "f0e56b4a-5159-44fe-b623-3e5288b988bb"
version = "2.2.1"

[[deps.Contour]]
git-tree-sha1 = "d05d9e7b7aedff4e5b51a029dced05cfb6125781"
uuid = "d38c429a-6771-53c6-b99e-75d170b6e991"
version = "0.6.2"

[[deps.DataAPI]]
git-tree-sha1 = "8da84edb865b0b5b0100c0666a9bc9a0b71c553c"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.15.0"

[[deps.DataStructures]]
deps = ["Compat", "InteractiveUtils", "OrderedCollections"]
git-tree-sha1 = "3dbd312d370723b6bb43ba9d02fc36abade4518d"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.18.15"

[[deps.DataValueInterfaces]]
git-tree-sha1 = "bfc1187b79289637fa0ef6d4436ebdfe6905cbd6"
uuid = "e2d170a0-9d28-54be-80f0-106bbe20a464"
version = "1.0.0"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"

[[deps.DelimitedFiles]]
deps = ["Mmap"]
git-tree-sha1 = "9e2f36d3c96a820c678f2f1f1782582fcf685bae"
uuid = "8bb1440f-4735-579b-a4ab-409b98df4dab"
version = "1.9.1"

[[deps.Distances]]
deps = ["LinearAlgebra", "Statistics", "StatsAPI"]
git-tree-sha1 = "b6def76ffad15143924a2199f72a5cd883a2e8a9"
uuid = "b4f34e82-e78d-54a5-968a-f98e89d6e8f7"
version = "0.10.9"
weakdeps = ["SparseArrays"]

    [deps.Distances.extensions]
    DistancesSparseArraysExt = "SparseArrays"

[[deps.Distributed]]
deps = ["Random", "Serialization", "Sockets"]
uuid = "8ba89e20-285c-5b6f-9357-94700520ee1b"

[[deps.Distributions]]
deps = ["FillArrays", "LinearAlgebra", "PDMats", "Printf", "QuadGK", "Random", "SpecialFunctions", "Statistics", "StatsAPI", "StatsBase", "StatsFuns", "Test"]
git-tree-sha1 = "938fe2981db009f531b6332e31c58e9584a2f9bd"
uuid = "31c24e10-a181-5473-b8eb-7969acd0382f"
version = "0.25.100"

    [deps.Distributions.extensions]
    DistributionsChainRulesCoreExt = "ChainRulesCore"
    DistributionsDensityInterfaceExt = "DensityInterface"

    [deps.Distributions.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    DensityInterface = "b429d917-457f-4dbc-8f4c-0cc954292b1d"

[[deps.DocStringExtensions]]
deps = ["LibGit2"]
git-tree-sha1 = "2fb1e02f2b635d0845df5d7c167fec4dd739b00d"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.9.3"

[[deps.Downloads]]
deps = ["ArgTools", "FileWatching", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"
version = "1.6.0"

[[deps.DualNumbers]]
deps = ["Calculus", "NaNMath", "SpecialFunctions"]
git-tree-sha1 = "5837a837389fccf076445fce071c8ddaea35a566"
uuid = "fa6b7ba4-c1ee-5f82-b5fc-ecf0adba8f74"
version = "0.6.8"

[[deps.EpollShim_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "8e9441ee83492030ace98f9789a654a6d0b1f643"
uuid = "2702e6a9-849d-5ed8-8c21-79e8b8f9ee43"
version = "0.0.20230411+0"

[[deps.ExceptionUnwrapping]]
deps = ["Test"]
git-tree-sha1 = "e90caa41f5a86296e014e148ee061bd6c3edec96"
uuid = "460bff9d-24e4-43bc-9d9f-a8973cb893f4"
version = "0.1.9"

[[deps.Expat_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "4558ab818dcceaab612d1bb8c19cee87eda2b83c"
uuid = "2e619515-83b5-522b-bb60-26c02a35a201"
version = "2.5.0+0"

[[deps.FFMPEG]]
deps = ["FFMPEG_jll"]
git-tree-sha1 = "b57e3acbe22f8484b4b5ff66a7499717fe1a9cc8"
uuid = "c87230d0-a227-11e9-1b43-d7ebe4e7570a"
version = "0.4.1"

[[deps.FFMPEG_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "JLLWrappers", "LAME_jll", "Libdl", "Ogg_jll", "OpenSSL_jll", "Opus_jll", "PCRE2_jll", "Pkg", "Zlib_jll", "libaom_jll", "libass_jll", "libfdk_aac_jll", "libvorbis_jll", "x264_jll", "x265_jll"]
git-tree-sha1 = "74faea50c1d007c85837327f6775bea60b5492dd"
uuid = "b22a6f82-2f65-5046-a5b2-351ab43fb4e5"
version = "4.4.2+2"

[[deps.FFTW]]
deps = ["AbstractFFTs", "FFTW_jll", "LinearAlgebra", "MKL_jll", "Preferences", "Reexport"]
git-tree-sha1 = "b4fbdd20c889804969571cc589900803edda16b7"
uuid = "7a1cc6ca-52ef-59f5-83cd-3a7055c09341"
version = "1.7.1"

[[deps.FFTW_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "c6033cc3892d0ef5bb9cd29b7f2f0331ea5184ea"
uuid = "f5851436-0d7a-5f13-b9de-f02708fd171a"
version = "3.3.10+0"

[[deps.FileWatching]]
uuid = "7b1f6079-737a-58dc-b8bc-7a2ca5c1b5ee"

[[deps.FillArrays]]
deps = ["LinearAlgebra", "Random"]
git-tree-sha1 = "a20eaa3ad64254c61eeb5f230d9306e937405434"
uuid = "1a297f60-69ca-5386-bcde-b61e274b549b"
version = "1.6.1"
weakdeps = ["SparseArrays", "Statistics"]

    [deps.FillArrays.extensions]
    FillArraysSparseArraysExt = "SparseArrays"
    FillArraysStatisticsExt = "Statistics"

[[deps.FixedPointNumbers]]
deps = ["Statistics"]
git-tree-sha1 = "335bfdceacc84c5cdf16aadc768aa5ddfc5383cc"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.4"

[[deps.Fontconfig_jll]]
deps = ["Artifacts", "Bzip2_jll", "Expat_jll", "FreeType2_jll", "JLLWrappers", "Libdl", "Libuuid_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "21efd19106a55620a188615da6d3d06cd7f6ee03"
uuid = "a3f928ae-7b40-5064-980b-68af3947d34b"
version = "2.13.93+0"

[[deps.Formatting]]
deps = ["Printf"]
git-tree-sha1 = "8339d61043228fdd3eb658d86c926cb282ae72a8"
uuid = "59287772-0a20-5a39-b81b-1366585eb4c0"
version = "0.4.2"

[[deps.FreeType2_jll]]
deps = ["Artifacts", "Bzip2_jll", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "d8db6a5a2fe1381c1ea4ef2cab7c69c2de7f9ea0"
uuid = "d7e528f0-a631-5988-bf34-fe36492bcfd7"
version = "2.13.1+0"

[[deps.FriBidi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "aa31987c2ba8704e23c6c8ba8a4f769d5d7e4f91"
uuid = "559328eb-81f9-559d-9380-de523a88c83c"
version = "1.0.10+0"

[[deps.GLFW_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libglvnd_jll", "Pkg", "Xorg_libXcursor_jll", "Xorg_libXi_jll", "Xorg_libXinerama_jll", "Xorg_libXrandr_jll"]
git-tree-sha1 = "d972031d28c8c8d9d7b41a536ad7bb0c2579caca"
uuid = "0656b61e-2033-5cc2-a64a-77c0f6c09b89"
version = "3.3.8+0"

[[deps.GR]]
deps = ["Artifacts", "Base64", "DelimitedFiles", "Downloads", "GR_jll", "HTTP", "JSON", "Libdl", "LinearAlgebra", "Pkg", "Preferences", "Printf", "Random", "Serialization", "Sockets", "TOML", "Tar", "Test", "UUIDs", "p7zip_jll"]
git-tree-sha1 = "d73afa4a2bb9de56077242d98cf763074ab9a970"
uuid = "28b8d3ca-fb5f-59d9-8090-bfdbd6d07a71"
version = "0.72.9"

[[deps.GR_jll]]
deps = ["Artifacts", "Bzip2_jll", "Cairo_jll", "FFMPEG_jll", "Fontconfig_jll", "FreeType2_jll", "GLFW_jll", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Libtiff_jll", "Pixman_jll", "Qt6Base_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "1596bab77f4f073a14c62424283e7ebff3072eca"
uuid = "d2c73de3-f751-5644-a686-071e5b155ba9"
version = "0.72.9+1"

[[deps.Gettext_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Libiconv_jll", "Pkg", "XML2_jll"]
git-tree-sha1 = "9b02998aba7bf074d14de89f9d37ca24a1a0b046"
uuid = "78b55507-aeef-58d4-861c-77aaff3498b1"
version = "0.21.0+0"

[[deps.Glib_jll]]
deps = ["Artifacts", "Gettext_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Libiconv_jll", "Libmount_jll", "PCRE2_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "d3b3624125c1474292d0d8ed0f65554ac37ddb23"
uuid = "7746bdde-850d-59dc-9ae8-88ece973131d"
version = "2.74.0+2"

[[deps.Graphite2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "344bf40dcab1073aca04aa0df4fb092f920e4011"
uuid = "3b182d85-2403-5c21-9c21-1e1f0cc25472"
version = "1.3.14+0"

[[deps.Grisu]]
git-tree-sha1 = "53bb909d1151e57e2484c3d1b53e19552b887fb2"
uuid = "42e2da0e-8278-4e71-bc24-59509adca0fe"
version = "1.0.2"

[[deps.HTTP]]
deps = ["Base64", "CodecZlib", "ConcurrentUtilities", "Dates", "ExceptionUnwrapping", "Logging", "LoggingExtras", "MbedTLS", "NetworkOptions", "OpenSSL", "Random", "SimpleBufferStream", "Sockets", "URIs", "UUIDs"]
git-tree-sha1 = "cb56ccdd481c0dd7f975ad2b3b62d9eda088f7e2"
uuid = "cd3eb016-35fb-5094-929b-558a96fad6f3"
version = "1.9.14"

[[deps.HarfBuzz_jll]]
deps = ["Artifacts", "Cairo_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "Graphite2_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Pkg"]
git-tree-sha1 = "129acf094d168394e80ee1dc4bc06ec835e510a3"
uuid = "2e76f6c2-a576-52d4-95c1-20adfe4de566"
version = "2.8.1+1"

[[deps.HypergeometricFunctions]]
deps = ["DualNumbers", "LinearAlgebra", "OpenLibm_jll", "SpecialFunctions"]
git-tree-sha1 = "f218fe3736ddf977e0e772bc9a586b2383da2685"
uuid = "34004b35-14d8-5ef3-9330-4cdb6864b03a"
version = "0.3.23"

[[deps.Hyperscript]]
deps = ["Test"]
git-tree-sha1 = "8d511d5b81240fc8e6802386302675bdf47737b9"
uuid = "47d2ed2b-36de-50cf-bf87-49c2cf4b8b91"
version = "0.0.4"

[[deps.HypertextLiteral]]
deps = ["Tricks"]
git-tree-sha1 = "c47c5fa4c5308f27ccaac35504858d8914e102f9"
uuid = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"
version = "0.9.4"

[[deps.IOCapture]]
deps = ["Logging", "Random"]
git-tree-sha1 = "d75853a0bdbfb1ac815478bacd89cd27b550ace6"
uuid = "b5f81e59-6552-4d32-b1f0-c071b021bf89"
version = "0.2.3"

[[deps.IntelOpenMP_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "ad37c091f7d7daf900963171600d7c1c5c3ede32"
uuid = "1d5cc7b8-4909-519e-a0f8-d0f5ad9712d0"
version = "2023.2.0+0"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"

[[deps.Interpolations]]
deps = ["Adapt", "AxisAlgorithms", "ChainRulesCore", "LinearAlgebra", "OffsetArrays", "Random", "Ratios", "Requires", "SharedArrays", "SparseArrays", "StaticArrays", "WoodburyMatrices"]
git-tree-sha1 = "721ec2cf720536ad005cb38f50dbba7b02419a15"
uuid = "a98d9a8b-a2ab-59e6-89dd-64a1c18fca59"
version = "0.14.7"

[[deps.IrrationalConstants]]
git-tree-sha1 = "630b497eafcc20001bba38a4651b327dcfc491d2"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.2.2"

[[deps.IteratorInterfaceExtensions]]
git-tree-sha1 = "a3f24677c21f5bbe9d2a714f95dcd58337fb2856"
uuid = "82899510-4779-5014-852e-03e436cf321d"
version = "1.0.0"

[[deps.JLFzf]]
deps = ["Pipe", "REPL", "Random", "fzf_jll"]
git-tree-sha1 = "f377670cda23b6b7c1c0b3893e37451c5c1a2185"
uuid = "1019f520-868f-41f5-a6de-eb00f4b6a39c"
version = "0.1.5"

[[deps.JLLWrappers]]
deps = ["Artifacts", "Preferences"]
git-tree-sha1 = "7e5d6779a1e09a36db2a7b6cff50942a0a7d0fca"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.5.0"

[[deps.JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "31e996f0a15c7b280ba9f76636b3ff9e2ae58c9a"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.4"

[[deps.JpegTurbo_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "6f2675ef130a300a112286de91973805fcc5ffbc"
uuid = "aacddb02-875f-59d6-b918-886e6ef4fbf8"
version = "2.1.91+0"

[[deps.JuliaInterpreter]]
deps = ["CodeTracking", "InteractiveUtils", "Random", "UUIDs"]
git-tree-sha1 = "81dc6aefcbe7421bd62cb6ca0e700779330acff8"
uuid = "aa1ae85d-cabe-5617-a682-6adf51b2e16a"
version = "0.9.25"

[[deps.KernelDensity]]
deps = ["Distributions", "DocStringExtensions", "FFTW", "Interpolations", "StatsBase"]
git-tree-sha1 = "90442c50e202a5cdf21a7899c66b240fdef14035"
uuid = "5ab0869b-81aa-558d-bb23-cbf5423bbe9b"
version = "0.6.7"

[[deps.LAME_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "f6250b16881adf048549549fba48b1161acdac8c"
uuid = "c1c5ebd0-6772-5130-a774-d5fcae4a789d"
version = "3.100.1+0"

[[deps.LERC_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "bf36f528eec6634efc60d7ec062008f171071434"
uuid = "88015f11-f218-50d7-93a8-a6af411a945d"
version = "3.0.0+1"

[[deps.LLVMOpenMP_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "f689897ccbe049adb19a065c495e75f372ecd42b"
uuid = "1d63c593-3942-5779-bab2-d838dc0a180e"
version = "15.0.4+0"

[[deps.LZO_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "e5b909bcf985c5e2605737d2ce278ed791b89be6"
uuid = "dd4b983a-f0e5-5f8d-a1b7-129d4a5fb1ac"
version = "2.10.1+0"

[[deps.LaTeXStrings]]
git-tree-sha1 = "f2355693d6778a178ade15952b7ac47a4ff97996"
uuid = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
version = "1.3.0"

[[deps.Latexify]]
deps = ["Formatting", "InteractiveUtils", "LaTeXStrings", "MacroTools", "Markdown", "OrderedCollections", "Printf", "Requires"]
git-tree-sha1 = "f428ae552340899a935973270b8d98e5a31c49fe"
uuid = "23fbe1c1-3f47-55db-b15f-69d7ec21a316"
version = "0.16.1"

    [deps.Latexify.extensions]
    DataFramesExt = "DataFrames"
    SymEngineExt = "SymEngine"

    [deps.Latexify.weakdeps]
    DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
    SymEngine = "123dc426-2d89-5057-bbad-38513e3affd8"

[[deps.LazyArtifacts]]
deps = ["Artifacts", "Pkg"]
uuid = "4af54fe1-eca0-43a8-85a7-787d91b784e3"

[[deps.LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"
version = "0.6.4"

[[deps.LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"
version = "8.4.0+0"

[[deps.LibGit2]]
deps = ["Base64", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"

[[deps.LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "MbedTLS_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"
version = "1.11.0+1"

[[deps.Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"

[[deps.Libffi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "0b4a5d71f3e5200a7dff793393e09dfc2d874290"
uuid = "e9f186c6-92d2-5b65-8a66-fee21dc1b490"
version = "3.2.2+1"

[[deps.Libgcrypt_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libgpg_error_jll", "Pkg"]
git-tree-sha1 = "64613c82a59c120435c067c2b809fc61cf5166ae"
uuid = "d4300ac3-e22c-5743-9152-c294e39db1e4"
version = "1.8.7+0"

[[deps.Libglvnd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll", "Xorg_libXext_jll"]
git-tree-sha1 = "6f73d1dd803986947b2c750138528a999a6c7733"
uuid = "7e76a0d4-f3c7-5321-8279-8d96eeed0f29"
version = "1.6.0+0"

[[deps.Libgpg_error_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "c333716e46366857753e273ce6a69ee0945a6db9"
uuid = "7add5ba3-2f88-524e-9cd5-f83b8a55f7b8"
version = "1.42.0+0"

[[deps.Libiconv_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "c7cb1f5d892775ba13767a87c7ada0b980ea0a71"
uuid = "94ce4f54-9a6c-5748-9c1c-f9c7231a4531"
version = "1.16.1+2"

[[deps.Libmount_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "9c30530bf0effd46e15e0fdcf2b8636e78cbbd73"
uuid = "4b2f31a3-9ecc-558c-b454-b3730dcb73e9"
version = "2.35.0+0"

[[deps.Libtiff_jll]]
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "LERC_jll", "Libdl", "XZ_jll", "Zlib_jll", "Zstd_jll"]
git-tree-sha1 = "2da088d113af58221c52828a80378e16be7d037a"
uuid = "89763e89-9b03-5906-acba-b20f662cd828"
version = "4.5.1+1"

[[deps.Libuuid_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "7f3efec06033682db852f8b3bc3c1d2b0a0ab066"
uuid = "38a345b3-de98-5d2b-a5d3-14cd9215e700"
version = "2.36.0+0"

[[deps.LinearAlgebra]]
deps = ["Libdl", "OpenBLAS_jll", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"

[[deps.LogExpFunctions]]
deps = ["DocStringExtensions", "IrrationalConstants", "LinearAlgebra"]
git-tree-sha1 = "7d6dd4e9212aebaeed356de34ccf262a3cd415aa"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "0.3.26"

    [deps.LogExpFunctions.extensions]
    LogExpFunctionsChainRulesCoreExt = "ChainRulesCore"
    LogExpFunctionsChangesOfVariablesExt = "ChangesOfVariables"
    LogExpFunctionsInverseFunctionsExt = "InverseFunctions"

    [deps.LogExpFunctions.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    ChangesOfVariables = "9e997f8a-9a97-42d5-a9f1-ce6bfc15e2c0"
    InverseFunctions = "3587e190-3f89-42d0-90ee-14403ec27112"

[[deps.Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"

[[deps.LoggingExtras]]
deps = ["Dates", "Logging"]
git-tree-sha1 = "a03c77519ab45eb9a34d3cfe2ca223d79c064323"
uuid = "e6f89c97-d47a-5376-807f-9c37f3926c36"
version = "1.0.1"

[[deps.LoweredCodeUtils]]
deps = ["JuliaInterpreter"]
git-tree-sha1 = "60168780555f3e663c536500aa790b6368adc02a"
uuid = "6f1432cf-f94c-5a45-995e-cdbf5db27b0b"
version = "2.3.0"

[[deps.MIMEs]]
git-tree-sha1 = "65f28ad4b594aebe22157d6fac869786a255b7eb"
uuid = "6c6e2e6c-3030-632d-7369-2d6c69616d65"
version = "0.1.4"

[[deps.MKL_jll]]
deps = ["Artifacts", "IntelOpenMP_jll", "JLLWrappers", "LazyArtifacts", "Libdl", "Pkg"]
git-tree-sha1 = "eb006abbd7041c28e0d16260e50a24f8f9104913"
uuid = "856f044c-d86e-5d09-b602-aeab76dc8ba7"
version = "2023.2.0+0"

[[deps.MacroTools]]
deps = ["Markdown", "Random"]
git-tree-sha1 = "9ee1618cbf5240e6d4e0371d6f24065083f60c48"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.11"

[[deps.Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"

[[deps.MbedTLS]]
deps = ["Dates", "MbedTLS_jll", "MozillaCACerts_jll", "Random", "Sockets"]
git-tree-sha1 = "03a9b9718f5682ecb107ac9f7308991db4ce395b"
uuid = "739be429-bea8-5141-9913-cc70e7f3736d"
version = "1.1.7"

[[deps.MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"
version = "2.28.2+0"

[[deps.Measures]]
git-tree-sha1 = "c13304c81eec1ed3af7fc20e75fb6b26092a1102"
uuid = "442fdcdd-2543-5da2-b0f3-8c86c306513e"
version = "0.3.2"

[[deps.Missings]]
deps = ["DataAPI"]
git-tree-sha1 = "f66bdc5de519e8f8ae43bdc598782d35a25b1272"
uuid = "e1d29d7a-bbdc-5cf2-9ac0-f12de2c33e28"
version = "1.1.0"

[[deps.Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"

[[deps.MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"
version = "2022.10.11"

[[deps.MultivariateStats]]
deps = ["Arpack", "LinearAlgebra", "SparseArrays", "Statistics", "StatsAPI", "StatsBase"]
git-tree-sha1 = "68bf5103e002c44adfd71fea6bd770b3f0586843"
uuid = "6f286f6a-111f-5878-ab1e-185364afe411"
version = "0.10.2"

[[deps.NaNMath]]
deps = ["OpenLibm_jll"]
git-tree-sha1 = "0877504529a3e5c3343c6f8b4c0381e57e4387e4"
uuid = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
version = "1.0.2"

[[deps.NearestNeighbors]]
deps = ["Distances", "StaticArrays"]
git-tree-sha1 = "2c3726ceb3388917602169bed973dbc97f1b51a8"
uuid = "b8a86587-4115-5ab1-83bc-aa920d37bbce"
version = "0.4.13"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"
version = "1.2.0"

[[deps.Observables]]
git-tree-sha1 = "6862738f9796b3edc1c09d0890afce4eca9e7e93"
uuid = "510215fc-4207-5dde-b226-833fc4488ee2"
version = "0.5.4"

[[deps.OffsetArrays]]
deps = ["Adapt"]
git-tree-sha1 = "2ac17d29c523ce1cd38e27785a7d23024853a4bb"
uuid = "6fe1bfb0-de20-5000-8ca7-80f57d26f881"
version = "1.12.10"

[[deps.Ogg_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "887579a3eb005446d514ab7aeac5d1d027658b8f"
uuid = "e7412a2a-1a6e-54c0-be00-318e2571c051"
version = "1.3.5+1"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"
version = "0.3.21+4"

[[deps.OpenLibm_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "05823500-19ac-5b8b-9628-191a04bc5112"
version = "0.8.1+0"

[[deps.OpenSSL]]
deps = ["BitFlags", "Dates", "MozillaCACerts_jll", "OpenSSL_jll", "Sockets"]
git-tree-sha1 = "51901a49222b09e3743c65b8847687ae5fc78eb2"
uuid = "4d8831e6-92b7-49fb-bdf8-b643e874388c"
version = "1.4.1"

[[deps.OpenSSL_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "bbb5c2115d63c2f1451cb70e5ef75e8fe4707019"
uuid = "458c3c95-2e84-50aa-8efc-19380b2a3a95"
version = "1.1.22+0"

[[deps.OpenSpecFun_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "13652491f6856acfd2db29360e1bbcd4565d04f1"
uuid = "efe28fd5-8261-553b-a9e1-b2916fc3738e"
version = "0.5.5+0"

[[deps.Opus_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "51a08fb14ec28da2ec7a927c4337e4332c2a4720"
uuid = "91d4177d-7536-5919-b921-800302f37372"
version = "1.3.2+0"

[[deps.OrderedCollections]]
git-tree-sha1 = "2e73fe17cac3c62ad1aebe70d44c963c3cfdc3e3"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.6.2"

[[deps.PCRE2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "efcefdf7-47ab-520b-bdef-62a2eaa19f15"
version = "10.42.0+0"

[[deps.PDMats]]
deps = ["LinearAlgebra", "SparseArrays", "SuiteSparse"]
git-tree-sha1 = "67eae2738d63117a196f497d7db789821bce61d1"
uuid = "90014a1f-27ba-587c-ab20-58faa44d9150"
version = "0.11.17"

[[deps.Parsers]]
deps = ["Dates", "PrecompileTools", "UUIDs"]
git-tree-sha1 = "716e24b21538abc91f6205fd1d8363f39b442851"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.7.2"

[[deps.Pipe]]
git-tree-sha1 = "6842804e7867b115ca9de748a0cf6b364523c16d"
uuid = "b98c9c47-44ae-5843-9183-064241ee97a0"
version = "1.3.0"

[[deps.Pixman_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "LLVMOpenMP_jll", "Libdl"]
git-tree-sha1 = "64779bc4c9784fee475689a1752ef4d5747c5e87"
uuid = "30392449-352a-5448-841d-b1acce4e97dc"
version = "0.42.2+0"

[[deps.Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "FileWatching", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "REPL", "Random", "SHA", "Serialization", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"
version = "1.9.2"

[[deps.PlotThemes]]
deps = ["PlotUtils", "Statistics"]
git-tree-sha1 = "1f03a2d339f42dca4a4da149c7e15e9b896ad899"
uuid = "ccf2f8ad-2431-5c83-bf29-c5338b663b6a"
version = "3.1.0"

[[deps.PlotUtils]]
deps = ["ColorSchemes", "Colors", "Dates", "PrecompileTools", "Printf", "Random", "Reexport", "Statistics"]
git-tree-sha1 = "f92e1315dadf8c46561fb9396e525f7200cdc227"
uuid = "995b91a9-d308-5afd-9ec6-746e21dbc043"
version = "1.3.5"

[[deps.Plots]]
deps = ["Base64", "Contour", "Dates", "Downloads", "FFMPEG", "FixedPointNumbers", "GR", "JLFzf", "JSON", "LaTeXStrings", "Latexify", "LinearAlgebra", "Measures", "NaNMath", "Pkg", "PlotThemes", "PlotUtils", "PrecompileTools", "Preferences", "Printf", "REPL", "Random", "RecipesBase", "RecipesPipeline", "Reexport", "RelocatableFolders", "Requires", "Scratch", "Showoff", "SparseArrays", "Statistics", "StatsBase", "UUIDs", "UnicodeFun", "UnitfulLatexify", "Unzip"]
git-tree-sha1 = "9f8675a55b37a70aa23177ec110f6e3f4dd68466"
uuid = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
version = "1.38.17"

    [deps.Plots.extensions]
    FileIOExt = "FileIO"
    GeometryBasicsExt = "GeometryBasics"
    IJuliaExt = "IJulia"
    ImageInTerminalExt = "ImageInTerminal"
    UnitfulExt = "Unitful"

    [deps.Plots.weakdeps]
    FileIO = "5789e2e9-d7fb-5bc7-8068-2c6fae9b9549"
    GeometryBasics = "5c1252a2-5f33-56bf-86c9-59e7332b4326"
    IJulia = "7073ff75-c697-5162-941a-fcdaad2a7d2a"
    ImageInTerminal = "d8c32880-2388-543b-8c61-d9f865259254"
    Unitful = "1986cc42-f94f-5a68-af5c-568840ba703d"

[[deps.PlutoHooks]]
deps = ["InteractiveUtils", "Markdown", "UUIDs"]
git-tree-sha1 = "072cdf20c9b0507fdd977d7d246d90030609674b"
uuid = "0ff47ea0-7a50-410d-8455-4348d5de0774"
version = "0.0.5"

[[deps.PlutoLinks]]
deps = ["FileWatching", "InteractiveUtils", "Markdown", "PlutoHooks", "Revise", "UUIDs"]
git-tree-sha1 = "8f5fa7056e6dcfb23ac5211de38e6c03f6367794"
uuid = "0ff47ea0-7a50-410d-8455-4348d5de0420"
version = "0.1.6"

[[deps.PlutoTeachingTools]]
deps = ["Downloads", "HypertextLiteral", "LaTeXStrings", "Latexify", "Markdown", "PlutoLinks", "PlutoUI", "Random"]
git-tree-sha1 = "542de5acb35585afcf202a6d3361b430bc1c3fbd"
uuid = "661c6b06-c737-4d37-b85c-46df65de6f69"
version = "0.2.13"

[[deps.PlutoUI]]
deps = ["AbstractPlutoDingetjes", "Base64", "ColorTypes", "Dates", "FixedPointNumbers", "Hyperscript", "HypertextLiteral", "IOCapture", "InteractiveUtils", "JSON", "Logging", "MIMEs", "Markdown", "Random", "Reexport", "URIs", "UUIDs"]
git-tree-sha1 = "e47cd150dbe0443c3a3651bc5b9cbd5576ab75b7"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.52"

[[deps.PrecompileTools]]
deps = ["Preferences"]
git-tree-sha1 = "03b4c25b43cb84cee5c90aa9b5ea0a78fd848d2f"
uuid = "aea7be01-6a6a-4083-8856-8a6e6704d82a"
version = "1.2.0"

[[deps.Preferences]]
deps = ["TOML"]
git-tree-sha1 = "7eb1686b4f04b82f96ed7a4ea5890a4f0c7a09f1"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.4.0"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"

[[deps.Qt6Base_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Fontconfig_jll", "Glib_jll", "JLLWrappers", "Libdl", "Libglvnd_jll", "OpenSSL_jll", "Pkg", "Xorg_libXext_jll", "Xorg_libXrender_jll", "Xorg_libxcb_jll", "Xorg_xcb_util_image_jll", "Xorg_xcb_util_keysyms_jll", "Xorg_xcb_util_renderutil_jll", "Xorg_xcb_util_wm_jll", "Zlib_jll", "xkbcommon_jll"]
git-tree-sha1 = "364898e8f13f7eaaceec55fd3d08680498c0aa6e"
uuid = "c0090381-4147-56d7-9ebc-da0b1113ec56"
version = "6.4.2+3"

[[deps.QuadGK]]
deps = ["DataStructures", "LinearAlgebra"]
git-tree-sha1 = "6ec7ac8412e83d57e313393220879ede1740f9ee"
uuid = "1fd47b50-473d-5c70-9696-f719f8f3bcdc"
version = "2.8.2"

[[deps.REPL]]
deps = ["InteractiveUtils", "Markdown", "Sockets", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"

[[deps.Random]]
deps = ["SHA", "Serialization"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[[deps.Ratios]]
deps = ["Requires"]
git-tree-sha1 = "1342a47bf3260ee108163042310d26f2be5ec90b"
uuid = "c84ed2f1-dad5-54f0-aa8e-dbefe2724439"
version = "0.4.5"
weakdeps = ["FixedPointNumbers"]

    [deps.Ratios.extensions]
    RatiosFixedPointNumbersExt = "FixedPointNumbers"

[[deps.RecipesBase]]
deps = ["PrecompileTools"]
git-tree-sha1 = "5c3d09cc4f31f5fc6af001c250bf1278733100ff"
uuid = "3cdcf5f2-1ef4-517c-9805-6587b60abb01"
version = "1.3.4"

[[deps.RecipesPipeline]]
deps = ["Dates", "NaNMath", "PlotUtils", "PrecompileTools", "RecipesBase"]
git-tree-sha1 = "45cf9fd0ca5839d06ef333c8201714e888486342"
uuid = "01d81517-befc-4cb6-b9ec-a95719d0359c"
version = "0.6.12"

[[deps.Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[deps.RelocatableFolders]]
deps = ["SHA", "Scratch"]
git-tree-sha1 = "90bc7a7c96410424509e4263e277e43250c05691"
uuid = "05181044-ff0b-4ac5-8273-598c1e38db00"
version = "1.0.0"

[[deps.Requires]]
deps = ["UUIDs"]
git-tree-sha1 = "838a3a4188e2ded87a4f9f184b4b0d78a1e91cb7"
uuid = "ae029012-a4dd-5104-9daa-d747884805df"
version = "1.3.0"

[[deps.Revise]]
deps = ["CodeTracking", "Distributed", "FileWatching", "JuliaInterpreter", "LibGit2", "LoweredCodeUtils", "OrderedCollections", "Pkg", "REPL", "Requires", "UUIDs", "Unicode"]
git-tree-sha1 = "1e597b93700fa4045d7189afa7c004e0584ea548"
uuid = "295af30f-e4ad-537b-8983-00126c2a3abe"
version = "3.5.3"

[[deps.Rmath]]
deps = ["Random", "Rmath_jll"]
git-tree-sha1 = "f65dcb5fa46aee0cf9ed6274ccbd597adc49aa7b"
uuid = "79098fc4-a85e-5d69-aa6a-4863f24498fa"
version = "0.7.1"

[[deps.Rmath_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "6ed52fdd3382cf21947b15e8870ac0ddbff736da"
uuid = "f50d1b31-88e8-58de-be2c-1cc44531875f"
version = "0.4.0+0"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"
version = "0.7.0"

[[deps.Scratch]]
deps = ["Dates"]
git-tree-sha1 = "30449ee12237627992a99d5e30ae63e4d78cd24a"
uuid = "6c6a2e73-6563-6170-7368-637461726353"
version = "1.2.0"

[[deps.SentinelArrays]]
deps = ["Dates", "Random"]
git-tree-sha1 = "04bdff0b09c65ff3e06a05e3eb7b120223da3d39"
uuid = "91c51154-3ec4-41a3-a24f-3f23e20d615c"
version = "1.4.0"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"

[[deps.SharedArrays]]
deps = ["Distributed", "Mmap", "Random", "Serialization"]
uuid = "1a1011a3-84de-559e-8e89-a11a2f7dc383"

[[deps.Showoff]]
deps = ["Dates", "Grisu"]
git-tree-sha1 = "91eddf657aca81df9ae6ceb20b959ae5653ad1de"
uuid = "992d4aef-0814-514b-bc4d-f2e9a6c4116f"
version = "1.0.3"

[[deps.SimpleBufferStream]]
git-tree-sha1 = "874e8867b33a00e784c8a7e4b60afe9e037b74e1"
uuid = "777ac1f9-54b0-4bf8-805c-2214025038e7"
version = "1.1.0"

[[deps.Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"

[[deps.SortingAlgorithms]]
deps = ["DataStructures"]
git-tree-sha1 = "c60ec5c62180f27efea3ba2908480f8055e17cee"
uuid = "a2af1166-a08f-5f64-846c-94a0d3cef48c"
version = "1.1.1"

[[deps.SparseArrays]]
deps = ["Libdl", "LinearAlgebra", "Random", "Serialization", "SuiteSparse_jll"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

[[deps.SpecialFunctions]]
deps = ["IrrationalConstants", "LogExpFunctions", "OpenLibm_jll", "OpenSpecFun_jll"]
git-tree-sha1 = "e2cfc4012a19088254b3950b85c3c1d8882d864d"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "2.3.1"
weakdeps = ["ChainRulesCore"]

    [deps.SpecialFunctions.extensions]
    SpecialFunctionsChainRulesCoreExt = "ChainRulesCore"

[[deps.StaticArrays]]
deps = ["LinearAlgebra", "Random", "StaticArraysCore"]
git-tree-sha1 = "9cabadf6e7cd2349b6cf49f1915ad2028d65e881"
uuid = "90137ffa-7385-5640-81b9-e52037218182"
version = "1.6.2"
weakdeps = ["Statistics"]

    [deps.StaticArrays.extensions]
    StaticArraysStatisticsExt = "Statistics"

[[deps.StaticArraysCore]]
git-tree-sha1 = "36b3d696ce6366023a0ea192b4cd442268995a0d"
uuid = "1e83bf80-4336-4d27-bf5d-d5a4f845583c"
version = "1.4.2"

[[deps.Statistics]]
deps = ["LinearAlgebra", "SparseArrays"]
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
version = "1.9.0"

[[deps.StatsAPI]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "45a7769a04a3cf80da1c1c7c60caf932e6f4c9f7"
uuid = "82ae8749-77ed-4fe6-ae5f-f523153014b0"
version = "1.6.0"

[[deps.StatsBase]]
deps = ["DataAPI", "DataStructures", "LinearAlgebra", "LogExpFunctions", "Missings", "Printf", "Random", "SortingAlgorithms", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "75ebe04c5bed70b91614d684259b661c9e6274a4"
uuid = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
version = "0.34.0"

[[deps.StatsFuns]]
deps = ["HypergeometricFunctions", "IrrationalConstants", "LogExpFunctions", "Reexport", "Rmath", "SpecialFunctions"]
git-tree-sha1 = "f625d686d5a88bcd2b15cd81f18f98186fdc0c9a"
uuid = "4c63d2b9-4356-54db-8cca-17b64c39e42c"
version = "1.3.0"

    [deps.StatsFuns.extensions]
    StatsFunsChainRulesCoreExt = "ChainRulesCore"
    StatsFunsInverseFunctionsExt = "InverseFunctions"

    [deps.StatsFuns.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    InverseFunctions = "3587e190-3f89-42d0-90ee-14403ec27112"

[[deps.StatsPlots]]
deps = ["AbstractFFTs", "Clustering", "DataStructures", "Distributions", "Interpolations", "KernelDensity", "LinearAlgebra", "MultivariateStats", "NaNMath", "Observables", "Plots", "RecipesBase", "RecipesPipeline", "Reexport", "StatsBase", "TableOperations", "Tables", "Widgets"]
git-tree-sha1 = "9115a29e6c2cf66cf213ccc17ffd61e27e743b24"
uuid = "f3b207a7-027a-5e70-b257-86293d7955fd"
version = "0.15.6"

[[deps.SuiteSparse]]
deps = ["Libdl", "LinearAlgebra", "Serialization", "SparseArrays"]
uuid = "4607b0f0-06f3-5cda-b6b1-a6196a1729e9"

[[deps.SuiteSparse_jll]]
deps = ["Artifacts", "Libdl", "Pkg", "libblastrampoline_jll"]
uuid = "bea87d4a-7f5b-5778-9afe-8cc45184846c"
version = "5.10.1+6"

[[deps.TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"
version = "1.0.3"

[[deps.TableOperations]]
deps = ["SentinelArrays", "Tables", "Test"]
git-tree-sha1 = "e383c87cf2a1dc41fa30c093b2a19877c83e1bc1"
uuid = "ab02a1b2-a7df-11e8-156e-fb1833f50b87"
version = "1.2.0"

[[deps.TableTraits]]
deps = ["IteratorInterfaceExtensions"]
git-tree-sha1 = "c06b2f539df1c6efa794486abfb6ed2022561a39"
uuid = "3783bdb8-4a98-5b6b-af9a-565f29a5fe9c"
version = "1.0.1"

[[deps.Tables]]
deps = ["DataAPI", "DataValueInterfaces", "IteratorInterfaceExtensions", "LinearAlgebra", "OrderedCollections", "TableTraits", "Test"]
git-tree-sha1 = "1544b926975372da01227b382066ab70e574a3ec"
uuid = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"
version = "1.10.1"

[[deps.Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"
version = "1.10.0"

[[deps.TensorCore]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "1feb45f88d133a655e001435632f019a9a1bcdb6"
uuid = "62fd8b95-f654-4bbd-a8a5-9c27f68ccd50"
version = "0.1.1"

[[deps.Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[deps.TranscodingStreams]]
deps = ["Random", "Test"]
git-tree-sha1 = "9a6ae7ed916312b41236fcef7e0af564ef934769"
uuid = "3bb67fe8-82b1-5028-8e26-92a6c54297fa"
version = "0.9.13"

[[deps.Tricks]]
git-tree-sha1 = "aadb748be58b492045b4f56166b5188aa63ce549"
uuid = "410a4b4d-49e4-4fbc-ab6d-cb71b17b3775"
version = "0.1.7"

[[deps.URIs]]
git-tree-sha1 = "b7a5e99f24892b6824a954199a45e9ffcc1c70f0"
uuid = "5c2747f8-b7ea-4ff2-ba2e-563bfd36b1d4"
version = "1.5.0"

[[deps.UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"

[[deps.Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"

[[deps.UnicodeFun]]
deps = ["REPL"]
git-tree-sha1 = "53915e50200959667e78a92a418594b428dffddf"
uuid = "1cfade01-22cf-5700-b092-accc4b62d6e1"
version = "0.4.1"

[[deps.Unitful]]
deps = ["Dates", "LinearAlgebra", "Random"]
git-tree-sha1 = "a72d22c7e13fe2de562feda8645aa134712a87ee"
uuid = "1986cc42-f94f-5a68-af5c-568840ba703d"
version = "1.17.0"

    [deps.Unitful.extensions]
    ConstructionBaseUnitfulExt = "ConstructionBase"
    InverseFunctionsUnitfulExt = "InverseFunctions"

    [deps.Unitful.weakdeps]
    ConstructionBase = "187b0558-2788-49d3-abe0-74a17ed4e7c9"
    InverseFunctions = "3587e190-3f89-42d0-90ee-14403ec27112"

[[deps.UnitfulLatexify]]
deps = ["LaTeXStrings", "Latexify", "Unitful"]
git-tree-sha1 = "e2d817cc500e960fdbafcf988ac8436ba3208bfd"
uuid = "45397f5d-5981-4c77-b2b3-fc36d6e9b728"
version = "1.6.3"

[[deps.Unzip]]
git-tree-sha1 = "ca0969166a028236229f63514992fc073799bb78"
uuid = "41fe7b60-77ed-43a1-b4f0-825fd5a5650d"
version = "0.2.0"

[[deps.Wayland_jll]]
deps = ["Artifacts", "EpollShim_jll", "Expat_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Pkg", "XML2_jll"]
git-tree-sha1 = "ed8d92d9774b077c53e1da50fd81a36af3744c1c"
uuid = "a2964d1f-97da-50d4-b82a-358c7fce9d89"
version = "1.21.0+0"

[[deps.Wayland_protocols_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4528479aa01ee1b3b4cd0e6faef0e04cf16466da"
uuid = "2381bf8a-dfd0-557d-9999-79630e7b1b91"
version = "1.25.0+0"

[[deps.Widgets]]
deps = ["Colors", "Dates", "Observables", "OrderedCollections"]
git-tree-sha1 = "fcdae142c1cfc7d89de2d11e08721d0f2f86c98a"
uuid = "cc8bc4a8-27d6-5769-a93b-9d913e69aa62"
version = "0.6.6"

[[deps.WoodburyMatrices]]
deps = ["LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "de67fa59e33ad156a590055375a30b23c40299d3"
uuid = "efce3f68-66dc-5838-9240-27a6d6f5f9b6"
version = "0.5.5"

[[deps.XML2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libiconv_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "93c41695bc1c08c46c5899f4fe06d6ead504bb73"
uuid = "02c8fc9c-b97f-50b9-bbe4-9be30ff0a78a"
version = "2.10.3+0"

[[deps.XSLT_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libgcrypt_jll", "Libgpg_error_jll", "Libiconv_jll", "Pkg", "XML2_jll", "Zlib_jll"]
git-tree-sha1 = "91844873c4085240b95e795f692c4cec4d805f8a"
uuid = "aed1982a-8fda-507f-9586-7b0439959a61"
version = "1.1.34+0"

[[deps.XZ_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "cf2c7de82431ca6f39250d2fc4aacd0daa1675c0"
uuid = "ffd25f8a-64ca-5728-b0f7-c24cf3aae800"
version = "5.4.4+0"

[[deps.Xorg_libX11_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libxcb_jll", "Xorg_xtrans_jll"]
git-tree-sha1 = "afead5aba5aa507ad5a3bf01f58f82c8d1403495"
uuid = "4f6342f7-b3d2-589e-9d20-edeb45f2b2bc"
version = "1.8.6+0"

[[deps.Xorg_libXau_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "6035850dcc70518ca32f012e46015b9beeda49d8"
uuid = "0c0b7dd1-d40b-584c-a123-a41640f87eec"
version = "1.0.11+0"

[[deps.Xorg_libXcursor_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXfixes_jll", "Xorg_libXrender_jll"]
git-tree-sha1 = "12e0eb3bc634fa2080c1c37fccf56f7c22989afd"
uuid = "935fb764-8cf2-53bf-bb30-45bb1f8bf724"
version = "1.2.0+4"

[[deps.Xorg_libXdmcp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "34d526d318358a859d7de23da945578e8e8727b7"
uuid = "a3789734-cfe1-5b06-b2d0-1dd0d9d62d05"
version = "1.1.4+0"

[[deps.Xorg_libXext_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "b7c0aa8c376b31e4852b360222848637f481f8c3"
uuid = "1082639a-0dae-5f34-9b06-72781eeb8cb3"
version = "1.3.4+4"

[[deps.Xorg_libXfixes_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "0e0dc7431e7a0587559f9294aeec269471c991a4"
uuid = "d091e8ba-531a-589c-9de9-94069b037ed8"
version = "5.0.3+4"

[[deps.Xorg_libXi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll", "Xorg_libXfixes_jll"]
git-tree-sha1 = "89b52bc2160aadc84d707093930ef0bffa641246"
uuid = "a51aa0fd-4e3c-5386-b890-e753decda492"
version = "1.7.10+4"

[[deps.Xorg_libXinerama_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll"]
git-tree-sha1 = "26be8b1c342929259317d8b9f7b53bf2bb73b123"
uuid = "d1454406-59df-5ea1-beac-c340f2130bc3"
version = "1.1.4+4"

[[deps.Xorg_libXrandr_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll", "Xorg_libXrender_jll"]
git-tree-sha1 = "34cea83cb726fb58f325887bf0612c6b3fb17631"
uuid = "ec84b674-ba8e-5d96-8ba1-2a689ba10484"
version = "1.5.2+4"

[[deps.Xorg_libXrender_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "19560f30fd49f4d4efbe7002a1037f8c43d43b96"
uuid = "ea2f1a96-1ddc-540d-b46f-429655e07cfa"
version = "0.9.10+4"

[[deps.Xorg_libpthread_stubs_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "8fdda4c692503d44d04a0603d9ac0982054635f9"
uuid = "14d82f49-176c-5ed1-bb49-ad3f5cbd8c74"
version = "0.1.1+0"

[[deps.Xorg_libxcb_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "XSLT_jll", "Xorg_libXau_jll", "Xorg_libXdmcp_jll", "Xorg_libpthread_stubs_jll"]
git-tree-sha1 = "b4bfde5d5b652e22b9c790ad00af08b6d042b97d"
uuid = "c7cfdc94-dc32-55de-ac96-5a1b8d977c5b"
version = "1.15.0+0"

[[deps.Xorg_libxkbfile_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll"]
git-tree-sha1 = "730eeca102434283c50ccf7d1ecdadf521a765a4"
uuid = "cc61e674-0454-545c-8b26-ed2c68acab7a"
version = "1.1.2+0"

[[deps.Xorg_xcb_util_image_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "0fab0a40349ba1cba2c1da699243396ff8e94b97"
uuid = "12413925-8142-5f55-bb0e-6d7ca50bb09b"
version = "0.4.0+1"

[[deps.Xorg_xcb_util_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libxcb_jll"]
git-tree-sha1 = "e7fd7b2881fa2eaa72717420894d3938177862d1"
uuid = "2def613f-5ad1-5310-b15b-b15d46f528f5"
version = "0.4.0+1"

[[deps.Xorg_xcb_util_keysyms_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "d1151e2c45a544f32441a567d1690e701ec89b00"
uuid = "975044d2-76e6-5fbe-bf08-97ce7c6574c7"
version = "0.4.0+1"

[[deps.Xorg_xcb_util_renderutil_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "dfd7a8f38d4613b6a575253b3174dd991ca6183e"
uuid = "0d47668e-0667-5a69-a72c-f761630bfb7e"
version = "0.3.9+1"

[[deps.Xorg_xcb_util_wm_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "e78d10aab01a4a154142c5006ed44fd9e8e31b67"
uuid = "c22f9ab0-d5fe-5066-847c-f4bb1cd4e361"
version = "0.4.1+1"

[[deps.Xorg_xkbcomp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libxkbfile_jll"]
git-tree-sha1 = "330f955bc41bb8f5270a369c473fc4a5a4e4d3cb"
uuid = "35661453-b289-5fab-8a00-3d9160c6a3a4"
version = "1.4.6+0"

[[deps.Xorg_xkeyboard_config_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_xkbcomp_jll"]
git-tree-sha1 = "691634e5453ad362044e2ad653e79f3ee3bb98c3"
uuid = "33bec58e-1273-512f-9401-5d533626f822"
version = "2.39.0+0"

[[deps.Xorg_xtrans_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "e92a1a012a10506618f10b7047e478403a046c77"
uuid = "c5fb5394-a638-5e4d-96e5-b29de1b5cf10"
version = "1.5.0+0"

[[deps.Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"
version = "1.2.13+0"

[[deps.Zstd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "49ce682769cd5de6c72dcf1b94ed7790cd08974c"
uuid = "3161d3a3-bdf6-5164-811a-617609db77b4"
version = "1.5.5+0"

[[deps.fzf_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "868e669ccb12ba16eaf50cb2957ee2ff61261c56"
uuid = "214eeab7-80f7-51ab-84ad-2988db7cef09"
version = "0.29.0+0"

[[deps.libaom_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "3a2ea60308f0996d26f1e5354e10c24e9ef905d4"
uuid = "a4ae2306-e953-59d6-aa16-d00cac43593b"
version = "3.4.0+0"

[[deps.libass_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "HarfBuzz_jll", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "5982a94fcba20f02f42ace44b9894ee2b140fe47"
uuid = "0ac62f75-1d6f-5e53-bd7c-93b484bb37c0"
version = "0.15.1+0"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"
version = "5.8.0+0"

[[deps.libfdk_aac_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "daacc84a041563f965be61859a36e17c4e4fcd55"
uuid = "f638f0a6-7fb0-5443-88ba-1cc74229b280"
version = "2.0.2+0"

[[deps.libpng_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "94d180a6d2b5e55e447e2d27a29ed04fe79eb30c"
uuid = "b53b4c65-9356-5827-b1ea-8c7a1a84506f"
version = "1.6.38+0"

[[deps.libvorbis_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Ogg_jll", "Pkg"]
git-tree-sha1 = "b910cb81ef3fe6e78bf6acee440bda86fd6ae00c"
uuid = "f27f6e37-5d2b-51aa-960f-b287f2bc3b7a"
version = "1.3.7+1"

[[deps.nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"
version = "1.52.0+1"

[[deps.p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
version = "17.4.0+0"

[[deps.x264_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4fea590b89e6ec504593146bf8b988b2c00922b2"
uuid = "1270edf5-f2f9-52d2-97e9-ab00b5d0237a"
version = "2021.5.5+0"

[[deps.x265_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "ee567a171cce03570d77ad3a43e90218e38937a9"
uuid = "dfaa095f-4041-5dcd-9319-2fabd8486b76"
version = "3.5.0+0"

[[deps.xkbcommon_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Wayland_jll", "Wayland_protocols_jll", "Xorg_libxcb_jll", "Xorg_xkeyboard_config_jll"]
git-tree-sha1 = "9ebfc140cc56e8c2156a15ceac2f0302e327ac0a"
uuid = "d8fb68d0-12a3-5cfd-a85a-d49703b185fd"
version = "1.4.1+0"
"""

# ╔═╡ Cell order:
# ╟─120a282a-91c1-11ec-346f-25d56e50d38c
# ╟─06d45497-b465-4370-8411-9651e33e70e6
# ╟─48bf8355-1bfe-41a1-97fc-5e2d0470a6c3
# ╟─646dd3d8-6092-4435-aee9-01fa6a281bdc
# ╟─6f051fad-2c4b-4a9e-9361-e9b62ba189c5
# ╟─be9bcfcb-7ec7-4851-bd3f-24d4c29462fe
# ╟─e61193e8-3f91-455b-9e33-101de4cfd4a3
# ╟─234a8b66-d84a-4f6f-aad2-60db238c1850
# ╟─203b859f-f6ea-4ff8-be1b-dc68a44c10d1
# ╟─edf96a48-1e6d-4866-85ba-7e1afd19ca61
# ╟─76d4b64b-471f-4688-9da2-f896d9d0d05c
# ╟─e136edb5-3e98-4355-83e2-55761eb8b15c
# ╟─f8b54f24-c144-4ac5-8083-706436a368e9
# ╟─abd46485-03c9-4077-8ace-34f9b897bb04
# ╟─3ea57a8e-8d15-4f41-acfb-e3bd1d65e585
# ╟─08c17e06-b9f4-40de-acb1-5481fa53e5e8
# ╟─cc38bdc2-4c1b-4a08-b8e6-1504a11924a5
# ╟─77f0d45a-c658-4143-ad14-3a3a20ce2b4b
# ╟─eb1c7d49-fffc-44ab-9d1b-026d7a39fc15
# ╟─0ee59618-d28c-4929-bb7c-cd590068c7b7
# ╟─4c2cf2db-0042-4074-a740-c6458c552ebe
# ╟─c63b4e0d-9f64-4248-ac7b-75ceeedddf0c
# ╟─38e89872-c1ff-4d50-ab99-1ed063f5118b
# ╟─2831de52-9f49-494e-8b2b-4cee810cee06
# ╟─7c9b404e-ef6c-4ed1-910e-2f409e579309
# ╟─434f1bf9-b6d2-4029-9ffa-36241c6ce23f
# ╟─b81503f3-1caf-498a-8dc9-639ea5a5d569
# ╟─63ce270b-c5ed-45c2-ae96-2146d0487b87
# ╟─99a717ee-1b29-42c3-9ef7-80201e715e79
# ╟─188b9f9d-78e3-4e78-8eae-795a9b012cab
# ╟─f6bb31fd-d30c-4e59-86bb-ebac1163609f
# ╟─29b0734a-41d4-43e0-938a-3ced8e70b85b
# ╟─f2b7b63a-fd54-4f80-af28-f255d186f6f9
# ╟─61fb0322-07f6-4e59-a5d3-dfdd7399d558
# ╟─bd670cde-bcbe-4104-8114-34f761f7f850
# ╟─e1ccac4e-1cba-43d2-b0e0-e8c650a79d9a
# ╟─ceaedbd8-6c88-407b-84f6-aedc0a59a22d
# ╟─00d85bc5-c320-41dc-8a85-4ad409b058a8
# ╟─8efbba52-7a34-4c17-bca9-52a417f69d56
# ╟─3873ef48-34c1-479a-b6f8-2cd05bcdc8bb
# ╟─a2b02ca9-6edc-48bf-b5e5-746d986d8252
# ╟─5276ff0e-bc4f-4b24-b98c-8f9e7c0d018d
# ╟─dd461997-cf3c-4124-8503-11f5b5a9e530
# ╟─20e84d48-0f5f-403e-a8aa-1cbd11cd3b04
# ╟─d2365f95-9b61-41fa-83d8-9bf4a8b5ee92
# ╟─b5dc223b-c5b5-4eda-889e-c39171c44499
# ╟─8d86466e-93c5-4263-b436-6fe684eeccd8
# ╟─a391d4a3-9fe3-4ccf-9a62-c2cb19ea8813
# ╟─a0ebfea7-bbe5-4798-8db3-163381bf67c2
# ╟─d8e832e1-bc53-400d-a5a9-7cda2c5b8799
# ╠═a414e554-3a8c-472d-af82-07c2f0843627
# ╠═33cc44b4-bd32-4112-bf33-6807ae53818c
# ╟─8c803aa8-de06-4194-bb55-67af083cfd74
# ╟─bba94ae7-a6a8-4958-bcf5-9380864a9147
# ╟─4dc678ce-8f35-44c7-b58c-628345350b49
# ╟─9f18a392-8edb-4f4e-a468-6d1d735d05bf
# ╟─40d1d8ec-3875-49c8-9425-f5cb6765603b
# ╟─c4c51b3c-c0d3-44d7-b5e8-1a65c3e737cc
# ╟─ba06623e-a9e6-4f39-8c4d-d8f86c765a8c
# ╟─542b0f6f-befa-4dfe-acb7-b7aa6ab9f56c
# ╟─ab0c4765-e9a0-4ba3-8a21-e81b214ab060
# ╟─05de404f-d2d7-4feb-ba70-a25d8eba71d4
# ╟─eb273a3a-60d6-439e-bb63-6af8130b4ec0
# ╟─b44fce15-1056-4f42-9ec7-7a70092c355b
# ╟─533efc1a-fd99-4a5d-9ef3-4ff7ce483f24
# ╟─c46949ad-46c8-4bcb-9e83-1eb646b4ea2a
# ╟─adb47976-1d3c-42d9-941c-8a791866eb97
# ╟─c8cb2535-7b65-4d86-9405-71880fdb906e
# ╟─caf21a20-8dd8-4a31-8463-96d3869d3ee9
# ╟─5f9ad998-c410-4358-925b-66e5d3b2f9e9
# ╟─9612d7c1-3c86-45bc-806a-a53bbf869d87
# ╟─9cbc7ef6-788c-4dfd-a98d-2a9409eb5127
# ╟─ca7f12d8-f823-448b-ab1f-0e28c85a3f7f
# ╟─6569c4e1-5d62-42ad-94c0-927dd6b6f504
# ╟─8d2778fd-edfe-4a02-92f9-2b9a67f2a2c0
# ╟─8107c1d9-58c4-41f0-8870-bfd7084e42b5
# ╟─f6b06459-0608-4981-bdb4-8f54d0cadfe1
# ╟─d9eaf592-1522-4f7e-a7db-cf503a152eb9
# ╟─ad943140-9510-463e-b57c-4956ed2cd5d5
# ╟─c7fd532d-d72a-439a-9e71-e85392c66f8c
# ╟─e0cfcb9b-794b-4731-abf7-5435f67ced42
# ╟─aabdfd01-c9b8-4c45-83c3-29881581e6e3
# ╟─bbd25bdf-e82d-4f65-bfad-7d8e8e9cca18
# ╟─4cc1f89c-6e1d-4457-9e84-fd5bdd0b575a
# ╟─a0465ae8-c843-4fc0-abaf-0497ada26652
# ╟─dafd1a68-715b-4f06-a4f2-287c123761f8
# ╠═76859d4c-f3e2-4576-b4d6-b637e9c99877
# ╠═620789b7-59bc-4e17-bcfb-728a329eed0f
# ╠═7b47cda6-d772-468c-a8f3-75e3d77369d8
# ╠═8d0c6fdc-4717-4203-b933-4b37fe60d512
# ╠═8d06ce32-2c8d-4317-8c38-108ec0e7fe23
# ╟─d66e373d-8443-4810-9332-305d9781a21a
# ╠═acfb80f0-f4d0-4870-b401-6e26c1c99e45
# ╠═e091ce93-9526-4c7f-9f14-7634419bfe57
# ╠═5a8cdbe7-6abe-4f07-8bcc-89dd71fc35f7
# ╠═c46e0b36-c3fd-4b7f-8f31-25c3315bb10c
# ╠═d44526f4-3051-47ee-8b63-f5e694c2e609
# ╠═27755688-f647-48e5-a939-bb0fa70c95d8
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
