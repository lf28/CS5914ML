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
	using Plots, Distributions,Random, StatsFuns, StatsBase, Clustering, LinearAlgebra
	using PlutoUI
	using StatsPlots
end

# ╔═╡ 0ae6c88f-3874-4cf7-86ab-28e400d0aca9
md"""
##### Initializing packages

*When running this notebook for the first time, this could take up to 10 minutes. Hang in there !*
"""

# ╔═╡ 75361c09-f201-457f-8081-25074ebe1b90
Random.seed!(1)

# ╔═╡ 2e8fcd55-70f9-4b7b-87dd-7e7d2413eb6f
md"""

> I recommend you download and run this Pluto notebook (on studres) when you study or review the lecture. A quick reference for Julia can be found here: [minimum starting syntax](https://computationalthinking.mit.edu/Spring21/basic_syntax/), and [getting-started with Julia](https://syl1.gitbook.io/julia-language-a-concise-tutorial/language-core/getting-started). Note that programming in Julia is not required for this module and Julia programs are only used to render the slides. With that said, it should greatly help you understand the mathematics by checking and understanding the code.
"""

# ╔═╡ 646dd3d8-6092-4435-aee9-01fa6a281bdc
html"<button onclick='present()'>present</button>"

# ╔═╡ 4f95e17a-9fbf-4a38-a7ad-f5ad31b1cfd0
md"""
# CS5014 Machine Learning
#### Lecture 11. Unsupervised learning (Clustering)
###### Lei Fang 
"""

# ╔═╡ cd91d8a9-b208-4e26-8c72-655a28c93b61
md"""

## Roadmap for the next two lectures

Unsupervised learning is a vast subject 
* clustering
* density estimation
* dimension reduction
* hidden representation learning
* etc.

We will talk about **clustering** (or **density estimation**) only
* the unsupervised learning counterpart of classification
* will adopt a **probabilistic approach**




"""

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

# ╔═╡ 8a217d25-e231-43a8-94b9-538f138dd331
md"""
## Today's topic

* Clustering problem

* A first clustering algorithm: K-means 

* Discover the probabilistic model behind K-means
  * finite mixture of Gaussians (actually the same model as QDA)
  * EM algorthm for mixture model

* Generalisation of mixture model (if we have time)


"""

# ╔═╡ abd46485-03c9-4077-8ace-34f9b897bb04
md"""

## Clustering

**Classification**: input data ${X}=\{x^1, x^2, \ldots, x^n\}$ and their targets $Y=\{y^1, y^2, \ldots, y^n\}$

  - ``x^i \in \mathbb{R}^d``: ``d`` dimensional input
  - ``y^i \in \{1,2,\ldots,K\}``: i.e. ``K`` class classification

"""

# ╔═╡ 3ea57a8e-8d15-4f41-acfb-e3bd1d65e585
begin
	K₁ =3
	n₁ = 600
	# D₁ = zeros(n₁, 2)
	# 200 per cluster
	truezs₁ = repeat(1:K₁; inner=200)
	trueμs₁ = zeros(2, K₁)
	trueμs₁[:,1] = [-3.0, 2.0]
	trueμs₁[:,2] = [3.0, 2.0]
	trueμs₁[:,3] = [0., -2]
	data₁ = trueμs₁[:,1]' .+ randn(200, 2)
	data₁ = vcat(data₁, trueμs₁[:,2]' .+ randn(200, 2))
	data₁ = vcat(data₁, trueμs₁[:,3]' .+ randn(200, 2))
	plt₁ = plot(ratio=1)
	for k in 1:K₁
		scatter!(data₁[truezs₁ .== k,1], data₁[truezs₁ .==k,2], label="Class"*string(k)) 
	end
	title!(plt₁, "Classification (dataset with labels)")
end

# ╔═╡ cc38bdc2-4c1b-4a08-b8e6-1504a11924a5
md"""
##

**Clustering**: input features ${X}=\{x^1, x^2, \ldots, x^n\}$ without targets
* figuratively, clustering is to color the dataset into $K$ groups ($K=3$ here seems reasonable)
"""

# ╔═╡ 8cb79133-2339-4366-b881-098d83449aee
plot(data₁[:,1], data₁[:,2], ratio=1, st=:scatter, label= "x",title="Clustering (dataset without labels)")

# ╔═╡ 2831de52-9f49-494e-8b2b-4cee810cee06
md"""

## K-means 


A very simple and intuitive clustering algorithm

**Initialisation**: start with some random guess of $K$ centroids $\mu_k$

**Repeat** the following two steps until converge:

**assignment step**: assign each ${x}^{i}$ to the closest centroid ${\mu}_k$: for $i= 1\ldots n,\; k= 1\ldots K$:

$${z}^{i} \leftarrow \arg\min_{k} \|{x}^{i} - {\mu}_k\|^2$$ 

where $$\|{x}^i - {\mu}_k\|^2 = \sum_{j=1}^d (x^i_j - \mu_{kj})^2$$, i.e. (square) Euclidean distance between $x^i, \mu_k$

**update step**: update ${\mu}_k$ by averaging the assigned samples
    - ``I`` indicator function: return ``1`` if true; ``0`` otherwise  
	
$${\mu}_{k} \leftarrow  \frac{\sum_{i=1}^n I({z}^{i}=k)\cdot {x}^{i}}{\sum_{i=1}^n I({z}^{i}=k)}\;$$


Convergence check on the sum of (assigned) distances 

$$\text{loss} = \sum_{i=1}^n \|x^i - \mu_{z_i}\|^2 = \sum_{i=1}^n\sum_{k=1}^K I(z^i=k) \cdot \|x^i - \mu_{k}\|^2$$

* distances between $x^i$ and its assigned cluster
* the loss will only decrease over the iterations
  * can be used to test/debug your implementation
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
		_,d = size(D)
		μs = zeros(d,K)
		# update
		for k in 1:K
			μₖ = mean(D[zs.==k,:], dims=1)[:]
			μs[:,k] = μₖ
		end
		return μs
	end
end

# ╔═╡ 7c9b404e-ef6c-4ed1-910e-2f409e579309
md"""

## A demonstration of K-means

"""

# ╔═╡ b81503f3-1caf-498a-8dc9-639ea5a5d569
begin
	gr()
	# random initialisation step: randn: zero mean Gaussian
	ms = randn(2,K₁)
	plt_demo = plot(data₁[:,1], data₁[:,2], ratio=1, st=:scatter, label= "data",title="K means initialisation; Iteration=0")
	for k in 1:K₁
		scatter!([ms[1,k]], [ms[2,k]], label="μ"*string(k), markerstrokewidth =3, markershape = :star4, color=k, markersize = 10)
	end
	plt_demo
end

# ╔═╡ f2b7b63a-fd54-4f80-af28-f255d186f6f9
begin
	gr()
	pltsₖₘ = []
	anim2 = @animate for iter in 1:5
		ls, initzs = assignment_step(data₁, ms)
	    p = plot_clusters(data₁, initzs, K₁, sum(ls), iter)
		push!(pltsₖₘ, p)
		for k in 1:K₁
			scatter!([ms[1,k]], [ms[2,k]], markershape = :star4, markerstrokewidth =3, markersize = 10, c= k, labels= "μ"*string(k))
		end
		ms = update_step(data₁, initzs, K₁)
	end
end;

# ╔═╡ 11519291-da33-4bb2-b0e4-cffe347cf085
pltsₖₘ

# ╔═╡ 474bace2-9586-45d3-b674-73806bbc85b8
gif(anim2, fps = 1.0)

# ╔═╡ e1ccac4e-1cba-43d2-b0e0-e8c650a79d9a
md"""
## A demonstration of K-means (cont.)

"""

# ╔═╡ 33cc44b4-bd32-4112-bf33-6807ae53818c
function kmeans(D, K=3; tol= 1e-4, maxIters= 100, seed= 123)
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
end

# ╔═╡ 00d85bc5-c320-41dc-8a85-4ad409b058a8
loss_km₁, zs_km₁, μs_km₁ = kmeans(data₁, K₁)

# ╔═╡ ceaedbd8-6c88-407b-84f6-aedc0a59a22d
plot(loss_km₁, label="loss", xlab="Iterations", title="Loss trajectory of K-means")

# ╔═╡ 8efbba52-7a34-4c17-bca9-52a417f69d56
md"""

## An alternative K-means formula

We can also start with initialising random guess on the assignments $z^i$
* and swap the order of **assignment** and **update** steps
* should usually converge to the same results
  
Specifically, the algorithm is 

**Initialisation**: random guess of $z^i$ for $i=1,\ldots,n$

$$z^i \leftarrow \text{rand}(1,\ldots,K)$$
**Repeat** until convergence:

  - **update step**: update ${\mu}_k$ by averaging the assigned samples
$${\mu}_{k} \leftarrow  \frac{\sum_{i=1}^n I({z}^{i}=k)\cdot {x}^{i}}{\sum_{i=1}^n I({z}^{i}=k)}\;$$
  * **assignment step**: assign each ${x}^{i}$ to the closest centroid ${\mu}_k$: for $i= 1\ldots n,\; k= 1\ldots K$:
    

$${z}^{i} \leftarrow \arg\min_{k} \|{x}^{i} - {\mu}_k\|^2$$

"""

# ╔═╡ 3873ef48-34c1-479a-b6f8-2cd05bcdc8bb
md"""

## A quick demon of alternative K-means

"""

# ╔═╡ 5276ff0e-bc4f-4b24-b98c-8f9e7c0d018d
begin
	gr()
	pltsₖₘ2 = []
	zs0 = rand(1:K₁, size(data₁)[1])
	l_ = Inf
	anim = @animate for iter in 1:6
		p = plot_clusters(data₁, zs0, K₁, l_, iter)
		ms0 = update_step(data₁, zs0, K₁)
		for k in 1:K₁
			scatter!([ms0[1,k]], [ms0[2,k]], markershape = :star4, markersize = 10, c=k, markerstrokewidth =3, color=k, labels= "μ"*string(k))
		end
		ls, zs0 = assignment_step(data₁, ms0)
		l_ = sum(ls)
		push!(pltsₖₘ2, p)
	end
end

# ╔═╡ 20e84d48-0f5f-403e-a8aa-1cbd11cd3b04
pltsₖₘ2

# ╔═╡ 1954eac7-f155-4c37-9a51-26440d79851d
gif(anim, fps = 1)

# ╔═╡ a391d4a3-9fe3-4ccf-9a62-c2cb19ea8813
md"""
##

For convenience, I have created a method `kmeansDemoGif` to produce gifs for us
(check the code in Appendix)

`kmeansDemoGif(data, K, iterations; init_step="a" or "u", add_contour=true or false)`

* 2-d input only
* produces three types of gifs (more example next)
"""

# ╔═╡ bc645a51-6e38-431d-b67d-1512126840cf
md"""

## Euclidean distance as quadratic form


(Squared) Euclidean distance is used in K-means

$$\|{x} - {\mu}_k\|^2 = \sum_{j=1}^d (x_j - \mu_{kj})^2$$
* or **straightline distance** between two points in d-dimensional Euclidean space $x\in \mathbb{R}^d, \mu_k\in \mathbb{R}^d$
* and ``x = [x_1, x_2, \ldots, x_d]``, and ``\mu_k = [\mu_{k1}, \ldots, \mu_{k}]``
* fix $\mu_k$, $x$ forms a circle in $\mathbb{R}^2$ (or sphere in $\mathbb{R}^3$ or hypersphere for $d>3$) when

$\|{x} - {\mu}_k\|^2= r$

Euclidean distance can also be written in a **quadratic form**

$$\begin{align}\sum_{j=1}^d (x_j - \mu_{kj})^2&=\begin{bmatrix}x_1- \mu_{k1}& \ldots & x_d- \mu_{kd}\end{bmatrix}\begin{bmatrix} x_1-\mu_{k1} \\ \vdots \\ x_d-\mu_{kd}\end{bmatrix} \\
&= \underbrace{(x - \mu_k)^\top}_{\text{row vector}} \underbrace{(x - \mu_k)}_{\text{column vector}}\\ 
&= \boxed{(x - \mu_k)^\top  I  (x - \mu_k)}\end{align}$$

* remember this **quadratic form** notation we will come back to it soon!
  *  **quadratic form**: *a row vector* ``\times`` *a square matrix* ``\times`` *a column vector*
  *  generalisation of scalar quadratic function ``x\cdot a \cdot x``
* where $I$ is the Identity matrix (diagonal matrix with ones on the diagonal entries)
  
$I = \begin{bmatrix} 1 &0& \ldots & 0 \\
  0 &1 & \ldots & 0  \\
  \vdots & \vdots & \vdots & \vdots \\
  0 & 0 & \ldots & 1
  \end{bmatrix}$
"""

# ╔═╡ 542b0f6f-befa-4dfe-acb7-b7aa6ab9f56c
md"""

## Decision boundary of K-means

K-means imposes linear (straightlines) **decision boundaries**

$$x \in R^d \;\; \text{s.t.}\; \|{x} - {\mu}_i\|^2 - \|{x} - {\mu}_j\|^2=0$$

* points $x\in R^d$ with equal distances between centers $\mu_i$ and $\mu_j$
* basic geometry tells us it is a straightline

"""

# ╔═╡ 05de404f-d2d7-4feb-ba70-a25d8eba71d4
md"""

## Limitations of K-means


How do you think K-means would fare on **this dataset** ?

* Euclidean distance (circular distance contour) do not make sense for cluster 3
* ellipse seems better
"""

# ╔═╡ ca7f12d8-f823-448b-ab1f-0e28c85a3f7f
md"""

## Limitations of K-means (cont.)

**How about this dataset ?**

* the datasets are overlapping
* the centers can even collide altogether 
* even axis aligned ellipses do not work here
  * rotated ellipses for all three 
"""

# ╔═╡ 8d2778fd-edfe-4a02-92f9-2b9a67f2a2c0
md"""

## Let's see how K-means fare

Two relatively challenging tasks
"""

# ╔═╡ 8107c1d9-58c4-41f0-8870-bfd7084e42b5
md"""

K-means fails at the boundries
* it does not take into account of cluster 3's variance on the vertical dimension
"""

# ╔═╡ 7f3440ec-5577-443c-8448-bf6cb4aeb1cb
md"""

## Let's see how K-means fare (conti.)

Complete failure!


Euclidean distance completely ignores the hidden structures! 
"""

# ╔═╡ 01544a34-0647-4589-b267-3d440c35d8ba
md"""

## Multivariate Gaussian


"""

# ╔═╡ a221a92c-7939-42de-84cb-85ab220f19f1
md"""

A d-dimensional multivariate Gaussian with mean $\mu$ and covariance $\Sigma$ has density

$$\begin{equation*}
p({x})={N}({x}| {\mu}, {\Sigma}) =  \underbrace{\frac{1}{(2\pi)^{d/2} |\Sigma|^{1/2}}}_{\text{normalising constant}}
\exp \left \{-\frac{1}{2} \underbrace{({x} - {\mu})^\top \Sigma^{-1}({x}-{\mu})}_{\text{distance btw } x \text{ and } \mu }\right\}
\end{equation*}$$

* ``\mu \in R^d`` is the mean vector 
* ``\Sigma`` is a $d$ by $d$ symmetric and *positive definite* matrix 
* the kernel is a **distance measure** between $x$ and $\mu$

  $({x} - {\mu})^\top \Sigma^{-1}({x}-{\mu})$
* the density is negatively related to the distance measure
  * the further away $x$ from $\mu$, then $p(x)$ is smaller
* ``\frac{1}{(2\pi)^{d/2} |{\Sigma}|^{1/2}}`` is the **normalising constant** (does not change w.r.t $x$) such that

$$\int p(x) dx = 1$$
* ``|\Sigma|`` is determinant, which measures the volume under the the bell surface (when it is a positive definite matrix)
"""

# ╔═╡ f93d15ba-300d-45bc-9b6f-6a24ce1be4ad
md"""
## Example when $Σ=I$

For simplicity, let's only consider $R^2$, when $\Sigma = I = \begin{bmatrix} 1 & 0 \\0 & 1\end{bmatrix}$

$({x} - {\mu})^\top I^{-1}({x}-{\mu})=({x} - {\mu})^\top({x}-{\mu})= (x_{1} -\mu_1)^2+ (x_2 - \mu_2)^2$

i.e. Euclidean distance between $x$ and $\mu$

* remember $I^{-1} = I$
"""

# ╔═╡ 5d04b864-15e9-4e91-9928-e9af7181c3f7
begin
	gr()
	μ₁ = [2,2]
	mvn1 = 	MvNormal(μ₁, Matrix(1.0I, 2,2))
	spl1 = rand(mvn1, 500)
	x₁s = μ₁[1]-3:0.1:μ₁[1]+3
	x₂s = μ₁[2]-3:0.1:μ₁[2]+3	
	mvnplt₁ = scatter(spl1[1,:], spl1[2,:], ratio=1, label="")	
	scatter!([μ₁[1]], [μ₁[2]], ratio=1, label="μ", markershape = :diamond, markersize=8)	
	plot!(x₁s, x₂s, (x1, x2)->pdf(mvn1, [x1, x2]), levels=4, linewidth=4, st=:contour, colorbar=false)	

	# plot(mvnplt₁, mvnplt₂)
end

# ╔═╡ dac8fd37-2e61-42d8-b9c1-17c46fa7b9b7
begin
	plotly()
	mvnplt₂ = surface(x₁s, x₂s, (x1, x2)->pdf(mvn1, [x1, x2]), color=:lighttest, st=:surface, colorbar=true)
end

# ╔═╡ e66784ce-ec45-49c6-a1b7-d5d186e147ef
md"""

## Example (Diagonal $Σ$)

When $\Sigma =  \begin{bmatrix} \sigma_1^2 & 0 \\0 & \sigma_2^2\end{bmatrix}$, then $\Sigma^{-1} = \begin{bmatrix} 1/\sigma_1^2 & 0 \\0 & 1/\sigma_2^2\end{bmatrix}$


The distance measure forms (axis aligned) ellipses


$({x} - {\mu})^\top \Sigma^{-1}({x}-{\mu})= \underbrace{\frac{(x_{1} -\mu_1)^2}{\sigma_1^2}+ \frac{(x_2 - \mu_2)^2}{\sigma_2^2}}_{\text{analytical form of an ellipse}}$

* you should verify yourself, note $x-\mu=\begin{bmatrix}x_1-\mu_1\\ x_2-\mu_2\end{bmatrix}$ is a column vector

* the inputs are mutually independent 
"""

# ╔═╡ eb50ce8b-a029-4599-9860-3594488187d0
@bind σ₁² Slider(0.5:0.1:5, default=1)

# ╔═╡ d6651a85-3980-4e30-bb5e-0f770574ca7e
md"``\sigma_1^2``=$(σ₁²)"

# ╔═╡ e5a57b78-77b8-4b07-8451-b37df36da736
@bind σ₂² Slider(0.5:0.1:5, default=1)

# ╔═╡ 0aee031f-b724-4ddb-a288-22650585948e
md"``\sigma_2^2``=$(σ₂²)"

# ╔═╡ fab36b7e-5339-4007-8b84-1ec0b0c3fc23
begin
	gr()
	mvnsample = randn(2, 500);
	μ₂ = [2,2]
	# Σ₂ = [1 0; 0 2]
	Σ₂ = [σ₁² 0; 0 σ₂²]
	L₂ = [sqrt(σ₁²) 0; 0 sqrt(σ₂²)]
	mvn₂ = 	MvNormal(μ₂, Σ₂)
	# μ + L * MvNormal(0, I) = MvNormal(μ, LLᵀ)
	spl₂ = μ₂.+ L₂ * mvnsample
	x₁s_ = μ₂[1]-6:0.1:μ₂[1]+6	
	x₂s_ = μ₂[2]-6:0.1:μ₂[2]+6	
	scatter(spl₂[1,:], spl₂[2,:], ratio=1, label="", xlabel="x₁", ylabel="x₂")	
	scatter!([μ₂[1]], [μ₂[2]], ratio=1, label="μ", markershape = :diamond, markersize=8)	
	plot!(x₁s, x₂s, (x1, x2)->pdf(mvn₂, [x1, x2]), levels=4, linewidth=4, st=:contour)	
end

# ╔═╡ 67e6b025-c9be-4cf8-8231-f8c01177eaf3
begin
	plotly()
	surface(x₁s, x₂s, (x1, x2)->pdf(mvn₂, [x1, x2]), color=:lighttest, st=:surface, colorbar=true, xlabel="x₁", ylabel="x₂", zlabel="p(x)")
end

# ╔═╡ aa93a75d-302c-4ede-b181-06d87a27eb23
md"""
## Example (full $Σ$)

When 

$\Sigma =  \begin{bmatrix} \sigma_1^2 & \sigma_{12} \\ \sigma_{21} & \sigma_2^2\end{bmatrix}$


Gaussian's distance measure forms (rotated) ellipses


$({x} - {\mu})^T \Sigma^{-1}({x}-{\mu})= \underbrace{\frac{(v_1^\top(x-\mu))^2}{\lambda_1} + \frac{(v_2^\top(x-\mu))^2}{\lambda_2}}_{\text{still an analytical form of ellipse}}$

* ``v_1`` and ``v_2`` are the eigen vectors of $\Sigma^{-1}$
* i.e. the rotated ellipse's basis (the $\textcolor{red}{\text{red vectors}}$ in the plot below)
* positive or negatively correlated inputs

"""

# ╔═╡ b5008102-cace-4c43-98eb-c2a8ca5c1c76
@bind σ₁₂ Slider(-1:0.02:1, default=0)

# ╔═╡ ebe5606a-4945-4dda-81d1-bde84a8d4fce
md"``\sigma_{12}=\sigma_{21}``=$(σ₁₂)"

# ╔═╡ 89f4e422-b603-44c9-87db-241457aeb080
begin
	gr()
	# mvnsample = randn(2, 500);
	# μ₂ = [2,2]
	# Σ₂ = [1 0; 0 2]
	Σ₃ = [1 σ₁₂; σ₁₂ 1]
	# cholesky decomposition of Σ (only to reuse the random samples)
	L₃ = cholesky(Σ₃).L
	mvn₃ = 	MvNormal(μ₂, Σ₃)
	# μ + L * MvNormal(0, I) = MvNormal(μ, LLᵀ)
	spl₃ = μ₂.+ L₃ * mvnsample
	# x₁s_ = μ₂[1]-6:0.1:μ₂[1]+6	
	# x₂s_ = μ₂[2]-6:0.1:μ₂[2]+6	
	scatter(spl₃[1,:], spl₃[2,:], ratio=1, label="", xlabel="x₁", ylabel="x₂")	
	scatter!([μ₂[1]], [μ₂[2]], ratio=1, label="μ", markershape = :diamond, markersize=8)	
	λs, vs =eigen(Σ₃)
	v1 = (vs .* λs')[:,1]
	v2 = (vs .* λs')[:,2]
	quiver!([μ₂[1]], [μ₂[2]],quiver=([v1[1]], [v1[2]]), linewidth=4, color=:red)
	quiver!([μ₂[1]], [μ₂[2]],quiver=([v2[1]], [v2[2]]), linewidth=4, color=:red)
	plot!(x₁s, x₂s, (x1, x2)->pdf(mvn₃, [x1, x2]), levels=4, linewidth=4, st=:contour)	
end

# ╔═╡ b449cd92-93de-4c75-bd45-3c8c2cbb8d33
md"""

## *Positive definiteness of $Σ$
When $\sigma_{12} = 1.0$ or $-1$ (or $|\sigma_{12}| >1$), 

$\Sigma = \begin{bmatrix} 1 & 1 \\ 1 & 1\end{bmatrix}\;\; \text{or}\;\; \begin{bmatrix} 1 & -1 \\ -1 & 1\end{bmatrix}$ is no longer positive definite
* one of the dimension collapses (with zero variance!)
* the normalising constant does not exist as $|\Sigma| =0$, therefore the Gaussian does not exist
* similar to the case when $\sigma^2=0$ for univariate Gaussian


**Positive definite**: that is for all $x\in R^d$

$x^\top \Sigma^{-1} x>0$ 

* strictly positive so it is a valid distance measure
"""

# ╔═╡ cc3beb3b-f5de-4571-8210-47e843b54965
md"""

## Review: generative model (QDA)


Recall discriminant analysis: quadratic discriminant analysis (QDA)

In a nutshell, a set of K multivariate Gaussians put together
* each class's input is represented as one Gaussian

Probabilistic model for both **input** $x^i\in R^d$ and its **label** $z^i\in 1,\ldots, K$

$$p(x^i, z^i) = p(z^i) p(x^i|z^i)$$


where  

* prior for $z^i$ ``p(z^i)``: how *popular* that class is in apriori

$$p(z^i=k) = \pi_k$$

* likelihood for $x^i$ ``p(x^i|z^i=k)``: likelihood model for input feature $x^i\in R^d$; given knowing the label, how likely to see a observation $x^i$

$$p(x^i|z^i=k) = N(x^i; \mu_k, \Sigma_k)$$

"""

# ╔═╡ 380b7426-5801-45ab-898a-a72f22f1e953
md"""

## Sampling from a generative model

Given the true parameters $\{\pi, \{\mu_k, \Sigma_k\}_{k=1}^K\}$, we can simulate or sample dataset according to the data generating process

for $i \in 1,\ldots,n$
1. sample $z^i \sim p(z^i)$
2. sample $x^i \sim N(x^i|\mu_{z^i}, \Sigma_{z^i})$
add $\{z^i, x^i\}$ to $D$

Simulated dataset can be very useful for debugging purpose

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

# ╔═╡ 853de250-143e-4add-b50d-2c73d1bc7910
md"""

## Example of QDA

Where assume the true parameters are known:

$\pi = [0.2, 0.2, 0.6]$

$\mu_1 = [-2 , 1]; \Sigma_1 = \begin{bmatrix}0.5, 0\\0, 0.5\end{bmatrix}$
$\mu_2 = [2 , 1]; \Sigma_2 = \begin{bmatrix}0.5, 0\\0, 0.5\end{bmatrix}$
$\mu_3 = [0 , -1]; \Sigma_3 = \begin{bmatrix}0.5, 0\\0, 2\end{bmatrix}$
"""

# ╔═╡ cc60e789-02d1-4944-8ad5-718ede99669c
begin
	K₂ = 3
	trueμs₂ = zeros(2,K₂)
	trueΣs₂ = zeros(2,2,K₂)
	trueμs₂[:,1], trueΣs₂[:,:,1] = [-2.0, 1.0], 0.5 * Matrix(1.0I, 2,2)
	trueμs₂[:,2], trueΣs₂[:,:,2] = [2.0, 1.0], 0.5 * Matrix(1.0I, 2,2)
	trueμs₂[:,3], trueΣs₂[:,:,3] = [0., -1],  Matrix([0.5 0; 0 2])
	trueπs₂ = [0.2, 0.2, 0.6]
	truemvns₂ = [MvNormal(trueμs₂[:,k], trueΣs₂[:,:,k]) for k in 1:K₂]
	n₂= 800
	truezs₂ = rand(Categorical(trueπs₂), n₂)
	data₂= vcat([rand(truemvns₂[z])' for z in truezs₂]...)
	# data₂, truezs₂ = sampleMixGaussian(n₂, truemvns₂, trueπs₂)
end;

# ╔═╡ c8cb2535-7b65-4d86-9405-71880fdb906e
_, zskm₂, ms₂ = kmeans(data₂, K₂);

# ╔═╡ 02ad4708-cb36-45dc-ab2b-34ab22c26ccc
begin
	gr()
	pltqda = plot(title="QDA example with ground truth params", ratio=1)
	xs_qda₂ = minimum(data₂[:,1])-0.1:0.1:maximum(data₂[:,1])+0.1
	ys_qda₂ = minimum(data₂[:,2])-0.1:0.1:maximum(data₂[:,2])+0.1
	for k in 1:K₂
		scatter!(data₂[truezs₂ .==k,1], data₂[truezs₂ .==k, 2], label="class "*string(k), c= k )
		scatter!([truemvns₂[k].μ[1]], [truemvns₂[k].μ[2]], color = k, label = "μ"*string(k), markersize = 10, markershape=:diamond, markerstrokewidth=3)
		contour!(xs_qda₂, ys_qda₂, (x,y)-> pdf(truemvns₂[k], [x,y]), levels=5, colorbar = false, ratio=1,lw=4) 
	end

	pltqda
end

# ╔═╡ 8d71174d-401e-4cf9-8afb-3e8bbd49a0b1
md"""

## Example of QDA 2

Where assume the true parameters are known:

$\pi = [0.25, 0.5, 0.25]$

$\mu_1 = [1 , 1]; \Sigma_1 = \begin{bmatrix}1, -0.9\\-0.9, 1\end{bmatrix}$
$\mu_2 = [0 , 0]; \Sigma_2 = \begin{bmatrix}1, 0.9\\0.9, 1\end{bmatrix}$
$\mu_3 = [-1 , -1]; \Sigma_3 = \begin{bmatrix}1, -0.9\\-0.9, 1\end{bmatrix}$

"""

# ╔═╡ e0cfcb9b-794b-4731-abf7-5435f67ced42
begin
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
end;

# ╔═╡ c7fd532d-d72a-439a-9e71-e85392c66f8c
_, zskm₃, ms₃ = kmeans(data₃, K₃) 

# ╔═╡ c86a3f0d-ea61-4f35-a116-a7aa7ab9bc2d
begin
	gr()
	pltqda₂ = plot(title="QDA example 2 with ground truth params", ratio=1)
	xs_qda₃ = minimum(data₃[:,1])-0.1:0.1:maximum(data₃[:,1])+0.1
	ys_qda₃ = minimum(data₃[:,2])-0.2:0.1:maximum(data₃[:,2])+0.2
	for k in 1:K₃
		scatter!(data₃[truezs₃ .==k,1], data₃[truezs₃ .==k, 2], label="class "*string(k), c= k)
		scatter!([truemvns₃[k].μ[1]], [truemvns₃[k].μ[2]], color = k, label = "μ"*string(k), markersize = 10, markershape=:diamond, markerstrokewidth=3)
		contour!(xs_qda₃, ys_qda₃, (x,y)-> pdf(truemvns₃[k], [x,y]), levels=3, colorbar = false, ratio=1,lw=5) 
	end

	pltqda₂
end

# ╔═╡ 4e038980-c531-4f7c-9c51-4e346eccc0aa
md"""

## (Supervsied) Learning of QDA: MLE

**Learning** or **training**: estimate a model's parameters given *observed* data

We can use maximum likelihood estimation

$$\hat \theta = \arg\max_{\theta} P(D|\theta)$$

* model parameters: ``\theta = \{\pi_k, \mu_k, \Sigma_k\}_{k=1}^K``
* observed data: ``D=\{x^i, z^i\}_{i=1}^n``

Assume independent observations, the likelihood is

$P(D|\theta) = \prod_{i=1}^n p(z^i, x^i) = \prod_{i=1}^n p(z^i)p(x^i|z^i)$

Take log 

$$\mathcal L(\theta) = \ln P(D|\theta) = \sum_{i=1}^n \ln p(z^i)+ \sum_{i=1}^n \ln p(x^i|z^i)$$




"""

# ╔═╡ cdf72ed6-0d70-4901-9b8f-a12ceacd359d
md"""
## *Further details

Write down the distribution with $I(\cdot)$ notation (you should verify they are the same)

$$p(z^i) = \prod_{k=1}^K \pi_k^{I(z^i=k)}$$ and also

$$p(x^i|z^i) = \prod_{k=1}^K N(x^i|\mu_k,\Sigma_k)^{I(z^i=k)}$$

Their logs are 

$$\ln p(z^i) = \sum_{k=1}^K {I(z^i=k)} \cdot \pi_k\;\; \ln p(x^i|z^i) =  \sum_{k=1}^K {I(z^i=k)} \cdot \ln N(x^i|\mu_k,\Sigma_k)$$

Then

$$\mathcal L(\theta) = \sum_{i=1}^n \sum_{k=1}^K {I(z^i=k)} \cdot \ln \pi_k+ \sum_{i=1}^n \sum_{k=1}^K {I(z^i=k)} \cdot \ln N(x^i|\mu_k,\Sigma_k)$$

Therefore, we can isolate the terms and write $\mathcal L$ as a function of $\mu_k, \Sigma_k$:

$\mathcal L(\mu_k,\Sigma_k) = \sum_{i=1}^n {I(z^i=k)} \cdot \ln N(x^i|\mu_k,\Sigma_k) +C$

which justifies why the MLE for $\mu_k, \Sigma_k$ are the pooled MLE for the k-th class's observations!


The first term is ordinary Multinoulli log-likelihood, its MLE is relative frequency (need to use Lagrange multiplier as $\sum_{k} \pi_k =1$).
"""

# ╔═╡ 359b407f-6371-4e8a-b822-956173e89a47
md"""

## (Supervised) Learning of QDA: MLE 


Then optimise $\mathcal L$: take derivatives and set to zero we find the MLE

$\frac{\partial \mathcal L}{\partial \pi} =0; \frac{\partial \mathcal L}{\partial \mu_k} =0; \frac{\partial \mathcal L}{\partial \Sigma_k} =0$

The **maximum likelihood estimators** are 

$$\hat \pi_k = \frac{\sum_{i=1}^n I(z^i= k)}{n}$$

$$\hat \mu_k = \frac{1}{\sum_{i=1}^n I(z^i=k)}{\sum_{i=1}^n I(z^i=k)\cdot x^i}$$

$$\hat \Sigma_k = \frac{1}{\sum_{i=1}^n I(z^i=k)} \sum_{i=1}^n I(z^i=k) (x^i-\mu_k)(x^i-\mu_k)^\top$$

* ``\hat \pi``: frequency of labels belong to each class 
* ``\hat \mu_k, \hat \Sigma_k``: sample mean and covariance of datasets belong to each class $k$
"""

# ╔═╡ e22b902a-cd90-450c-80f1-8b1ff00ec4a7
md"""

## Demonstration of supervised learning of QDA

"""

# ╔═╡ 16f7831e-8fdf-4843-8ab9-934e4bd163d4
md"For supervised learning of QDA, the observed data are both input features and labels"

# ╔═╡ f347146d-b8a2-4487-9d05-5417bdb1c5d1
data₂, truezs₂;

# ╔═╡ 4360022f-3111-453e-8585-4616341a174b
md"""

The ground truth is : $\pi = [0.2, 0.2, 0.6]$ $\mu_1 = [-2 , 1];\mu_2 = [2 , 1]; \mu_3 = [0 , -1]; $ and $\Sigma_1 = \begin{bmatrix}.5, 0\\0, .5\end{bmatrix},  \Sigma_2 = \begin{bmatrix}.5, 0\\0, .5\end{bmatrix}, \Sigma_3 = \begin{bmatrix}0.5, 0\\0, 2\end{bmatrix}$
"""

# ╔═╡ ae701c75-8ba8-43e8-95dc-fceec823a867
md"MLE for ``\hat \pi``"

# ╔═╡ b0e16123-df7e-429c-a795-9e5ba788171a
πₘₗ = counts(truezs₂)/length(truezs₂)

# ╔═╡ 9ab9cffa-e5e9-4a17-84ab-2e1aa25af9ce
md"""MLE for ``\mu_k, \Sigma_k``; 

MLE are close to the ground truth (but not exactly the same)
* it can be shown more data observed, MLE asymptotically converges to the ground truth (due to the *Large Number Theory*)
"""

# ╔═╡ 2f8e92fc-3f3f-417f-9171-c2c755d5e0f0
begin
	μ_ml, Σ_ml = zeros(2,K₂), zeros(2,2,K₂)
	for k in 1:K₂
		data_in_ck = data₂[truezs₂ .==k,:]
		μ_ml[:,k] = mean(data_in_ck, dims=1)
		Σ_ml[:,:, k] = cov(data_in_ck)
	end
end

# ╔═╡ 5d28e09c-891d-44c0-98a4-ef4cf3a235f1
μsₘₗ = μ_ml

# ╔═╡ 58663741-fa05-4804-8734-8ccb1fa90b2d
Σsₘₗ = Σ_ml

# ╔═╡ 889093e8-5e14-4211-8807-113adbac9a46
begin
	gr()
	mvnsₘₗ = [MvNormal(μsₘₗ[:,k], Σsₘₗ[:,:,k]) for k in 1:K₂]
	pltqdaₘₗ = plot(title="QDA MLE params", ratio=1)
	for k in 1:K₂
		scatter!(data₂[truezs₂ .==k,1], data₂[truezs₂ .==k, 2], label="", c= k )
		scatter!([mvnsₘₗ[k].μ[1]], [mvnsₘₗ[k].μ[2]], color = k, label = "", markersize = 10, markershape=:diamond, markerstrokewidth=3)
		contour!(xs_qda₂, ys_qda₂, (x,y)-> pdf(mvnsₘₗ[k], [x,y]), levels=5, colorbar = false, ratio=1,lw=4) 
	end
	title!(pltqda, "QDA truth")
	plot(pltqda, pltqdaₘₗ, layout=(1,2))
end

# ╔═╡ c9fb5e3a-77cc-4545-9139-5b6697810762
md"""
## Classification 
Then classification is just 

$$\begin{align}p(z^i=k|x^i)&\propto p(z^i=k) p(x^i|z^i=k)\\
&= \pi_k N(x^i; \mu_k, \Sigma_k)\end{align}$$

* prior ``\times`` likelihood then normalise, such that :
  $\sum_{k=1}^K p(z^i=k|x^i) =1$

We can assign based on the posterior distribution

$\hat z^i = \arg\max_k p(z^i=k|x^i)$

"""

# ╔═╡ 6b732f1b-f412-43bd-b8fa-ccd41df8b1e2
md"""

## Demo of the classification
"""

# ╔═╡ a6c90edd-97eb-4282-9f7d-6f89d48b672b
md"Calculate the posterior for $i\in 1,\ldots,n;\; k=1,\ldots, K$:

$p(z^i=k|x^i)=w_{ik}\propto p(z^i=k)p(x^i|z^i=k)$

* note it is a ``n \times K`` matrix: each row is a probability vector for an observation's membership"

# ╔═╡ 99fbcb20-0ef5-4d2e-8fe7-7316e195b0e2
begin
	postc_ = zeros(n₂, K₂)
	postc_[:,1] = πₘₗ[1] .* pdf(MvNormal(μsₘₗ[:,1], Σsₘₗ[:,:,1]), data₂')
	postc_[:,2] = πₘₗ[2] .* pdf(MvNormal(μsₘₗ[:,2], Σsₘₗ[:,:,2]), data₂');
	postc_[:,3] =πₘₗ[3] .* pdf(MvNormal(μsₘₗ[:,3], Σsₘₗ[:,:,3]), data₂');
end;

# ╔═╡ 18fac74c-e181-49a9-b763-3d42ab66522a
postc = postc_ ./ sum(postc_, dims=2) ;

# ╔═╡ ad502388-3dca-4d01-8a2d-2247f632f398
@bind iₜₕ Slider(1:length(truezs₂), default=1)

# ╔═╡ ff0d536c-ec33-437b-a2c0-21e9c414a806
md" For example, ``p(z^i = k|x^i)`` when $i=$ $(iₜₕ)"

# ╔═╡ 6820b40b-423d-4b1d-942d-9a960b8ff390
round.(postc[iₜₕ,:],digits=2)

# ╔═╡ cefb1570-de86-40e2-9139-f7b3d159982b
md"Classification: find the largest entry of each row"

# ╔═╡ 39fd6577-edf2-4e3d-a636-2f19cc17d351
zsₘₗ=argmax.(eachrow(postc))

# ╔═╡ 45aa6de1-82f2-41ae-bbed-33d38f14b05b
md"""

## Mixture of Gaussian models

The same model as QDA but assume $z^i$ **hidden**
* unsupervised learning, we do not have labels $z^i$
* in other words, the data is $D=\{x^i\}_{i=1}^n$ 

We can apply marginalisation rule or sum rule to QDA model to find the mariginal likelihood for $x^i$:

$$\begin{align}p(x^i) &= \sum_{k=1}^K p(z^i=k, x^i) = \sum_{k=1}^K p(z^i=k)p(x^i|z^i=k)\\
&= \sum_{k=1}^K \pi_k \cdot p(x^i|z^i=k)\end{align}$$

This is called **finite mixture of Gaussians**
* a linear superposition of K distributions ``p(x^i|z^i=k)`` (Gaussian here but can be others)
* interpretation: since we do not know which $z^i=k$ is responsible for $x^i$, we add all possibilities up 
"""

# ╔═╡ a8beed6c-62ce-452c-adc2-1173fd201cb2
md"""

## Mixture of Gaussian models (cont.)

Finite mixture of Gaussians 

$p(x)= \sum_{k=1}^K \pi_k \cdot N(x; \mu_k, \Sigma_k)$

* is a distribution on $x\in R^d$: ``p(x)>0`` and ``\int p(x) dx = 1`` 
* it is more flexible than singular Gaussian
* it can approximate any distribution arbitrarily well when $K$ is large enough
  * a popular density estimation model

"""

# ╔═╡ fc07ba43-c534-4a9b-a177-a4241a17c92b
md"## Demonstration of mixture of Gaussians (1 dimensional)

The three univariate Gaussians are $N(-2,1), N(0, 0.5^2), N(2, 2^2)$

The effect of the prior:"

# ╔═╡ 6d3ee930-9960-4b7f-915d-16398c41a3c0
@bind n₁0 Slider(1:50, default=1)	

# ╔═╡ 39cf4800-7bb2-4802-9da6-012eb1e2086a
@bind n₂0 Slider(1:50, default=1)

# ╔═╡ bf809430-0c5d-44ad-ae45-68ac558bd494
@bind n₃0 Slider(1:50, default=1)

# ╔═╡ 135d1b0a-81e7-4802-a987-391550b22ecf
md"``p(z) = \pi`` is"

# ╔═╡ 242a181a-a670-421a-a1ed-8e39e7a63324
begin
	πs0 = [n₁0, n₂0, n₃0]
	πs0 = πs0/sum(πs0)
end

# ╔═╡ fa621263-43a5-4151-86e9-098e0d8537ff
begin
	plotly()
	cpt1, cpt2, cpt3 =  Normal(-2,1), Normal(0,0.5), Normal(2,2)
	plot(cpt1, label="c₁", alpha=0.5, lw=2)
	plot!(cpt2,label="c₂", alpha=0.5, lw=2)
	plot!(cpt3, label="c₃", alpha=0.5, lw=2)
	mixgaussians(πs,x) = πs[1]* pdf(cpt1, x)+ πs[2]*pdf(cpt2, x)+ πs[3] *pdf(cpt3, x)
	
	plot!(-6:0.1:10, (x)-> mixgaussians(πs0, x), label="mixture", lw=3)
end

# ╔═╡ c2435c4d-55ac-4a08-b0cd-72cc505db099
md"""

## Demon of mixture of Gaussians (multi-dimensional)

"""

# ╔═╡ d7a0d175-5487-40b2-b491-28f05ed055ff
md"""

## Unsupervised learning of mixture model


**Mixture of Gaussians** is a model we use for both *classification* and *clustering* 
* supervised learning: i.e. classification, QDA when $z^i$ are observed
* unsupervised learning: i.e. clustering, when $z^i$ are missing


So next to tackle is how to learn a mixture model in a **unsupervised learning** setting:


!!! correct "Unsupervised learning of mixture model: objective"
	Estimate parameters $\theta=\{\pi, \{\mu_k, \Sigma_k\}_{k=1}^K\}$, given only $D=\{x^i\}_{i=1}^n$; 
 	
    As you may have expected, we use maximum likelihood estimation:
 
	$$\hat \theta =\arg\max_{\theta} p(D|\theta)$$
"""

# ╔═╡ fc4c2785-4e5b-412a-8f54-b7ba6cd3828c
md"""

## EM algorithm for mixture of Gaussians

Recall **maximum likelihood estimators** for QDA when $c$ are observed: 

$$\hat \pi_k = \frac{\sum_{i=1}^n I(z^i= k)}{n}, \hat \mu_k = \frac{1}{\sum_{i=1}^n I(z^i=k)}{\sum_{i=1}^n I(z^i=k)\cdot x^i}$$

$$\hat \Sigma_k = \frac{1}{\sum_{i=1}^n I(z^i=k)} \sum_{i=1}^n I(z^i=k) (x^i-\mu_k)(x^i-\mu_k)^\top$$


The unsupervised learning or **EM algorithm** just replaces 

$I(z^i=k) \Rightarrow \underbrace{p(z^i=k|x^i)}_{w_{ik}}$

* since ``z^i`` is not observed, it is a random variable endowed with a posterior distribution $p(z^i|x^i)$
* e.g. the mean's reestimation is

  $\hat \mu_k = \frac{1}{\sum_{i=1}^n w_{ik}}{\sum_{i=1}^n w_{ik} \cdot x^i}$
  * weighted mean of all data pointed weighted by the **responsibilities**
* if we believe one data point $x^i$ is more likely to be generated by k-th component, it will contribute more (higher weight) towards that component's estimation    

"""

# ╔═╡ c25f303b-7596-4901-9063-aa1c36249ace
md"""

## EM algorithm for mixture of Gaussians


Initilisation: random guess $\{{\pi_k}, \mu_k, \Sigma_k\}$


* Expectation step (E step): for $i= 1\ldots n,\; k= 1\ldots K$
$$w_{ik} \leftarrow p(z^i=k|{x}^i) = \frac{\pi_k N(x^i; {\mu}_k, {\Sigma}_k)}{\sum_{j=1}^K \pi_j N(x^i; {\mu}_j, {\Sigma}_j)}$$


* Maximisation step (M step): update ${\theta}$, for $k=1\ldots K$

$\pi_k \leftarrow \frac{1}{n}\sum_{i=1}^n w_{ik}$

${\mu}_{k} \leftarrow \frac{1}{\sum_{i=1}^n w_{ik}} \sum_{i=1}^n w_{ik}x^i$

${{\Sigma}}_{k} \leftarrow \frac{1}{\sum_{i=1}^n w_{ik}} \sum_{i=1}^n w_{ik} (x^i-{{\mu}}_{k})(x^i-{{\mu}}_{k})^\top$

Repeat above two steps until converge

It can be shown that the (log-)likelihood (we will prove it next time)

$p(D|\theta_{\text{iter}-1}) < p(D|\theta_{\text{iter}})$ will improve over the iterations and finally converge to a (local) maximum 
* monitor the (log) likelihood for debugging and convergence check
"""

# ╔═╡ 4c87bf89-8070-47c2-abcb-e68d708a7fac
md"""

The **likelihood** can be calculated in E-step

$p(D|\theta) = \prod_{i=1}^n p(x^i|\theta) = \prod_{i=1}^n \underbrace{\sum_{k=1}^K \pi_k N(x^i|\mu_k, \Sigma_k)}_{\text{sum of i-th row of } (w_{ik})}$

* first equation is due to independence assumption
* this might underflow easily when dataset is large; better monitor log-likelihood
"""

# ╔═╡ 3e415e55-39af-4f81-a408-3336338d69e2
md"""

## Revisit K-means

K-means is a specific case of EM with the following assumptions

* first, the prior is uniform distributed

$$p(z^i=k) = \pi_k = 1/K$$
* second, $\Sigma_k = I$, covariances are tied but also fixed to be identity matrix
* then, assignment step is just a **hard E step** (winner takes all)

  $$w_{ik} \leftarrow \begin{cases} 1, \text{ if } k=\arg\max_{k'} p(z^i=k'|{x}^i)& \\ 0, \text{ otherwise} \end{cases}$$ 
$$\begin{align*}
  \arg\max_{k'} p(z^i=k'|{x}^i) &=\arg\max_{k'}\frac{\bcancel{\pi_{k'}} N(x^i; {\mu}_{k'}, {I})}{\sum_{j=1}^K \bcancel{\pi_j} N(x^i; {\mu}_j, {I})} \\
  &= \arg\max_{k'}\frac{c\cdot \exp(-\frac{1}{2}(x^i-\mu_{k'})^\top(x^i-\mu_{k'}))}{\sum_{j}c\cdot \exp(-\frac{1}{2}(x^i-\mu_{j})^\top(x^i-\mu_{j}))}\\
  &= \arg\max_{k'} {\exp\left (-\frac{1}{2} \|{x}^i-{\mu}_{k'}\|_2^2\right )} \\
  &= \arg\min_{k'} \|{x}^i-{\mu}_{k'}\|_2^2
  \end{align*}$$

* update step follows due to the above hard assignment and the assumption
  * only update the mean $\mu_k$
  * no reestimation for $\pi$ and $\Sigma_k$ they are assumed known or fixed

"""

# ╔═╡ 87515234-3bdb-46c2-bcf4-d5d9c675d31c
md"""
## Implementation 

Best way is to implement E and M step separately as two methods and test them independently


For **E-step**
* it is more numerically stable to calculate it in log-probability space and use *logsumexp* trick (more on this in lab session next week)

* **testing**: test it on similated dataset (with known labels) to see whether the responsibility matrix makes sense

Similar for **M-step**
* **testing**: test it with one-hot encoded of the true labels as a responsiblity matrix, it should return MLE estimator of QDA

Then finally, assemble them together to form an complete EM algorithm
"""

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

# ╔═╡ d4941adc-253b-407b-a4e6-e02a939634cc
md"""
## Demonstration on EM algorithm

"""

# ╔═╡ a83f34e7-34a5-4578-928c-479f135c2b9c
md"""Similar to K-means, a method `mixGaussiansDemoGif` is implemented to produce demo gifs (check Appendix)

`mixGaussiansDemoGif(data, K, iterations; init_step="e" or "m", add_contour=true or false)`
* it runs the E and M steps a few steps and record everything as a frame in a gif
"""

# ╔═╡ 28dc6e55-7494-468a-b03a-652e3547daa5
md"Recall ground truth for the first challenging dataset"

# ╔═╡ cdd9ce07-9099-430f-a97a-c2347e282a16
md"Clustering plot"

# ╔═╡ d60ef237-32f8-4b91-960b-ae0ab258f317
md"Decision boundary plot"

# ╔═╡ c488b48a-1f6a-4ee5-b2df-98f16206dac5
md"Density plot"

# ╔═╡ 4b34d1a6-c854-4b58-8f36-42fb009d0ed6
md"""

## Demonstration on EM (cont.)

"""

# ╔═╡ 416f552e-79a1-4aa2-9738-9299f2d718a6
md"Remember the ground truth for the dataset"

# ╔═╡ 72748220-3cc7-4dd4-be9d-3a6d53035081
md"""

You can also use `em_mix_gaussian` directly
"""

# ╔═╡ 2ed1d3b2-50f3-4d1a-9c13-b1613056b798
ll, gss₂, pps₂, zs₂=em_mix_gaussian(data₂, K₂; init_step="m");

# ╔═╡ 77edb64c-6acf-4610-b3a4-f2798224584d
begin
	plotly()
	plot(ll, xlabel="iteration", ylabel="log likelihood", label="")
end

# ╔═╡ f61e3851-fc4f-43bf-b516-348e4a8edbca
ll₃, gss₃, pps₃, zs₃=em_mix_gaussian(data₃, K₃; init_step="m")

# ╔═╡ daedd76d-2156-40d2-b17e-bbafebbc2336
plot(ll₃, xlabel="iteration", ylabel="log-likelihood", label="")

# ╔═╡ 02fc47ab-7b4d-4cbc-a6d0-55d2fdab1585
md"""

## Measure clustering performance

If true labels were available 
* need to deals with **labelling** problem
  * K-means might index the labels differently: e.g. 1,2,3 or 3,2,1
* measure accuracy: percentage of accurately clustered labels
* or information based criteria: e.g. **normalised mutual information** (NMI)
  * how much correlation between two clusters 
  * NMI is between 0 and 1; 1 means perfectly correlated
* there are others such as random index (RI), adjusted random index (ARI)
* due to time limit: we just use them as blackbox metrics 

"""

# ╔═╡ d37b3d73-eb9c-4b40-8c5a-f6b1b7afc06d
begin
	nmi_em_d₃ = mutualinfo(argmax.(eachrow(zs₃)), truezs₃)
	nmi_em_d₂ = mutualinfo(argmax.(eachrow(zs₂)), truezs₂)
end;

# ╔═╡ af17a502-45ad-4055-a527-e562f705d000
begin
	nmi_km_d₂ = mutualinfo(zskm₂, truezs₂)
	nmi_km_d₃ =mutualinfo(zskm₃, truezs₃)
end

# ╔═╡ 62afcad3-a92f-42cc-b535-dc8cb444aef6
md"Confusion table"

# ╔═╡ 7dfee1bb-4422-422d-96dd-cd4cc954b645
counts(truezs₂, argmax.(eachrow(zs₂)));

# ╔═╡ 7ea802f1-d18c-4438-840c-9d5e1c20b13a
counts(truezs₃, argmax.(eachrow(zs₃)));

# ╔═╡ f3512172-661d-45a5-8a7b-499df6657266
begin
	# best possible estimate of cluster labels
	zs_ub₃, _ = e_step(data₃, truemvns₃, trueπs₃);
	zs_ub₂, _ = e_step(data₂, truemvns₂, trueπs₂);
	# performance upper bound
	nmi_ub_d₃ = mutualinfo(argmax.(eachrow(zs_ub₃)), truezs₃)
	nmi_ub_d₂ = mutualinfo(argmax.(eachrow(zs_ub₂)), truezs₂)
end;

# ╔═╡ 56a35a21-b454-4f46-aadb-ceb65237e6ac
md"""
## Compare K-means and EM

The NMI for the two challenging datasets 

||dataset 2| dataset 3|
|:---:|:---:|:---:|
|Kmeans| $(round(nmi_km_d₂, digits=2))| $(round(nmi_km_d₃, digits=2))|
|EM |$(round(nmi_em_d₂, digits=2))| $(round(nmi_em_d₃, digits=2)) |
|True parms|$(round(nmi_ub_d₂, digits=2)) | $(round(nmi_ub_d₃, digits=2))|

The last row is the best performance you can possibly achieve
* use true parameters to do the E-step
* *irreducible error* or performance upper bound
"""

# ╔═╡ dbb657d0-72d7-48c8-8b10-53fad2911ade
md"The best performance of NMI is about $(round(nmi_ub_d₃, digits=2)) i.e. use true parameter to estimate the cluster labels"

# ╔═╡ 64d31497-9009-49f2-b132-07a81331ac2f
md"""

## Suggested reading

Machine learning: a probabilistic approach by Kevin Murphy
* 4.2: Gaussian discriminant analysis
* 11.2 and 11.4: mixture of Gaussians 


"""

# ╔═╡ a0465ae8-c843-4fc0-abaf-0497ada26652
md"""

## Appendix

Utility functions
"""

# ╔═╡ 620789b7-59bc-4e17-bcfb-728a329eed0f
qdform(x, S) = dot(x, S, x)

# ╔═╡ 7b47cda6-d772-468c-a8f3-75e3d77369d8
begin
# decision boundary function of input [x,y] 
function decisionBdry(x,y, mvns, πs)
	z, _ = e_step([x,y]', mvns, πs)
	findmax(z[:])
end

end

# ╔═╡ b4bfb1ba-73f2-45c9-9303-baee4345f8f6
begin
	plotly()
	pqda_class = contour(xs_qda₂, ys_qda₂, (x,y) -> decisionBdry(x,y, mvnsₘₗ, πₘₗ)[2], nlev=K₂, fill=true, c=cgrad(:lighttest, K₂+1, categorical = true),  title="Decision boundary by supervised learning QDA", ratio=0.9, colorbar=false)
	for k in 1:K₂
		scatter!(data₂[zsₘₗ .==k, 1], data₂[zsₘₗ .==k, 2], label="")
	end
	scatter!([data₂[iₜₕ, 1]], [data₂[iₜₕ, 2]], markersize = 12, markershape=:xcross, markerstrokewidth=3, c= :white, label="xⁱ")
	pqda_class
end

# ╔═╡ 8d0c6fdc-4717-4203-b933-4b37fe60d512
function logLikMixGuassian(x, mvns, πs, logLik=true) 
	l = logsumexp(log.(πs) .+ [logpdf(mvn, x) for mvn in mvns])
	logLik ? l : exp(l)
end

# ╔═╡ 21c01e65-6f2e-4a5c-a4a9-0397c52317e5
begin
	plotly()
	xs = (minimum(data₂[:,1])-3):0.1: (maximum(data₂[:,1])+3)
	ys = (minimum(data₂[:,2])-3):0.1: (maximum(data₂[:,2])+3)
	plt_mix_contour =plot(xs, ys, (x,y) -> logLikMixGuassian([x,y], truemvns₂, trueπs₂), st=:contour,fill = true, ratio=1, colorbar=false, title="contour plot")
	plt_mix_surface=plot(xs, ys, (x,y) -> logLikMixGuassian([x,y], truemvns₂, trueπs₂), st=:surface,fill = true, ratio=1,colorbar=false, title="surface plot")
	plot(plt_mix_contour, plt_mix_surface)
end

# ╔═╡ 7ee6f500-099a-46c6-aca7-8598f71ada1a
begin
	plot(pltqda, plt_mix_contour, title="Ground truth")
end

# ╔═╡ 9ae3a788-0516-4455-972d-b41ce26fa851
begin
	pltqda₂_=plot(xs₃, ys₃, (x,y) -> logLikMixGuassian([x,y], truemvns₃, trueπs₃), st=:contour,fill = true, ratio=1)
	title!(pltqda₂,"")
	plot(pltqda₂, pltqda₂_, title="Ground truth")
end

# ╔═╡ d66e373d-8443-4810-9332-305d9781a21a
md"""

Functions used to plot and produce the gifs

"""

# ╔═╡ acfb80f0-f4d0-4870-b401-6e26c1c99e45
function plot_clusters(D, zs, K, loss=nothing, iter=nothing)
	title_string = ""
	if !isnothing(iter)
		title_string ="Iteration: "*string(iter)*";"
	end
	if !isnothing(loss)
		title_string *= " L = "*string(round(loss; digits=2))
	end
	plt = plot(title=title_string, ratio=1)
	for k in 1:K
		scatter!(D[zs .==k,1], D[zs .==k, 2], label="cluster "*string(k))
	end
	return plt
end

# ╔═╡ 4fec39f6-d367-49f7-83f2-47b2b2dee538
begin
	plt₂=plot_clusters(data₂, truezs₂, K₂)
	title!(plt₂, "Ground truth for dataset 2")
end

# ╔═╡ 06164f6e-9bda-4a17-b65f-fd66a3f9eb4e
begin
	plt₃=plot_clusters(data₃, truezs₃, K₃)
	title!(plt₃, "Ground truth for dataset 3")
end

# ╔═╡ e091ce93-9526-4c7f-9f14-7634419bfe57
# plot clustering results: scatter plot + Gaussian contours
function plot_clustering_rst(data, K, zs, mvns, πs= 1/K .* ones(K); title="")
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
		plot!(xs, ys, (x,y)-> qdform([x,y]-mvns[k].μ, inv(mvns[k].Σ)), levels=[2.0],  st=:contour, colorbar = false, ratio=1, color=k, linewidth=3) 
		scatter!([mvns[k].μ[1]], [mvns[k].μ[2]], color = k, label = "", markersize = 10, markershape=:star4, markerstrokewidth=3)
	end
	title!(p, title)
	return p
end

# ╔═╡ 9cbc7ef6-788c-4dfd-a98d-2a9409eb5127
begin
	gr()
	plt₂_ =plot_clustering_rst(data₂, K₂, truezs₂,  truemvns₂, trueπs₂)
	title!(plt₂_, "Ground truth for dataset 2 with distance contours")
end

# ╔═╡ 0434dd27-4349-4731-80d5-b71ab99b53e2
begin
	gr()
	plt₃_= plot_clustering_rst(data₃,  K₃, truezs₃, truemvns₃, trueπs₃)
	title!(plt₃_, "Ground truth for dataset 3 with distance contours")
end

# ╔═╡ 3799d0ab-44b9-459f-9ca8-241c7861f84e
begin
	gr()
	plot_clustering_rst(data₃, K₃, zskm₃, ms₃; title= "K-means result for dataset 2")
end;

# ╔═╡ 5fa30e8a-bc79-4fae-a946-865af7f495d4

begin
	gr()
	plot_clustering_rst(data₂, K₂, zs₂, gss₂, pps₂)
end

# ╔═╡ 92646ae3-b80c-4734-ba2e-503c13bd14b4
begin
	gr()
	plot_clustering_rst(data₃, K₃, zs₃, gss₃, pps₃)
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
	cs = cgrad(:lighttest, K+1, categorical = true)
	ps = 1/K .* ones(K)
	for iter in 1:iters
		ms = update_step(data, zs, K)
		# animation 1: classification evolution
		p1 = plot_clusters(data, zs, K, l, iter)
		if add_contour
			for k in 1:K 
				plot!(xs, ys, (x,y)-> sum((ms[:, k] - [x,y]).^2), levels=[1.5],  st=:contour, colorbar = false, ratio=1, color=k, linewidth=3)  
				scatter!([ms[1,k]], [ms[2,k]], color = k, label = "", markersize = 10, markershape=:star4, markerstrokewidth=2)
			end
		end
		frame(anims[1], p1)
		# animation 2: decision boundary
		mvns = [MvNormal(ms[:,k], Matrix(1.0I, dim, dim)) for k in 1:K]
		p2 = contour(xs, ys, (x,y) -> decisionBdry(x,y, mvns, ps)[2], nlev=K, fill=true, c=cgrad(:lighttest, K+1, categorical = true), leg=:none, title="Iteration: "*string(iter)*"; L="*string(round(l; digits=2)), ratio=1)
		for k in 1:K
			scatter!(data[zs .==k, 1], data[zs .==k, 2], c= cs[k])
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
end

# ╔═╡ bd670cde-bcbe-4104-8114-34f761f7f850
gif(kmAnim₁[1], fps=2)

# ╔═╡ d2da6a9e-0e00-412a-a0d0-86a4eb34360d
gif(kmAnim₁[2], fps=2)

# ╔═╡ 533efc1a-fd99-4a5d-9ef3-4ff7ce483f24
begin
	gr()
	kmAnim3 = kmeansDemoGif(data₂, K₂, 10; init_step="a", add_contour=true)
end;

# ╔═╡ de9c10a8-3a93-44cb-9c20-a0bc8bea594c
gif(kmAnim3[1], fps=1)

# ╔═╡ 2a3683f6-92aa-4a98-bc24-e71e456ea0ce
gif(kmAnim3[2], fps=1)

# ╔═╡ 3c7e9888-31ed-47c1-be42-3e25e425ea5d
gif(kmAnim3[3], fps=1);

# ╔═╡ ad943140-9510-463e-b57c-4956ed2cd5d5
begin
	gr()
	kmAnim₃ = kmeansDemoGif(data₃, K₃, 10; init_step="a", add_contour=true)
end

# ╔═╡ 0889db75-5b6f-4eb4-a35a-3637b20da628
gif(kmAnim₃[1], fps=1)

# ╔═╡ b825ebd5-028b-41f6-84d4-92129de6959c
gif(kmAnim₃[2], fps=0.5)

# ╔═╡ bf53be52-2de7-4a98-8110-e8cd96b2ed7b
gif(kmAnim₃[3], fps=1);

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
	cs = cgrad(:lighttest, K+1, categorical = true)

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
		p2 = contour(xs, ys, (x,y) -> decisionBdry(x,y, mvns, ps)[2], nlev=K, fill=true, c=cgrad(:lighttest, K+1, categorical = true), leg=:none, title="Iteration: "*string(iter)*"; L="*string(round(l; digits=2)), ratio=1)
		for k in 1:K
			scatter!(data[zs_ .==k, 1], data[zs_ .==k, 2], c= cs[k])
		end
		frame(anims[2], p2)

		# animation 3: contour plot
		# p3 = plot_clusters(data, zs_, K, l, iter)
		p3 = plot(xs, ys, (x,y) -> logLikMixGuassian([x,y], mvns, ps), st=:contour, fill=true, colorbar=false, title="Iteration: "*string(iter)*"; L="*string(round(l; digits=2)), ratio=1)
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

# ╔═╡ d9456461-1b59-4fee-a265-6f1cdc8bb2e6
begin
	gr()
	mixAnims = mixGaussiansDemoGif(data₂, K₂, 100; init_step="e", add_contour=true)
end

# ╔═╡ 6d53d19b-38a3-45fa-9c62-94b7d4a69cad
gif(mixAnims[1], fps=10)

# ╔═╡ 44069d22-92cb-44b8-9775-eab0d9b6240e
gif(mixAnims[2], fps=20)

# ╔═╡ 9ec71356-485b-423e-98f0-eb73b046e5f5
gif(mixAnims[3], fps=20)

# ╔═╡ 23d95e32-4999-4032-90dd-570e33af5895
begin
	gr()
	mixAnims₃ = mixGaussiansDemoGif(data₃, K₃, 100; init_step="m", add_contour=true)
end

# ╔═╡ 6f9f4116-d887-42c7-84af-d6a2fbb16af7
gif(mixAnims₃[1], fps=5)

# ╔═╡ e40f7779-9640-4714-881d-a9802b548ed9
gif(mixAnims₃[2], fps=20)

# ╔═╡ 2b9d6c0d-8006-49bb-9fb4-807bb02fa049
gif(mixAnims₃[3], fps=20)

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
Clustering = "aaaa29a8-35af-508c-8bc3-b662a17a0fe5"
Distributions = "31c24e10-a181-5473-b8eb-7969acd0382f"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
Plots = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
StatsBase = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
StatsFuns = "4c63d2b9-4356-54db-8cca-17b64c39e42c"
StatsPlots = "f3b207a7-027a-5e70-b257-86293d7955fd"

[compat]
Clustering = "~0.15.3"
Distributions = "~0.25.98"
Plots = "~1.38.16"
PlutoUI = "~0.7.51"
StatsBase = "~0.34.0"
StatsFuns = "~1.3.0"
StatsPlots = "~0.15.5"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.9.2"
manifest_format = "2.0"
project_hash = "2e9df7b887c7ef21eee38b60fec8a32799cbee52"

[[deps.AbstractFFTs]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "cad4c758c0038eea30394b1b671526921ca85b21"
uuid = "621f4979-c628-5d54-868e-fcf4e3e8185c"
version = "1.4.0"
weakdeps = ["ChainRulesCore"]

    [deps.AbstractFFTs.extensions]
    AbstractFFTsChainRulesCoreExt = "ChainRulesCore"

[[deps.AbstractPlutoDingetjes]]
deps = ["Pkg"]
git-tree-sha1 = "8eaf9f1b4921132a4cff3f36a1d9ba923b14a481"
uuid = "6e696c72-6542-2067-7265-42206c756150"
version = "1.1.4"

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
git-tree-sha1 = "42fe66dbc8f1d09a44aa87f18d26926d06a35f84"
uuid = "aaaa29a8-35af-508c-8bc3-b662a17a0fe5"
version = "0.15.3"

[[deps.CodecZlib]]
deps = ["TranscodingStreams", "Zlib_jll"]
git-tree-sha1 = "9c209fb7536406834aa938fb149964b985de6c83"
uuid = "944b1d66-785c-5afd-91f1-9de20f533193"
version = "0.7.1"

[[deps.ColorSchemes]]
deps = ["ColorTypes", "ColorVectorSpace", "Colors", "FixedPointNumbers", "PrecompileTools", "Random"]
git-tree-sha1 = "be6ab11021cd29f0344d5c4357b163af05a48cba"
uuid = "35d6a980-a343-548e-a6ea-1d62b119f2f4"
version = "3.21.0"

[[deps.ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "eb7f0f8307f71fac7c606984ea5fb2817275d6e4"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.11.4"

[[deps.ColorVectorSpace]]
deps = ["ColorTypes", "FixedPointNumbers", "LinearAlgebra", "SpecialFunctions", "Statistics", "TensorCore"]
git-tree-sha1 = "600cc5508d66b78aae350f7accdb58763ac18589"
uuid = "c3611d14-8923-5661-9e6a-0046d554d3a4"
version = "0.9.10"

[[deps.Colors]]
deps = ["ColorTypes", "FixedPointNumbers", "Reexport"]
git-tree-sha1 = "fc08e5930ee9a4e03f84bfb5211cb54e7769758a"
uuid = "5ae59095-9a9b-59fe-a467-6f913c188581"
version = "0.12.10"

[[deps.Compat]]
deps = ["UUIDs"]
git-tree-sha1 = "4e88377ae7ebeaf29a047aa1ee40826e0b708a5d"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "4.7.0"
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
git-tree-sha1 = "cf25ccb972fec4e4817764d01c82386ae94f77b4"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.18.14"

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
deps = ["LinearAlgebra", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "49eba9ad9f7ead780bfb7ee319f962c811c6d3b2"
uuid = "b4f34e82-e78d-54a5-968a-f98e89d6e8f7"
version = "0.10.8"

[[deps.Distributed]]
deps = ["Random", "Serialization", "Sockets"]
uuid = "8ba89e20-285c-5b6f-9357-94700520ee1b"

[[deps.Distributions]]
deps = ["FillArrays", "LinearAlgebra", "PDMats", "Printf", "QuadGK", "Random", "SparseArrays", "SpecialFunctions", "Statistics", "StatsAPI", "StatsBase", "StatsFuns", "Test"]
git-tree-sha1 = "e76a3281de2719d7c81ed62c6ea7057380c87b1d"
uuid = "31c24e10-a181-5473-b8eb-7969acd0382f"
version = "0.25.98"

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
deps = ["LinearAlgebra", "Random", "SparseArrays", "Statistics"]
git-tree-sha1 = "e5556303fd8c9ad4a8fceccd406ef3433ddb4c45"
uuid = "1a297f60-69ca-5386-bcde-b61e274b549b"
version = "1.4.0"

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
git-tree-sha1 = "8b8a2fd4536ece6e554168c21860b6820a8a83db"
uuid = "28b8d3ca-fb5f-59d9-8090-bfdbd6d07a71"
version = "0.72.7"

[[deps.GR_jll]]
deps = ["Artifacts", "Bzip2_jll", "Cairo_jll", "FFMPEG_jll", "Fontconfig_jll", "GLFW_jll", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Libtiff_jll", "Pixman_jll", "Qt5Base_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "19fad9cd9ae44847fe842558a744748084a722d1"
uuid = "d2c73de3-f751-5644-a686-071e5b155ba9"
version = "0.72.7+0"

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
git-tree-sha1 = "3e008a7aa28d717a5badd05cb70c834e31001077"
uuid = "cd3eb016-35fb-5094-929b-558a96fad6f3"
version = "1.9.13"

[[deps.HarfBuzz_jll]]
deps = ["Artifacts", "Cairo_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "Graphite2_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Pkg"]
git-tree-sha1 = "129acf094d168394e80ee1dc4bc06ec835e510a3"
uuid = "2e76f6c2-a576-52d4-95c1-20adfe4de566"
version = "2.8.1+1"

[[deps.HypergeometricFunctions]]
deps = ["DualNumbers", "LinearAlgebra", "OpenLibm_jll", "SpecialFunctions"]
git-tree-sha1 = "ce7ea9cc5db29563b1fe20196b6d23ab3b111384"
uuid = "34004b35-14d8-5ef3-9330-4cdb6864b03a"
version = "0.3.18"

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
git-tree-sha1 = "0cb9352ef2e01574eeebdb102948a58740dcaf83"
uuid = "1d5cc7b8-4909-519e-a0f8-d0f5ad9712d0"
version = "2023.1.0+0"

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
deps = ["Preferences"]
git-tree-sha1 = "abc9885a7ca2052a736a600f7fa66209f96506e1"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.4.1"

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
version = "0.6.3"

[[deps.LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"
version = "7.84.0+0"

[[deps.LibGit2]]
deps = ["Base64", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"

[[deps.LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "MbedTLS_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"
version = "1.10.2+0"

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
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "LERC_jll", "Libdl", "Pkg", "Zlib_jll", "Zstd_jll"]
git-tree-sha1 = "3eb79b0ca5764d4799c06699573fd8f533259713"
uuid = "89763e89-9b03-5906-acba-b20f662cd828"
version = "4.4.0+0"

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
git-tree-sha1 = "c3ce8e7420b3a6e071e0fe4745f5d4300e37b13f"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "0.3.24"

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
git-tree-sha1 = "cedb76b37bc5a6c702ade66be44f831fa23c681e"
uuid = "e6f89c97-d47a-5376-807f-9c37f3926c36"
version = "1.0.0"

[[deps.MIMEs]]
git-tree-sha1 = "65f28ad4b594aebe22157d6fac869786a255b7eb"
uuid = "6c6e2e6c-3030-632d-7369-2d6c69616d65"
version = "0.1.4"

[[deps.MKL_jll]]
deps = ["Artifacts", "IntelOpenMP_jll", "JLLWrappers", "LazyArtifacts", "Libdl", "Pkg"]
git-tree-sha1 = "154d7aaa82d24db6d8f7e4ffcfe596f40bff214b"
uuid = "856f044c-d86e-5d09-b602-aeab76dc8ba7"
version = "2023.1.0+0"

[[deps.MacroTools]]
deps = ["Markdown", "Random"]
git-tree-sha1 = "42324d08725e200c23d4dfb549e0d5d89dede2d2"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.10"

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
git-tree-sha1 = "1aa4b74f80b01c6bc2b89992b861b5f210e665b5"
uuid = "458c3c95-2e84-50aa-8efc-19380b2a3a95"
version = "1.1.21+0"

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
git-tree-sha1 = "d321bf2de576bf25ec4d3e4360faca399afca282"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.6.0"

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
git-tree-sha1 = "4b2e829ee66d4218e0cef22c0a64ee37cf258c29"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.7.1"

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
git-tree-sha1 = "75ca67b2c6512ad2d0c767a7cfc55e75075f8bbc"
uuid = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
version = "1.38.16"

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

[[deps.PlutoUI]]
deps = ["AbstractPlutoDingetjes", "Base64", "ColorTypes", "Dates", "FixedPointNumbers", "Hyperscript", "HypertextLiteral", "IOCapture", "InteractiveUtils", "JSON", "Logging", "MIMEs", "Markdown", "Random", "Reexport", "URIs", "UUIDs"]
git-tree-sha1 = "b478a748be27bd2f2c73a7690da219d0844db305"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.51"

[[deps.PrecompileTools]]
deps = ["Preferences"]
git-tree-sha1 = "9673d39decc5feece56ef3940e5dafba15ba0f81"
uuid = "aea7be01-6a6a-4083-8856-8a6e6704d82a"
version = "1.1.2"

[[deps.Preferences]]
deps = ["TOML"]
git-tree-sha1 = "7eb1686b4f04b82f96ed7a4ea5890a4f0c7a09f1"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.4.0"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"

[[deps.Qt5Base_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Fontconfig_jll", "Glib_jll", "JLLWrappers", "Libdl", "Libglvnd_jll", "OpenSSL_jll", "Pkg", "Xorg_libXext_jll", "Xorg_libxcb_jll", "Xorg_xcb_util_image_jll", "Xorg_xcb_util_keysyms_jll", "Xorg_xcb_util_renderutil_jll", "Xorg_xcb_util_wm_jll", "Zlib_jll", "xkbcommon_jll"]
git-tree-sha1 = "0c03844e2231e12fda4d0086fd7cbe4098ee8dc5"
uuid = "ea2cea3b-5b76-57ae-a6ef-0a8af62496e1"
version = "5.15.3+2"

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
git-tree-sha1 = "7beb031cf8145577fbccacd94b8a8f4ce78428d3"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "2.3.0"
weakdeps = ["ChainRulesCore"]

    [deps.SpecialFunctions.extensions]
    SpecialFunctionsChainRulesCoreExt = "ChainRulesCore"

[[deps.StaticArrays]]
deps = ["LinearAlgebra", "Random", "StaticArraysCore"]
git-tree-sha1 = "0da7e6b70d1bb40b1ace3b576da9ea2992f76318"
uuid = "90137ffa-7385-5640-81b9-e52037218182"
version = "1.6.0"
weakdeps = ["Statistics"]

    [deps.StaticArrays.extensions]
    StaticArraysStatisticsExt = "Statistics"

[[deps.StaticArraysCore]]
git-tree-sha1 = "1d5708d926c76a505052d0d24a846d5da08bc3a4"
uuid = "1e83bf80-4336-4d27-bf5d-d5a4f845583c"
version = "1.4.1"

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
git-tree-sha1 = "14ef622cf28b05e38f8af1de57bc9142b03fbfe3"
uuid = "f3b207a7-027a-5e70-b257-86293d7955fd"
version = "0.15.5"

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
git-tree-sha1 = "074f993b0ca030848b897beff716d93aca60f06a"
uuid = "5c2747f8-b7ea-4ff2-ba2e-563bfd36b1d4"
version = "1.4.2"

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
git-tree-sha1 = "c4d2a349259c8eba66a00a540d550f122a3ab228"
uuid = "1986cc42-f94f-5a68-af5c-568840ba703d"
version = "1.15.0"

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
deps = ["Artifacts", "Expat_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Pkg", "XML2_jll"]
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
version = "1.48.0+0"

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
# ╟─0ae6c88f-3874-4cf7-86ab-28e400d0aca9
# ╟─120a282a-91c1-11ec-346f-25d56e50d38c
# ╟─75361c09-f201-457f-8081-25074ebe1b90
# ╟─2e8fcd55-70f9-4b7b-87dd-7e7d2413eb6f
# ╟─646dd3d8-6092-4435-aee9-01fa6a281bdc
# ╟─4f95e17a-9fbf-4a38-a7ad-f5ad31b1cfd0
# ╟─cd91d8a9-b208-4e26-8c72-655a28c93b61
# ╟─e136edb5-3e98-4355-83e2-55761eb8b15c
# ╟─8a217d25-e231-43a8-94b9-538f138dd331
# ╟─abd46485-03c9-4077-8ace-34f9b897bb04
# ╟─3ea57a8e-8d15-4f41-acfb-e3bd1d65e585
# ╟─cc38bdc2-4c1b-4a08-b8e6-1504a11924a5
# ╟─8cb79133-2339-4366-b881-098d83449aee
# ╟─2831de52-9f49-494e-8b2b-4cee810cee06
# ╠═a414e554-3a8c-472d-af82-07c2f0843627
# ╟─7c9b404e-ef6c-4ed1-910e-2f409e579309
# ╟─b81503f3-1caf-498a-8dc9-639ea5a5d569
# ╟─f2b7b63a-fd54-4f80-af28-f255d186f6f9
# ╠═11519291-da33-4bb2-b0e4-cffe347cf085
# ╠═474bace2-9586-45d3-b674-73806bbc85b8
# ╟─e1ccac4e-1cba-43d2-b0e0-e8c650a79d9a
# ╠═33cc44b4-bd32-4112-bf33-6807ae53818c
# ╠═00d85bc5-c320-41dc-8a85-4ad409b058a8
# ╟─ceaedbd8-6c88-407b-84f6-aedc0a59a22d
# ╟─8efbba52-7a34-4c17-bca9-52a417f69d56
# ╟─3873ef48-34c1-479a-b6f8-2cd05bcdc8bb
# ╟─5276ff0e-bc4f-4b24-b98c-8f9e7c0d018d
# ╠═20e84d48-0f5f-403e-a8aa-1cbd11cd3b04
# ╟─1954eac7-f155-4c37-9a51-26440d79851d
# ╟─a391d4a3-9fe3-4ccf-9a62-c2cb19ea8813
# ╟─a0ebfea7-bbe5-4798-8db3-163381bf67c2
# ╟─bd670cde-bcbe-4104-8114-34f761f7f850
# ╟─bc645a51-6e38-431d-b67d-1512126840cf
# ╟─542b0f6f-befa-4dfe-acb7-b7aa6ab9f56c
# ╟─d2da6a9e-0e00-412a-a0d0-86a4eb34360d
# ╟─05de404f-d2d7-4feb-ba70-a25d8eba71d4
# ╟─4fec39f6-d367-49f7-83f2-47b2b2dee538
# ╟─9cbc7ef6-788c-4dfd-a98d-2a9409eb5127
# ╟─ca7f12d8-f823-448b-ab1f-0e28c85a3f7f
# ╟─06164f6e-9bda-4a17-b65f-fd66a3f9eb4e
# ╟─0434dd27-4349-4731-80d5-b71ab99b53e2
# ╟─8d2778fd-edfe-4a02-92f9-2b9a67f2a2c0
# ╟─533efc1a-fd99-4a5d-9ef3-4ff7ce483f24
# ╟─8107c1d9-58c4-41f0-8870-bfd7084e42b5
# ╠═de9c10a8-3a93-44cb-9c20-a0bc8bea594c
# ╠═2a3683f6-92aa-4a98-bc24-e71e456ea0ce
# ╟─3c7e9888-31ed-47c1-be42-3e25e425ea5d
# ╠═c8cb2535-7b65-4d86-9405-71880fdb906e
# ╟─7f3440ec-5577-443c-8448-bf6cb4aeb1cb
# ╟─ad943140-9510-463e-b57c-4956ed2cd5d5
# ╟─0889db75-5b6f-4eb4-a35a-3637b20da628
# ╟─b825ebd5-028b-41f6-84d4-92129de6959c
# ╠═bf53be52-2de7-4a98-8110-e8cd96b2ed7b
# ╠═c7fd532d-d72a-439a-9e71-e85392c66f8c
# ╟─3799d0ab-44b9-459f-9ca8-241c7861f84e
# ╟─01544a34-0647-4589-b267-3d440c35d8ba
# ╟─a221a92c-7939-42de-84cb-85ab220f19f1
# ╟─f93d15ba-300d-45bc-9b6f-6a24ce1be4ad
# ╟─5d04b864-15e9-4e91-9928-e9af7181c3f7
# ╟─dac8fd37-2e61-42d8-b9c1-17c46fa7b9b7
# ╟─e66784ce-ec45-49c6-a1b7-d5d186e147ef
# ╟─d6651a85-3980-4e30-bb5e-0f770574ca7e
# ╟─eb50ce8b-a029-4599-9860-3594488187d0
# ╟─0aee031f-b724-4ddb-a288-22650585948e
# ╟─e5a57b78-77b8-4b07-8451-b37df36da736
# ╟─fab36b7e-5339-4007-8b84-1ec0b0c3fc23
# ╟─67e6b025-c9be-4cf8-8231-f8c01177eaf3
# ╟─aa93a75d-302c-4ede-b181-06d87a27eb23
# ╟─ebe5606a-4945-4dda-81d1-bde84a8d4fce
# ╟─b5008102-cace-4c43-98eb-c2a8ca5c1c76
# ╟─89f4e422-b603-44c9-87db-241457aeb080
# ╟─b449cd92-93de-4c75-bd45-3c8c2cbb8d33
# ╟─cc3beb3b-f5de-4571-8210-47e843b54965
# ╟─380b7426-5801-45ab-898a-a72f22f1e953
# ╟─dafd1a68-715b-4f06-a4f2-287c123761f8
# ╟─853de250-143e-4add-b50d-2c73d1bc7910
# ╟─cc60e789-02d1-4944-8ad5-718ede99669c
# ╟─02ad4708-cb36-45dc-ab2b-34ab22c26ccc
# ╟─8d71174d-401e-4cf9-8afb-3e8bbd49a0b1
# ╟─e0cfcb9b-794b-4731-abf7-5435f67ced42
# ╟─c86a3f0d-ea61-4f35-a116-a7aa7ab9bc2d
# ╟─4e038980-c531-4f7c-9c51-4e346eccc0aa
# ╟─cdf72ed6-0d70-4901-9b8f-a12ceacd359d
# ╟─359b407f-6371-4e8a-b822-956173e89a47
# ╟─e22b902a-cd90-450c-80f1-8b1ff00ec4a7
# ╟─16f7831e-8fdf-4843-8ab9-934e4bd163d4
# ╠═f347146d-b8a2-4487-9d05-5417bdb1c5d1
# ╟─4360022f-3111-453e-8585-4616341a174b
# ╟─ae701c75-8ba8-43e8-95dc-fceec823a867
# ╟─b0e16123-df7e-429c-a795-9e5ba788171a
# ╟─9ab9cffa-e5e9-4a17-84ab-2e1aa25af9ce
# ╟─2f8e92fc-3f3f-417f-9171-c2c755d5e0f0
# ╟─5d28e09c-891d-44c0-98a4-ef4cf3a235f1
# ╟─58663741-fa05-4804-8734-8ccb1fa90b2d
# ╟─889093e8-5e14-4211-8807-113adbac9a46
# ╟─c9fb5e3a-77cc-4545-9139-5b6697810762
# ╟─6b732f1b-f412-43bd-b8fa-ccd41df8b1e2
# ╟─a6c90edd-97eb-4282-9f7d-6f89d48b672b
# ╠═99fbcb20-0ef5-4d2e-8fe7-7316e195b0e2
# ╠═18fac74c-e181-49a9-b763-3d42ab66522a
# ╟─ff0d536c-ec33-437b-a2c0-21e9c414a806
# ╟─ad502388-3dca-4d01-8a2d-2247f632f398
# ╟─6820b40b-423d-4b1d-942d-9a960b8ff390
# ╟─cefb1570-de86-40e2-9139-f7b3d159982b
# ╟─39fd6577-edf2-4e3d-a636-2f19cc17d351
# ╟─b4bfb1ba-73f2-45c9-9303-baee4345f8f6
# ╟─45aa6de1-82f2-41ae-bbed-33d38f14b05b
# ╟─a8beed6c-62ce-452c-adc2-1173fd201cb2
# ╟─fc07ba43-c534-4a9b-a177-a4241a17c92b
# ╟─6d3ee930-9960-4b7f-915d-16398c41a3c0
# ╟─39cf4800-7bb2-4802-9da6-012eb1e2086a
# ╟─bf809430-0c5d-44ad-ae45-68ac558bd494
# ╟─135d1b0a-81e7-4802-a987-391550b22ecf
# ╟─242a181a-a670-421a-a1ed-8e39e7a63324
# ╟─fa621263-43a5-4151-86e9-098e0d8537ff
# ╟─c2435c4d-55ac-4a08-b0cd-72cc505db099
# ╟─21c01e65-6f2e-4a5c-a4a9-0397c52317e5
# ╟─d7a0d175-5487-40b2-b491-28f05ed055ff
# ╟─fc4c2785-4e5b-412a-8f54-b7ba6cd3828c
# ╟─c25f303b-7596-4901-9063-aa1c36249ace
# ╟─4c87bf89-8070-47c2-abcb-e68d708a7fac
# ╟─3e415e55-39af-4f81-a408-3336338d69e2
# ╟─87515234-3bdb-46c2-bcf4-d5d9c675d31c
# ╠═d44526f4-3051-47ee-8b63-f5e694c2e609
# ╠═27755688-f647-48e5-a939-bb0fa70c95d8
# ╠═8d06ce32-2c8d-4317-8c38-108ec0e7fe23
# ╟─d4941adc-253b-407b-a4e6-e02a939634cc
# ╟─a83f34e7-34a5-4578-928c-479f135c2b9c
# ╟─28dc6e55-7494-468a-b03a-652e3547daa5
# ╟─7ee6f500-099a-46c6-aca7-8598f71ada1a
# ╟─d9456461-1b59-4fee-a265-6f1cdc8bb2e6
# ╟─cdd9ce07-9099-430f-a97a-c2347e282a16
# ╟─6d53d19b-38a3-45fa-9c62-94b7d4a69cad
# ╟─d60ef237-32f8-4b91-960b-ae0ab258f317
# ╠═44069d22-92cb-44b8-9775-eab0d9b6240e
# ╟─c488b48a-1f6a-4ee5-b2df-98f16206dac5
# ╠═9ec71356-485b-423e-98f0-eb73b046e5f5
# ╟─4b34d1a6-c854-4b58-8f36-42fb009d0ed6
# ╟─416f552e-79a1-4aa2-9738-9299f2d718a6
# ╟─9ae3a788-0516-4455-972d-b41ce26fa851
# ╟─23d95e32-4999-4032-90dd-570e33af5895
# ╠═6f9f4116-d887-42c7-84af-d6a2fbb16af7
# ╠═e40f7779-9640-4714-881d-a9802b548ed9
# ╠═2b9d6c0d-8006-49bb-9fb4-807bb02fa049
# ╟─72748220-3cc7-4dd4-be9d-3a6d53035081
# ╠═2ed1d3b2-50f3-4d1a-9c13-b1613056b798
# ╟─77edb64c-6acf-4610-b3a4-f2798224584d
# ╟─5fa30e8a-bc79-4fae-a946-865af7f495d4
# ╠═f61e3851-fc4f-43bf-b516-348e4a8edbca
# ╟─daedd76d-2156-40d2-b17e-bbafebbc2336
# ╟─92646ae3-b80c-4734-ba2e-503c13bd14b4
# ╟─02fc47ab-7b4d-4cbc-a6d0-55d2fdab1585
# ╟─56a35a21-b454-4f46-aadb-ceb65237e6ac
# ╠═d37b3d73-eb9c-4b40-8c5a-f6b1b7afc06d
# ╠═af17a502-45ad-4055-a527-e562f705d000
# ╟─62afcad3-a92f-42cc-b535-dc8cb444aef6
# ╠═7dfee1bb-4422-422d-96dd-cd4cc954b645
# ╠═7ea802f1-d18c-4438-840c-9d5e1c20b13a
# ╟─dbb657d0-72d7-48c8-8b10-53fad2911ade
# ╠═f3512172-661d-45a5-8a7b-499df6657266
# ╟─64d31497-9009-49f2-b132-07a81331ac2f
# ╟─a0465ae8-c843-4fc0-abaf-0497ada26652
# ╠═620789b7-59bc-4e17-bcfb-728a329eed0f
# ╠═7b47cda6-d772-468c-a8f3-75e3d77369d8
# ╠═8d0c6fdc-4717-4203-b933-4b37fe60d512
# ╟─d66e373d-8443-4810-9332-305d9781a21a
# ╠═acfb80f0-f4d0-4870-b401-6e26c1c99e45
# ╠═e091ce93-9526-4c7f-9f14-7634419bfe57
# ╠═5a8cdbe7-6abe-4f07-8bcc-89dd71fc35f7
# ╠═c46e0b36-c3fd-4b7f-8f31-25c3315bb10c
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002