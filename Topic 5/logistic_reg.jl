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

# ╔═╡ 86ac8000-a595-4162-863d-8720ff9dd3bd
begin
	using StatsPlots
	using LogExpFunctions
	using StatsBase
	using Distributions
end

# ╔═╡ 9f90a18b-114f-4039-9aaf-f52c77205a49
begin
	using LinearAlgebra
	using PlutoUI
	using PlutoTeachingTools
	using LaTeXStrings
	using Latexify
	using Random
	using Statistics
	using HypertextLiteral
	using Plots; default(fontfamily="Computer Modern", framestyle=:box) # LaTex-style
	
end

# ╔═╡ ece21354-0718-4afb-a905-c7899f41883b
begin
	using Logging
	Logging.disable_logging(Logging.Info); # or e.g. Logging.Info
end;

# ╔═╡ 1aa0cc79-e9ca-44bc-b9ab-711eed853f00
using MLUtils

# ╔═╡ 3e2e1ea8-3a7d-462f-ac38-43a087907a14
TableOfContents()

# ╔═╡ 7bbf37e1-27fd-4871-bc1d-c9c3ecaac076
ChooseDisplayMode()

# ╔═╡ bc96a33d-9011-41ec-a19e-d472cbaafb70
md"""

# CS5914 Machine Learning Algorithms


#### Logistic regression 
\

$(Resource("https://www.st-andrews.ac.uk/assets/university/brand/logos/standard-vertical-black.png", :width=>130, :align=>"right"))

Lei Fang(@lf28 $(Resource("https://raw.githubusercontent.com/edent/SuperTinyIcons/bed6907f8e4f5cb5bb21299b9070f4d7c51098c0/images/svg/github.svg", :width=>10)))

*School of Computer Science*

*University of St Andrews, UK*

"""

# ╔═╡ 7091d2cf-9237-45b2-b609-f442cd1cdba5
md"""

## Topics to cover
	
"""

# ╔═╡ 0a7f37e1-51bc-427d-a947-31a6be5b765e
aside((md"""$(@bind next1 Button("next")) 
$(@bind init1 Button("init"))
	"""))

# ╔═╡ 595a5ef3-4f54-4502-a943-ace4146efa31
begin
	init1
	next_idx = [0];
end;

# ╔═╡ a696c014-2070-4041-ada3-da79f50c9140
begin
	next1
	topics = ["Binary linear classifier: logistic regression", "Probabilistic model and Cross entropy loss", "Regularisation", "Evaluation", "Case study: MNIST dataset"]
	@htl "<ul>$([@htl("""<li>$b</li><br>""") for b in topics[1:min(next_idx[1], length(topics))]])</ul>"
end

# ╔═╡ bc1ee08d-9376-44d7-968c-5e114b09a5e0
let
	next1
	next_idx[1] += 1
end;

# ╔═╡ bcc3cea5-0564-481b-883a-a45a1b870ba3
md"""
## Binary classification



"""

# ╔═╡ 2d70884d-c922-45e6-853e-5608affdd860
md"""

## A first attempt -- use regression


Just treat the categorical labels ``y^{(i)}`` as numerical values

For better visualisation, we replace

```math
	y=0\;\; \text{with}\;\; \tilde{y}=-1
``` 
  * to differentiate from the original label; use ``\tilde{y}``
  * the targets are categorical labels, the label scheme is really arbitrary
"""

# ╔═╡ a7cbd3b5-494a-45ef-9bed-1aa1ba2d7924
md"""

## A first attempt -- use regression (cont.)

The fitting result by **least squared estimation**

"""

# ╔═╡ 8c69c448-c104-4cae-9352-10d4cec512a9
md"""

## Least square -- classification decision boundary


"""

# ╔═╡ bf83caa9-d9f9-48a2-9563-152fcbb8afdc
md"""

##
"""

# ╔═╡ ea55660a-090b-453c-a141-b6867670ba48
md"""

## Least square classifier fails


##### What if we *add some more data* ?

* **_ideally_**, the old decision boundary **shouldn't** change much
"""

# ╔═╡ 3315ef07-17df-4fe1-a7a5-3e8f039c0cc1
linear_reg(X, y; λ = 1) = (X' * X + λ *I) \ X' * y;

# ╔═╡ c936fe77-827b-4245-93e5-d239d467bffd
md"""

## Least square classifier fails


##### The least-square classifier is overly *sensitive* to the extra data

"""

# ╔═╡ cdccf955-19f3-45bb-8dfe-2e15addcdd32
md"""

## Why?


**Firstly**, the sum of squared error (SSE) loss doesn't make sense

```math
\large
\text{SSE Loss}(\mathbf{w}) = \frac{1}{2} \sum_{i=1}^{n} (y^{(i)} - \mathbf{w}^\top \mathbf{x}^{(i)})^2
```

* least square tries to please every data point (equally) in squared loss sense


"""

# ╔═╡ 331571e5-f314-42db-ae56-66e64e485b85
md"""

## Why? (conti.)


**Secondly**, the prediction function 

```math
\large
\begin{align}
h(\mathbf{x}; \mathbf{w}) &= \mathbf{w}^{\top} \mathbf{x}^{(i)} \\
&\in (-\infty, \infty)
\end{align}
```
The target ``y`` is **binary**, but the hyperplane ``\large h(\cdot)\in (-\infty, +\infty)`` is **unbounded**
  * *e.g.* the output ``h(\mathbf{x}) = 100``, or ``-100`` does not make sense


"""

# ╔═╡ 324ea2b8-c350-438d-8c8f-6404045fc19f
md"""

## Towards logistic regression

"""

# ╔═╡ 3a50c68d-5cb1-45f5-a6af-c07ab280b1ad
TwoColumn(
md"""
\
\

> Idea: **squeeze** ``h(\mathbf{x})`` to 0 and 1
> * then interpret it as some **probability** of ``y^{(i)}`` being class 1

"""
,

html"<center><img src='https://carpentries-incubator.github.io/ml4bio-workshop/assets/Linreg-vs-logit_.png
' width = '400' /></center>"	
	
)

# ╔═╡ c8e55a60-0829-4cc7-bc9b-065809ac791c
md"""

## Towards logistic regression


"""

# ╔═╡ 9d9c91b3-ed65-4929-8cfe-4de7a0d6f807
TwoColumn(md"""

\

Recall **logistic function** (also known as **sigmoid**)

```math
\large
\texttt{logistic}(x) \triangleq \sigma(x) = \frac{1}{1+e^{-x}}.
``` 

* ``\sigma`` as a shorthand notation for the logistic activation

* it squeezes a line to 0 and 1
  * ``x\rightarrow \infty``, ``\sigma(x) \rightarrow 1``
   * ``x\rightarrow -\infty``, ``\sigma(x) \rightarrow 0``""", begin
gr()
plot(-10:0.2:10, logistic, c=7, xlabel=L"x",  label=L"\texttt{logistic}(x)", legend=:topleft, lw=3, size=(350,300), ylabel=L"P(y=1)")
end

)

# ╔═╡ 8da5d36d-7fe0-45ee-bbcc-abb9eb2831f6
md"""

## Logistic regression function


Now feed a linear function ``\large h(x) = w_1 x + w_0`` to ``\large \sigma(\cdot)``


```math
\Large
(\sigma \circ h) (x) = \sigma(h(x)) = \sigma(w_1 x+ w_0) = \frac{1}{1+e^{-w_1x -w_0}}
```
"""

# ╔═╡ 6444df20-363c-4db3-acb0-efb3b17d7368
md"Slope: ``w_1`` $(@bind w₁_ Slider([(0.0:0.0001:1.0)...; (1.0:0.005:20)...], default=1.0, show_value=true)),
Intercept: ``w_0`` $(@bind w₀_ Slider(-10:0.5:10, default=0, show_value=true))"

# ╔═╡ 5ffef289-18e0-40c8-be74-de9871a45687
begin
	k, b= w₁_, w₀_
	gr()

	plot(-20:0.1:20, (x) -> k* x+ b, ylim=[-0.2, 1.2], label=L"h(x) =%$(round(w₁_; digits=2)) x + %$(w₀_)", lw=2, l=:gray, ls=:dash, ylabel=L"\sigma(x) \triangleq P(y=1|x)")
	
	plot!(-20:0.1:20, (x) -> logistic( x ), xlabel=L"x", label=L"\sigma(x)", legend=:topleft, lw=2, size=(450,300), c=1)
	
	plot!(-20:0.1:20, (x) -> logistic(k * x + b), xlabel=L"x", label=L"\sigma(%$(round(w₁_; digits=2)) x + %$(w₀_))", legend=:topleft, lw=2, c=7, size=(450,300))
end

# ╔═╡ caef3a1c-c85d-4083-b8e3-4389d81ad6c1
md"""

## Logistic regression function 
##### higher-dimensional
\
It transforms a hyperplane instead of a line:

```math
\large
h(\mathbf{x}) = \mathbf{w}^\top \mathbf{x}
```


```math
\large
\texttt{logistic}(h(\mathbf{x})) = σ(\mathbf{w}^\top \mathbf{x}) = \frac{1}{1+e^{-\mathbf{w}^\top \mathbf{x}}}
```
"""

# ╔═╡ e5853cf2-657c-473d-8617-db74e8f59ea2
wv_ = [1, 1] * 1

# ╔═╡ 206c82ee-389e-4c92-adbf-9578f7125418
meshgrid(x, y) = (repeat(x, outer=length(y)), repeat(y, inner=length(x))) # helper function to create a quiver grid.

# ╔═╡ c03ab032-13b1-41a4-9477-6c79bb4fecd6
begin
	function ∇σ(w, x) 
		wx = w' * x
		logistic(wx) * (1-logistic(wx)) * x
	end
end

# ╔═╡ 85cf441b-8b42-4dba-9bef-e54855cf0ce1
let
	gr()
	w₀ = 0
	w₁, w₂ = wv_[1], wv_[2]
	plot(-5:0.5:5, -5:0.5:5, (x, y) -> logistic(w₀+ w₁* x + w₂ * y), st=:contourf, c=:jet, colorbar=false, alpha=0.5, xlim=[-5, 5], ylim=[-5, 5],  xlabel=L"x_1", ylabel=L"x_2", title="contour plot of "* L"σ(w^\top x)", ratio=1)
	α = 2
	xs_, ys_ = meshgrid(range(-5, 5, length=10), range(-5, 5, length=15))
	∇f_d(x, y) = ∇σ([1, x, y], [w₀, w₁, w₂])[2:end] * α
	quiver!(xs_, ys_, quiver = ∇f_d, c=:black)
end

# ╔═╡ 1a062043-8cfe-4946-8eb0-688d3896229e
md"""

## Logistic regression -- as a graph


The function is actually a simpliest **neural network**


```math
\large 
\sigma(\mathbf{x}) = \frac{1}{1+ e^{- \mathbf{w}^\top\mathbf{x}}}
``` 


which can be represented as a **computational graph**

* ``z = \mathbf{w}^\top \mathbf{x}`` is called the *logit* value
"""

# ╔═╡ 88093782-d180-4422-bd92-357c716bfc89
html"<center><img src='https://leo.host.cs.st-andrews.ac.uk/figs/CS3105/logistic_reg_neuron.png
' width = '400' /></center>"

# ╔═╡ 2459a2d9-4f48-46ab-82e5-3968e713f15f
html"<center><img src='https://carpentries-incubator.github.io/ml4bio-workshop/assets/logit_nodes.png
' width = '400' /></center>";

# ╔═╡ 52ff5315-002c-480b-9a4b-c04124498277
md"""
## Recap: probabilistic linear regression model


> $\large \begin{align}p(y^{(i)}|\mathbf{x}^{(i)}, \mathbf{w}, \sigma^2) &= \mathcal{N}(y^{(i)};  \mathbf{w}^\top \mathbf{x}^{(i)} , \sigma^2)\end{align}$

* ``y^{(i)}`` is a univariate Gaussian with mean $\mathbf{w}^\top \mathbf{x}^{(i)}$ and variance $\sigma^2$ 


"""

# ╔═╡ 0469b52f-ce98-4cfa-abf3-be53630b30dd
md"""

## Probabilistic model for logistic regression



Since ``y^{(i)} \in \{0,1\}``, a natural choice of **likelihood** function is Bernoulli (the outcome is binary)

> ```math
> \large
> 
> p(y^{(i)}|\mathbf{w}, \mathbf{x}^{(i)}) = \texttt{Bernoulli}(\sigma^{(i)}) =\begin{cases}\sigma(\mathbf{w}^\top\mathbf{x}^{(i)}) & y^{(i)} =1
> \\
> 1-\sigma(\mathbf{w}^\top\mathbf{x}^{(i)}) & y^{(i)} = 0   \end{cases}
> ```

* short-hand notation ``\sigma^{(i)} =\sigma(\mathbf{w}^\top\mathbf{x}^{(i)})``


"""

# ╔═╡ 4f884fee-4934-46c3-8352-0105652f8537
md"""


## Probabilistic model for linear regression (cont.)



Since ``y^{(i)} \in \{0,1\}``, a natural choice of **likelihood** function is Bernoulli (the outcome is binary)

> ```math
> \large
> 
> p(y^{(i)}|\mathbf{w}, \mathbf{x}^{(i)}) = \texttt{Bernoulli}(\sigma^{(i)}) =\begin{cases}\sigma(\mathbf{w}^\top\mathbf{x}^{(i)}) & y^{(i)} =1
> \\
> 1-\sigma(\mathbf{w}^\top\mathbf{x}^{(i)}) & y^{(i)} = 0   \end{cases}
> ```

* short-hand notation ``\sigma^{(i)} =\sigma(\mathbf{w}^\top\mathbf{x}^{(i)})``

##### The generative story therefore is

---


Given fixed ``\{\mathbf{x}^{(i)}\}``, which are assumed fixed and non-random


for each ``\mathbf{x}^{(i)}``
  * compute regression function ``\sigma(\mathbf{x}^{(i)}) =\sigma(\mathbf{w}^\top \mathbf{x}^{(i)})``
  * *generate* ``y^{(i)} \sim \texttt{Bernoulli}(\sigma(\mathbf{x}^{(i)}))``

---
"""

# ╔═╡ 9dbf5502-fa44-404f-88ae-be3488e3e41c
md"""

##

"""

# ╔═╡ 6e92fa09-cf58-4d3c-9864-c6617f1e54d7
md"Add true function ``\sigma(x; \mathbf{w})``: $(@bind add_h CheckBox(default=false)),
Add ``p(y^{(i)}|x^{(i)})``: $(@bind add_pyi CheckBox(default=false)),
Add ``y^{(i)}\sim p(y^{(i)}|x^{(i)})``: $(@bind add_yi CheckBox(default=false))
"

# ╔═╡ 6173a35d-1e28-45ac-93b8-d2fb1117bf02
begin
	Random.seed!(123)
	n_obs = 20
	# the input x is fixed; non-random
	# xs = range(-1, 1; length = n_obs)
	xs = sort(rand(n_obs) * 2 .- 1)
	true_w = [0.0, 1.0] * 10
	# true_σ² = 0.05
	ys = zeros(Bool, n_obs)
	for (i, xⁱ) in enumerate(xs)
		hⁱ = true_w' * [1, xⁱ]
		# ys[i] = hⁱ + rand(Normal(0, sqrt(true_σ²)))
		ys[i] = rand() < logistic(hⁱ)
	end
end

# ╔═╡ 8fd7703b-db80-4aa4-977f-c7e3ad1f9fb6
TwoColumn(md"""

To be more specific, the "story" for ``y^{(i)}`` is


---


Given fixed ``\{\mathbf{x}^{(i)}\}``, which are assumed fixed and non-random


for each ``\mathbf{x}^{(i)}``
  * *true signal* ``h(\mathbf{x}^{(i)}) =\mathbf{w}^\top \mathbf{x}^{(i)}``
  * *generate* ``y^{(i)} \sim \mathcal{N}(\mu^{(i)}=h(\mathbf{x}^{(i)}), \sigma^2)``

---

""", let

	plt = plot(xs, zeros(length(xs)), st=:scatter, framestyle=:origin, labels=L"x", color=:black, ms=5, markershape=:x, xlabel=L"x", ylim=[-2.2, 2.5], ylabel=L"y", legend=:outerbottom, size=(400,450))
	true_w =[0, 1]
	plot!(-1:0.1:1.1, (x) -> true_w[1] + true_w[2]*x, lw=2, label="the true signal: " * L"h(x)")
	true_σ² = 0.05
	σ²0 = true_σ²
	xis = xs

	ys = xs * true_w[2] .+ true_w[1] + randn(length(xs)) * sqrt(σ²0)
	anim = @animate for i in 1:length(xis)
		x = xis[i]
		μi = dot(true_w, [1, x])
		σ = sqrt(σ²0)
		xs_ = μi- 4 * σ :0.05:μi+ 4 * σ
		ys_ = pdf.(Normal(μi, sqrt(σ²0)), xs_)
		ys_ = 0.1 *ys_ ./ maximum(ys_)
		# scatter!([x],[μi], markerstrokewidth =1, markershape = :diamond, c=:grey, label="", markersize=3)
		plot!(ys_ .+x, xs_, c=:grey, label="", linewidth=.5)
		scatter!([xis[i]],[ys[i]], markershape = :circle, label="", c=1, markersize=4)
	end

	gif(anim; fps=4)
end)

# ╔═╡ 866e6bfe-748b-424c-b0af-b6f51c15be21
md"
Select ``x^{(i)}``: $(@bind add_i Slider(1:length(xs); show_value=true))
"

# ╔═╡ 1ae4fa36-0faa-438b-8be3-292ec7b617a0
TwoColumnWideLeft(let
	gr()
	plt = plot(xs, zeros(length(xs)), st=:scatter, framestyle=:origin, labels=L"x", color=:black, ms=5, markershape=:x, xlabel=L"x", ylim=[-0.2, 1.2], ylabel=L"y", legend=:outerbottom, size=(450,350))

	if add_h
		plot!(-1.1:0.01:1.1, (x) -> logistic(true_w[1] + true_w[2]*x), lw=2, label="the true signal: " * L"h(x)")
	end
	# σ²0 = true_σ²
	xis = xs
	i = add_i

	if add_yi
		shapes = [:diamond, :circle]
		scatter!([xis[i]],[ys[i]], markershape = shapes[ys[i] + 1], label="observation: "*L"y^{(i)}\sim \texttt{Bern}(\sigma^{(i)})", c = ys[i] +1, markersize=8)
	end

	plt
end, 
	let 
	gr()
	xis = xs
	i = add_i
	if add_pyi
		x = xis[i]
		μi = dot(true_w, [1, x])
		σ = logistic(μi)
		# scatter!([x],[μi], markerstrokewidth =1, markershape = :diamond, c=:grey, label="signal: "*L"h(x)", markersize=3)
		# plot!(ys_ .+x, xs_, c=:grey, label="", linewidth=2)
		plot(["y = 0", "y = 1"], [1-σ, σ], st=:bar, orientation=:v, size=(250,250), title="Bias: "*L"\sigma(x^{(i)})=%$(round(σ, digits=2))", ylim =(0, 1.02), label="", ylabel=L"P(y|x)" )
	else
		plot(size=(250,250))
	end
	
end)

# ╔═╡ 3e980014-daf7-4d8b-b9e6-14a5d078e3b6
md"""
## Demonstration (animation)
"""

# ╔═╡ 6cbddc5d-ae3f-43ac-9b7a-bbc779739353
begin
	bias = 0.0 # 2
	slope = 1 # 10, 0.1
end;

# ╔═╡ 5d2f56e8-21b2-4aa9-b450-40f7881489e0
let
	gr()
	n_obs = 20
	# logistic = σ
	Random.seed!(4321)
	xs = sort(rand(n_obs) * 10 .- 5)
	true_w = [bias, slope]
	# true_σ² = 0.05
	ys = zeros(Bool, n_obs)
	for (i, xⁱ) in enumerate(xs)
		hⁱ = true_w' * [1, xⁱ]
		# ys[i] = hⁱ + rand(Normal(0, sqrt(true_σ²)))
		ys[i] = rand() < logistic(hⁱ)
	end

	x_centre = -true_w[1]/true_w[2]

	plt = plot(xs, zeros(length(xs)), st=:scatter, framestyle=:origin, labels=L"x", color=:black, ms=5, markershape=:x, xlabel=L"x", ylim=[-0.1, 1.2], ylabel=L"y", legend=:outerbottom)
	# true_w =[0, 1]
	plot!(plt, min(-5 + x_centre, -5):0.01:max(x_centre +5, 5), (x) -> logistic(true_w[1] + true_w[2]*x), lw=1.5, label=L"\sigma(x)", title="Probabilistic model for logistic regression")
	plot!(plt, min(-5 + x_centre, -5):0.01:max(x_centre +5, 5), (x) -> 1-logistic(true_w[1] + true_w[2]*x),lc=1, lw=1.5, label=L"1-\sigma(x)")

	xis = xs

	anim = @animate for i in 1:length(xis)
		x = xis[i]
		scatter!(plt, [xis[i]],[ys[i]], markershape = :circle, label="", c=ys[i]+1, markersize=5)
		vline!(plt, [x], ls=:dash, lc=:gray, lw=0.2, label="")
		plt2 = plot(Bernoulli(logistic(true_w[1] + true_w[2]*x)), st=:bar, yticks=(0:1, ["negative", "positive"]), xlim=[0,1.01], orientation=:h, yflip=true, label="", title=L"p(y|{x})", color=1:2)
		plot(plt, plt2, layout=grid(2, 1, heights=[0.85, 0.15]), size=(650,500))
	end
	# ys = xs * true_w[2] .+ true_w[1] + randn(length(xs)) * sqrt(σ²0)
	
	# 	x = xis[i]
	# 	
	# end

	gif(anim; fps=4)
end

# ╔═╡ 64a5e292-14b4-4df0-871d-65d9fec6201d
# let
# 	gr()
# 	Random.seed!(4321)
# 	xs = sort(rand(n_obs) * 10 .- 5)
# 	true_w = [bias, slope]
# 	# true_σ² = 0.05
# 	ys = zeros(Bool, n_obs)
# 	for (i, xⁱ) in enumerate(xs)
# 		hⁱ = true_w' * [1, xⁱ]
# 		# ys[i] = hⁱ + rand(Normal(0, sqrt(true_σ²)))
# 		ys[i] = rand() < logistic(hⁱ)
# 	end

# 	x_centre = -true_w[1]/true_w[2]

# 	plt = plot(xs, zeros(length(xs)), st=:scatter, framestyle=:origin, labels=L"x", color=:black, ms=5, markershape=:x, xlabel=L"x", ylim=[-0.1, 1.2], ylabel=L"y", legend=:outerbottom)
# 	# true_w =[0, 1]
# 	plot!(min(-5 + x_centre, -5):0.01:max(x_centre +5, 5), (x) -> logistic(true_w[1] + true_w[2]*x), lw=2, label="the regression function: " * L"\sigma(x)", title="Probabilistic model for logistic regression")

# 	xis = xs

# 	# ys = xs * true_w[2] .+ true_w[1] + randn(length(xs)) * sqrt(σ²0)
# 	anim = @animate for i in 1:length(xis)
# 		x = xis[i]
# 		scatter!([xis[i]],[ys[i]], markershape = :circle, label="", c=ys[i]+1, markersize=5)
# 	end

# 	gif(anim; fps=5)
# end

# ╔═╡ 96b18c60-87c6-4c09-9882-fbbc5a53f046
md"""

## Probabilistic model for binary classification



> ```math
> \large
> 
> p(y^{(i)}|\mathbf{w}, \mathbf{x}^{(i)}) = \texttt{Bernoulli}(\sigma^{(i)}) =\begin{cases}\sigma(\mathbf{w}^\top\mathbf{x}^{(i)}) & y^{(i)} =1
> \\
> 1-\sigma(\mathbf{w}^\top\mathbf{x}^{(i)}) & y^{(i)} = 0   \end{cases}
> ```

* short-hand notation ``\sigma^{(i)} =\sigma(\mathbf{w}^\top\mathbf{x}^{(i)})``


Recall the above likelihood can be written in one-line

```math
\large
p(y^{(i)}|\mathbf{w}, \mathbf{x}^{(i)}) =  (\sigma^{(i)})^{y^{(i)}} (1-\sigma^{(i)})^{1- y^{(i)}}
```


And its log transformation (log-likelihood) is 


```math
\large
\ln p(y^{(i)}|\mathbf{w}, \mathbf{x}^{(i)}) =  {y^{(i)}} \ln \sigma^{(i)}+ (1- y^{(i)}) \ln (1-\sigma^{(i)})
```
"""

# ╔═╡ 35d5f991-5194-4174-badc-324ad7d15ac3
md"""

## (Log-)likelihood function



For logistic regression model

* the unknown (hypothesis) are: ``\mathbf{w}``

* the data observations are: ``\mathcal{D} =\{y^{(1)}, y^{(2)}, \ldots, y^{(n)}\}``

* as usual, ``\{\mathbf{x}^{(i)}\}`` are assumed fixed



Therefore, the log-likelihood function is

```math
\large
\begin{align}
\ln p(\mathcal{D}|\mathbf{w}, \sigma^2, \{\mathbf{x}^{i}\}) &= \sum_{i=1}^n \ln p(y^{(i)}|\mathbf{w}, \mathbf{x}^{(i)}) \\

&= \sum_{i=1}^n {y^{(i)}} \ln \sigma^{(i)}+ (1- y^{(i)}) \ln (1-\sigma^{(i)})
\end{align}
```


In practice, we use take the average/mean loss rather than the sum

```math
\large
\frac{1}{n}\ln p(\mathcal{D}|\mathbf{w}, \sigma^2, \{\mathbf{x}^{i}\}) =\colorbox{pink}{$\frac{1}{n}$}\sum_{i=1}^n {y^{(i)}} \ln \sigma^{(i)}+ (1- y^{(i)}) \ln (1-\sigma^{(i)})
```



## MLE

**Learning** is just to maximise the log likelihood function *w.r.t.* ``\mathbf{w}``, which is **MLE**


```math
\hat{\mathbf{w}}_{\text{MLE}} \leftarrow \arg\max_{\mathbf{w}} \ln p(\mathcal{D}|\mathbf{w}, \sigma^2, \{\mathbf{x}^{i}\})
```

or minimise its negative (also known as **cross-entropy** (CE) loss)


```math
\large
\hat{\mathbf{w}}_{\text{MLE}} \leftarrow \arg\min_{\mathbf{w}} \underbrace{-\ln p(\mathcal{D}|\mathbf{w}, \sigma^2, \{\mathbf{x}^{i}\})}_{\text{CE loss}(\mathbf{w})}
```

"""

# ╔═╡ f693814d-9639-4abb-a5b4-e83fe7a28a77
md"""

## Cross-entropy loss -- how ?


##### *Cross-entropy* loss but _WHY_ and _How_?
\

Consider one observation ``\{\mathbf{x}^{(i)}, y^{(i)}\}`` only, the loss is

```math
\large
\begin{align}
 \text{CE}(\mathbf{w}) = - {y^{(i)}} \ln \sigma^{(i)}- (1- y^{(i)}) \ln (1-\sigma^{(i)})
\end{align}
```
When ``y^{(i)} = 1``, *i.e.* the true label is 1, the loss becomes 

```math
\large
\begin{align}
y^{(i)} = 1\; \Rightarrow\; \text{CE error}^{(i)} &= - 1 \ln \sigma^{(i)}- (1- 1) \ln (1-\sigma^{(i)})\\
&= - \ln (\sigma^{(i)})
\end{align}
```
"""

# ╔═╡ 67f7449f-19b8-4607-9e32-0f8a16a806c0
TwoColumn(md"""

\
\


When ``y^{(i)} = 1``, *i.e.* the true label is 1, the loss becomes 

\

```math
\begin{align}
 \text{CE}(\mathbf{w}) &= - 1 \ln \sigma^{(i)}- (1- 1) \ln (1-\sigma^{(i)})\\
&= - \ln (\sigma^{(i)})
\end{align}
```


* when the prediction is correct,  the loss is zero
* when the prediction is wrong, the loss is `Inf` loss

""", let
	gr()
	plot(0:0.005:1, (x) -> -log(x), lw=2, xlabel="Predicted probability: "* L"\sigma^{(i)}", ylabel="loss", label=L"-\ln \sigma^{(i)}", title=L"y^{(i)}=1"* ": i.e. class label is 1", annotate = [(1, 0.9, text("perfect pred", "Computer Modern", :right, rotation = 270 ,:green, 12)), (0.11, 3.5, text("worst pred", :right, "Computer Modern", rotation = 270 ,:red, 12))], size=(350,350))


	quiver!([1], [0.8], quiver=([1-1], [0-0.8]), c=:green, lw=3)
	quiver!([0.07], [5], quiver=([0.0- 0.06], [5-5]), c=:red, lw=3)

	quiver!([0.25], [2], quiver=([0.75- 0.25], [2-2]), c=:green, lw=3)
	quiver!([0.75], [3], quiver=([0.25- 0.75], [3-3]), c=:red, lw=3)
	annotate!(0.5, 2.1, text("better","Computer Modern", :green, :bottom))
	annotate!(0.5, 3.1, text("worse","Computer Modern", :red, :bottom))
end)

# ╔═╡ fb0361a6-c967-4dcd-9002-55ae25e225a5
aside(tip(md"""

Recall ``\sigma^{(i)} = p(y^{(i)}=1|\mathbf{x}^{(i)})``

"""))

# ╔═╡ 5c0eaab3-de6d-457f-a0c3-8ea6b5da2c88
md"""

##
"""

# ╔═╡ de93ac5e-bec6-4764-ac6d-f84076ff20ee
TwoColumn(md"""

\
\


When ``y^{(i)} = 0``, *i.e.* the true label is 0, the loss becomes 

\

```math
\begin{align}
 \text{CE}(\mathbf{w}) &= - 0 \ln \sigma^{(i)}- (1- 0) \ln (1-\sigma^{(i)})\\
&= - \ln (1-\sigma^{(i)})
\end{align}
```
""", let
	gr()
	plot(0:0.005:1, (x) -> -log(1-x), lw=2, xlabel="Predicted probability: "* L"\sigma^{(i)}", ylabel="loss", label=L"-\ln(1-\sigma^{(i)})", title=L"y^{(i)}=0"* ": the true class label is 0", size=(350,350))


	# quiver!([0], [0.8], quiver=([1-1], [0-0.8]), c=:red, lw=3)
	
	# quiver!([0.07], [5], quiver=([0.0- 0.06], [5-5]), c=:green, lw=3)

	quiver!([0.25], [2], quiver=([0.75- 0.25], [2-2]), c=:red, lw=3)
	quiver!([0.75], [3], quiver=([0.25- 0.75], [3-3]), c=:green, lw=3)
	annotate!(0.5, 2.1, text("worse", "Computer Modern", :red, :bottom))
	annotate!(0.5, 3.1, text("better", "Computer Modern", :green, :bottom))
end)

# ╔═╡ a50cc950-45ca-4745-9f47-a3fbb28db782
md"""


## Surrogate loss


"""

# ╔═╡ 1421c748-083b-4977-8a88-6a39c9b9906d
TwoColumn(md"""

\
\

For classification, a more **natural** **loss** is 1/0 classification accuracy:


```math
\large
\begin{align}
\ell_{1/0}(\mathbf{y}, \hat{\mathbf{y}}) = \frac{1}{n}\sum_{i=1}^n \mathbb{I}(y^{(i)} \neq \hat{y}^{(i)})\\
\text{where }\; \hat{y}^{(i)} = \mathbb{I}(\sigma^{(i)} > 0.5)
\end{align}
```

* however, this loss is **not** differentiable 
* the gradients are zero _everywhere_

""", let
	gr()

	plot(0:0.001:1, (x) -> x < .5, lw=4, xlabel="Predicted probability: "* L"\sigma^{(i)}", ylabel="loss", label=L"1/0"* " loss", title="When the class label is 1", ylim =[-0.2, 4], size=(350,350))

	# plot(0:0.005:1, (x) -> -log(x), lw=2, xlabel="Predicted probability: "* L"\sigma^{(i)}", ylabel="loss", label=L"-\ln \sigma^{(i)}", title=L"y^{(i)}=1"* ": i.e. class label is 1",  size=(550,350))


	# quiver!([1], [0.8], quiver=([1-1], [0-0.8]), c=:green, lw=3)
	# quiver!([0.07], [5], quiver=([0.0- 0.06], [5-5]), c=:red, lw=3)

	quiver!([0.25], [2], quiver=([0.75- 0.25], [2-2]), c=:green, la=0.5, lw=3)
	quiver!([0.75], [3], quiver=([0.25- 0.75], [3-3]), c=:red, lw=3)
	annotate!(0.5, 2.1, text("better", "Computer Modern", :green, :bottom))
	annotate!(0.5, 3.1, text("worse", "Computer Modern", :red, :bottom))
end)

# ╔═╡ 82baa991-6df5-4561-9643-52db96c5e99b
md"""


## Surrogate loss

"""

# ╔═╡ 9a02f4f8-38d6-44a4-9118-1440bfc4d271
TwoColumn(md"""

\
\
\
\

**Cross-entropy**: is a **surrogate loss** for the **1/0** loss
* it **approximates**/**traces** the desired loss

""", let
	gr()

	plot(0:0.001:1, (x) -> x < .5, lw=2, xlabel="Predicted probability: "* L"\sigma^{(i)}", ylabel="loss", label=L"1/0"* " loss", title="When the class label is 1", ylim =[-0.2, 5], size=(350,350))

	plot!(0:0.005:1, (x) -> -log(x), lw=2, xlabel="Predicted probability: "* L"\sigma^{(i)}", ylabel="loss", label="Cross Entropy loss")


	# plot!(0:0.005:1, (x) -> (1-x)^2+ (x-1)^2, lw=2, lc=4, ls=:dash, label="Bier loss")
	# quiver!([1], [0.8], quiver=([1-1], [0-0.8]), c=:green, lw=3)
	# quiver!([0.07], [5], quiver=([0.0- 0.06], [5-5]), c=:red, lw=3)

	quiver!([0.25], [2], quiver=([0.75- 0.25], [2-2]), c=:green, la=0.5, lw=3)
	quiver!([0.75], [3], quiver=([0.25- 0.75], [3-3]), c=:red, lw=3)
	annotate!(0.5, 2.1, text("better", "Computer Modern", :green, :bottom))
	annotate!(0.5, 3.1, text("worse", "Computer Modern", :red, :bottom))
end)

# ╔═╡ ce822f87-d2c4-434b-9b5d-b629962d2df2
md"""

# Logistic regression -- Learning
"""

# ╔═╡ 9d25c603-07b3-4758-9b28-0d2041e18569
md"""

## Learning -- gradient descent


**Learning** is to minimise the **cross-entropy** (CE) loss


```math
\large
\hat{\mathbf{w}}_{\text{MLE}} \leftarrow \arg\min_{\mathbf{w}} \underbrace{-\frac{1}{n}\ln p(\mathcal{D}|\mathbf{w}, \{\mathbf{x}^{(i)}\})}_{\text{CE loss}(\mathbf{w})}
```

> Optimisation: **gradient descent**

##

----


**Bach gradient descent algorithm:**



* random guess ``\large\mathbf{w}_0``

* while **not converge**
  * ``\large\mathbf{w}_t \leftarrow \mathbf{w}_{t-1} - \gamma \nabla L(\mathbf{w}_{t-1})``
-----

and recall the binary **C**ross **E**ntropy (CE) _loss_ is 


```math
\large
L(\mathbf{w}) =\frac{1}{n} \sum_{i=1}^n L^{(i)}(\mathbf{w}) = \frac{1}{n}\sum_{i=1}^n \underbrace{-{y^{(i)}} \ln \sigma^{(i)}- (1- y^{(i)}) \ln (1-\sigma^{(i)})}_{L^{(i)}(\mathbf{w})}
```

"""

# ╔═╡ 929916d2-9abf-40b2-9399-85c3ba05989b
md"""

## Learning -- gradient descent


**Learning** is to minimise the **cross-entropy** (CE) loss


```math
\large
\hat{\mathbf{w}}_{\text{MLE}} \leftarrow \arg\min_{\mathbf{w}} \underbrace{-\frac{1}{n}\sum_{i=1}^n {{y^{(i)}} \ln \sigma^{(i)}+ (1- y^{(i)}) \ln (1-\sigma^{(i)})}}_{\text{CE loss}(\mathbf{w})}
```

> Optimisation: **gradient descent**


##

The **gradient** for _logistic regression_ is


```math
\large
\nabla_{\mathbf{w}}L(\mathbf{w})  = -\frac{1}{n}\sum_{i=1}^n \underbrace{(y^{(i)} - \sigma^{(i)})}_{\text{pred. error for }i} \cdot \mathbf{x}^{(i)} 
```

Recall the **gradient** for _linear regression_ is


```math
\large
\nabla_{\mathbf{w}}L(\mathbf{w})  = -\frac{1}{n}\sum_{i=1}^n \underbrace{(y^{(i)} - \mathbf{w}^{\top} \mathbf{x}^{(i)})}_{\text{pred. error for }i} \cdot \mathbf{x}^{(i)} 
```

> The same idea: gradient is proportional to the prediction quality
> * prediction is perfect: gradient is near zero
> * otherwise, increase/decrease accordingly
"""

# ╔═╡ 8743242e-7f7f-4f54-b27c-1f08d1a110e2
aside(tip(md"""


Use the fact

```math
\sigma'(x) = \sigma(x) (1-\sigma(x))
```

Note that 


```math

\sigma^{(i)} = \sigma(\mathbf{w}^\top\mathbf{x}^{(i)}),
```

The gradient *w.r.t.* ``\mathbf{w}`` can be easily derived based on chain rule


Let ``z = \mathbf{w}^\top\mathbf{x}^{(i)}``, and ``\sigma^{(i)} = \sigma(z)``,

Based on the chain rule:

```math
\begin{align}
\nabla_{\mathbf{w}} \sigma^{(i)} &= \frac{\partial \sigma^{(i)}}{\partial z} \frac{\partial z}{\partial \mathbf{w}} \\
&= \sigma^{(i)} (1- \sigma^{(i)}) \cdot \mathbf{x}^{(i)}

\end{align}
```

"""))

# ╔═╡ aaaadaa8-d2e9-4647-82de-5547b2d6ddb4
md"""

## Gradient -- matrix notation

The gradient is


```math
\large
\nabla_{\mathbf{w}}L(\mathbf{w})  = - \frac{1}{n}\sum_{i=1}^n \underbrace{(y^{(i)} - \sigma^{(i)})}_{\text{pred. error for }i} \cdot \mathbf{x}^{(i)} 
```

In matrix notation, 

```math
\large
\nabla L(\mathbf{w}) =\frac{1}{n} \mathbf{X}^\top (\boldsymbol{\sigma} - \mathbf{y})
```

where 

```math
\large
\boldsymbol{\sigma} = \begin{bmatrix}
\sigma({\mathbf{w}^\top\mathbf{x}^{(1)}})\\
\sigma({\mathbf{w}^\top\mathbf{x}^{(2)}})\\
\vdots\\

\sigma({\mathbf{w}^\top\mathbf{x}^{(n)}})
\end{bmatrix} = \begin{bmatrix}
\sigma^{(1)}\\
\sigma^{(2)}\\
\vdots\\

\sigma^{(n)}
\end{bmatrix}
```
"""

# ╔═╡ 6e62b53d-68d8-45de-9a0f-8223b5d2df92
md"""

## Gradient derivation (details)


Consider one training sample ``(\mathbf{x}^{(i)}, y^{(i)})``'s loss ``L^{(i)} (\mathbf{w})`` and its gradient ``\nabla{L}^{(i)}(\mathbf{w})``, where

```math
\begin{align}
L^{(i)}(\mathbf{w}) 
&= - {y^{(i)}} \ln \sigma^{(i)}- (1- y^{(i)}) \ln (1-\sigma^{(i)})
\end{align}
```

Note that the full batch gradient is the mean of the gradients ``\nabla{L}(\mathbf{w}) = \frac{1}{n}\sum_{i=1}^n \nabla{L}^{(i)}(\mathbf{w})``.



```math
\begin{align}
\nabla{L}^{(i)}(\mathbf{w}) &= \nabla\left (- {y^{(i)}} \ln \sigma^{(i)}- (1- y^{(i)}) \ln (1-\sigma^{(i)})\right )\\
&= - {y^{(i)}}  \nabla \ln \sigma^{(i)}- (1- y^{(i)}) \nabla \ln (1-\sigma^{(i)})\\
&= - {y^{(i)}} \frac{1}{\sigma^{(i)}} \nabla \sigma^{(i)}- (1- y^{(i)})\frac{1}{1-\sigma^{(i)}} \nabla  (1-\sigma^{(i)})\\
&= - {y^{(i)}} \frac{\sigma^{(i)} (1-\sigma^{(i)})}{\sigma^{(i)}}  \nabla \mathbf{w}^\top \mathbf{x}^{(i)} - (1- y^{(i)})\frac{-\sigma^{(i)} (1-\sigma^{(i)})}{1-\sigma^{(i)}}  \nabla \mathbf{w}^\top \mathbf{x}^{(i)}\\
&= - {y^{(i)}} (1-\sigma^{(i)})\cdot\mathbf{x}^{(i)} - (1- y^{(i)})(-\sigma^{(i)}) \cdot\mathbf{x}^{(i)}\\
&= (- {y^{(i)}} +  {y^{(i)}}\sigma^{(i)} + \sigma^{(i)} -  {y^{(i)}}\sigma^{(i)} ) \cdot\mathbf{x}^{(i)}\\
&= -( {y^{(i)}}-\sigma^{(i)} ) \cdot \mathbf{x}^{(i)}
\end{align}
```
"""

# ╔═╡ 7ed0de94-b513-40c1-9f83-6ed75fcd4cdd
md"""

## Learing algorithm -- Implementation


----


**Bach gradient descent algorithm:**



* random guess ``\large\mathbf{w}_0``

* while **not converge**
  * ``\large\mathbf{w}_t \leftarrow \mathbf{w}_{t-1} - \gamma \nabla L(\mathbf{w}_{t-1})``
-----

"""

# ╔═╡ 188e9981-9597-483f-b1e3-689e26389b61
md"""


**Implementation in Python**


```python
def logistic(x):    
    return 1/ (1 + np.exp(-x))
```


```python
for i in range(0, max_iters):
	yhats = logistic(Xs@w)
    grad =  Xs.T@(yhats - ys) / np.size(ys)
    w = w - gamma * grad 
```


"""

# ╔═╡ 50d9a624-20d8-4feb-a565-4efcaf22fe27
# md"""

# ## Gradient implementation and check


# Now check the gradient as a routine task
# * checking gradient should become your habit
# """

# ╔═╡ ec78bc4f-884b-4525-8d1b-138b37274ee7
begin
	function logistic_loss(w, X, y)
		σ = logistic.(X * w)
		# deal with boundary cases such as σ = 0 or 1, log(0) gracefully
		# sum(y .* log.(σ) + (1 .- y).* log.(1 .- σ))
		# rather you should use xlogy and xlog1py
		-sum(xlogy.(y, σ) + xlog1py.(1 .-y, -σ))
	end
end;

# ╔═╡ e88db991-bf54-4548-80cb-7cd307300ec9
function ∇logistic_loss(w, X, y)
	σ = logistic.(X * w)
	X' * (σ - y)
end;

# ╔═╡ 99c3f5ee-63d0-4f6f-90d9-2e524a7e945a
md"""

## Demonstration -- gradient descent


"""

# ╔═╡ ff397d54-42b3-40af-bf36-4716fd9a4419
md"""

## Stochastic gradient descent



**_Batch_ gradient descent algorithm:**
 
* *batch*: means the **whole** ``n`` training instances

----
* random guess ``\large\mathbf{w}_0``

* for each *epoch*
  * run through all observations in training data
  * ``\large\mathbf{w}_t \leftarrow \mathbf{w}_{t-1} - \gamma \nabla L(\mathbf{w}_{t-1})``
-----



_However_, the batch gradient can be prehibitively **expensive** to compute

```math
\Large
\nabla L(\mathbf{w})  = \boxed{\frac{1}{n}\sum_{i=1}^n \underbrace{\colorbox{pink}{${-(y^{(i)} - \sigma^{(i)})}\cdot \mathbf{x}^{(i)}$}}_{\nabla L^{(i)}(\mathbf{w})}}_{\text{too expensive!}}
```

* it requires to store all ``n`` training data ``\{\mathbf{x}^{(i)}, y^{(i)}\}`` in memory: may not be feasible!





"""

# ╔═╡ fcc49f3c-6219-40b8-a5d0-85fa70738a8d
md"""


## Solution: stochastic gradient descent (SGD)

"""

# ╔═╡ 6dc74312-0e3b-4ead-b369-f9c2b70ab3d3
TwoColumn(md"""

\
\
\
\
\

**Idea**: instead using all ``n``, use just **one random observation**'s gradient
\

```math
\begin{align}
\nabla L(\mathbf{w})  &= {\frac{1}{n}\sum_{i=1}^n \underbrace{\colorbox{pink}{${-(y^{(i)} - \sigma^{(i)})}\cdot \mathbf{x}^{(i)}$}}_{\nabla L^{(i)}(\mathbf{w})}}\\
&\approx \nabla L^{(i)}(\mathbf{w})
\end{align}
```

* noisy gradient instead of the mean
* might even help avoid local optimum as a by-product


""", let
	gr()
	vv = [.5, .5] *2
	plt = plot([0, vv[1]], [0, vv[2]], lc=:blue, arrow=Plots.Arrow(:closed, :head, 10, 10),  st=:path, lw=2, c=:red,  xlim=[-.2, 2], ylim=[-.2, 2], ratio=1, label="",framestyle=:none, legend=:bottomright, size=(350,450), legendfontsize=12)
	annotate!([vv[1] + .1], [vv[2] + .05], (L"\nabla L",14, :blue))
	Random.seed!(123)
	v = vv + randn(2) ./ 2.5
	annotate!([v[1] + .3], [v[2] + .1], (L"\nabla L^{(i)}", 12, :gray))
	plot!([0, v[1]], [0, v[2]], lc=:gray, arrow=Plots.Arrow(:open, :head,1,1),  st=:path, lw=.3, label="")
	for i in 1:15
		v = vv + randn(2) ./ 2.5
		plot!([0, v[1]], [0, v[2]], lc=:gray, arrow = Plots.Arrow(:open, :head,1,1),  st=:path, lw=.3, label="")
	end
	plt
end)

# ╔═╡ e0f3cee1-b6ee-4399-8e4f-c0d70b94723e
md"""

## SGD




**Stochastic gradient descent algorithm:**

-----

* random guess ``\large\mathbf{w}_0``

* for each *epoch*
  * randomly shuffle the data
  * for each ``i \in 1\ldots n``
    * ``\large\mathbf{w}_t \leftarrow \mathbf{w}_{t-1} - \gamma_t \nabla L^{(i)}(\mathbf{w}_{t-1})``
-----


The learning rate ``\gamma_t`` usually is adaptive or delaying (but small constant works reasonably well)

```math 
\large \gamma_t = \frac{1}{{iter}}\;\; \text{or}\;\; \gamma_t = \frac{1}{1\, +\, \eta\, \cdot\, iter}
```

* as the stochastic gradient can be noisy at the end


"""

# ╔═╡ aa5b5ee6-b5fe-4245-80fd-ab3ab3963d59
md"""

## Random shuffle

##### Three variants:

* ###### SGD: without shuffling


* ###### SS: single shuffle



* ###### RR: repeated random shuffle
"""

# ╔═╡ 819d01e1-4b81-4210-82de-eaf838b6a337
html"<center><img src='https://leo.host.cs.st-andrews.ac.uk/figs/CS5914/sgdcompare_logist.png
' width = '800' /></center>"

# ╔═╡ 4d4a45b5-e221-4713-b3fa-3eb36a813385
md"""

\* Koloskova *et al* (2023) *Shuffle SGD is Always Better than SGD: Improved Analysis of SGD with Arbitrary Data Orders*
"""

# ╔═╡ ea08837f-3535-4807-92e5-8091c3911948
md"""
## Mini-batch SGD


We can also use mini-batch instead of just one observation

* *e.g.* with a batch size ``5``
* a trade-off between SGD and full batch GD



**Mini-batch stochastic gradient descent**

-----

* random guess ``\large\mathbf{w}_0``

* for each *epoch*
  * split the data into equal batches ``\{B_1, B_2,\ldots, B_m\}``
  * for each batch `b` in ``\{B_1, B_2,\ldots, B_m\}``
    * ``\large\mathbf{w}_t \leftarrow \mathbf{w}_{t-1} - \gamma_t \frac{1}{|\mathtt{b}|} \sum_{i\in \mathtt{b}}\nabla L^{(i)}(\mathbf{w}_{t-1})``
-----



"""

# ╔═╡ dc70f9a4-9b52-490a-9a94-95fe903401ce
md"""

## GD vs Mini-batch
"""

# ╔═╡ f7499d8d-e511-4540-ab68-84454b3d9cd9
html"<center><img src='https://leo.host.cs.st-andrews.ac.uk/figs/CS5914/ml_stochastic.png
' width = '800' /></center>"

# ╔═╡ 0cf4d7db-5545-4b1c-ba6c-d9a6a3501e0e
md"""
[*Watt, Borhani, and Katsaggelos (2023), Machine Learning Refined
Foundations, Algorithms, and Applications](https://www.cambridge.org/highereducation/books/machine-learning-refined/0A64B2370C2F7CE3ACF535835E9D7955#overview)
"""

# ╔═╡ 9522421a-1f8e-494d-a61f-f5f8dff72c83
md"""

## SGD variants summary
"""

# ╔═╡ 39082958-54ed-4120-a7a1-f40f3dc9d965
html"<center><img src='https://leo.host.cs.st-andrews.ac.uk/figs/CS5914/sgd_compares.png
' width = '800' /></center>"

# ╔═╡ 3d0c370e-d3e5-40ec-8e66-745e4f990e18
md"""
[Source](avichawla.substack.com)
"""

# ╔═╡ 3583f790-bcc3-466b-95e8-e99a080e5658
begin
	Random.seed!(321)
	true_ww = [1., 1] 
	nobs = 100
	xxs = range(-2, 2, nobs)
	Xs = [ones(length(xxs)) xxs]
	Ys = rand(nobs) .< logistic.(Xs * true_ww)
end;

# ╔═╡ 06a80959-58de-4e21-bfdf-5b06caf157f1
w00 = [-10, 20];

# ╔═╡ eb5b710c-70a2-4237-8913-cd69e34b8e50
md"""

## Demonstration


A simulated logistic regression dataset
* true ``\mathbf{w} = [1, 1]^\top``
* learning rate ``\lambda = 0.25``
* ``n=100`` training observations
* ``100`` epochs (full data passes)
"""

# ╔═╡ ba0397a7-f514-42f8-b094-2ce5bd95b081
batch_loss, batch_wws=let
	# a very bad starting point: completely wrong prediction function
	ww = w00
	γ = 0.25
	iters = 100
	losses = zeros(iters+1)
	wws = Matrix(undef, length(ww), iters+1)
	losses[1] = logistic_loss(ww, Xs, Ys)
	wws[:, 1] = ww 
	nobs = length(Ys)
	for i in 1:iters
		gw = ∇logistic_loss(ww, Xs, Ys)/nobs
		# Flux.Optimise.update!(opt, ww, -gt)
		ww = ww - γ * gw
		wws[:, i+1] = ww 
		losses[i+1] = logistic_loss(ww, Xs, Ys)
	end
	losses, wws
end;

# ╔═╡ e1659838-ecf2-4205-b23c-227c586f6cc3
sgd_loss, sgd_wws=let
	# a very bad starting point: completely wrong prediction function
	ww = w00
	γ = 0.25
	iters = 100
	losses = zeros(iters+1)
	wws = Matrix(undef, length(ww), iters+1)
	losses[1] = logistic_loss(ww, Xs, Ys)
	wws[:, 1] = ww 
	for i in 1:iters
		ts = shuffle(1:length(Ys))
		for t in ts
			gw = ∇logistic_loss(ww, Xs[t, :]', Ys[t])
			ww = ww - γ * gw
		end
		wws[:, i+1] = ww 
		losses[i+1] = logistic_loss(ww, Xs, Ys)
	end
	losses, wws
end;

# ╔═╡ 1039a13b-6665-4bb1-82f8-11b740c3b5ba
let
	gr()
	plot(-12:0.1:10, -10:0.1:25, (x,y) ->logistic_loss([x,y], Xs, Ys), st=:contour, levels=10, framestyle=:none, colorbar=false, title="SGD with constant learning rate", legend=:outerbottom, size=(500, 450))
	step = 1
	# plot!(batch_wws[1,1:step:end], batch_wws[2, 1:step:end];  marker=(:diamond,3), label="Batch gradient descent", ls=:dashdot)

	plot!(sgd_wws[1,1:step:end], sgd_wws[2, 1:step:end];  c=3, marker=(:x, 5), la=0.5, ls=:dashdot, label="Stochastic gradient descent")

	# plot!(mini_b_sgd_wws[1,1:step:end], mini_b_sgd_wws[2, 1:step:end];marker=(:circle,5), ls=:dash, label="Minibatch SGD")
	plot!([w00[1]], [w00[2]], st=:scatter, markershape=:diamond, mc=:black, markersize=10, label="")
	annotate!([w00[1]], [w00[2] + .9], text("Random start", 12,  :bottom, :"Computer Modern"))
end

# ╔═╡ bf7cd9d6-c7a4-4715-a57c-d0acaef1e7d8
mini_b_sgd_loss, mini_b_sgd_wws = let
	ww = w00
	γ = 0.25
	iters = 100
	losses = zeros(iters+1)
	wws = Matrix(undef, length(ww), iters+1)
	losses[1] = logistic_loss(ww, Xs, Ys)
	wws[:, 1] = ww 
	train_loader = DataLoader((data=Xs', label=Ys), batchsize=5, shuffle=true);
	for i in 1:iters
		for (xs, ys) in train_loader
			gw = ∇logistic_loss(ww, xs', ys)/length(ys)
			ww = ww - γ * gw
		end
		wws[:, i+1] = ww 
		losses[i+1] = logistic_loss(ww, Xs, Ys)
	end
	losses, wws

end;

# ╔═╡ 2aa229e0-f578-4ff6-80a3-4897e2ad187f
let
	gr()
	plot(-12:0.1:10, -10:0.1:25, (x,y) ->logistic_loss([x,y], Xs, Ys), st=:contour, levels=10, framestyle=:none, colorbar=false, title="Batch GD vs SGD vs MiniBatch", legend=:outerbottom, size=(550, 600))


	step = 4
	plot!(batch_wws[1,1:step+2:end], batch_wws[2, 1:step+2:end];  marker=(:diamond,4, 0.5), label="Batch gradient descent", ls=:dashdot)

	plot!(sgd_wws[1,1:step:end], sgd_wws[2, 1:step:end];  marker=(:cross,4), ls=:dashdot, label="Stochastic gradient descent")

	plot!(mini_b_sgd_wws[1,1:step:end], mini_b_sgd_wws[2, 1:step:end];marker=(:circle,5), ls=:dash, label="Minibatch SGD")
	plot!([w00[1]], [w00[2]], st=:scatter, markershape=:diamond, mc=:black, markersize=10, label="")

	# plot!([true_ww[1]], [true_ww[1]], st=:scatter, markershape=:circle, mc=:black, markersize=2, label="")
	annotate!([w00[1]], [w00[2] + .9], text("random start", 12,  :bottom, :"Computer Modern"))


	annotate!([true_ww[1]], [true_ww[2] - 0.5], text("true w", 12,  :top, :"Computer Modern"))
end

# ╔═╡ 0734ddb1-a9a0-4fe1-b5ee-9a839a33d1dc
md"""

## Appendix
"""

# ╔═╡ 3a083374-afd6-4e64-95bd-d7e6385ab403
md"""

### Data generation
"""

# ╔═╡ 6cd96390-8565-499c-a788-fd0070331f25
D₂, targets_D₂, targets_D₂_=let
	Random.seed!(123)
	D_class_1 = rand(MvNormal(zeros(2), Matrix([1 -0.8; -0.8 1.0])), 30)' .+2
	D_class_2 = rand(MvNormal(zeros(2), Matrix([1 -0.8; -0.8 1.0])), 30)' .-2
	data₂ = [D_class_1; D_class_2]
	D₂ = [ones(size(data₂)[1]) data₂]
	targets_D₂ = [ones(size(D_class_1)[1]); zeros(size(D_class_2)[1])]
	targets_D₂_ = [ones(size(D_class_1)[1]); -1 *ones(size(D_class_2)[1])]
	D₂, targets_D₂,targets_D₂_
end

# ╔═╡ 67dae6d0-aa32-4b46-9046-aa24295aa117
plt_binary_2d = let
	gr()
	plot(D₂[targets_D₂ .== 1, 2], D₂[targets_D₂ .== 1, 3], st=:scatter, label=L"y^{(i)} = 1", xlabel=L"x_1", ylabel=L"x_2", title="Binary classification example", c=2, size=(400,300))
	plot!(D₂[targets_D₂ .== 0, 2], D₂[targets_D₂ .== 0, 3], st=:scatter, c=1, framestyle=:origin, label=L"y^{(i)} = 0", xlim=[-8, 8], ylim=[-6, 6])
end;

# ╔═╡ e77f6f3b-f18d-4260-ab7a-db61e7b4131d
TwoColumn(md"""
\
\
\

**input** features: ``\mathbf{x}^{(i)} \in \mathbb{R}^m``
* *e.g.* the right is a two-dimensional, *i.e.* ``m=2``


**output** label: ``y^{(i)} \in \{0,1\}``


""", plt_binary_2d)

# ╔═╡ b640fc38-2ada-4c35-82f0-53fd801d14e1
TwoColumn(plot(plt_binary_2d, title="2-d view", size=(300,300)), let
	gr()
	plot(D₂[targets_D₂ .== 1, 2], D₂[targets_D₂ .== 1, 3], ones(sum(targets_D₂ .== 1)), st=:scatter, label=L"\tilde{y}^{(i)} = 1", zlim=[-1.1, 1.1], xlabel=L"x_1", ylabel=L"x_2", c=2, size=(400,300))
	plot!(D₂[targets_D₂ .== 0, 2], D₂[targets_D₂ .== 0, 3], -1* ones(sum(targets_D₂ .== 0)), st=:scatter, framestyle=:origin, label=L"\tilde{y}^{(i)} = -1", xlim=[-8, 8], ylim=[-6, 6], title="3-d view", c=1)
end)

# ╔═╡ 8765a8e8-6f73-4ad4-9604-6a9eb5991241
begin
	plotly()
	scatter(D₂[targets_D₂ .== 1, 2], D₂[targets_D₂ .== 1, 3], ones(sum(targets_D₂ .== 1)), zlim=[-1.1, 1.1], label="ỹ=1", c=2)
	scatter!(D₂[targets_D₂ .== 0, 2], D₂[targets_D₂ .== 0, 3], -1 * ones(sum(targets_D₂ .== 0)), label="ỹ=-1", c=1, framestyle=:zerolines)
	w_d₂ = linear_reg(D₂, targets_D₂_;λ=0.0)
	plot!(-5:1:5, -5:1:5, (x,y) -> w_d₂[1] + w_d₂[2] * x + w_d₂[3] * y, alpha =0.9, st=:surface, colorbar=false, c=:jet, title="First attempt: least square regression")


	# plot!(-5:1:5, -5:1:5, (x,y) -> 0, alpha =0.5, st=:surface, c=:gray)
end

# ╔═╡ 68a0247b-9346-42fc-b2d2-4373e70a1789
TwoColumn(md"""


Note that the fitted hyperplane ``h(\mathbf{x})``

* ``h(\mathbf{x}^{(i)}) = 0``: the decision boundary
* ``h(\mathbf{x}^{(i)}) > 0``: one side of the boundary
* ``h(\mathbf{x}^{(i)}) < 0``: the other side of the boundary


To classify ``\hat{y}^{(i)}``:

```math
\hat{y}^{(i)} = \begin{cases} \text{class 1} & h(\mathbf{x}^{(i)}) \geq 0 \\

\text{class 2}& h(\mathbf{x}^{(i)}) < 0 
\end{cases}
```

""" , let
	plotly()
	scatter(D₂[targets_D₂ .== 1, 2], D₂[targets_D₂ .== 1, 3], ones(sum(targets_D₂ .== 1)), zlim=[-1.1, 1.1], label="ỹ=1", c=2, titlefontsize=10, legend=:bottom)
	scatter!(D₂[targets_D₂ .== 0, 2], D₂[targets_D₂ .== 0, 3], -1 * ones(sum(targets_D₂ .== 0)), label="ỹ=-1", framestyle=:zerolines, c=1)
	w_d₂ = linear_reg(D₂, targets_D₂_;λ=0.0)
	plot!(-5:1:5, -5:1:5, (x,y) -> w_d₂[1] + w_d₂[2] * x + w_d₂[3] * y, alpha =0.9, st=:surface, colorbar=false, c=:jet, title="First attempt: decision boundary h(x) =0")
	if true
		# the intersection
		xs = -5:1:5
		ys = @. (- w_d₂[1] - w_d₂[2] * xs) / w_d₂[3]
		plot!(xs,  ys, zeros(length(xs)), lw=3, lc=:gray, ls=:solid, label="h(x)=0")
	end

	plot!(-5:1:5, -5:1:5, (x,y) -> 0, alpha =0.5, st=:surface, c=:gray)
end)

# ╔═╡ 7b4502d1-3847-4472-b706-65cb68413927
plot_ls_class=let
	gr()
	plt = plot(D₂[targets_D₂ .== 1, 2], D₂[targets_D₂ .== 1, 3], st=:scatter, label="class 1", ratio=1, c=2, legend=:topleft)
	plot!(D₂[targets_D₂ .== 0, 2], D₂[targets_D₂ .== 0, 3], st=:scatter, framestyle=:origin, label="class 2", xlim=[-8, 8], ylim=[-6, 6], c=1)
	plot!(-6:1:6, (x) -> - w_d₂[1]/w_d₂[3] - w_d₂[2]/w_d₂[3] * x, lw=4, lc=:gray, label="Decision bounary: "*L"h(\mathbf{x}) =0", title="Least square classifier")

	plot!([0, w_d₂[2]*3], [0, w_d₂[3]*3], line = (:arrow, 3), c=2, label="")

	minx, maxx = minimum(D₂[:,2])-2, maximum(D₂[:,2])+2
	miny, maxy = minimum(D₂[:,3])-2, maximum(D₂[:,3])+2
	xs, ys = meshgrid(minx:0.2:maxx, miny:0.2:maxy)
	colors = [dot(w_d₂ , [1, x, y]) > 0.0   for (x, y) in zip(xs, ys)] .+ 1
	scatter!(xs, ys, color = colors,
            markersize = 1.5, label = nothing, markerstrokewidth = 0, markeralpha=0.5, xlim= [minx, maxx], ylim =[miny, maxy], framestyle=:origin)
	plt
end;

# ╔═╡ c78fbd4c-1e5f-4307-9ab3-9303b0744de0
plot_ls_class

# ╔═╡ cbbcf999-1d31-4f53-850e-d37a28cff849
begin
	plotly()
	scatter(D₂[targets_D₂ .== 1, 2], D₂[targets_D₂ .== 1, 3], ones(sum(targets_D₂ .== 1)),  label="ỹ=1", c=2)
	scatter!(D₂[targets_D₂ .== 0, 2], D₂[targets_D₂ .== 0, 3], -1 *ones(sum(targets_D₂ .== 0)), label="ỹ=-1", framestyle=:zerolines, c=1)
	# w = linear_reg(D₂, targets_D₂;λ=0.0)
	plot!(-10:1:10, -50:1:50, (x,y) -> w_d₂[1] + w_d₂[2] * x + w_d₂[3] * y, alpha =0.8, st=:surface, colorbar=false,c=:jet, title="First attempt: unbounded prediction")
end

# ╔═╡ 8fe5631a-1f10-4af4-990d-5a23c96fb73b
begin
	D1 = [
	    7 4;
	    5 6;
	    8 6;
	    9.5 5;
	    9 7
	]

	# D1 = randn(5, 2) .+ 2
	
	D2 = [
	    2 3;
	    3 2;
	    3 6;
	    5.5 4.5;
	    5 3;
	]

	# D2 = randn(5,2) .- 2

	D = [D1; D2]
	D = [ones(10) D]
	targets = [ones(5); zeros(5)]
	AB = [1.5 8; 10 1.9]
end;

# ╔═╡ 619df17d-9f75-463f-afe5-6cbffb0762d5
begin
	n3_ = 30
	extraD = randn(n3_, 2)/2 .+ [2 -6]
	D₃ = [copy(D₂); [ones(n3_) extraD]]
	targets_D₃ = [targets_D₂; zeros(n3_)]
	targets_D₃_ = [targets_D₂; -ones(n3_)]
end

# ╔═╡ ffb7f8f1-739c-43a5-b13e-4045b50c9e05
let

	gr()
	p2 = plot(D₃[targets_D₃ .== 1, 2], D₃[targets_D₃ .== 1, 3], st=:scatter, label="class 1", ratio = 1, c = 2)
	plot!(D₃[targets_D₃ .== 0, 2], D₃[targets_D₃ .== 0, 3], st=:scatter, framestyle=:origin, label="class 2", xlim=[-8, 8], legend=:topleft,  title="How about adding more data?", c=1)
	tmin = 0
	tmax = 2π
	tvec = range(tmin, tmax, length = 100)

	plot!(2.2 .+ sin.(tvec) *1.6, -6 .+ cos.(tvec) * 2, lc=1, lw=3, label="", alpha=1.0 , fill=true, fillalpha=0.2, fillcolor=1)
	plot(plot(plot_ls_class, legend=:outerbottom), p2, titlefontsize=10)
end

# ╔═╡ 69d2fe20-4e8d-4982-8f22-c4a6bfb39b4e
let
	gr()
	plt = plot(D₃[targets_D₃ .== 1, 2], D₃[targets_D₃ .== 1, 3], st=:scatter, c= 2, label="class 1", ratio=1, legend=:bottomright)
	plot!(D₃[targets_D₃ .== 0, 2], D₃[targets_D₃ .== 0, 3], st=:scatter, c=1, framestyle=:origin, label="class 2", xlim=[-10, 8])
	w_d₃ = linear_reg(D₃, targets_D₃_;λ=0.0)
	# w_d₂
	plot!(-7.5:1:7, (x) -> - w_d₂[1]/w_d₂[3] - w_d₂[2]/w_d₂[3] * x, lw=2.5, lc=:gray,ls=:solid,  label="", title="")
	plot!(-6:1:6, (x) -> - w_d₃[1]/w_d₃[3] - w_d₃[2]/w_d₃[3] * x, lw=4, lc=2, label="New decision boundary: "*L"h(\mathbf{x}) =0", title="Least square classifier with new data")

	minx, maxx = minimum(D₃[:,2])-2, maximum(D₃[:,2])+2
	miny, maxy = minimum(D₃[:,3])-2, maximum(D₃[:,3])+2
	xs, ys = meshgrid(minx:0.25:maxx, miny:0.25:maxy)
	colors = [dot(w_d₃ , [1, x, y]) > 0.0   for (x, y) in zip(xs, ys)] .+ 1
	scatter!(xs, ys, color = colors,
            markersize = 1.5, label = nothing, markerstrokewidth = 0,
		markeralpha=0.5, xlim= [minx, maxx], ylim =[miny, maxy], framestyle=:origin)


	plot(plot_ls_class, plt, legendfontsize=6, titlefontsize=8, legend=:outerbottom)
end

# ╔═╡ 4d474d9f-50dc-4c9f-9505-65e024c83f38
let
	plotly()
	scatter(D₃[targets_D₃ .== 1, 2], D₃[targets_D₃ .== 1, 3], ones(sum(targets_D₂ .== 1)), zlim=[-2,2], label="y=1", c=2)
	scatter!(D₃[targets_D₃ .== 0, 2], D₃[targets_D₃ .== 0, 3], -1 * ones(sum(targets_D₃ .== 0)), label="y=-1", framestyle=:zerolines, c=1)
	w_d₃ = linear_reg(D₃, targets_D₃_;λ=0.0)
	plot!(-10:1:10, -10:1:10, (x,y) -> w_d₃[1] + w_d₃[2] * x + w_d₃[3] * y, alpha =0.8, st=:surface, color=:jet, colorbar=false, title="Why least square classifier fails")
end

# ╔═╡ a270b1b3-7bb4-4612-9e57-3ea9b7daedc0
losses, wws=let
	# a very bad starting point: completely wrong prediction function
	ww = [0, -5, -5]
	γ = 0.008
	iters = 2000
	losses = zeros(iters+1)
	wws = Matrix(undef, 3, iters+1)
	losses[1] = logistic_loss(ww, D₃, targets_D₃)
	wws[:, 1] = ww 
	for i in 1:iters
		gw = ∇logistic_loss(ww, D₃, targets_D₃)
		# Flux.Optimise.update!(opt, ww, -gt)
		ww = ww - γ * gw
		wws[:, i+1] = ww 
		losses[i+1] = logistic_loss(ww, D₃, targets_D₃)
	end
	losses, wws
end;

# ╔═╡ 872aa766-2ed8-4cb3-a029-4a2bb42c90a8
let
	gr()
	plot(losses[1:50], label="loss", xlabel="Epoch",ylabel="loss", title="Batch gradient descent's loss vs epoch")
end

# ╔═╡ daf19517-4123-49a8-affb-5d869e08480a
anim_logis=let
	gr()
	# xs_, ys_ = meshgrid(range(-5, 5, length=20), range(-5, 5, length=20))

	anim = @animate for t in [1:25;26:10:200;]
		plot(D₃[targets_D₃ .== 1, 2], D₃[targets_D₃ .== 1, 3], st=:scatter, label="class 1", c=2)
		plot!(D₃[targets_D₃ .== 0, 2], D₃[targets_D₃ .== 0, 3], st=:scatter, framestyle=:origin, label="class 2", xlim=[-8, 8], legend=:topleft, c=1)
		w₀, w₁, w₂ = (wws[:, t])
		plot!(-5:0.1:5, -5:0.1:5, (x, y) -> logistic(w₀+ w₁* x + w₂ * y), st=:contour, levels=6, c=:jet, colorbar=false, alpha=0.5, xlim=[-5, 5], ylim=[-5, 5],  xlabel=L"x_1", ylabel=L"x_2", title="Epoch: "*L"%$(t);"*" loss: " *L"%$(round(losses[t]; digits=1))", ratio=1)
		quiver!([0.0], [0.0], quiver=([w₁- 0.0], [w₂-0.0]), c=2, lw=2)
	end

	anim
	# ∇f_d(x, y) = ∇σ([1, x, y], [w₀, w₁, w₂])[2:end] * 1
	# quiver!(xs_, ys_, quiver = ∇f_d, c=:black)
	# w_d₃ = linear_reg(D₃, targets_D₃_;λ=0.0)
	# w_d₂
	# plot!(-6:1:6, (x) -> - w_d₃[1]/w_d₃[3] - w_d₃[2]/w_d₃[3] * x, lw=4, lc=:gray, label="Decision boundary: "*L"h(\mathbf{x}) =0", title="Least square classifier fails")
end;

# ╔═╡ 03840aa7-4258-47f7-b802-92ab93c92911
gif(anim_logis, fps=5)

# ╔═╡ 8398cb20-b9f4-427c-8b10-2a572f8a5632
anim_logis2=let

	gr()

	anim = @animate for t in [1:25...]
		plot(D₃[targets_D₃ .== 1, 2], D₃[targets_D₃ .== 1, 3], ones(sum(targets_D₃ .== 1)), st=:scatter, label="class 1", c=2)
		plot!(D₃[targets_D₃ .== 0, 2], D₃[targets_D₃ .== 0, 3], zeros(sum(targets_D₃ .== 0)), st=:scatter, framestyle=:origin, label="class 2", xlim=[-8, 8], legend=:topleft, c=1)
		w₀, w₁, w₂ = (wws[:, t])
		plot!(-5:0.1:5, -5:0.1:5, (x, y) -> logistic(w₀+ w₁* x + w₂ * y), st=:surface, c=:jet, colorbar=false, alpha=0.5, xlim=[-5, 5], ylim=[-5, 5],  xlabel=L"x_1", ylabel=L"x_2", title="Epoch: "*L"%$(t);"*" loss: " *L"%$(round(losses[t]; digits=1))", ratio=1)
	end

	anim
	# ∇f_d(x, y) = ∇σ([1, x, y], [w₀, w₁, w₂])[2:end] * 1
	# quiver!(xs_, ys_, quiver = ∇f_d, c=:black)
	# w_d₃ = linear_reg(D₃, targets_D₃_;λ=0.0)
	# w_d₂
	# plot!(-6:1:6, (x) -> - w_d₃[1]/w_d₃[3] - w_d₃[2]/w_d₃[3] * x, lw=4, lc=:gray, label="Decision boundary: "*L"h(\mathbf{x}) =0", title="Least square classifier fails")
end;

# ╔═╡ 12991c72-6b19-47ed-9ebe-90e3d615f985
gif(anim_logis2, fps=2)

# ╔═╡ 8687dbd1-4857-40e4-b9cb-af469b8563e2
function perp_square(origin, vx, vy; δ=0.1) 
	x = δ * vx/sqrt(norm(vx))
	y = δ * vy/sqrt(norm(vy))
	xyunit = origin+ x + y
	xunit = origin + x
	yunit = origin +y
	Shape([origin[1], xunit[1], xyunit[1], yunit[1]], [origin[2], xunit[2], xyunit[2], yunit[2]])
end

# ╔═╡ fab7a0dd-3a9e-463e-a66b-432a6b2d8a1b
# as: arrow head size 0-1 (fraction of arrow length)
# la: arrow alpha transparency 0-1
function arrow3d!(x, y, z,  u, v, w; as=0.1, lc=:black, la=1, lw=0.4, scale=:identity)
    (as < 0) && (nv0 = -maximum(norm.(eachrow([u v w]))))
    for (x,y,z, u,v,w) in zip(x,y,z, u,v,w)
        nv = sqrt(u^2 + v^2 + w^2)
        v1, v2 = -[u,v,w]/nv, nullspace(adjoint([u,v,w]))[:,1]
        v4 = (3*v1 + v2)/3.1623  # sqrt(10) to get unit vector
        v5 = v4 - 2*(v4'*v2)*v2
        (as < 0) && (nv = nv0) 
        v4, v5 = -as*nv*v4, -as*nv*v5
        plot!([x,x+u], [y,y+v], [z,z+w], lc=lc, la=la, lw=lw, scale=scale, label=false)
        plot!([x+u,x+u-v5[1]], [y+v,y+v-v5[2]], [z+w,z+w-v5[3]], lc=lc, la=la, lw=lw, label=false)
        plot!([x+u,x+u-v4[1]], [y+v,y+v-v4[2]], [z+w,z+w-v4[3]], lc=lc, la=la, lw=lw, label=false)
    end
end

# ╔═╡ 7424b8a4-355e-408b-9b39-8af53a717134
let
	plotly()
	w₀ = 0
	w₁, w₂ = wv_[1], wv_[2]
	p1 = plot(-5:0.5:5, -5:0.5:5, (x, y) -> w₀+ w₁* x + w₂ * y, st=:surface, c=:jet, colorbar=false, alpha=0.8, framestyle=:zerolines, xlabel="x₁", ylabel="x₂", title="h(x) = wᵀ x")
	arrow3d!([0], [0], [0], [w₁], [w₂], [0]; as=0.5, lc=2, la=1, lw=2, scale=:identity)
	p2 = plot(-5:0.5:5, -5:0.5:5, (x, y) -> logistic(w₀+ w₁* x + w₂ * y), st=:surface, c=:jet, colorbar=false, alpha=0.8, zlim=[-0.1, 1.1],  xlabel="x₁", ylabel="x₂", title="σ(wᵀx)")

	arrow3d!([0], [0], [0], [w₁], [w₂], [0]; as=0.5, lc=2, la=1, lw=2, scale=:identity)
	plot(p1, p2)
end

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
Distributions = "31c24e10-a181-5473-b8eb-7969acd0382f"
HypertextLiteral = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"
LaTeXStrings = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
Latexify = "23fbe1c1-3f47-55db-b15f-69d7ec21a316"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
LogExpFunctions = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
Logging = "56ddb016-857b-54e1-b83d-db4d58db5568"
MLUtils = "f1d291b0-491e-4a28-83b9-f70985020b54"
Plots = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
PlutoTeachingTools = "661c6b06-c737-4d37-b85c-46df65de6f69"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
Statistics = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
StatsBase = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
StatsPlots = "f3b207a7-027a-5e70-b257-86293d7955fd"

[compat]
Distributions = "~0.25.98"
HypertextLiteral = "~0.9.4"
LaTeXStrings = "~1.3.0"
Latexify = "~0.16.1"
LogExpFunctions = "~0.3.24"
MLUtils = "~0.4.3"
Plots = "~1.38.16"
PlutoTeachingTools = "~0.2.12"
PlutoUI = "~0.7.51"
StatsBase = "~0.34.0"
StatsPlots = "~0.15.5"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.9.3"
manifest_format = "2.0"
project_hash = "43079daeec3b9c57f773016699a98546325333d3"

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

[[deps.ArgCheck]]
git-tree-sha1 = "a3a402a35a2f7e0b87828ccabbd5ebfbebe356b4"
uuid = "dce04be8-c92d-5529-be00-80e4d2c0e197"
version = "2.3.0"

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

[[deps.Atomix]]
deps = ["UnsafeAtomics"]
git-tree-sha1 = "c06a868224ecba914baa6942988e2f2aade419be"
uuid = "a9b6321e-bd34-4604-b9c9-b65b8de01458"
version = "0.1.0"

[[deps.AxisAlgorithms]]
deps = ["LinearAlgebra", "Random", "SparseArrays", "WoodburyMatrices"]
git-tree-sha1 = "66771c8d21c8ff5e3a93379480a2307ac36863f7"
uuid = "13072b0f-2c55-5437-9ae7-d433b7a33950"
version = "1.0.1"

[[deps.BangBang]]
deps = ["Compat", "ConstructionBase", "InitialValues", "LinearAlgebra", "Requires", "Setfield", "Tables"]
git-tree-sha1 = "e28912ce94077686443433c2800104b061a827ed"
uuid = "198e06fe-97b7-11e9-32a5-e1d131e6ad66"
version = "0.3.39"

    [deps.BangBang.extensions]
    BangBangChainRulesCoreExt = "ChainRulesCore"
    BangBangDataFramesExt = "DataFrames"
    BangBangStaticArraysExt = "StaticArrays"
    BangBangStructArraysExt = "StructArrays"
    BangBangTypedTablesExt = "TypedTables"

    [deps.BangBang.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"
    StructArrays = "09ab397b-f2b6-538f-b94a-2f83cf4a842a"
    TypedTables = "9d95f2ec-7b3d-5a63-8d20-e2491e220bb9"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"

[[deps.Baselet]]
git-tree-sha1 = "aebf55e6d7795e02ca500a689d326ac979aaf89e"
uuid = "9718e550-a3fa-408a-8086-8db961cd8217"
version = "0.1.1"

[[deps.BitFlags]]
git-tree-sha1 = "43b1a4a8f797c1cddadf60499a8a077d4af2cd2d"
uuid = "d1d4a3ce-64b1-5f1a-9ba4-7e7e69966f35"
version = "0.1.7"

[[deps.Bzip2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "19a35467a82e236ff51bc17a3a44b69ef35185a2"
uuid = "6e34b625-4abd-537c-b88f-471c36dfa7a0"
version = "1.0.8+0"

[[deps.CEnum]]
git-tree-sha1 = "eb4cb44a499229b3b8426dcfb5dd85333951ff90"
uuid = "fa961155-64e5-5f13-b03f-caf6b980ea82"
version = "0.4.2"

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

[[deps.CompositionsBase]]
git-tree-sha1 = "802bb88cd69dfd1509f6670416bd4434015693ad"
uuid = "a33af91c-f02d-484b-be07-31d278c5ca2b"
version = "0.1.2"

    [deps.CompositionsBase.extensions]
    CompositionsBaseInverseFunctionsExt = "InverseFunctions"

    [deps.CompositionsBase.weakdeps]
    InverseFunctions = "3587e190-3f89-42d0-90ee-14403ec27112"

[[deps.ConcurrentUtilities]]
deps = ["Serialization", "Sockets"]
git-tree-sha1 = "5372dbbf8f0bdb8c700db5367132925c0771ef7e"
uuid = "f0e56b4a-5159-44fe-b623-3e5288b988bb"
version = "2.2.1"

[[deps.ConstructionBase]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "fe2838a593b5f776e1597e086dcd47560d94e816"
uuid = "187b0558-2788-49d3-abe0-74a17ed4e7c9"
version = "1.5.3"

    [deps.ConstructionBase.extensions]
    ConstructionBaseIntervalSetsExt = "IntervalSets"
    ConstructionBaseStaticArraysExt = "StaticArrays"

    [deps.ConstructionBase.weakdeps]
    IntervalSets = "8197267c-284f-5f27-9208-e0e47529a953"
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"

[[deps.ContextVariablesX]]
deps = ["Compat", "Logging", "UUIDs"]
git-tree-sha1 = "25cc3803f1030ab855e383129dcd3dc294e322cc"
uuid = "6add18c4-b38d-439d-96f6-d6bc489c04c5"
version = "0.1.3"

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

[[deps.DefineSingletons]]
git-tree-sha1 = "0fba8b706d0178b4dc7fd44a96a92382c9065c2c"
uuid = "244e2a9f-e319-4986-a169-4d1fe445cd52"
version = "0.1.2"

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

[[deps.FLoops]]
deps = ["BangBang", "Compat", "FLoopsBase", "InitialValues", "JuliaVariables", "MLStyle", "Serialization", "Setfield", "Transducers"]
git-tree-sha1 = "ffb97765602e3cbe59a0589d237bf07f245a8576"
uuid = "cc61a311-1640-44b5-9fba-1b764f453329"
version = "0.2.1"

[[deps.FLoopsBase]]
deps = ["ContextVariablesX"]
git-tree-sha1 = "656f7a6859be8673bf1f35da5670246b923964f7"
uuid = "b9860ae5-e623-471e-878b-f6a53c775ea6"
version = "0.1.1"

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

[[deps.Future]]
deps = ["Random"]
uuid = "9fa8497b-333b-5362-9e8d-4d0656e87820"

[[deps.GLFW_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libglvnd_jll", "Pkg", "Xorg_libXcursor_jll", "Xorg_libXi_jll", "Xorg_libXinerama_jll", "Xorg_libXrandr_jll"]
git-tree-sha1 = "d972031d28c8c8d9d7b41a536ad7bb0c2579caca"
uuid = "0656b61e-2033-5cc2-a64a-77c0f6c09b89"
version = "3.3.8+0"

[[deps.GPUArraysCore]]
deps = ["Adapt"]
git-tree-sha1 = "2d6ca471a6c7b536127afccfa7564b5b39227fe0"
uuid = "46192b85-c4d5-4398-a991-12ede77f4527"
version = "0.1.5"

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

[[deps.InitialValues]]
git-tree-sha1 = "4da0f88e9a39111c2fa3add390ab15f3a44f3ca3"
uuid = "22cec73e-a1b8-11e9-2c92-598750a2cf9c"
version = "0.3.1"

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

[[deps.JuliaVariables]]
deps = ["MLStyle", "NameResolution"]
git-tree-sha1 = "49fb3cb53362ddadb4415e9b73926d6b40709e70"
uuid = "b14d175d-62b4-44ba-8fb7-3064adc8c3ec"
version = "0.2.4"

[[deps.KernelAbstractions]]
deps = ["Adapt", "Atomix", "InteractiveUtils", "LinearAlgebra", "MacroTools", "PrecompileTools", "Requires", "SparseArrays", "StaticArrays", "UUIDs", "UnsafeAtomics", "UnsafeAtomicsLLVM"]
git-tree-sha1 = "4c5875e4c228247e1c2b087669846941fb6e0118"
uuid = "63c18a36-062a-441e-b654-da1e3ab1ce7c"
version = "0.9.8"

    [deps.KernelAbstractions.extensions]
    EnzymeExt = "EnzymeCore"

    [deps.KernelAbstractions.weakdeps]
    EnzymeCore = "f151be2c-9106-41f4-ab19-57ee4f262869"

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

[[deps.LLVM]]
deps = ["CEnum", "LLVMExtra_jll", "Libdl", "Printf", "Unicode"]
git-tree-sha1 = "8695a49bfe05a2dc0feeefd06b4ca6361a018729"
uuid = "929cbde3-209d-540e-8aea-75f648917ca0"
version = "6.1.0"

[[deps.LLVMExtra_jll]]
deps = ["Artifacts", "JLLWrappers", "LazyArtifacts", "Libdl", "TOML"]
git-tree-sha1 = "c35203c1e1002747da220ffc3c0762ce7754b08c"
uuid = "dad2f222-ce93-54a1-a47d-0025e8a3acab"
version = "0.0.23+0"

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
git-tree-sha1 = "0d097476b6c381ab7906460ef1ef1638fbce1d91"
uuid = "e6f89c97-d47a-5376-807f-9c37f3926c36"
version = "1.0.2"

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

[[deps.MLStyle]]
git-tree-sha1 = "bc38dff0548128765760c79eb7388a4b37fae2c8"
uuid = "d8e11817-5142-5d16-987a-aa16d5891078"
version = "0.4.17"

[[deps.MLUtils]]
deps = ["ChainRulesCore", "Compat", "DataAPI", "DelimitedFiles", "FLoops", "NNlib", "Random", "ShowCases", "SimpleTraits", "Statistics", "StatsBase", "Tables", "Transducers"]
git-tree-sha1 = "3504cdb8c2bc05bde4d4b09a81b01df88fcbbba0"
uuid = "f1d291b0-491e-4a28-83b9-f70985020b54"
version = "0.4.3"

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

[[deps.MicroCollections]]
deps = ["BangBang", "InitialValues", "Setfield"]
git-tree-sha1 = "629afd7d10dbc6935ec59b32daeb33bc4460a42e"
uuid = "128add7d-3638-4c79-886c-908ea0c25c34"
version = "0.1.4"

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

[[deps.NNlib]]
deps = ["Adapt", "Atomix", "ChainRulesCore", "GPUArraysCore", "KernelAbstractions", "LinearAlgebra", "Pkg", "Random", "Requires", "Statistics"]
git-tree-sha1 = "3d42748c725c3f088bcda47fa2aca89e74d59d22"
uuid = "872c559c-99b0-510c-b3b7-b6c96a88d5cd"
version = "0.9.4"

    [deps.NNlib.extensions]
    NNlibAMDGPUExt = "AMDGPU"
    NNlibCUDACUDNNExt = ["CUDA", "cuDNN"]
    NNlibCUDAExt = "CUDA"

    [deps.NNlib.weakdeps]
    AMDGPU = "21141c5a-9bdb-4563-92ae-f87d6854732e"
    CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba"
    cuDNN = "02a925ec-e4fe-4b08-9a7e-0d78e3d38ccd"

[[deps.NaNMath]]
deps = ["OpenLibm_jll"]
git-tree-sha1 = "0877504529a3e5c3343c6f8b4c0381e57e4387e4"
uuid = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
version = "1.0.2"

[[deps.NameResolution]]
deps = ["PrettyPrint"]
git-tree-sha1 = "1a0fa0e9613f46c9b8c11eee38ebb4f590013c5e"
uuid = "71a1bf82-56d0-4bbc-8a3c-48b961074391"
version = "0.1.5"

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

[[deps.PrettyPrint]]
git-tree-sha1 = "632eb4abab3449ab30c5e1afaa874f0b98b586e4"
uuid = "8162dcfd-2161-5ef2-ae6c-7681170c5f98"
version = "0.2.0"

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

[[deps.Setfield]]
deps = ["ConstructionBase", "Future", "MacroTools", "StaticArraysCore"]
git-tree-sha1 = "e2cc6d8c88613c05e1defb55170bf5ff211fbeac"
uuid = "efcf1570-3423-57d1-acb7-fd33fddbac46"
version = "1.1.1"

[[deps.SharedArrays]]
deps = ["Distributed", "Mmap", "Random", "Serialization"]
uuid = "1a1011a3-84de-559e-8e89-a11a2f7dc383"

[[deps.ShowCases]]
git-tree-sha1 = "7f534ad62ab2bd48591bdeac81994ea8c445e4a5"
uuid = "605ecd9f-84a6-4c9e-81e2-4798472b76a3"
version = "0.1.0"

[[deps.Showoff]]
deps = ["Dates", "Grisu"]
git-tree-sha1 = "91eddf657aca81df9ae6ceb20b959ae5653ad1de"
uuid = "992d4aef-0814-514b-bc4d-f2e9a6c4116f"
version = "1.0.3"

[[deps.SimpleBufferStream]]
git-tree-sha1 = "874e8867b33a00e784c8a7e4b60afe9e037b74e1"
uuid = "777ac1f9-54b0-4bf8-805c-2214025038e7"
version = "1.1.0"

[[deps.SimpleTraits]]
deps = ["InteractiveUtils", "MacroTools"]
git-tree-sha1 = "5d7e3f4e11935503d3ecaf7186eac40602e7d231"
uuid = "699a6c99-e7fa-54fc-8d76-47d257e15c1d"
version = "0.9.4"

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

[[deps.SplittablesBase]]
deps = ["Setfield", "Test"]
git-tree-sha1 = "e08a62abc517eb79667d0a29dc08a3b589516bb5"
uuid = "171d559e-b47b-412a-8079-5efa626c420e"
version = "0.1.15"

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

[[deps.Transducers]]
deps = ["Adapt", "ArgCheck", "BangBang", "Baselet", "CompositionsBase", "ConstructionBase", "DefineSingletons", "Distributed", "InitialValues", "Logging", "Markdown", "MicroCollections", "Requires", "Setfield", "SplittablesBase", "Tables"]
git-tree-sha1 = "53bd5978b182fa7c57577bdb452c35e5b4fb73a5"
uuid = "28d57a85-8fef-5791-bfe6-a80928e7c999"
version = "0.4.78"

    [deps.Transducers.extensions]
    TransducersBlockArraysExt = "BlockArrays"
    TransducersDataFramesExt = "DataFrames"
    TransducersLazyArraysExt = "LazyArrays"
    TransducersOnlineStatsBaseExt = "OnlineStatsBase"
    TransducersReferenceablesExt = "Referenceables"

    [deps.Transducers.weakdeps]
    BlockArrays = "8e7c35d0-a365-5155-bbbb-fb81a777f24e"
    DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
    LazyArrays = "5078a376-72f3-5289-bfd5-ec5146d43c02"
    OnlineStatsBase = "925886fa-5bf2-5e8e-b522-a9147a512338"
    Referenceables = "42d2dcc6-99eb-4e98-b66c-637b7d73030e"

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

[[deps.UnsafeAtomics]]
git-tree-sha1 = "6331ac3440856ea1988316b46045303bef658278"
uuid = "013be700-e6cd-48c3-b4a1-df204f14c38f"
version = "0.2.1"

[[deps.UnsafeAtomicsLLVM]]
deps = ["LLVM", "UnsafeAtomics"]
git-tree-sha1 = "323e3d0acf5e78a56dfae7bd8928c989b4f3083e"
uuid = "d80eeb9a-aca5-4d75-85e5-170c8b632249"
version = "0.1.3"

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
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libiconv_jll", "Zlib_jll"]
git-tree-sha1 = "04a51d15436a572301b5abbb9d099713327e9fc4"
uuid = "02c8fc9c-b97f-50b9-bbe4-9be30ff0a78a"
version = "2.10.4+0"

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
# ╟─9f90a18b-114f-4039-9aaf-f52c77205a49
# ╟─86ac8000-a595-4162-863d-8720ff9dd3bd
# ╟─ece21354-0718-4afb-a905-c7899f41883b
# ╟─3e2e1ea8-3a7d-462f-ac38-43a087907a14
# ╟─7bbf37e1-27fd-4871-bc1d-c9c3ecaac076
# ╟─bc96a33d-9011-41ec-a19e-d472cbaafb70
# ╟─7091d2cf-9237-45b2-b609-f442cd1cdba5
# ╟─0a7f37e1-51bc-427d-a947-31a6be5b765e
# ╟─a696c014-2070-4041-ada3-da79f50c9140
# ╟─595a5ef3-4f54-4502-a943-ace4146efa31
# ╟─bc1ee08d-9376-44d7-968c-5e114b09a5e0
# ╟─bcc3cea5-0564-481b-883a-a45a1b870ba3
# ╟─e77f6f3b-f18d-4260-ab7a-db61e7b4131d
# ╟─67dae6d0-aa32-4b46-9046-aa24295aa117
# ╟─2d70884d-c922-45e6-853e-5608affdd860
# ╟─b640fc38-2ada-4c35-82f0-53fd801d14e1
# ╟─a7cbd3b5-494a-45ef-9bed-1aa1ba2d7924
# ╟─8765a8e8-6f73-4ad4-9604-6a9eb5991241
# ╟─8c69c448-c104-4cae-9352-10d4cec512a9
# ╟─68a0247b-9346-42fc-b2d2-4373e70a1789
# ╟─bf83caa9-d9f9-48a2-9563-152fcbb8afdc
# ╟─c78fbd4c-1e5f-4307-9ab3-9303b0744de0
# ╟─7b4502d1-3847-4472-b706-65cb68413927
# ╟─ea55660a-090b-453c-a141-b6867670ba48
# ╟─ffb7f8f1-739c-43a5-b13e-4045b50c9e05
# ╟─3315ef07-17df-4fe1-a7a5-3e8f039c0cc1
# ╟─c936fe77-827b-4245-93e5-d239d467bffd
# ╟─69d2fe20-4e8d-4982-8f22-c4a6bfb39b4e
# ╟─cdccf955-19f3-45bb-8dfe-2e15addcdd32
# ╟─4d474d9f-50dc-4c9f-9505-65e024c83f38
# ╟─331571e5-f314-42db-ae56-66e64e485b85
# ╟─cbbcf999-1d31-4f53-850e-d37a28cff849
# ╟─324ea2b8-c350-438d-8c8f-6404045fc19f
# ╟─3a50c68d-5cb1-45f5-a6af-c07ab280b1ad
# ╟─c8e55a60-0829-4cc7-bc9b-065809ac791c
# ╟─9d9c91b3-ed65-4929-8cfe-4de7a0d6f807
# ╟─8da5d36d-7fe0-45ee-bbcc-abb9eb2831f6
# ╟─6444df20-363c-4db3-acb0-efb3b17d7368
# ╟─5ffef289-18e0-40c8-be74-de9871a45687
# ╟─caef3a1c-c85d-4083-b8e3-4389d81ad6c1
# ╟─e5853cf2-657c-473d-8617-db74e8f59ea2
# ╟─7424b8a4-355e-408b-9b39-8af53a717134
# ╟─85cf441b-8b42-4dba-9bef-e54855cf0ce1
# ╟─206c82ee-389e-4c92-adbf-9578f7125418
# ╟─c03ab032-13b1-41a4-9477-6c79bb4fecd6
# ╟─1a062043-8cfe-4946-8eb0-688d3896229e
# ╟─88093782-d180-4422-bd92-357c716bfc89
# ╟─2459a2d9-4f48-46ab-82e5-3968e713f15f
# ╟─52ff5315-002c-480b-9a4b-c04124498277
# ╟─8fd7703b-db80-4aa4-977f-c7e3ad1f9fb6
# ╟─0469b52f-ce98-4cfa-abf3-be53630b30dd
# ╟─4f884fee-4934-46c3-8352-0105652f8537
# ╟─9dbf5502-fa44-404f-88ae-be3488e3e41c
# ╟─6e92fa09-cf58-4d3c-9864-c6617f1e54d7
# ╟─6173a35d-1e28-45ac-93b8-d2fb1117bf02
# ╟─866e6bfe-748b-424c-b0af-b6f51c15be21
# ╟─1ae4fa36-0faa-438b-8be3-292ec7b617a0
# ╟─3e980014-daf7-4d8b-b9e6-14a5d078e3b6
# ╟─5d2f56e8-21b2-4aa9-b450-40f7881489e0
# ╟─6cbddc5d-ae3f-43ac-9b7a-bbc779739353
# ╟─64a5e292-14b4-4df0-871d-65d9fec6201d
# ╟─96b18c60-87c6-4c09-9882-fbbc5a53f046
# ╟─35d5f991-5194-4174-badc-324ad7d15ac3
# ╟─f693814d-9639-4abb-a5b4-e83fe7a28a77
# ╟─67f7449f-19b8-4607-9e32-0f8a16a806c0
# ╟─fb0361a6-c967-4dcd-9002-55ae25e225a5
# ╟─5c0eaab3-de6d-457f-a0c3-8ea6b5da2c88
# ╟─de93ac5e-bec6-4764-ac6d-f84076ff20ee
# ╟─a50cc950-45ca-4745-9f47-a3fbb28db782
# ╟─1421c748-083b-4977-8a88-6a39c9b9906d
# ╟─82baa991-6df5-4561-9643-52db96c5e99b
# ╟─9a02f4f8-38d6-44a4-9118-1440bfc4d271
# ╟─ce822f87-d2c4-434b-9b5d-b629962d2df2
# ╟─9d25c603-07b3-4758-9b28-0d2041e18569
# ╟─929916d2-9abf-40b2-9399-85c3ba05989b
# ╟─8743242e-7f7f-4f54-b27c-1f08d1a110e2
# ╟─aaaadaa8-d2e9-4647-82de-5547b2d6ddb4
# ╟─6e62b53d-68d8-45de-9a0f-8223b5d2df92
# ╟─7ed0de94-b513-40c1-9f83-6ed75fcd4cdd
# ╟─188e9981-9597-483f-b1e3-689e26389b61
# ╟─50d9a624-20d8-4feb-a565-4efcaf22fe27
# ╟─ec78bc4f-884b-4525-8d1b-138b37274ee7
# ╟─e88db991-bf54-4548-80cb-7cd307300ec9
# ╟─99c3f5ee-63d0-4f6f-90d9-2e524a7e945a
# ╟─a270b1b3-7bb4-4612-9e57-3ea9b7daedc0
# ╟─872aa766-2ed8-4cb3-a029-4a2bb42c90a8
# ╟─12991c72-6b19-47ed-9ebe-90e3d615f985
# ╟─03840aa7-4258-47f7-b802-92ab93c92911
# ╟─daf19517-4123-49a8-affb-5d869e08480a
# ╟─8398cb20-b9f4-427c-8b10-2a572f8a5632
# ╟─ff397d54-42b3-40af-bf36-4716fd9a4419
# ╟─fcc49f3c-6219-40b8-a5d0-85fa70738a8d
# ╟─6dc74312-0e3b-4ead-b369-f9c2b70ab3d3
# ╟─e0f3cee1-b6ee-4399-8e4f-c0d70b94723e
# ╟─1039a13b-6665-4bb1-82f8-11b740c3b5ba
# ╟─aa5b5ee6-b5fe-4245-80fd-ab3ab3963d59
# ╟─819d01e1-4b81-4210-82de-eaf838b6a337
# ╟─4d4a45b5-e221-4713-b3fa-3eb36a813385
# ╟─ea08837f-3535-4807-92e5-8091c3911948
# ╟─dc70f9a4-9b52-490a-9a94-95fe903401ce
# ╟─f7499d8d-e511-4540-ab68-84454b3d9cd9
# ╟─0cf4d7db-5545-4b1c-ba6c-d9a6a3501e0e
# ╟─9522421a-1f8e-494d-a61f-f5f8dff72c83
# ╟─39082958-54ed-4120-a7a1-f40f3dc9d965
# ╟─3d0c370e-d3e5-40ec-8e66-745e4f990e18
# ╟─3583f790-bcc3-466b-95e8-e99a080e5658
# ╟─06a80959-58de-4e21-bfdf-5b06caf157f1
# ╟─eb5b710c-70a2-4237-8913-cd69e34b8e50
# ╟─2aa229e0-f578-4ff6-80a3-4897e2ad187f
# ╟─ba0397a7-f514-42f8-b094-2ce5bd95b081
# ╟─e1659838-ecf2-4205-b23c-227c586f6cc3
# ╟─1aa0cc79-e9ca-44bc-b9ab-711eed853f00
# ╟─bf7cd9d6-c7a4-4715-a57c-d0acaef1e7d8
# ╟─0734ddb1-a9a0-4fe1-b5ee-9a839a33d1dc
# ╟─3a083374-afd6-4e64-95bd-d7e6385ab403
# ╟─6cd96390-8565-499c-a788-fd0070331f25
# ╟─8fe5631a-1f10-4af4-990d-5a23c96fb73b
# ╟─619df17d-9f75-463f-afe5-6cbffb0762d5
# ╟─8687dbd1-4857-40e4-b9cb-af469b8563e2
# ╟─fab7a0dd-3a9e-463e-a66b-432a6b2d8a1b
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
