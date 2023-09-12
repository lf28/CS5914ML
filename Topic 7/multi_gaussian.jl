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
	using StatsPlots
	using LogExpFunctions
end

# ╔═╡ 50752620-a604-442c-bf92-992963b1dd7a
begin
	using Images
	using Distributions
	using StatsBase
	
end

# ╔═╡ 3e2e1ea8-3a7d-462f-ac38-43a087907a14
TableOfContents()

# ╔═╡ 7bbf37e1-27fd-4871-bc1d-c9c3ecaac076
ChooseDisplayMode()

# ╔═╡ bc96a33d-9011-41ec-a19e-d472cbaafb70
md"""

# CS5914 Machine Learning Algorithms


#### Multivariate Gaussian
\

$(Resource("https://www.st-andrews.ac.uk/assets/university/brand/logos/standard-vertical-black.png", :width=>130, :align=>"right"))

Lei Fang(@lf28 $(Resource("https://raw.githubusercontent.com/edent/SuperTinyIcons/bed6907f8e4f5cb5bb21299b9070f4d7c51098c0/images/svg/github.svg", :width=>10)))

*School of Computer Science*

*University of St Andrews, UK*

"""

# ╔═╡ ff61cd9d-a193-44b3-a715-3c372ade7f79
md"# Multivariate Gaussian"

# ╔═╡ b89ac105-597e-44ac-9b58-c1c3c5ac59e9
md"""
## Recap: univariate Gaussian



"""

# ╔═╡ 40f334b3-60fa-4ca1-8146-06e00fa14d45
md"##
"

# ╔═╡ bb02b467-0269-4047-8c0b-ee61e0185df8
html"<center><img src='https://leo.host.cs.st-andrews.ac.uk/figs/CS5914/gaussian_eq_1d_alt.png' width = '900' /></center>"

# ╔═╡ d9829c67-d346-4145-ae75-028f9fdd103d
md"""
* ``\mu``: mean or location
* ``\sigma^2``: variance or scale, controls the spread



"""

# ╔═╡ 72af797b-5340-482e-be00-2cda375dd734
md"""

## Dissect Gaussian

```math
\Large
{\color{darkorange}{-\frac{1}{2}\left(\frac{x -\mu}{\sigma}\right)^2}} \Longrightarrow  { \color{green}{ e^{-\frac{1}{2}\left(\frac{x -\mu}{\sigma}\right)^2} }} \Longrightarrow  {\color{purple}{\frac{1}{\sigma \sqrt{2\pi}}} e^{-\frac{1}{2}\left(\frac{x -\mu}{\sigma}\right)^2}}
```

"""

# ╔═╡ 723365e7-1fad-4899-8ac1-fb8674e2b9a7
md"``\mu``: $(@bind μ4_ Slider(-5:.1:5.0, default=1.0, show_value=true)),
``\sigma``: $(@bind σ4_ Slider(0.1:.1:2, default=1.0, show_value=true))"

# ╔═╡ ff303022-1eff-4278-be92-edf46c747bec
md"
Add ``-\frac{1}{2}\left(\frac{x -\mu}{\sigma}\right)^2``: $(@bind add_kernel CheckBox(default=false)), 
Add exp ``e^{-\frac{1}{2}\left(\frac{x -\mu}{\sigma}\right)^2}``: $(@bind add_ekernel CheckBox(default=false)), 
Add final ``p(x)``: $(@bind add_px CheckBox(default=false))
"

# ╔═╡ 6dba8be7-b62b-42d4-835a-d5ed827befa9
begin
	f1(x; μ=0, σ=1) = ((x - μ)/σ )^2
	f2(x; μ=0, σ=1) = -0.5 * f1(x; μ=μ, σ=σ)
	f3(x; μ=0, σ=1) = exp(f2(x; μ=μ, σ=σ))
	f4(x; μ=0, σ=1) = 1/(σ * sqrt(2π)) *exp(f2(x; μ=μ, σ=σ))
end;

# ╔═╡ a862e9d6-c31d-4b21-80c0-e359a5435b6b
let
	μ = μ4_
	σ = σ4_
	# f1(x) = ((x - μ)/σ )^2
	# f2(x) = -0.5* f1(x)
	# f3(x) = exp(f2(x))
	maxy = f4(μ; μ=μ, σ=σ) 
	# f4(x) = 1/(σ * sqrt(2π)) *exp(f2(x))
	# plot(f1, lw=2, label=L"\left(\frac{x-\mu}{\sigma}\right)^2", title=L"f(x)= \left(({x-\mu})/{\sigma}\right)^2")
	if add_kernel
		plt = plot(range(μ -5, μ+5, 100), (x) -> f2(x; μ=μ, σ=σ), lw=1.5, lc=2,label=L"-\frac{1}{2}\left(\frac{x-\mu}{\sigma}\right)^2", title=L"f(x)= -\frac{1}{2} \left(({x-\mu})/{\sigma}\right)^2", framestyle=:origin, ylim=[-2, max(maxy + 0.1, 1.5)])
	else
		plt = plot(framestyle=:origin, ylim=[-2,1.5], xlim=[-2,2])
	end
	if add_ekernel
		plot!((x) -> f3(x; μ=μ, σ=σ), lw=1.5, lc=3, label=L"e^{-\frac{1}{2}\left(\frac{x-\mu}{\sigma}\right)^2}", title=L"f(x)= e^{-\frac{1}{2}\left(({x-\mu})/{\sigma}\right)^2}")
	end
	vline!([μ], label="", ls=:dash, lw=2, lc=2, la=0.5)
	if add_px
		plot!(x -> f4(x; μ=μ, σ=σ), lw=3, lc=4, label=L"\frac{1}{\sigma\sqrt{2\pi}}e^{-\frac{1}{2}\left(\frac{x-\mu}{\sigma}\right)^2}", title="Gaussian prob. density")
	end
	plt
end

# ╔═╡ 4e55339d-4b61-4584-81c8-a9f35312830d
ThreeColumn(let
	μ = μ4_
	σ = 1
	maxy = f4(μ; μ=μ, σ=σ) 
	plt = plot(range(μ -4, μ+4, 100), (x) -> f2(x; μ=μ, σ=σ), lw=2, lc=2, title = "", framestyle=:origin, ylim=[-2, max(maxy + 0.1, 1.5)], size=(230,250), label=L"-\frac{1}{2} \left(({x-\mu})/{\sigma}\right)^2", legendfontsize=8, legend=:outertop)
	vline!([μ], label="", ls=:dash, lw=2, lc=2, la=0.5)
	plt
end, let
	μ = μ4_
	σ = 1
	maxy = f4(μ; μ=μ, σ=σ) 
	plt = plot(range(μ -4, μ+4, 100), (x) -> f2(x; μ=μ, σ=σ), lw=1., lc=2,label="" , ls=:dash, framestyle=:origin, ylim=[-2,max(maxy + 0.1, 1.5)], size=(230,250))
	
	plot!(x -> f3(x; μ=μ, σ=σ), lw=2, lc=3, title="", ylim=[-2,max(maxy + 0.1, 1.5)], titlefontsize=15, label=L"e^{-\frac{1}{2}\left(({x-\mu})/{\sigma}\right)^2}", legendfontsize=8, legend=:outertop)

	vline!([μ], label="", ls=:dash, lw=2, lc=2, la=0.5)
	plt
end, let
	μ = μ4_
	σ = 1
	maxy = f4(μ; μ=μ, σ=σ) 
	plt = plot(range(μ -4, μ+4, 100), (x) -> f2(x; μ=μ, σ=σ), lw=1, ls=:dash, lc=2,label="", title="", framestyle=:origin, ylim=[-2,max(maxy + 0.1, 1.5)],size=(230,250))
	plot!(x -> f3(x; μ=μ, σ=σ), lw=1, ls=:dash, lc=3, label="", title="")
	# end
	vline!([μ], label="", ls=:dash, lw=2, lc=2, la=0.5)
	plot!(x -> f4(x; μ=μ, σ=σ), lw=3, lc=4, title="", label=L"1/{\sigma\sqrt{2\pi}}\,e^{-\frac{1}{2}\left(\frac{x-\mu}{\sigma}\right)^2}", legendfontsize=8, legend=:outertop)

	plt
end)

# ╔═╡ 3f189707-d21d-4bf6-bc86-35304206281a
md"""

## *Normalising constant

```math
\Large
p(x)=\colorbox{lightgreen}{$\frac{1}{\sigma \sqrt{2\pi}}$} e^{-\frac{1}{2}\left(\frac{x -\mu}{\sigma}\right)^2}
```


* ``\large \colorbox{lightgreen}{$\frac{1}{\sigma \sqrt{2\pi}}$}``: normalising constant, a contant from ``x``'s perspective

* it normalises the density such that 

$$\large\int p(x)\mathrm{d}x = 1$$

* *in other words*


```math
\large
\int e^{-\frac{1}{2}\left(\frac{x -\mu}{\sigma}\right)^2}dx = \left (\frac{1}{\sigma \sqrt{2\pi}}\right )^{-1} = \sigma \sqrt{2\pi}
```

"""

# ╔═╡ 66a08217-481a-4956-ba8a-c4c822f3d0d2
figure_url = "https://leo.host.cs.st-andrews.ac.uk/figs/figs4CS5010/";

# ╔═╡ b02d6493-8da9-4d4d-a090-353f96addbdd
# $$\begin{equation*}
# p(\mathbf{x})=\mathcal{N}(\mathbf{x}| \boldsymbol{\mu}, \boldsymbol{\Sigma}) =  \underbrace{\frac{1}{(2\pi)^{d/2} |\boldsymbol\Sigma|^{1/2}}}_{\text{normalising constant}}
# \exp \left \{-\frac{1}{2} \underbrace{({x} - {\mu})^\top \Sigma^{-1}({x}-{\mu})}_{\text{distance btw } x \text{ and } \mu }\right\}
# \end{equation*}$$
# #

# ╔═╡ b88ce59d-a715-48af-b7f6-cc867ef2b50e
md"""

## Recap: Euclidean distance 


"""

# ╔═╡ dc64c064-6cee-4fba-8a36-e9b3264ebbbf
TwoColumn(md"""

\
\

_Squared_ Euclidean distance between $\mathbf{x}\in \mathbb{R}^d,\boldsymbol{\mu}\in \mathbb{R}^d$ is

$$\|\mathbf{x} - \boldsymbol{\mu}\|^2 = \sum_{j=1}^d (x_j - \mu_{j})^2$$
* or **straightline distance** between them 
* note ``\mathbf{x} = [x_1, \ldots, x_d]^\top`` and ``\boldsymbol\mu = [\mu_{1}, \ldots, \mu_{d}]^\top`` are vectors

""", 
	
	let
	μ = [1,1]
 	t = 0:0.01:2π
	r = 1 
	# plot(μ[1] .+ cos.(t), μ[2] .+ sin.(t), ratio=1,  xlim = μ .+ [-1.5, 1.5],lw=2, label="", framestyle=:origin,  size=(300,300))

	plt = scatter([μ[1]], [μ[2]], label=L"\mu", marker=:x, markerstrokewidth=5, markersize=6, size=(300,300), ratio=1,c=2,  xlim = μ .+ [-1.5, 1.5], ylim = μ .+ [-1.5, 1.5], framestyle=:zerolines)
	θs = range(0, 2π, 15)

	vs = [cos.(θs) sin.(θs)] .+ μ'
	ii = 3
	scatter!(vs[ii:ii,1], vs[ii:ii,2], c=1,  markersize=5, label="")
	plot!([μ[1], vs[ii, 1]],  [μ[2], vs[ii, 2]], lc=:gray, arrow=Plots.Arrow(:close, :both, 1,1),  lw=0.6, st=:path, label="")

		
	annotate!(μ[1], μ[2], text(L"μ", 20, :red,:top))
	annotate!(vs[ii, 1], vs[ii, 2], text(L"\mathbf{x}", 20, :blue,:top))
	if false 
		plot!(μ[1] .+ cos.(t), μ[2] .+ sin.(t), lw=2, label="", framestyle=:origin,  size=(300,300))
		scatter!(vs[:,1], vs[:,2], label=L"\mathbf{x}", c=1,  markersize=2)
			for i in 1:size(vs)[1]
			# 	quiver!([μ[1]], [μ[2]], quiver=([vs[i, 1]], [vs[i, 2]]), lw=1, lc=:gray)
				# quiver!([μ[1]], [μ[2]], quiver=([vs[i, 1]-μ[1]], vs[i, 2]-μ[2]), lc=:gray, lw=1)
			plot!([μ[1], vs[i, 1]],  [μ[2], vs[i, 2]], lc=:gray, arrow=Plots.Arrow(:close, :both, 1,1),  lw=0.6, st=:path, label="")
			end

	end

	plt

end)

# ╔═╡ 2d72550f-6dd4-496b-a763-125ec7f9d251
md"""

## Recap: Euclidean distance 


"""

# ╔═╡ 7d625061-fdef-440b-b541-0dd141a50c95
md"""
Add all ``\mathbf{x}``s: $(@bind add_xs CheckBox(default=false))
"""

# ╔═╡ 848c79ff-61b3-47e1-a574-4549afe9900f
TwoColumn(md"""
_Squared_ Euclidean distance between $\mathbf{x}\in \mathbb{R}^d,\boldsymbol{\mu}\in \mathbb{R}^d$ is

$$\|\mathbf{x} - \boldsymbol{\mu}\|^2 = \sum_{j=1}^d (x_j - \mu_{j})^2$$
* or **straightline distance** between them 
* note ``\mathbf{x} = [x_1, \ldots, x_d]^\top`` and ``\boldsymbol\mu = [\mu_{1}, \ldots, \mu_{d}]^\top`` are vectors


> if we fix $\boldsymbol\mu$ and a **distance**, 
> * ``\mathbf{x}``s forms a circle in $\mathbb{R}^2$ (or sphere in $\mathbb{R}^3$ or hypersphere for $d>3$) 
""", 
	
	let
	μ = [1,1]
 	t = 0:0.01:2π
	r = 1 
	# plot(μ[1] .+ cos.(t), μ[2] .+ sin.(t), ratio=1,  xlim = μ .+ [-1.5, 1.5],lw=2, label="", framestyle=:origin,  size=(300,300))

	plt = scatter([μ[1]], [μ[2]], label=L"\mu", marker=:x, markerstrokewidth=5, markersize=6, size=(300,300), ratio=1,c=2,  xlim = μ .+ [-1.5, 1.5], ylim = μ .+ [-1.5, 1.5])
	θs = range(0, 2π, 15)

	vs = [cos.(θs) sin.(θs)] .+ μ'
		ii = 2
	scatter!(vs[ii:ii,1], vs[ii:ii,2], label="", c=1,  markersize=5)
	plot!([μ[1], vs[ii, 1]],  [μ[2], vs[ii, 2]], lc=:gray, arrow=Plots.Arrow(:close, :both, 1,1),  lw=0.6, st=:path, label="")

		
	annotate!(μ[1], μ[2], text(L"μ", 20, :red,:top))

	if add_xs 
		plot!(μ[1] .+ cos.(t), μ[2] .+ sin.(t), lw=2, label="", framestyle=:origin, lc=1)
		scatter!(vs[:,1], vs[:,2], label=L"\mathbf{x}", c=1,  markersize=2)
		for i in 1:size(vs)[1]
			plot!([μ[1], vs[i, 1]],  [μ[2], vs[i, 2]], lc=:gray, arrow=Plots.Arrow(:close, :both, 1,1),  lw=0.6, st=:path, label="")
		end

	end

	plt

end)

# ╔═╡ aecdb616-d38b-4db6-af4c-55c988a3139d
md"""


## Recap: Euclidean distance as Quadratic form


"""

# ╔═╡ 82c78222-131f-4a38-a014-00cfee6d09ed
TwoColumn(md"""

Euclidean distance can also be formed as
\
\


$$\begin{align}\sum_{j=1}^d (x_j - \mu_{j})^2&=\begin{bmatrix}x_1- \mu_{1}& \ldots & x_d- \mu_{d}\end{bmatrix}\begin{bmatrix} x_1-\mu_{1} \\ \vdots \\ x_d-\mu_{d}\end{bmatrix} \\
&= \underbrace{(\mathbf{x} - \boldsymbol\mu)^\top}_{\text{row vector}} \underbrace{(\mathbf{x} - \boldsymbol\mu)}_{\text{column vector}}\\ 
&= \boxed{(\mathbf{x} - \boldsymbol\mu)^\top  \mathbf{I}^{-1}  (\mathbf{x} - \boldsymbol\mu)}\end{align}$$



""",let
	μ = [1.25, 0.5]
	# plot(μ[1] .+ cos.(t), μ[2] .+ sin.(t), ratio=1,  xlim = μ .+ [-1.5, 1.5],lw=2, label="", framestyle=:origin,  size=(300,300))

	plt = scatter([μ[1]], [μ[2]], label="", marker=:x, markerstrokewidth=0, markersize=0, size=(300,300), ratio=1,c=2,  xlim = [-0.5, 2], ylim =  [-0.5, 2.2], framestyle=:zerolines)
	# θ = π/4

	quiver!([0], [0], quiver=([μ[1]], [μ[2]]), lw=1.5, lc=2)
	v = [1.5, 1.8]
	ii = 3
	scatter!([v[1]], [v[2]], c=1,  markersize=0, label="")
	quiver!([0], [0], quiver=([v[1]], [v[2]]), lw=1.5, lc=1)
	# plot!([μ[1], v[1]],  [μ[2], v[2]], lc=:gray, arrow=Plots.Arrow(:close, :both, 1,1),  lw=0.6, st=:path, label="")
	xmu = - v + μ 
	quiver!([v[1]], [v[2]], quiver=([xmu[1]], [xmu[2]]), lw=3, lc=3)	
	annotate!(μ[1], μ[2], text(L"μ", 20, :red,:top))
	annotate!(v[1], v[2], text(L"\mathbf{x}", 20, :blue,:bottom))
	
	θ = acos(dot(-xmu, [1,0])/ norm(xmu)) * 180/π
	annotate!(.5 * (μ[1] + v[1]), .5 * (μ[2] + v[2]),text(L"\mathbf{x}-\mu", 20, :green, :top, rotation=θ))

		# annotate!(0.5*b[1], 0.5*b[2], text(L"\sqrt{\mathbf{b}^\top\mathbf{b}}", 18, :bottom, rotation = θb ))
	plt

end )

# ╔═╡ 3864434a-0f66-4872-81c6-84353d870f50
md"""


## Recap: Euclidean distance as Quadratic form


"""

# ╔═╡ a0a1eb6a-74e1-4d1e-8097-b7b3c04cffa2
TwoColumn(md"""

Euclidean distance can also be formed as a **quadratic form**

$$\begin{align}\sum_{j=1}^d (x_j - \mu_{j})^2&=\begin{bmatrix}x_1- \mu_{1}& \ldots & x_d- \mu_{d}\end{bmatrix}\begin{bmatrix} x_1-\mu_{1} \\ \vdots \\ x_d-\mu_{d}\end{bmatrix} \\
&= \underbrace{(\mathbf{x} - \boldsymbol\mu)^\top}_{\text{row vector}} \underbrace{(\mathbf{x} - \boldsymbol\mu)}_{\text{column vector}}\\ 
&= \boxed{(\mathbf{x} - \boldsymbol\mu)^\top  \mathbf{I}^{-1}  (\mathbf{x} - \boldsymbol\mu)}\end{align}$$

* **quadratic form**: *a row vector* ``\times`` *a square matrix* ``\times`` *a column vector*
  *  generalisation of scalar quadratic function ``x\cdot a \cdot x``
* note that $\mathbf{I}$ is the Identity matrix 
  
$\mathbf{I} = \begin{bmatrix} 1 &0& \ldots & 0 \\
  0 &1 & \ldots & 0  \\
  \vdots & \vdots & \vdots & \vdots \\
  0 & 0 & \ldots & 1
  \end{bmatrix}$

""",let
	μ = [1.25, 0.5]
	# plot(μ[1] .+ cos.(t), μ[2] .+ sin.(t), ratio=1,  xlim = μ .+ [-1.5, 1.5],lw=2, label="", framestyle=:origin,  size=(300,300))

	plt = scatter([μ[1]], [μ[2]], label="", marker=:x, markerstrokewidth=0, markersize=0, size=(300,300), ratio=1,c=2,  xlim = [-0.5, 2], ylim =  [-0.5, 2.2], framestyle=:zerolines)
	# θ = π/4

	quiver!([0], [0], quiver=([μ[1]], [μ[2]]), lw=1.5, lc=2)
	v = [1.5, 1.8]
	ii = 3
	scatter!([v[1]], [v[2]], c=1,  markersize=0, label="")
	quiver!([0], [0], quiver=([v[1]], [v[2]]), lw=1.5, lc=1)
	# plot!([μ[1], v[1]],  [μ[2], v[2]], lc=:gray, arrow=Plots.Arrow(:close, :both, 1,1),  lw=0.6, st=:path, label="")
	xmu = - v + μ 
	quiver!([v[1]], [v[2]], quiver=([xmu[1]], [xmu[2]]), lw=3, lc=3)	
	annotate!(μ[1], μ[2], text(L"μ", 20, :red,:top))
	annotate!(v[1], v[2], text(L"\mathbf{x}", 20, :blue,:bottom))
	
	θ = acos(dot(-xmu, [1,0])/ norm(xmu)) * 180/π
	annotate!(.5 * (μ[1] + v[1]), .5 * (μ[2] + v[2]),text(L"\mathbf{x}-\mu", 20, :green, :top, rotation=θ))

		# annotate!(0.5*b[1], 0.5*b[2], text(L"\sqrt{\mathbf{b}^\top\mathbf{b}}", 18, :bottom, rotation = θb ))
	plt

end )

# ╔═╡ 85a0636b-5475-4ece-a03d-cbbc4603cbc5
md"""

## Multivariate Gaussian
"""

# ╔═╡ 4ce3cb20-e70d-4d39-8572-ee7ac6d8e441
html"<center><img src='https://leo.host.cs.st-andrews.ac.uk/figs/CS5914/gaussian_eq_md_alt.png' width = '900' /></center>"

# ╔═╡ b0f0628e-ceff-4e6c-819b-2778ca4d4b08
md"""


* mean: ``\boldsymbol\mu \in \mathbb{R}^d`` is the  vector 


```math
\large
\boldsymbol{\mu} =\begin{bmatrix}\mu_1 \\ \mu_2 \\\vdots \\ \mu_d \end{bmatrix}
```
* variance: ``\boldsymbol\Sigma`` is a $d \times d$ *symmetric* and *positive definite* **matrix**
  * *symmetric*: ``\boldsymbol{\Sigma}^\top = \boldsymbol{\Sigma}``
  * _positive definite_: ``\mathbf{x}^\top \boldsymbol{\Sigma} \mathbf{x} > 0`` and ``\mathbf{x}^\top \boldsymbol{\Sigma}^{-1} \mathbf{x} > 0`` for all ``\mathbf{x} \in \mathbb{R}^d`` (except ``\mathbf{0}``)
```math
\large
\boldsymbol{\Sigma} =\begin{bmatrix}\sigma_1^2 & \sigma_{12} & \ldots & \sigma_{1d} \\ \sigma_{21}  & \sigma_2^2 & \ldots & \sigma_{2d}\\\vdots & \vdots & \vdots & \vdots \\ \sigma_{d1} &  \sigma_{d2} & \ldots & \sigma_{d}^2\end{bmatrix}
```
  
"""

# ╔═╡ 8c72fd4f-d783-48bb-8f62-c00e16399e2a
md"""

* **distance measure** between $x$ and $\mu$ (adjusted with correlations) is known as **mahalanobis distance**

  $\large \text{mahalanobis distance:}\;\;d_{\boldsymbol{\Sigma}} = \sqrt{(\mathbf{x} - \boldsymbol{\mu})^\top \boldsymbol\Sigma^{-1}(\mathbf{x}-\boldsymbol{\mu})}$

"""

# ╔═╡ ae216b5f-59bb-4d0e-ab58-ddea82d5ee2f
md"""

## Dissect multivariate Gaussian
"""

# ╔═╡ c1ac8263-f226-4ae9-b0de-ab4eacb4ff42
html"<center><img src='https://leo.host.cs.st-andrews.ac.uk/figs/CS5914/gaussian_eq_md_alt.png' width = '900' /></center>"

# ╔═╡ c07428c0-1c9d-41ba-9702-e3dffe2e0159
md"
Add ``-\frac{1}{2}(x -\mu){\Sigma}^{-1}(x -\mu)``: $(@bind add_kernel2 CheckBox(default=true)), 
Add exp ``e^{-\frac{1}{2}(x -\mu){\Sigma}^{-1}(x -\mu)}``: $(@bind add_ekernel2 CheckBox(default=false)), 
Add final ``p(x)``: $(@bind add_px2 CheckBox(default=false))
"

# ╔═╡ bfedf482-2ef2-4d3d-93d1-796ae61ef48d
let
	plotly()
	μ = [0, 0]
	Σ = Matrix(I, 2, 2) / 15
	Σinv = inv(Σ)
	f1(x) = (x -μ)' * Σinv * (x -μ)
	f2(x) = -0.5 * f1(x)
	f3(x) = exp(f2(x))
	f4(x) = pdf(MvNormal(μ, Σ), x)
	xlen = 2
	ylen = 2
	plt = plot(range(μ[1] -xlen, μ[1]+xlen, 50), range(μ[2] -ylen, μ[2]+ylen, 50), (x, y) -> f1([x, y]), st=:surface, alpha=0.55, label=L"-\frac{1}{2}\left(\frac{x-\mu}{\sigma}\right)^2", title=L"f(x)= -\frac{1}{2} ({x-\mu})\Sigma^{-1}(x-\mu)", framestyle=:zerolines,  zlim=[-5,25], colorbar=false, ratio=1)
	if add_kernel2
		plot(range(μ[1] -xlen, μ[1]+xlen, 50), range(μ[2] -ylen, μ[2]+ylen, 50), (x, y) -> f1([x, y]), st=:surface, alpha=0.25, label=L"-\frac{1}{2}\left(\frac{x-\mu}{\sigma}\right)^2", title=L"f(x)= -\frac{1}{2} ({x-\mu})\Sigma^{-1}(x-\mu)", framestyle=:zerolines,  zlim=[-5,25], colorbar=false, ratio=1)
		plt=plot!(range(μ[1] -xlen, μ[1]+xlen, 50), range(μ[2] -ylen, μ[2]+ylen, 50), (x, y) -> f2([x, y]), st=:surface, alpha=0.55, label=L"-\frac{1}{2}\left(\frac{x-\mu}{\sigma}\right)^2", title=L"f(x)= -\frac{1}{2} ({x-\mu})\Sigma^{-1}(x-\mu)", framestyle=:zerolines,  zlim=[-20,20], colorbar=false, ratio=1)
	end
	if add_ekernel2
		plt=plot(range(μ[1] -xlen, μ[1]+xlen, 50), range(μ[2] -ylen, μ[2]+ylen, 50), (x, y) -> f2([x, y]), st=:surface, alpha=0.3, label=L"-\frac{1}{2}\left(\frac{x-\mu}{\sigma}\right)^2", title=L"f(x)= -\frac{1}{2} ({x-\mu})\Sigma^{-1}(x-\mu)", framestyle=:zerolines,  zlim=[-20,20], colorbar=false, ratio=1)
		maxz = f3(μ)
		plot!(range(μ[1] -xlen, μ[1]+xlen, 50), range(μ[2] -ylen, μ[2]+ylen, 50), (x, y) -> f3([x,y]), st=:surface,  label=L"e^{-\frac{1}{2}\left(\frac{x-\mu}{\sigma}\right)^2}", title=L"f(x)= e^{-\frac{1}{2} ({x-\mu})\Sigma^{-1}(x-\mu)}", zlim=[-15,maxz+0.5], alpha=0.8, colorbar=false)
	end
	# vline!([μ], label=L"\mu", ls=:dash, lw=2, lc=:gray, la=0.5)
	if add_px2
		maxz = f4(μ)
		plot!(range(μ[1] -xlen, μ[1]+xlen, 80), range(μ[2] -ylen, μ[2]+ylen, 80), (x, y) -> f4([x,y]), label=L"p(x)", alpha=0.8, title="Gaussian prob. density", st=:surface, zlim=[-10, maxz+0.5])
	end
	plt
end

# ╔═╡ a08f53bd-91d3-45b2-ab73-42e9b4bae563
md"""
## Example when $\boldsymbol{\Sigma}=\mathbf{I}$

Consider $\mathbb{R}^2$, and 

$\large \boldsymbol{\Sigma}=\mathbf{I}= \begin{bmatrix} 1 & 0 \\0 & 1\end{bmatrix}$

then the kernel

$(\mathbf{x} - \boldsymbol{\mu})^\top \mathbf I^{-1}(\mathbf{x}-\boldsymbol{\mu})=(\mathbf{x} - \boldsymbol{\mu})^\top(\mathbf{x}-\boldsymbol{\mu})= (x_{1} -\mu_1)^2+ (x_2 - \mu_2)^2$

*i.e.* Euclidean distance between $\mathbf x$ and $\boldsymbol\mu$

* remember $\mathbf I^{-1} = \mathbf I$
"""

# ╔═╡ e69bbba2-3167-4edd-b9db-6aba7a0dceed
begin
	gr()
	Random.seed!(234)
	μ₁ = [2,2]
	mvn1 = 	MvNormal(μ₁, Matrix(1.0I, 2,2))
	spl1 = rand(mvn1, 500)
	x₁s = μ₁[1]-3:0.1:μ₁[1]+3
	x₂s = μ₁[2]-3:0.1:μ₁[2]+3	
	mvnplt₁ = scatter(spl1[1,:], spl1[2,:], ratio=1, label="", xlabel=L"x_1", ylabel=L"x_2", alpha=0.5)	
	scatter!([μ₁[1]], [μ₁[2]], ratio=1, label=L"{\mu}=%$(μ₁)", markershape = :x, markerstrokewidth=5, markersize=8)	
	plot!(x₁s, x₂s, (x1, x2)->pdf(mvn1, [x1, x2]), levels=4, linewidth=3, c=:jet, st=:contour, colorbar=false)	

	# plot(mvnplt₁, mvnplt₂)
end

# ╔═╡ d2085d25-1867-4037-92e0-8cbd2d1ada39
begin
	gr()
	mvnplt₂ = surface(x₁s, x₂s, (x1, x2)->pdf(mvn1, [x1, x2]), color=:lighttest, st=:surface, colorbar=true, title="Probability density plot")
end

# ╔═╡ bec564e6-fea8-4645-8f25-8babbe0b13df
md"""

## Example (Diagonal $\boldsymbol \Sigma$)

When $\boldsymbol\Sigma =  \begin{bmatrix} \sigma_1^2 & 0 \\0 & \sigma_2^2\end{bmatrix}$, then 


$\large \boldsymbol\Sigma^{-1} = \begin{bmatrix} 1/\sigma_1^2 & 0 \\0 & 1/\sigma_2^2\end{bmatrix}$


The distance measure forms *axis aligned* ellipses


$\large (\mathbf{x} - \boldsymbol{\mu})^\top \boldsymbol\Sigma^{-1}(\mathbf{x}-\boldsymbol{\mu})= \underbrace{\frac{(x_{1} -\mu_1)^2}{\sigma_1^2}+ \frac{(x_2 - \mu_2)^2}{\sigma_2^2}}_{\text{analytical form of an ellipse}}$

* note 
  $\mathbf x-\boldsymbol\mu=\begin{bmatrix}x_1-\mu_1\\ x_2-\mu_2\end{bmatrix}$
  is a column vector

"""

# ╔═╡ 55b63786-4e04-497a-abae-e6e229a7edf2
md"""

## Example (Diagonal $\boldsymbol \Sigma$)

When $\boldsymbol\Sigma =  \begin{bmatrix} \sigma_1^2 & 0 \\0 & \sigma_2^2\end{bmatrix}$, then 


$\large \boldsymbol\Sigma^{-1} = \begin{bmatrix} 1/\sigma_1^2 & 0 \\0 & 1/\sigma_2^2\end{bmatrix}$


The distance measure forms *axis aligned* ellipses


$\large (\mathbf{x} - \boldsymbol{\mu})^\top \boldsymbol\Sigma^{-1}(\mathbf{x}-\boldsymbol{\mu})= \underbrace{\frac{(x_{1} -\mu_1)^2}{\sigma_1^2}+ \frac{(x_2 - \mu_2)^2}{\sigma_2^2}}_{\text{analytical form of an ellipse}}$


#### Which also implies $\Rightarrow$ *Independent* dimensions


```math
\large 
p(\mathbf{x}) = p(x_1) p(x_2)= \mathcal{N}(x_1;\mu_1, \sigma_1^2)\cdot \mathcal{N}(x_2;\mu_2, \sigma_2^2) 
```
"""

# ╔═╡ 0d38b7f0-39e7-44db-8343-3c80f32a5c30
md"""

## Diagonal ``\mathbf\Sigma`` ``\Leftrightarrow`` _Independence_ 

When $\boldsymbol\Sigma =  \begin{bmatrix} \sigma_1^2 & 0 \\0 & \sigma_2^2\end{bmatrix}$ *i.e.* diagonal, then

$(\mathbf{x} - \boldsymbol{\mu})^\top \boldsymbol\Sigma^{-1}(\mathbf{x}-\boldsymbol{\mu})= \frac{(x_{1} -\mu_1)^2}{\sigma_1^2}+ \frac{(x_2 - \mu_2)^2}{\sigma_2^2}$

* where $\mathbf{x} = \begin{bmatrix} x_1\\ x_2 \end{bmatrix}$
The joint probability distribution ``p(\mathbf{x})`` becomes

$$\begin{align}p(\mathbf{x}) 
&= \frac{1}{(\sqrt{2\pi})^2 \sigma_1 \sigma_2} \exp{\left \{ -\frac{1}{2} \left (\frac{(x_1-\mu_1)^2}{\sigma_1^2}+\frac{(x_2-\mu_2)^2}{\sigma_2^2}\right ) \right\}}\\
 &=\underbrace{\frac{1}{\sqrt{2\pi}\sigma_1}\exp\left [-\frac{1}{2} \frac{(x_1-\mu_1)^2}{\sigma_1^2}\right ]}_{{p(x_1)}} \cdot \underbrace{\frac{1}{\sqrt{2\pi}\sigma_2}\exp\left [-\frac{1}{2} \frac{(x_2-\mu_2)^2}{\sigma_2^2}\right ]}_{p(x_2)} \\
&= p(x_1)p(x_2) 
\end{align}$$



Generalise the idea to ``d`` dimensional ``\mathbf{x} \in \mathbb{R}^d``

$$\begin{align}p(\mathbf{x}) 
&= \frac{1}{\left (\sqrt{2\pi} \right )^d \prod_i \sigma_i} \exp{\left \{ -\frac{1}{2} \sum_{i=1}^d\frac{(x_i-\mu_i)^2}{\sigma_i^2}  \right\}} = \prod_{i=1}^d p(x_i; \mu_i, \sigma_i^2)
\end{align}$$
"""

# ╔═╡ 434a9163-bddb-4702-b97b-b193a6f08b77
aside(tip(md"""

```math
e^{a+b} = e^a\cdot e^b
```
"""))

# ╔═╡ 001496ae-65ca-4de1-a211-4cbbafc7777f
md"""

##  Example (diagonal $\boldsymbol \Sigma$)

When $\boldsymbol\Sigma =  \begin{bmatrix} \sigma_1^2 & 0 \\0 & \sigma_2^2\end{bmatrix}$, then 
The distance measure forms *axis aligned* ellipses


$\large (\mathbf{x} - \boldsymbol{\mu})^\top \boldsymbol\Sigma^{-1}(\mathbf{x}-\boldsymbol{\mu})= \underbrace{\frac{(x_{1} -\mu_1)^2}{\sigma_1^2}+ \frac{(x_2 - \mu_2)^2}{\sigma_2^2}}_{\text{analytical form of an ellipse}}$


#### Which also implies $\Rightarrow$ *Independent* dimensions


```math
\large 
p(\mathbf{x}) = p(x_1) p(x_2)= \mathcal{N}(x_1;\mu_1, \sigma_1^2)\cdot \mathcal{N}(x_2;\mu_2, \sigma_2^2) 
```
"""

# ╔═╡ 1663b630-a57a-44c5-90c6-95be2ac42aa8
md"``\sigma_1^2``: $(@bind σ₁² Slider(0.5:0.1:5, default=1.9, show_value=true))"

# ╔═╡ 5acb28a1-49f1-412d-9c6e-b840cab687aa
md"``\sigma_2^2``: $(@bind σ₂² Slider(0.1:0.1:5, default=1, show_value=true))"

# ╔═╡ 2c152ff3-050f-42de-b434-21b2658d572f
md"""Add ``p(x_2|X_1= x_1)``: $(@bind add_px2x1 CheckBox(default=false)),

``X_1=`` $(@bind x1_ Slider(-1.5:0.1:4))
"""

# ╔═╡ da1fd88b-0e2a-49e6-8fb5-135a9d18be6e
begin
	gr()
	Random.seed!(123)
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
	plt_gau2=scatter(spl₂[1,:], spl₂[2,:], ratio=1, label="", xlabel=L"x_1", ylabel=L"x_2", alpha=0.5, framestyle=:zerolines)	
	scatter!([μ₂[1]], [μ₂[2]], ratio=1, label=L"\mu=%$(μ₂)", markershape = :x, markerstrokewidth=5, markersize=8)	
	plot!(x₁s, x₂s, (x1, x2)->pdf(mvn₂, [x1, x2]), levels=4, linewidth=4, c=:jet, st=:contour, colorbar=:false)	

	if add_px2x1
		x = x1_
		# μi = dot(true_w, [1, x])
		μi = μ₂[2] + Σ₂[2, 1] * Σ₂[1,1]^(-1) * (x - μ₂[1])
		σ = Σ₂[2,2] - Σ₂[2, 1] * Σ₂[1,1]^(-1) * Σ₂[1,2]
		xs_ = μi- 4 * sqrt(σ) :0.05:μi+ 4 * sqrt(σ)
		ys_ = pdf.(Normal(μi, sqrt(σ)), xs_)
		# ys_ = ys_ ./ maximum(ys_)
		plot!((x) -> μ₂[2] + Σ₂[2, 1] * Σ₂[1,1]^(-1) * (x - μ₂[1]), c=2, lw=3, ls=:dash, label="")
			# scatter!([x],[μi], markerstrokewidth =1, markershape = :diamond, c=:grey, label="", markersize=3)
		plot!(ys_ .+x, xs_, c=2, lw=2, label="")
		
	end
	plt_gau2
	# scatter!([xis[i]],[ys[i]], markershape = :circle, label="", c=1, markersize=4)
end

# ╔═╡ bda9fe31-88ae-4ce3-8038-9ca7f0a24378
begin
	plotly()
	surface(x₁s, x₂s, (x1, x2)->pdf(mvn₂, [x1, x2]), color=:lighttest, st=:surface, colorbar=true, xlabel="x1", ylabel="x2", zlabel="p(x)")
end

# ╔═╡ b20f121e-781a-4d17-8833-43d34396bf44
md"""
## Example (full covriance $\boldsymbol{\Sigma}$)

When 

$\large \boldsymbol\Sigma =  \begin{bmatrix} \sigma_1^2 & \sigma_{12} \\ \sigma_{21} & \sigma_2^2\end{bmatrix}$


``\sigma_{12} =\sigma_{21}``: **covariance** between ``x_1`` and ``x_2``, defined as

```math
\sigma_{12} = \mathbb{E}[(X_1 -\mu_1)(X_2 -\mu_2)]
```

* ``\sigma_{12} >0``: positively correlated
* ``\sigma_{12} < 0``: negatively correlated

##  Example (full covriance $\boldsymbol{\Sigma}$)

When 

$\large \boldsymbol\Sigma =  \begin{bmatrix} \sigma_1^2 & \sigma_{12} \\ \sigma_{21} & \sigma_2^2\end{bmatrix}$


* ``\sigma_{12} =\sigma_{21}``: covariance between ``x_1`` and ``x_2``, defined as


The distance measure still forms (rotated) **ellipses**


$(\mathbf{x} - \boldsymbol{\mu})^\top \boldsymbol\Sigma^{-1}(\mathbf{x}-\boldsymbol{\mu})= \underbrace{\frac{(\mathbf{v}_1^\top(\mathbf{x}-\boldsymbol\mu))^2}{\lambda_1} + \frac{(\mathbf{v}_2^\top(\mathbf{x}-\boldsymbol\mu))^2}{\lambda_2}}_{\text{still an analytical form of ellipse}}$

* ``\mathbf{v}_1`` and ``\mathbf{v}_2`` are the **eigen vectors** of $\boldsymbol\Sigma$; and ``\lambda_1, \lambda_2`` are the **eigen values**
* *i.e.* the rotated ellipse's basis (the $\textcolor{red}{\text{red vectors}}$ in the plot below)
* **positive** or **negatively correlated** inputs

"""

# ╔═╡ dd2fc1a7-4684-4956-aff3-25b23e1548e0
@bind σ₁₂ Slider(-1:0.02:1, default=0)

# ╔═╡ 1fb9b4c4-e7c7-4ceb-9193-3c1e83244445
md"``\sigma_{12}=\sigma_{21}``=$(σ₁₂)"

# ╔═╡ 0f7e65e5-5eea-4e4d-8ea4-0f04ce52f91d
md"""Add ``p(x_2|X_1= x_1)``: $(@bind add_px2x12 CheckBox(default=false)),

``X_1=`` $(@bind x1_2 Slider(-1.5:0.1:4))
"""

# ╔═╡ 492c3edd-772c-4d07-88bc-92a046190f29
let
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
	plt_gau3 = scatter(spl₃[1,:], spl₃[2,:], ratio=1, label="", xlabel=L"x_1", alpha=0.5, ylabel=L"x_2", framestyle=:zerolines)	
	scatter!([μ₂[1]], [μ₂[2]], ratio=1, label=L"\mu", markershape = :x, markerstrokewidth=5, markersize=8)	
	


	if add_px2x12
		x = x1_2
		# μi = dot(true_w, [1, x])
		μi = μ₂[2] + Σ₃[2, 1] * Σ₃[1,1]^(-1) * (x - μ₂[1])
		σ = Σ₃[2,2] - Σ₃[2, 1] * Σ₃[1,1]^(-1) * Σ₃[1,2]
		xs_ = μi - 4 * sqrt(σ) :0.05:μi+ 4 * sqrt(σ)
		ys_ = pdf.(Normal(μi, sqrt(σ)), xs_)
		# ys_ = ys_ ./ maximum(ys_)
			# scatter!([x],[μi], markerstrokewidth =1, markershape = :diamond, c=:grey, label="", markersize=3)
		plot!(ys_ .+x, xs_, c=2, lw=2, label=L"p(X_2|x_1)")

		plot!((x) -> μ₂[2] + Σ₃[2, 1] * Σ₃[1,1]^(-1) * (x - μ₂[1]), c=2, lw=3, ls=:dash, label="")

	else
		λs, vs = eigen(Σ₃)
		v1 = (vs .* λs')[:,1]
		v2 = (vs .* λs')[:,2]
		quiver!([μ₂[1]], [μ₂[2]], quiver=([v1[1]], [v1[2]]), linewidth=4, color=:red, alpha=0.8)
		quiver!([μ₂[1]], [μ₂[2]], quiver=([v2[1]], [v2[2]]), linewidth=4, color=:red, alpha=0.8)
		plot!(x₁s, x₂s, (x1, x2)->pdf(mvn₃, [x1, x2]), levels=4, linewidth=3, c=:jet, st=:contour, colorbar=false)
		
	end
	plt_gau3
end

# ╔═╡ 295644e3-b8ee-4d06-a17d-9a37b5f965ce
md"""

## *Positive definiteness of $\boldsymbol{\Sigma}$
When $\sigma_{12} = 1.0$ or $-1$ (or $|\sigma_{12}| >1$), 

$\boldsymbol\Sigma = \begin{bmatrix} 1 & 1 \\ 1 & 1\end{bmatrix}\;\; \text{or}\;\; \begin{bmatrix} 1 & -1 \\ -1 & 1\end{bmatrix}$ is no longer **positive definite**
* one of the dimension collapses (with zero variance!)
* the normalising constant does not exist as $|\boldsymbol\Sigma| =0$, therefore the Gaussian does not exist
* similar to the case when $\sigma^2=0$ for univariate Gaussian


**Positive definite**: that is for all $\mathbf{x}\in \mathbb{R}^d$

$\mathbf{x}^\top \boldsymbol\Sigma^{-1} \mathbf{x}>0$ 

* strictly positive so it needs to be a valid distance measure
"""

# ╔═╡ 1d0a3487-363e-4353-904a-73e2718748af
md"""

## Summary
"""

# ╔═╡ 2d94f5ba-9a3b-4615-b451-3ef3b393057d
html"<center><img src='https://leo.host.cs.st-andrews.ac.uk/figs/CS5914/gaussian_eq_md_alt.png' width = '900' /></center>"

# ╔═╡ f870fc82-56d4-4a39-991e-d42bc31cd9c8
md"""


* mean: ``\boldsymbol\mu \in \mathbb{R}^d`` is the  vector 
* variance: ``\boldsymbol\Sigma`` is a $d \times d$ *symmetric* and *positive definite* matrix 
* the kernel is a **distance measure** between $x$ and $\mu$

  $(\mathbf{x} - \boldsymbol{\mu})^\top \boldsymbol\Sigma^{-1}(\mathbf{x}-\boldsymbol{\mu})$
* the density is negatively related to the distance measure
  * the further away $\mathbf x$ from $\boldsymbol{\mu}$, then $p(\mathbf{x})$ is smaller
* ``\colorbox{lightblue}{$\frac{1}{(2\pi)^{d/2} | \boldsymbol{\Sigma}|^{1/2}}$}`` is the **normalising constant** (does not change w.r.t $\mathbf{x}$) such that

$$\int p(\mathbf x) d\mathbf x = 1$$
* ``| \boldsymbol\Sigma|`` is determinant, which measures the volume under the the bell surface (when it is a positive definite matrix)
"""

# ╔═╡ 0734ddb1-a9a0-4fe1-b5ee-9a839a33d1dc
md"""

## Appendix
"""

# ╔═╡ 9fc9a421-194e-4163-87f2-30989354932a
md"""


## *Eigen decomposition


**Eigen**-decompose ``\mathbf\Sigma``
```math
\boldsymbol\Sigma = \mathbf{V}\mathbf{\Lambda}\mathbf{V}^\top
```

* where ``\mathbf{V}`` are formed by the eigen-vectors and ``\mathbf\Lambda`` is a diagonal matrix with the eigen values

```math
\mathbf{V} = \begin{bmatrix}\vert & \vert & \vert & \vert\\\mathbf{v}_1 & \mathbf{v}_2 & \ldots & \mathbf{v}_d\\ \vert & \vert & \vert & \vert\end{bmatrix}, \;\;\mathbf{\Lambda} = \begin{bmatrix}\lambda_1 & 0 & \ldots & 0 \\ 0 & \lambda_2 & \ldots & 0\\ \vdots & \vdots & \vdots & \vdots \\ 0 & 0 & \ldots & \lambda_d\end{bmatrix}
```

* and since ``\mathbf{\Sigma}`` is symmetric and real, the eigen matrix is orthognormal, *i.e.* ``\mathbf{V}^\top =\mathbf{V}^{-1}`` 

Then, the precision matrix 

```math
\boldsymbol\Sigma^{-1} = (\mathbf{V}\mathbf{\Lambda}\mathbf{V}^\top)^{-1} = (\mathbf{V}^\top)^{-1} \mathbf{\Lambda}^{-1}\mathbf{V}^{-1}  = \mathbf{V}\mathbf{\Lambda}^{-1}\mathbf{V}^\top
```
and since ``\mathbf{\Lambda}`` and ``\mathbf{\Lambda}^{-1}`` are diagonal matrices, therefore

```math
\boldsymbol\Sigma^{-1}  = \mathbf{V}\mathbf{\Lambda}^{-1}\mathbf{V}^\top = \mathbf{V}\mathbf{\Lambda}^{-\frac{1}{2}}\mathbf{\Lambda}^{-\frac{1}{2}}\mathbf{V}^\top
```


The quadratic form becomes 



```math
(\mathbf{x}-\boldsymbol{\mu})^\top\underbrace{(\mathbf{V}\mathbf{\Lambda}^{-\frac{1}{2}}\mathbf{\Lambda}^{-\frac{1}{2}}\mathbf{V}^\top)}_{\mathbf{\Sigma}^{-1}}(\mathbf{x}-\boldsymbol{\mu}) = (\mathbf{\Lambda}^{-\frac{1}{2}}\mathbf{V}^\top(\mathbf{x}-\boldsymbol{\mu}))^\top(\mathbf{\Lambda}^{-\frac{1}{2}}\mathbf{V}^\top(\mathbf{x}-\boldsymbol{\mu}))
```

Note that ``\mathbf{\Lambda}^{-\frac{1}{2}}\mathbf{V}^\top(\mathbf{x}-\boldsymbol{\mu})`` is some linear transformation of ``\mathbf{x}-\boldsymbol{\mu}``
with transformation matrix ``\mathbf{\Lambda}^{-\frac{1}{2}}\mathbf{V}^\top``; the transformed vector can be expanded (for a 2-dimensional case) as 

```math
\begin{bmatrix}\lambda_1^{-1/2} & 0 \\ 0 &  \lambda_2^{-1/2}\end{bmatrix}\begin{bmatrix}\mathbf{v}_1^\top \\ \mathbf{v}_2^\top\end{bmatrix}\cdot (\mathbf{x}-\boldsymbol{\mu})= \begin{bmatrix}\lambda_1^{-1/2} & 0 \\ 0 &  \lambda_2^{-1/2}\end{bmatrix}\begin{bmatrix}\mathbf{v}_1^\top(\mathbf{x}-\boldsymbol{\mu}) \\ \mathbf{v}_2^\top(\mathbf{x}-\boldsymbol{\mu})\end{bmatrix}= \begin{bmatrix}\lambda_1^{-1/2} \mathbf{v}_1^\top(\mathbf{x}-\boldsymbol{\mu}) \\ \lambda_2^{-1/2} \mathbf{v}_2^\top(\mathbf{x}-\boldsymbol{\mu})\end{bmatrix}
```

its inner product or (squared) ``L_2`` norm is


```math
\begin{align}
&\begin{bmatrix}\lambda_1^{-1/2} \mathbf{v}_1^\top(\mathbf{x}-\boldsymbol{\mu}) & &\lambda_2^{-1/2} \mathbf{v}_2^\top(\mathbf{x}-\boldsymbol{\mu})\end{bmatrix}\begin{bmatrix}\lambda_1^{-1/2} \mathbf{v}_1^\top(\mathbf{x}-\boldsymbol{\mu}) \\ \lambda_2^{-1/2} \mathbf{v}_2^\top(\mathbf{x}-\boldsymbol{\mu})\end{bmatrix} \\
&= \frac{(\mathbf{v}_1^\top(\mathbf{x}-\boldsymbol{\mu}))^2}{\lambda_1}+ \frac{(\mathbf{v}_2^\top(\mathbf{x}-\boldsymbol{\mu}))^2}{\lambda_2}
\end{align}
```
"""

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

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
Distributions = "31c24e10-a181-5473-b8eb-7969acd0382f"
HypertextLiteral = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"
Images = "916415d5-f1e6-5110-898d-aaa5f9f070e0"
LaTeXStrings = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
Latexify = "23fbe1c1-3f47-55db-b15f-69d7ec21a316"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
LogExpFunctions = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
Plots = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
PlutoTeachingTools = "661c6b06-c737-4d37-b85c-46df65de6f69"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
Statistics = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
StatsBase = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
StatsPlots = "f3b207a7-027a-5e70-b257-86293d7955fd"

[compat]
Distributions = "~0.25.100"
HypertextLiteral = "~0.9.4"
Images = "~0.26.0"
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

julia_version = "1.9.3"
manifest_format = "2.0"
project_hash = "064da895d49e16c5a29303908d5165cc8d33a732"

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

[[deps.ArnoldiMethod]]
deps = ["LinearAlgebra", "Random", "StaticArrays"]
git-tree-sha1 = "62e51b39331de8911e4a7ff6f5aaf38a5f4cc0ae"
uuid = "ec485272-7323-5ecc-a04f-4719b315124d"
version = "0.2.0"

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

[[deps.ArrayInterface]]
deps = ["Adapt", "LinearAlgebra", "Requires", "SparseArrays", "SuiteSparse"]
git-tree-sha1 = "f83ec24f76d4c8f525099b2ac475fc098138ec31"
uuid = "4fba245c-0d91-5ea0-9b3e-6abc04ee57a9"
version = "7.4.11"

    [deps.ArrayInterface.extensions]
    ArrayInterfaceBandedMatricesExt = "BandedMatrices"
    ArrayInterfaceBlockBandedMatricesExt = "BlockBandedMatrices"
    ArrayInterfaceCUDAExt = "CUDA"
    ArrayInterfaceGPUArraysCoreExt = "GPUArraysCore"
    ArrayInterfaceStaticArraysCoreExt = "StaticArraysCore"
    ArrayInterfaceTrackerExt = "Tracker"

    [deps.ArrayInterface.weakdeps]
    BandedMatrices = "aae01518-5342-5314-be14-df237901396f"
    BlockBandedMatrices = "ffab5731-97b5-5995-9138-79e8c1846df0"
    CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba"
    GPUArraysCore = "46192b85-c4d5-4398-a991-12ede77f4527"
    StaticArraysCore = "1e83bf80-4336-4d27-bf5d-d5a4f845583c"
    Tracker = "9f7883ad-71c0-57eb-9f7f-b5c9e6d3789c"

[[deps.ArrayInterfaceCore]]
deps = ["LinearAlgebra", "SnoopPrecompile", "SparseArrays", "SuiteSparse"]
git-tree-sha1 = "e5f08b5689b1aad068e01751889f2f615c7db36d"
uuid = "30b0a656-2188-435a-8636-2ec0e6a096e2"
version = "0.1.29"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"

[[deps.AxisAlgorithms]]
deps = ["LinearAlgebra", "Random", "SparseArrays", "WoodburyMatrices"]
git-tree-sha1 = "66771c8d21c8ff5e3a93379480a2307ac36863f7"
uuid = "13072b0f-2c55-5437-9ae7-d433b7a33950"
version = "1.0.1"

[[deps.AxisArrays]]
deps = ["Dates", "IntervalSets", "IterTools", "RangeArrays"]
git-tree-sha1 = "16351be62963a67ac4083f748fdb3cca58bfd52f"
uuid = "39de3d68-74b9-583c-8d2d-e117c070f3a9"
version = "0.4.7"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"

[[deps.BitFlags]]
git-tree-sha1 = "43b1a4a8f797c1cddadf60499a8a077d4af2cd2d"
uuid = "d1d4a3ce-64b1-5f1a-9ba4-7e7e69966f35"
version = "0.1.7"

[[deps.BitTwiddlingConvenienceFunctions]]
deps = ["Static"]
git-tree-sha1 = "0c5f81f47bbbcf4aea7b2959135713459170798b"
uuid = "62783981-4cbd-42fc-bca8-16325de8dc4b"
version = "0.1.5"

[[deps.Bzip2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "19a35467a82e236ff51bc17a3a44b69ef35185a2"
uuid = "6e34b625-4abd-537c-b88f-471c36dfa7a0"
version = "1.0.8+0"

[[deps.CEnum]]
git-tree-sha1 = "eb4cb44a499229b3b8426dcfb5dd85333951ff90"
uuid = "fa961155-64e5-5f13-b03f-caf6b980ea82"
version = "0.4.2"

[[deps.CPUSummary]]
deps = ["CpuId", "IfElse", "PrecompileTools", "Static"]
git-tree-sha1 = "89e0654ed8c7aebad6d5ad235d6242c2d737a928"
uuid = "2a0fbf3d-bb9c-48f3-b0a9-814d99fd7ab9"
version = "0.2.3"

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

[[deps.CatIndices]]
deps = ["CustomUnitRanges", "OffsetArrays"]
git-tree-sha1 = "a0f80a09780eed9b1d106a1bf62041c2efc995bc"
uuid = "aafaddc9-749c-510e-ac4f-586e18779b91"
version = "0.2.2"

[[deps.ChainRulesCore]]
deps = ["Compat", "LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "e30f2f4e20f7f186dc36529910beaedc60cfa644"
uuid = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
version = "1.16.0"

[[deps.CloseOpenIntervals]]
deps = ["Static", "StaticArrayInterface"]
git-tree-sha1 = "70232f82ffaab9dc52585e0dd043b5e0c6b714f1"
uuid = "fb6a15b2-703c-40df-9091-08a04967cfa9"
version = "0.1.12"

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

[[deps.ComputationalResources]]
git-tree-sha1 = "52cb3ec90e8a8bea0e62e275ba577ad0f74821f7"
uuid = "ed09eef8-17a6-5b46-8889-db040fac31e3"
version = "0.3.2"

[[deps.ConcurrentUtilities]]
deps = ["Serialization", "Sockets"]
git-tree-sha1 = "5372dbbf8f0bdb8c700db5367132925c0771ef7e"
uuid = "f0e56b4a-5159-44fe-b623-3e5288b988bb"
version = "2.2.1"

[[deps.Contour]]
git-tree-sha1 = "d05d9e7b7aedff4e5b51a029dced05cfb6125781"
uuid = "d38c429a-6771-53c6-b99e-75d170b6e991"
version = "0.6.2"

[[deps.CoordinateTransformations]]
deps = ["LinearAlgebra", "StaticArrays"]
git-tree-sha1 = "f9d7112bfff8a19a3a4ea4e03a8e6a91fe8456bf"
uuid = "150eb455-5306-5404-9cee-2592286d6298"
version = "0.6.3"

[[deps.CpuId]]
deps = ["Markdown"]
git-tree-sha1 = "fcbb72b032692610bfbdb15018ac16a36cf2e406"
uuid = "adafc99b-e345-5852-983c-f28acb93d879"
version = "0.3.1"

[[deps.CustomUnitRanges]]
git-tree-sha1 = "1a3f97f907e6dd8983b744d2642651bb162a3f7a"
uuid = "dc8bdbbb-1ca9-579f-8c36-e416f6a65cce"
version = "1.0.2"

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

[[deps.FFTViews]]
deps = ["CustomUnitRanges", "FFTW"]
git-tree-sha1 = "cbdf14d1e8c7c8aacbe8b19862e0179fd08321c2"
uuid = "4f61f5a4-77b1-5117-aa51-3ab5ef4ef0cd"
version = "0.3.2"

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

[[deps.FileIO]]
deps = ["Pkg", "Requires", "UUIDs"]
git-tree-sha1 = "299dc33549f68299137e51e6d49a13b5b1da9673"
uuid = "5789e2e9-d7fb-5bc7-8068-2c6fae9b9549"
version = "1.16.1"

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
git-tree-sha1 = "8e2d86e06ceb4580110d9e716be26658effc5bfd"
uuid = "28b8d3ca-fb5f-59d9-8090-bfdbd6d07a71"
version = "0.72.8"

[[deps.GR_jll]]
deps = ["Artifacts", "Bzip2_jll", "Cairo_jll", "FFMPEG_jll", "Fontconfig_jll", "GLFW_jll", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Libtiff_jll", "Pixman_jll", "Qt5Base_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "da121cbdc95b065da07fbb93638367737969693f"
uuid = "d2c73de3-f751-5644-a686-071e5b155ba9"
version = "0.72.8+0"

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

[[deps.Graphics]]
deps = ["Colors", "LinearAlgebra", "NaNMath"]
git-tree-sha1 = "d61890399bc535850c4bf08e4e0d3a7ad0f21cbd"
uuid = "a2bd30eb-e257-5431-a919-1863eab51364"
version = "1.1.2"

[[deps.Graphite2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "344bf40dcab1073aca04aa0df4fb092f920e4011"
uuid = "3b182d85-2403-5c21-9c21-1e1f0cc25472"
version = "1.3.14+0"

[[deps.Graphs]]
deps = ["ArnoldiMethod", "Compat", "DataStructures", "Distributed", "Inflate", "LinearAlgebra", "Random", "SharedArrays", "SimpleTraits", "SparseArrays", "Statistics"]
git-tree-sha1 = "1cf1d7dcb4bc32d7b4a5add4232db3750c27ecb4"
uuid = "86223c79-3864-5bf0-83f7-82e725a168b6"
version = "1.8.0"

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

[[deps.HistogramThresholding]]
deps = ["ImageBase", "LinearAlgebra", "MappedArrays"]
git-tree-sha1 = "7194dfbb2f8d945abdaf68fa9480a965d6661e69"
uuid = "2c695a8d-9458-5d45-9878-1b8a99cf7853"
version = "0.3.1"

[[deps.HostCPUFeatures]]
deps = ["BitTwiddlingConvenienceFunctions", "IfElse", "Libdl", "Static"]
git-tree-sha1 = "eb8fed28f4994600e29beef49744639d985a04b2"
uuid = "3e5b6fbb-0976-4d2c-9146-d79de83f2fb0"
version = "0.1.16"

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

[[deps.IfElse]]
git-tree-sha1 = "debdd00ffef04665ccbb3e150747a77560e8fad1"
uuid = "615f187c-cbe4-4ef1-ba3b-2fcf58d6d173"
version = "0.1.1"

[[deps.ImageAxes]]
deps = ["AxisArrays", "ImageBase", "ImageCore", "Reexport", "SimpleTraits"]
git-tree-sha1 = "2e4520d67b0cef90865b3ef727594d2a58e0e1f8"
uuid = "2803e5a7-5153-5ecf-9a86-9b4c37f5f5ac"
version = "0.6.11"

[[deps.ImageBase]]
deps = ["ImageCore", "Reexport"]
git-tree-sha1 = "eb49b82c172811fd2c86759fa0553a2221feb909"
uuid = "c817782e-172a-44cc-b673-b171935fbb9e"
version = "0.1.7"

[[deps.ImageBinarization]]
deps = ["HistogramThresholding", "ImageCore", "LinearAlgebra", "Polynomials", "Reexport", "Statistics"]
git-tree-sha1 = "f5356e7203c4a9954962e3757c08033f2efe578a"
uuid = "cbc4b850-ae4b-5111-9e64-df94c024a13d"
version = "0.3.0"

[[deps.ImageContrastAdjustment]]
deps = ["ImageBase", "ImageCore", "ImageTransformations", "Parameters"]
git-tree-sha1 = "eb3d4365a10e3f3ecb3b115e9d12db131d28a386"
uuid = "f332f351-ec65-5f6a-b3d1-319c6670881a"
version = "0.3.12"

[[deps.ImageCore]]
deps = ["AbstractFFTs", "ColorVectorSpace", "Colors", "FixedPointNumbers", "MappedArrays", "MosaicViews", "OffsetArrays", "PaddedViews", "PrecompileTools", "Reexport"]
git-tree-sha1 = "fc5d1d3443a124fde6e92d0260cd9e064eba69f8"
uuid = "a09fc81d-aa75-5fe9-8630-4744c3626534"
version = "0.10.1"

[[deps.ImageCorners]]
deps = ["ImageCore", "ImageFiltering", "PrecompileTools", "StaticArrays", "StatsBase"]
git-tree-sha1 = "24c52de051293745a9bad7d73497708954562b79"
uuid = "89d5987c-236e-4e32-acd0-25bd6bd87b70"
version = "0.1.3"

[[deps.ImageDistances]]
deps = ["Distances", "ImageCore", "ImageMorphology", "LinearAlgebra", "Statistics"]
git-tree-sha1 = "08b0e6354b21ef5dd5e49026028e41831401aca8"
uuid = "51556ac3-7006-55f5-8cb3-34580c88182d"
version = "0.2.17"

[[deps.ImageFiltering]]
deps = ["CatIndices", "ComputationalResources", "DataStructures", "FFTViews", "FFTW", "ImageBase", "ImageCore", "LinearAlgebra", "OffsetArrays", "PrecompileTools", "Reexport", "SparseArrays", "StaticArrays", "Statistics", "TiledIteration"]
git-tree-sha1 = "432ae2b430a18c58eb7eca9ef8d0f2db90bc749c"
uuid = "6a3955dd-da59-5b1f-98d4-e7296123deb5"
version = "0.7.8"

[[deps.ImageIO]]
deps = ["FileIO", "IndirectArrays", "JpegTurbo", "LazyModules", "Netpbm", "OpenEXR", "PNGFiles", "QOI", "Sixel", "TiffImages", "UUIDs"]
git-tree-sha1 = "bca20b2f5d00c4fbc192c3212da8fa79f4688009"
uuid = "82e4d734-157c-48bb-816b-45c225c6df19"
version = "0.6.7"

[[deps.ImageMagick]]
deps = ["FileIO", "ImageCore", "ImageMagick_jll", "InteractiveUtils"]
git-tree-sha1 = "b0b765ff0b4c3ee20ce6740d843be8dfce48487c"
uuid = "6218d12a-5da1-5696-b52f-db25d2ecc6d1"
version = "1.3.0"

[[deps.ImageMagick_jll]]
deps = ["JpegTurbo_jll", "Libdl", "Libtiff_jll", "Pkg", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "1c0a2295cca535fabaf2029062912591e9b61987"
uuid = "c73af94c-d91f-53ed-93a7-00f77d67a9d7"
version = "6.9.10-12+3"

[[deps.ImageMetadata]]
deps = ["AxisArrays", "ImageAxes", "ImageBase", "ImageCore"]
git-tree-sha1 = "355e2b974f2e3212a75dfb60519de21361ad3cb7"
uuid = "bc367c6b-8a6b-528e-b4bd-a4b897500b49"
version = "0.9.9"

[[deps.ImageMorphology]]
deps = ["DataStructures", "ImageCore", "LinearAlgebra", "LoopVectorization", "OffsetArrays", "Requires", "TiledIteration"]
git-tree-sha1 = "6f0a801136cb9c229aebea0df296cdcd471dbcd1"
uuid = "787d08f9-d448-5407-9aad-5290dd7ab264"
version = "0.4.5"

[[deps.ImageQualityIndexes]]
deps = ["ImageContrastAdjustment", "ImageCore", "ImageDistances", "ImageFiltering", "LazyModules", "OffsetArrays", "PrecompileTools", "Statistics"]
git-tree-sha1 = "783b70725ed326340adf225be4889906c96b8fd1"
uuid = "2996bd0c-7a13-11e9-2da2-2f5ce47296a9"
version = "0.3.7"

[[deps.ImageSegmentation]]
deps = ["Clustering", "DataStructures", "Distances", "Graphs", "ImageCore", "ImageFiltering", "ImageMorphology", "LinearAlgebra", "MetaGraphs", "RegionTrees", "SimpleWeightedGraphs", "StaticArrays", "Statistics"]
git-tree-sha1 = "3ff0ca203501c3eedde3c6fa7fd76b703c336b5f"
uuid = "80713f31-8817-5129-9cf8-209ff8fb23e1"
version = "1.8.2"

[[deps.ImageShow]]
deps = ["Base64", "ColorSchemes", "FileIO", "ImageBase", "ImageCore", "OffsetArrays", "StackViews"]
git-tree-sha1 = "3b5344bcdbdc11ad58f3b1956709b5b9345355de"
uuid = "4e3cecfd-b093-5904-9786-8bbb286a6a31"
version = "0.3.8"

[[deps.ImageTransformations]]
deps = ["AxisAlgorithms", "CoordinateTransformations", "ImageBase", "ImageCore", "Interpolations", "OffsetArrays", "Rotations", "StaticArrays"]
git-tree-sha1 = "7ec124670cbce8f9f0267ba703396960337e54b5"
uuid = "02fcd773-0e25-5acc-982a-7f6622650795"
version = "0.10.0"

[[deps.Images]]
deps = ["Base64", "FileIO", "Graphics", "ImageAxes", "ImageBase", "ImageBinarization", "ImageContrastAdjustment", "ImageCore", "ImageCorners", "ImageDistances", "ImageFiltering", "ImageIO", "ImageMagick", "ImageMetadata", "ImageMorphology", "ImageQualityIndexes", "ImageSegmentation", "ImageShow", "ImageTransformations", "IndirectArrays", "IntegralArrays", "Random", "Reexport", "SparseArrays", "StaticArrays", "Statistics", "StatsBase", "TiledIteration"]
git-tree-sha1 = "d438268ed7a665f8322572be0dabda83634d5f45"
uuid = "916415d5-f1e6-5110-898d-aaa5f9f070e0"
version = "0.26.0"

[[deps.Imath_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "3d09a9f60edf77f8a4d99f9e015e8fbf9989605d"
uuid = "905a6f67-0a94-5f89-b386-d35d92009cd1"
version = "3.1.7+0"

[[deps.IndirectArrays]]
git-tree-sha1 = "012e604e1c7458645cb8b436f8fba789a51b257f"
uuid = "9b13fd28-a010-5f03-acff-a1bbcff69959"
version = "1.0.0"

[[deps.Inflate]]
git-tree-sha1 = "5cd07aab533df5170988219191dfad0519391428"
uuid = "d25df0c9-e2be-5dd7-82c8-3ad0b3e990b9"
version = "0.1.3"

[[deps.IntegralArrays]]
deps = ["ColorTypes", "FixedPointNumbers", "IntervalSets"]
git-tree-sha1 = "be8e690c3973443bec584db3346ddc904d4884eb"
uuid = "1d092043-8f09-5a30-832f-7509e371ab51"
version = "0.1.5"

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

[[deps.IntervalSets]]
deps = ["Dates", "Random"]
git-tree-sha1 = "8e59ea773deee525c99a8018409f64f19fb719e6"
uuid = "8197267c-284f-5f27-9208-e0e47529a953"
version = "0.7.7"
weakdeps = ["Statistics"]

    [deps.IntervalSets.extensions]
    IntervalSetsStatisticsExt = "Statistics"

[[deps.IrrationalConstants]]
git-tree-sha1 = "630b497eafcc20001bba38a4651b327dcfc491d2"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.2.2"

[[deps.IterTools]]
git-tree-sha1 = "4ced6667f9974fc5c5943fa5e2ef1ca43ea9e450"
uuid = "c8e1da08-722c-5040-9ed9-7db0dc04731e"
version = "1.8.0"

[[deps.IteratorInterfaceExtensions]]
git-tree-sha1 = "a3f24677c21f5bbe9d2a714f95dcd58337fb2856"
uuid = "82899510-4779-5014-852e-03e436cf321d"
version = "1.0.0"

[[deps.JLD2]]
deps = ["FileIO", "MacroTools", "Mmap", "OrderedCollections", "Pkg", "Printf", "Reexport", "Requires", "TranscodingStreams", "UUIDs"]
git-tree-sha1 = "aa6ffef1fd85657f4999030c52eaeec22a279738"
uuid = "033835bb-8acc-5ee8-8aae-3f567f8a3819"
version = "0.4.33"

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

[[deps.JpegTurbo]]
deps = ["CEnum", "FileIO", "ImageCore", "JpegTurbo_jll", "TOML"]
git-tree-sha1 = "327713faef2a3e5c80f96bf38d1fa26f7a6ae29e"
uuid = "b835a17e-a41a-41e7-81f0-2f016b05efe0"
version = "0.1.3"

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

[[deps.LayoutPointers]]
deps = ["ArrayInterface", "LinearAlgebra", "ManualMemory", "SIMDTypes", "Static", "StaticArrayInterface"]
git-tree-sha1 = "88b8f66b604da079a627b6fb2860d3704a6729a1"
uuid = "10f19ff3-798f-405d-979b-55457f8fc047"
version = "0.1.14"

[[deps.LazyArtifacts]]
deps = ["Artifacts", "Pkg"]
uuid = "4af54fe1-eca0-43a8-85a7-787d91b784e3"

[[deps.LazyModules]]
git-tree-sha1 = "a560dd966b386ac9ae60bdd3a3d3a326062d3c3e"
uuid = "8cdb02fc-e678-4876-92c5-9defec4f444e"
version = "0.3.1"

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

[[deps.LoopVectorization]]
deps = ["ArrayInterface", "ArrayInterfaceCore", "CPUSummary", "CloseOpenIntervals", "DocStringExtensions", "HostCPUFeatures", "IfElse", "LayoutPointers", "LinearAlgebra", "OffsetArrays", "PolyesterWeave", "PrecompileTools", "SIMDTypes", "SLEEFPirates", "Static", "StaticArrayInterface", "ThreadingUtilities", "UnPack", "VectorizationBase"]
git-tree-sha1 = "c88a4afe1703d731b1c4fdf4e3c7e77e3b176ea2"
uuid = "bdcacae8-1622-11e9-2a5c-532679323890"
version = "0.12.165"

    [deps.LoopVectorization.extensions]
    ForwardDiffExt = ["ChainRulesCore", "ForwardDiff"]
    SpecialFunctionsExt = "SpecialFunctions"

    [deps.LoopVectorization.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210"
    SpecialFunctions = "276daf66-3868-5448-9aa4-cd146d93841b"

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

[[deps.ManualMemory]]
git-tree-sha1 = "bcaef4fc7a0cfe2cba636d84cda54b5e4e4ca3cd"
uuid = "d125e4d3-2237-4719-b19c-fa641b8a4667"
version = "0.1.8"

[[deps.MappedArrays]]
git-tree-sha1 = "2dab0221fe2b0f2cb6754eaa743cc266339f527e"
uuid = "dbb5928d-eab1-5f90-85c2-b9b0edb7c900"
version = "0.4.2"

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

[[deps.MetaGraphs]]
deps = ["Graphs", "JLD2", "Random"]
git-tree-sha1 = "1130dbe1d5276cb656f6e1094ce97466ed700e5a"
uuid = "626554b9-1ddb-594c-aa3c-2596fe9399a5"
version = "0.7.2"

[[deps.Missings]]
deps = ["DataAPI"]
git-tree-sha1 = "f66bdc5de519e8f8ae43bdc598782d35a25b1272"
uuid = "e1d29d7a-bbdc-5cf2-9ac0-f12de2c33e28"
version = "1.1.0"

[[deps.Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"

[[deps.MosaicViews]]
deps = ["MappedArrays", "OffsetArrays", "PaddedViews", "StackViews"]
git-tree-sha1 = "7b86a5d4d70a9f5cdf2dacb3cbe6d251d1a61dbe"
uuid = "e94cdb99-869f-56ef-bcf0-1ae2bcbe0389"
version = "0.3.4"

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

[[deps.Netpbm]]
deps = ["FileIO", "ImageCore", "ImageMetadata"]
git-tree-sha1 = "d92b107dbb887293622df7697a2223f9f8176fcd"
uuid = "f09324ee-3d7c-5217-9330-fc30815ba969"
version = "1.1.1"

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

[[deps.OpenEXR]]
deps = ["Colors", "FileIO", "OpenEXR_jll"]
git-tree-sha1 = "327f53360fdb54df7ecd01e96ef1983536d1e633"
uuid = "52e1d378-f018-4a11-a4be-720524705ac7"
version = "0.3.2"

[[deps.OpenEXR_jll]]
deps = ["Artifacts", "Imath_jll", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "a4ca623df1ae99d09bc9868b008262d0c0ac1e4f"
uuid = "18a262bb-aa17-5467-a713-aee519bc75cb"
version = "3.1.4+0"

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

[[deps.PNGFiles]]
deps = ["Base64", "CEnum", "ImageCore", "IndirectArrays", "OffsetArrays", "libpng_jll"]
git-tree-sha1 = "9b02b27ac477cad98114584ff964e3052f656a0f"
uuid = "f57f5aa1-a3ce-4bc8-8ab9-96f992907883"
version = "0.4.0"

[[deps.PaddedViews]]
deps = ["OffsetArrays"]
git-tree-sha1 = "0fac6313486baae819364c52b4f483450a9d793f"
uuid = "5432bcbf-9aad-5242-b902-cca2824c8663"
version = "0.5.12"

[[deps.Parameters]]
deps = ["OrderedCollections", "UnPack"]
git-tree-sha1 = "34c0e9ad262e5f7fc75b10a9952ca7692cfc5fbe"
uuid = "d96e819e-fc66-5662-9728-84c9c7592b0a"
version = "0.12.3"

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

[[deps.PkgVersion]]
deps = ["Pkg"]
git-tree-sha1 = "f9501cc0430a26bc3d156ae1b5b0c1b47af4d6da"
uuid = "eebad327-c553-4316-9ea0-9fa01ccd7688"
version = "0.3.3"

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

[[deps.PolyesterWeave]]
deps = ["BitTwiddlingConvenienceFunctions", "CPUSummary", "IfElse", "Static", "ThreadingUtilities"]
git-tree-sha1 = "240d7170f5ffdb285f9427b92333c3463bf65bf6"
uuid = "1d0040c9-8b98-4ee7-8388-3f51789ca0ad"
version = "0.2.1"

[[deps.Polynomials]]
deps = ["LinearAlgebra", "RecipesBase"]
git-tree-sha1 = "3aa2bb4982e575acd7583f01531f241af077b163"
uuid = "f27b6e38-b328-58d1-80ce-0feddd5e7a45"
version = "3.2.13"

    [deps.Polynomials.extensions]
    PolynomialsChainRulesCoreExt = "ChainRulesCore"
    PolynomialsMakieCoreExt = "MakieCore"
    PolynomialsMutableArithmeticsExt = "MutableArithmetics"

    [deps.Polynomials.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    MakieCore = "20f20a25-4f0e-4fdf-b5d1-57303727442b"
    MutableArithmetics = "d8a4904e-b15c-11e9-3269-09a3773c0cb0"

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

[[deps.ProgressMeter]]
deps = ["Distributed", "Printf"]
git-tree-sha1 = "ae36206463b2395804f2787ffe172f44452b538d"
uuid = "92933f4c-e287-5a05-a399-4b506db050ca"
version = "1.8.0"

[[deps.QOI]]
deps = ["ColorTypes", "FileIO", "FixedPointNumbers"]
git-tree-sha1 = "18e8f4d1426e965c7b532ddd260599e1510d26ce"
uuid = "4b34888f-f399-49d4-9bb3-47ed5cae4e65"
version = "1.0.0"

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

[[deps.Quaternions]]
deps = ["LinearAlgebra", "Random", "RealDot"]
git-tree-sha1 = "da095158bdc8eaccb7890f9884048555ab771019"
uuid = "94ee1d12-ae83-5a48-8b1c-48b8ff168ae0"
version = "0.7.4"

[[deps.REPL]]
deps = ["InteractiveUtils", "Markdown", "Sockets", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"

[[deps.Random]]
deps = ["SHA", "Serialization"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[[deps.RangeArrays]]
git-tree-sha1 = "b9039e93773ddcfc828f12aadf7115b4b4d225f5"
uuid = "b3c3ace0-ae52-54e7-9d0b-2c1406fd6b9d"
version = "0.3.2"

[[deps.Ratios]]
deps = ["Requires"]
git-tree-sha1 = "1342a47bf3260ee108163042310d26f2be5ec90b"
uuid = "c84ed2f1-dad5-54f0-aa8e-dbefe2724439"
version = "0.4.5"
weakdeps = ["FixedPointNumbers"]

    [deps.Ratios.extensions]
    RatiosFixedPointNumbersExt = "FixedPointNumbers"

[[deps.RealDot]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "9f0a1b71baaf7650f4fa8a1d168c7fb6ee41f0c9"
uuid = "c1ae055f-0cd5-4b69-90a6-9a35b1a98df9"
version = "0.1.0"

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

[[deps.RegionTrees]]
deps = ["IterTools", "LinearAlgebra", "StaticArrays"]
git-tree-sha1 = "4618ed0da7a251c7f92e869ae1a19c74a7d2a7f9"
uuid = "dee08c22-ab7f-5625-9660-a9af2021b33f"
version = "0.3.2"

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

[[deps.Rotations]]
deps = ["LinearAlgebra", "Quaternions", "Random", "StaticArrays"]
git-tree-sha1 = "54ccb4dbab4b1f69beb255a2c0ca5f65a9c82f08"
uuid = "6038ab10-8711-5258-84ad-4b1120ba62dc"
version = "1.5.1"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"
version = "0.7.0"

[[deps.SIMDTypes]]
git-tree-sha1 = "330289636fb8107c5f32088d2741e9fd7a061a5c"
uuid = "94e857df-77ce-4151-89e5-788b33177be4"
version = "0.1.0"

[[deps.SLEEFPirates]]
deps = ["IfElse", "Static", "VectorizationBase"]
git-tree-sha1 = "4b8586aece42bee682399c4c4aee95446aa5cd19"
uuid = "476501e8-09a2-5ece-8869-fb82de89a1fa"
version = "0.6.39"

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

[[deps.SimpleTraits]]
deps = ["InteractiveUtils", "MacroTools"]
git-tree-sha1 = "5d7e3f4e11935503d3ecaf7186eac40602e7d231"
uuid = "699a6c99-e7fa-54fc-8d76-47d257e15c1d"
version = "0.9.4"

[[deps.SimpleWeightedGraphs]]
deps = ["Graphs", "LinearAlgebra", "Markdown", "SparseArrays"]
git-tree-sha1 = "4b33e0e081a825dbfaf314decf58fa47e53d6acb"
uuid = "47aef6b3-ad0c-573a-a1e2-d07658019622"
version = "1.4.0"

[[deps.Sixel]]
deps = ["Dates", "FileIO", "ImageCore", "IndirectArrays", "OffsetArrays", "REPL", "libsixel_jll"]
git-tree-sha1 = "2da10356e31327c7096832eb9cd86307a50b1eb6"
uuid = "45858cf5-a6b0-47a3-bbea-62219f50df47"
version = "0.1.3"

[[deps.SnoopPrecompile]]
deps = ["Preferences"]
git-tree-sha1 = "e760a70afdcd461cf01a575947738d359234665c"
uuid = "66db9d55-30c0-4569-8b51-7e840670fc0c"
version = "1.0.3"

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

[[deps.StackViews]]
deps = ["OffsetArrays"]
git-tree-sha1 = "46e589465204cd0c08b4bd97385e4fa79a0c770c"
uuid = "cae243ae-269e-4f55-b966-ac2d0dc13c15"
version = "0.1.1"

[[deps.Static]]
deps = ["IfElse"]
git-tree-sha1 = "f295e0a1da4ca425659c57441bcb59abb035a4bc"
uuid = "aedffcd0-7271-4cad-89d0-dc628f76c6d3"
version = "0.8.8"

[[deps.StaticArrayInterface]]
deps = ["ArrayInterface", "Compat", "IfElse", "LinearAlgebra", "PrecompileTools", "Requires", "SparseArrays", "Static", "SuiteSparse"]
git-tree-sha1 = "03fec6800a986d191f64f5c0996b59ed526eda25"
uuid = "0d7ed370-da01-4f52-bd93-41d350b8b718"
version = "1.4.1"
weakdeps = ["OffsetArrays", "StaticArrays"]

    [deps.StaticArrayInterface.extensions]
    StaticArrayInterfaceOffsetArraysExt = "OffsetArrays"
    StaticArrayInterfaceStaticArraysExt = "StaticArrays"

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

[[deps.ThreadingUtilities]]
deps = ["ManualMemory"]
git-tree-sha1 = "eda08f7e9818eb53661b3deb74e3159460dfbc27"
uuid = "8290d209-cae3-49c0-8002-c8c24d57dab5"
version = "0.5.2"

[[deps.TiffImages]]
deps = ["ColorTypes", "DataStructures", "DocStringExtensions", "FileIO", "FixedPointNumbers", "IndirectArrays", "Inflate", "Mmap", "OffsetArrays", "PkgVersion", "ProgressMeter", "UUIDs"]
git-tree-sha1 = "8621f5c499a8aa4aa970b1ae381aae0ef1576966"
uuid = "731e570b-9d59-4bfa-96dc-6df516fadf69"
version = "0.6.4"

[[deps.TiledIteration]]
deps = ["OffsetArrays", "StaticArrayInterface"]
git-tree-sha1 = "1176cc31e867217b06928e2f140c90bd1bc88283"
uuid = "06e1c1a7-607b-532d-9fad-de7d9aa2abac"
version = "0.5.0"

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

[[deps.UnPack]]
git-tree-sha1 = "387c1f73762231e86e0c9c5443ce3b4a0a9a0c2b"
uuid = "3a884ed6-31ef-47d7-9d2a-63182c4928ed"
version = "1.0.2"

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

[[deps.VectorizationBase]]
deps = ["ArrayInterface", "CPUSummary", "HostCPUFeatures", "IfElse", "LayoutPointers", "Libdl", "LinearAlgebra", "SIMDTypes", "Static", "StaticArrayInterface"]
git-tree-sha1 = "b182207d4af54ac64cbc71797765068fdeff475d"
uuid = "3d5dd08c-fd9d-11e8-17fa-ed2836048c2f"
version = "0.21.64"

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

[[deps.libsixel_jll]]
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Pkg", "libpng_jll"]
git-tree-sha1 = "d4f63314c8aa1e48cd22aa0c17ed76cd1ae48c3c"
uuid = "075b6546-f08a-558a-be8f-8157d0f608a5"
version = "1.10.3+0"

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
# ╟─50752620-a604-442c-bf92-992963b1dd7a
# ╟─3e2e1ea8-3a7d-462f-ac38-43a087907a14
# ╟─7bbf37e1-27fd-4871-bc1d-c9c3ecaac076
# ╟─bc96a33d-9011-41ec-a19e-d472cbaafb70
# ╟─ff61cd9d-a193-44b3-a715-3c372ade7f79
# ╟─b89ac105-597e-44ac-9b58-c1c3c5ac59e9
# ╟─40f334b3-60fa-4ca1-8146-06e00fa14d45
# ╟─bb02b467-0269-4047-8c0b-ee61e0185df8
# ╟─d9829c67-d346-4145-ae75-028f9fdd103d
# ╟─72af797b-5340-482e-be00-2cda375dd734
# ╟─723365e7-1fad-4899-8ac1-fb8674e2b9a7
# ╟─ff303022-1eff-4278-be92-edf46c747bec
# ╟─a862e9d6-c31d-4b21-80c0-e359a5435b6b
# ╟─6dba8be7-b62b-42d4-835a-d5ed827befa9
# ╟─4e55339d-4b61-4584-81c8-a9f35312830d
# ╟─3f189707-d21d-4bf6-bc86-35304206281a
# ╟─66a08217-481a-4956-ba8a-c4c822f3d0d2
# ╟─b02d6493-8da9-4d4d-a090-353f96addbdd
# ╟─b88ce59d-a715-48af-b7f6-cc867ef2b50e
# ╟─dc64c064-6cee-4fba-8a36-e9b3264ebbbf
# ╟─2d72550f-6dd4-496b-a763-125ec7f9d251
# ╟─848c79ff-61b3-47e1-a574-4549afe9900f
# ╟─7d625061-fdef-440b-b541-0dd141a50c95
# ╟─aecdb616-d38b-4db6-af4c-55c988a3139d
# ╟─82c78222-131f-4a38-a014-00cfee6d09ed
# ╟─3864434a-0f66-4872-81c6-84353d870f50
# ╟─a0a1eb6a-74e1-4d1e-8097-b7b3c04cffa2
# ╟─85a0636b-5475-4ece-a03d-cbbc4603cbc5
# ╟─4ce3cb20-e70d-4d39-8572-ee7ac6d8e441
# ╟─b0f0628e-ceff-4e6c-819b-2778ca4d4b08
# ╟─8c72fd4f-d783-48bb-8f62-c00e16399e2a
# ╟─ae216b5f-59bb-4d0e-ab58-ddea82d5ee2f
# ╟─c1ac8263-f226-4ae9-b0de-ab4eacb4ff42
# ╟─c07428c0-1c9d-41ba-9702-e3dffe2e0159
# ╟─bfedf482-2ef2-4d3d-93d1-796ae61ef48d
# ╟─a08f53bd-91d3-45b2-ab73-42e9b4bae563
# ╟─e69bbba2-3167-4edd-b9db-6aba7a0dceed
# ╟─d2085d25-1867-4037-92e0-8cbd2d1ada39
# ╟─bec564e6-fea8-4645-8f25-8babbe0b13df
# ╟─55b63786-4e04-497a-abae-e6e229a7edf2
# ╟─0d38b7f0-39e7-44db-8343-3c80f32a5c30
# ╟─434a9163-bddb-4702-b97b-b193a6f08b77
# ╟─001496ae-65ca-4de1-a211-4cbbafc7777f
# ╟─1663b630-a57a-44c5-90c6-95be2ac42aa8
# ╟─5acb28a1-49f1-412d-9c6e-b840cab687aa
# ╟─2c152ff3-050f-42de-b434-21b2658d572f
# ╟─da1fd88b-0e2a-49e6-8fb5-135a9d18be6e
# ╟─bda9fe31-88ae-4ce3-8038-9ca7f0a24378
# ╟─b20f121e-781a-4d17-8833-43d34396bf44
# ╟─1fb9b4c4-e7c7-4ceb-9193-3c1e83244445
# ╟─dd2fc1a7-4684-4956-aff3-25b23e1548e0
# ╟─0f7e65e5-5eea-4e4d-8ea4-0f04ce52f91d
# ╟─492c3edd-772c-4d07-88bc-92a046190f29
# ╟─295644e3-b8ee-4d06-a17d-9a37b5f965ce
# ╟─1d0a3487-363e-4353-904a-73e2718748af
# ╟─2d94f5ba-9a3b-4615-b451-3ef3b393057d
# ╟─f870fc82-56d4-4a39-991e-d42bc31cd9c8
# ╟─0734ddb1-a9a0-4fe1-b5ee-9a839a33d1dc
# ╟─9fc9a421-194e-4163-87f2-30989354932a
# ╟─8687dbd1-4857-40e4-b9cb-af469b8563e2
# ╟─fab7a0dd-3a9e-463e-a66b-432a6b2d8a1b
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
