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

# ╔═╡ 9ca39f56-5eea-11ed-03e1-81eb0fa2a37d
begin
	using PlutoTeachingTools
	using PlutoUI
	# using Plots
	using LinearAlgebra
	# using StatsPlots
	using LogExpFunctions
	using StatsBase
	using LaTeXStrings
	using Latexify
	using Random
	using HypertextLiteral
	using Plots; default(fontfamily="Computer Modern", framestyle=:box) # LaTex-style
end

# ╔═╡ dcacb71b-bf45-4b36-a2d2-477839f52411
using Logging

# ╔═╡ f9736ecc-053f-444a-9ef5-cdbe85795fce
figure_url = "https://leo.host.cs.st-andrews.ac.uk/figs/";

# ╔═╡ 8d096119-f6db-4f62-a091-6f00372468ec
function show_img(path_to_file; center=true, h = 400, w = nothing)
	if center
		if isnothing(w)
			@htl """<center><img src= $(figure_url * path_to_file) height = '$(h)' /></center>"""
		else
			@htl """<center><img src= $(figure_url * path_to_file) width = '$(w)' /></center>"""
		end

	else
		if isnothing(w)
			@htl """<img src= $(figure_url * path_to_file) height = '$(h)' />"""
		else
			@htl """<img src= $(figure_url * path_to_file) width = '$(w)' />"""
		end
	end
end;

# ╔═╡ d7a55322-0d9f-44e8-a2c6-4f0cead25f9d
Logging.disable_logging(Logging.Info) ; # or e.g. Logging.Info

# ╔═╡ 5bb178a2-f119-405f-a65b-ec6d59a112e0
TableOfContents()

# ╔═╡ 799ead95-2f68-4bcb-aae8-dd8b0b00560c
ChooseDisplayMode()

# ╔═╡ afb99b70-a418-4379-969b-355fbcfe8f14
md"""

# CS5914 Machine Learning Algorithms


#### Linear regression 3
##### Gradient descent

\

$(Resource("https://www.st-andrews.ac.uk/assets/university/brand/logos/standard-vertical-black.png", :width=>130, :align=>"right"))

Lei Fang(@lf28 $(Resource("https://raw.githubusercontent.com/edent/SuperTinyIcons/bed6907f8e4f5cb5bb21299b9070f4d7c51098c0/images/svg/github.svg", :width=>10)))

*School of Computer Science*

*University of St Andrews, UK*

"""

# ╔═╡ ee92f14a-784a-407c-8955-9ea1dd9cf942
md"""

## Notations


Super-script index with brackets ``.^{(i)}``: ``i \in \{1,2,\ldots, n\}`` index for observations/data
* ``n`` total number of observations
* *e.g.* ``y^{(i)}`` the i-th observation's label
* ``\mathbf{x}^{(i)}`` the i-th observation's predictor vector

Sub-script index ``j \in \{1,2,,\ldots, m\} ``
* ``m`` total number of features
* *e.g.* ``\mathbf{x}^{(i)}_2``: the second element/feature of ``i``-th observation


Vectors: **Bold-face** smaller case:
* ``\mathbf{x},\mathbf{y}, \boldsymbol{\beta}``
* ``\mathbf{x}^\top``: row vector

Matrices: **Bold-face** capital case: 
* ``\mathbf{X},\mathbf{A}, \boldsymbol{\Gamma}``  


Scalars: normal letters
* ``x,y,\beta,\gamma``

"""

# ╔═╡ c49a6965-727e-419b-b66e-57dc61415edf
md"""

# Gradient descent


"""

# ╔═╡ e83d5763-bc02-4d88-a8d0-f9fb38314ae8
aside((md"""$(@bind next1 Button("next")) 
$(@bind init1 Button("init"))
	"""))

# ╔═╡ 38a119e0-517f-492a-a675-0f318c6d8696
begin
	init1
	next_idx = [0];
end;

# ╔═╡ 828ceffb-1423-4e13-a92e-9d85f9f3f4b3
begin
	next1
	topics = ["recap gradient", "gradient descent: as an optimisation algorithm", "some caveats: local minimums and learning rate", "linear regression: case study"]
	@htl "<ul>$([@htl("""<li>$b</li><br>""") for b in topics[1:min(next_idx[1], length(topics))]])</ul>"
end

# ╔═╡ 768c34c9-d7a9-4bd9-abee-59a9a176ffb7
let
	next1
	next_idx[1] += 1
end;

# ╔═╡ 852be981-a741-42b2-b681-6b8fbc09a41c
md"""

## Motivation

Many machine learning problems reduce to an **optimisation** problem

```math
\large
\mathbf{w} \leftarrow \arg\min_{\mathbf{w}}\; \text{loss}(h_\mathbf{w}; \mathcal{D}_{train})
```

* _loss_ function: intuitively measures `the (negative) goodness` of a function 
  * parameterised by ``\mathbf{w}`` 
  * based on the training data ``\mathcal{D}_{train}``
"""

# ╔═╡ d02ec857-04aa-4128-a172-e49955bbb062
show_img("/mlgoodness.png", w=250)

# ╔═╡ f95f299a-3ad9-47f0-8b21-f7ee0e278cc1
md"""
##

**Example**: linear regression and least square

* sum of squared error (SSE)
```math
\large
\hat{\mathbf{w}} \leftarrow \arg\min_{\mathbf{w}} \frac{1}{2}\sum_{i=1}^n (y^{(i)} - \mathbf{w}^\top\mathbf{x}^{(i)})^2
```

* **mean squared error** (MSE)

```math
\large
\hat{\mathbf{w}} \leftarrow \arg\min_{\mathbf{w}} \frac{1}{2\colorbox{orange}{$n$}}\sum_{i=1}^n (y^{(i)} - \mathbf{w}^\top\mathbf{x}^{(i)})^2
```
"""

# ╔═╡ 520c22ab-7fa9-49ef-bb13-f50461f1a928
aside(tip(md"The two objectives differ by a constant scalar ``n``
* they lead to the same ``\hat{\mathbf{w}}``"))

# ╔═╡ 5921c8dc-25d6-4acb-9f60-1bcf8644ee12
md"""
##

To find the optimal ``\mathbf{w}``, we can simply solve 


```math
\large
\nabla l(\mathbf{w}) = \mathbf{0} \;\;\; \# \texttt{solve it for w}
```

For linear regression, the solution is the **normal equation**

```math
\large
\begin{align}
\hat{\mathbf{w}} = ({\mathbf{X}} ^\top{\mathbf{X}} )^{-1}{\mathbf{X}}^\top \mathbf{y}

\end{align}
```

* this is an rare example
* and also ``\mathbf{X}^\top\mathbf{X}`` can be expensive to compute and invert!

## But...

```math
\Large
	\boxed{\nabla l(\mathbf{w}) = \mathbf{0} \;\;\; \# \texttt{very hard to solve}}
``` **CAN NOT** be solved analytically for _most problems_


* linear regression is one of the rare exceptions
* **gradient descent algorithm** to rescue



"""

# ╔═╡ 39486178-559b-4b87-915e-b77475de02a5
md"""
## Recap: gradient


Given function $f(\mathbf{w}): \mathbb{R}^m\rightarrow \mathbb{R}$, 

**Gradient** is the collection of the partial derivatives:

$$\large \nabla f(\mathbf{w})=\text{grad} f(\mathbf{w}) = \begin{bmatrix}\frac{\partial f(\mathbf{w})}{\partial w_1}\\ \frac{\partial f(\mathbf{w})}{\partial w_2}\\ \vdots \\ \frac{\partial f(\mathbf{w})}{\partial w_m}\end{bmatrix}$$

- gradient is a vector-to-vector function! 
  - **input**:  a vector ``\mathbf{w} \in \mathbb{R}^m`` (interpreted as an input location) 
  - **output**: a vector ``\nabla f(\mathbf{w})`` (interpreted as a direction)


"""

# ╔═╡ f34c38d3-3ed1-4d59-8e23-6eb2f029de4f
md"""

## Recap: gradient direction


$$\large \nabla f(\mathbf{w})=\text{grad} f(\mathbf{w}) = \begin{bmatrix}\frac{\partial f(\mathbf{w})}{\partial w_1}\\ \frac{\partial f(\mathbf{w})}{\partial w_2}\\ \vdots \\ \frac{\partial f(\mathbf{w})}{\partial w_m}\end{bmatrix}$$


- gradient is a (vector to vector) function! 
  - **input**:  a vector ``\mathbf{w} \in \mathbb{R}^m`` (interpreted as an input location) 
  - **output**: a vector ``\nabla f(\mathbf{w})`` (interpreted as a direction)



!!! infor "Gradient direction"
	Gradient vector ``\nabla f(\mathbf{w})`` points to the _**greatest ascent direction**_ locally at the input location ``\mathbf{w}``


"""

# ╔═╡ 540564bf-74db-43c8-8c1f-8dc471dc8ec8
md"""

## Example: ``f(\mathbf{w}) = \mathbf{w}^\top \mathbf{w}``


For ``\mathbf{w} =[w_1, w_2]^\top \in \mathbb{R}^2``

```math
\large
f(\mathbf{w}) =\mathbf{w}^\top \mathbf{w} = w_1^2 + w_2^2
```

The gradient is 

$$\large \nabla f(\mathbf{w}) = \begin{bmatrix}\frac{\partial f(\mathbf{w})}{\partial w_1}\\ \frac{\partial f(\mathbf{w})}{\partial w_2}\end{bmatrix} = \begin{bmatrix} 2 w_1\\ 2w_2\end{bmatrix} = 2 \mathbf{w}$$


"""

# ╔═╡ 74977378-9c43-4638-8be9-cbf91c82c7ac
md"""

##

The gradient of ``f`` 

$$\large \nabla f(\mathbf{w}) = \begin{bmatrix}\frac{\partial f(\mathbf{w})}{\partial w_1}\\ \frac{\partial f(\mathbf{w})}{\partial w_2}\end{bmatrix} = \begin{bmatrix} 2 w_1\\ 2w_2\end{bmatrix} = 2 \mathbf{w}$$

"""

# ╔═╡ 35050b2d-4f25-4a96-88db-c8d193bb0a35
TwoColumn(let
	x0 = [0, 0]
	plotly()
	A = Matrix(I,2,2)
	μ = [0,0]
	f(x1, x2) = dot([x1, x2]- μ, A, [x1, x2]-μ)
	plot(μ[1]-5:1:μ[1]+5, μ[2]-5:1:μ[2]+5, f, st=:surface, xlabel="x₁", ylabel="x₂", zlabel="f",  alpha=0.5, color=:jet, framestyle=:zerolines, ratio=1, colorbar=false, xlim=[-6, 6], ylim = [-6, 6], size=(300,300))
	scatter!([x0[1]], [x0[2]], [0], ms=1, markershape=:cross, label="")
	scatter!([x0[1]], [x0[2]], [f(x0...)], ms=1, label="")
	vs = 0:0.1:f(x0...)
	plot!(2 * ones(length(vs)), 2 * ones(length(vs)), vs, lw=3, ls=:dash, lc=:gray, label="")

end, let
	gr()
	A = Matrix(I, 2, 2)
	f(x₁, x₂) = dot([x₁, x₂], A, [x₁, x₂])
	∇f(x₁, x₂) = 2 * A* [x₁, x₂] / 5
	xs = -20:0.5:20
	ys= -20:0.5:20
	cont = contour(xs, ys, (x, y)->f(x,y), c=:jet, xlabel=L"x_1", ylabel=L"x_2", framestyle=:origin, title="Gradient field plot", ratio=1, size=(300,300))
	# for better visualisation
	meshgrid(x, y) = (repeat(x, outer=length(y)), repeat(y, inner=length(x))) # helper function to create a quiver grid.
	xs_, ys_ = meshgrid(range(-15, 15, length=8), range(-15, 15, length=6))
	quiver!(xs_, ys_, quiver = ∇f, c=:green)
end)

# ╔═╡ 0d1dc5cf-36f1-4f28-befc-c96a02dc97a0
md"""

!!! note "Observation"
	The gradients "vanish" near the centre 
	* (smaller arrows: smaller scale) 
    * no surprise $$\nabla f(\mathbf{w}) =  \begin{bmatrix} 2 w_1\\ 2w_2\end{bmatrix}$$
      * the output vector gets smaller when ``\mathbf{w}\rightarrow \mathbf{0}``


"""

# ╔═╡ 0e74a142-70d1-4a36-bf7e-48411cc7c12b
md"""

## More example
"""

# ╔═╡ 7e21b526-da08-464b-8efc-f39f295a7836
md"""


```math
\large
f(w_1, w_2) = \frac{1}{4} (w_1^4 + w_2^4) -\frac{1}{3} (w_1^3 +w_2^3) - w_1^2 -w_2^2 +4
```
"""

# ╔═╡ 22e8e869-175e-45a3-83d0-652aa346a78f
aside(tip(md"Surface plot in Julia: 

`plot(xs, ys, (x,y) -> f(x,y), st=:surface)`


in Pyton, use 

`ax.plot_surface()` method
"))

# ╔═╡ 6c8ae8d2-a924-48f8-9ec7-beb0c9a7c08a
md"

##
The **contour plot** is shown below:
* there are four local minimums: ``[2,2], [2,-1],[-1,-1], [-1, 2]``
"

# ╔═╡ af78bc1d-4586-4c4e-a12f-c34b1ed9fbac
aside(tip(md"Contour plot in Julia: 

`plot(xs, ys, (x,y) -> f(x,y), st=:contour)`;


Python:

`matplotlib.pyplot.contour()`"))

# ╔═╡ 1437a46a-3882-4719-ad49-94d7d065ead0
md"""

## Let's derive the gradient



```math
\large
f(w_1, w_2) = \frac{1}{4} (w_1^4 + w_2^4) -\frac{1}{3} (w_1^3 +w_2^3) - w_1^2 -w_2^2 +4
```

!!! hint "Question: derive the gradient"
	```math
	\large
		\nabla f(\mathbf{w}) = \begin{bmatrix} \frac{\partial f}{\partial w_1} \\ \frac{\partial f}{\partial w_2}\end{bmatrix} =\begin{bmatrix} w_1^3 - w_1^2 - 2w_1 \\ w_2^3 - w_2^2 - 2w_2\end{bmatrix}
	```
"""

# ╔═╡ 9c3d80bb-20a4-4c81-8ce8-8d611cf24a6e
md"""
## Visualisation of the gradients
Let's view the gradients on top of contour plot
* note where the gradient vectors point to
* and also the length/scale of the gradients (where do they vanish?)

"""

# ╔═╡ b6fd9d29-b5ac-4940-90d8-fcf0d8063003
begin
	f_demo(w₁, w₂) = 1/4 * (w₁^4 + w₂^4) - 1/3 *(w₁^3 + w₂^3) - w₁^2 - w₂^2 + 4
	f_demo(w::Vector{T}) where T <: Real = f_demo(w...)
	∇f_demo(w₁, w₂) = [w₁^3 - w₁^2 - 2 * w₁, w₂^3 - w₂^2 - 2 * w₂]
	∇f_demo(w::Vector{T}) where T <: Real = ∇f_demo(w...)
end;

# ╔═╡ f477c82e-f4d6-4918-85d1-19ce21e90b5b
more_ex_surface = let
	gr()
	plot(-2:0.1:3, -2:0.1:3, f_demo, st=:surface, color=:jet, colorbar=false, aspect_ratio=1.0, xlabel=L"w_1", ylabel=L"w_2")
end;

# ╔═╡ b8156163-2a1f-48c6-85fd-9ecd15d1fb76
more_ex_surface

# ╔═╡ 22296f1a-3ed6-44ab-b9dd-47c58801afa6
let
	gr()
	plot(-2:0.05:3, -2:0.05:3, f_demo, st=:contour, color=:jet, colorbar=false, aspect_ratio=1.0, framestyle=:semi, xlim=[-2.1,3.1])
	# xs_, ys_ = meshgrid(range(-2, 3, length=10), range(-2, 3, length=10))
	# ∇f_d(x, y) = ∇f_demo(x, y)/20
	# quiver!(xs_, ys_, quiver = ∇f_d, c=:green)
end

# ╔═╡ a6bd512f-0c74-4fff-a894-f40526edc84b
# md"""
# ## More example

# """

# ╔═╡ 0c3f1ca9-d021-454a-a6db-eef0e92dc2bf
# let
# 	gr()
# 	plt = plot(-3.:0.05:1, -2.:0.05:1, f_demo_2, st=:contourf, color=:jet, alpha=.5, colorbar=true, xlim=[-3, 1], ylim=[-2, 1])
# 	xs = range(-3, 1, length = 20)
# 	ys = range(-2, 1, length = 20)
# 	α = 0.25
# 	xs_, ys_ = meshgrid(range(-3, 1, length=20), range(-2, 1, length=20))
# 	∇f_d(x, y) = ∇f_demo_2(x, y) * α
# 	quiver!(xs_, ys_, quiver = ∇f_d, c=:black)
# 	plt
# end

# ╔═╡ 241065bb-1733-497f-8797-8d29ec41546e
# md"""

# _**sanity check**_

# * gradients are always _perpendicular_ to level curves
# """

# ╔═╡ 2c16ffab-bb3b-44b1-9b64-5708085358e7
md"""

## Gradient descent

**Objective**: we want to minimise some function ``f(\mathbf{w})``


**Gradient descent**: an iterative optimisation algorithm to find the minimum

* **idea**: find the location ``\hat{\mathbf{w}}`` where the gradient ``\nabla f(\hat{\mathbf{w}})`` vanishes
* **how**: for current value of ``\mathbf{w}``, calculate gradient of ``f(\mathbf{w})``, then take small step in direction of negative gradient. Then **repeat**.



"""

# ╔═╡ 4c515d77-632c-4aa3-af4a-ab04abddf133
show_img("CS5914/gradient-descent-1.jpg", w=350)

# ╔═╡ 91c4a88a-dff9-491d-b213-3ed480ca304e
md"[source](https://web.stanford.edu/class/cs224n/slides/cs224n-2023-lecture01-wordvecs1.pdf)"

# ╔═╡ 20f57c9d-3543-4cb6-8844-30b30e3b08ec
md"""

## Gradient descent


```math
\LARGE
\mathbf{w}_{new} = \mathbf{w}_{old} - \underbrace{\colorbox{lightgreen}{$\gamma$}}_{{\color{green}\small \rm learning\; rate}} \nabla_{\mathbf{w}} f(\mathbf{w}_{old})
```


What gradient descent needs:
* ``\nabla f(\mathbf{w})``, the gradient function
* ``\colorbox{lightgreen}{$\gamma$}``, a step size called _learning rate_ *e.g.* 0.01


----


**Bach gradient descent algorithm:**



* random guess ``\large\mathbf{w}_0``

* while **not converge**
  * ``\large\mathbf{w}_t \leftarrow \mathbf{w}_{t-1} - \gamma \nabla f(\mathbf{w}_{t-1})``
-----



"""

# ╔═╡ f4a1f7db-0bcb-45b6-be9d-1c57dd6e2b99
function gradient_descent(f, ∇f; w_init= zeros(2), max_iters = 200, γ = 0.01)
	w₀ = w_init
	losses = Vector{Float64}(undef, max_iters+1)
	traces = zeros(length(w₀), max_iters+1)
	losses[1], traces[:, 1] = f(w₀), w₀
	for t in 1:max_iters
		# calculate the gradient at t
		∇w = ∇f(w₀)
		# follow a small gradient step
		w₀ = w₀ - γ * ∇w
		losses[t+1], traces[:, t+1] = f(w₀), w₀ # book keeping for visualisation
	end
	return losses, traces
end;

# ╔═╡ a5bfaa1e-9d40-4f2e-b0d5-73ab9bbceddf
md"""

## Maximisation: gradient _ascent_

To **maximise** a function?
* follow the positive gradient direction (rather than the negative direction)




```math
\LARGE
\mathbf{w}_{new} = \mathbf{w}_{old}\;  \colorbox{orange}{$+$} \;\gamma \nabla_{\mathbf{w}} f(\mathbf{w}_{old})
```



The two problems are exchangeable though

```math
\large \arg\max f(\mathbf{w}) \Leftrightarrow \arg\min \{-f(\mathbf{w})\}
```


* maximising a function is the same as minimising its negative



"""

# ╔═╡ 46dd39e0-95bf-4079-a982-4f93b16c3a5e
md"""

## Local optimum


!!! note ""
	Limitation: gradient descent **cannot** _backtrack_, it will only converge to a local optimum

	* depending on the starting location ``\mathbf{w}_0``

"""

# ╔═╡ 90ea67c0-6f15-4fdb-830d-c7629fae0fcd
let
	gr()
	plt=plot(-2:0.05:3, -2:0.05:3, f_demo, st=:contour, color=:jet, colorbar=false, aspect_ratio=1.0, framestyle=:semi, xlim=[-2.1,3.1])

	xs = [2 2; 2 -1; -1 -1; -1 2]

	for x in eachrow(xs)
		scatter!([x[1]], [x[2]], label="", markershape=:star, ms=8, xlim=(-2,3), ylim=(-2,3))
	end
	# scatter!([2], [2])
	# scatter!([2], [-1])
	
	plt
	# xs_, ys_ = meshgrid(range(-2, 3, length=10), range(-2, 3, length=10))
	# ∇f_d(x, y) = ∇f_demo(x, y)/20
	# quiver!(xs_, ys_, quiver = ∇f_d, c=:green)
end

# ╔═╡ 8d878e1a-6191-42f2-a632-6d0b44f7d702
md"""

## Demonstration 

Recall the function 

```math

f(w_1, w_2) = \frac{1}{4} (w_1^4 + w_2^4) -\frac{1}{3} (w_1^3 +w_2^3) - w_1^2 -w_2^2 +4
```

"""

# ╔═╡ 5d52d818-21b2-4e8f-8efa-486274b68e57
fs, traces = gradient_descent(f_demo, ∇f_demo; w_init=[3.5,2.5]);

# ╔═╡ b732f3da-7be0-4470-9a5b-a389c1d1c166
anim_demo = let
	gr()

	plt = plot(-2.:0.05:3, -2.:0.05:3, f_demo, st=:contour, color=:jet, colorbar=false, aspect_ratio=1.0, framestyle=:none)
	wt = [3.5, 2.5]
	anim = @animate for t in 1:5:size(traces)[2]
		plot!([traces[1, t]], [traces[2, t]], st=:scatter, color=1, label="", markersize=3)
		plot!([wt[1], traces[1, t]], [wt[2], traces[2, t]], line = (:arrow, 0.8, :gray), label="")
		wt = traces[1:2, t]
	end
end;

# ╔═╡ 22e3f0e5-6a3c-439e-8160-978b24dc0a7d
md"""

##

Starting **Gradient descent** from the top two corners
* Gradient descent converges to two different locations (local minimums)

"""

# ╔═╡ f15c498f-a4af-4cd5-a2db-2ea42516fb0f
TwoColumn(gif(anim_demo, fps=5), let
	gr()
	w0 = [-2.5, 3]
	_, traces = gradient_descent(f_demo, ∇f_demo; w_init = w0)
	plt=plot(-2.:0.05:3, -2.:0.05:3, f_demo, st=:contour, color=:jet, colorbar=false, aspect_ratio=1.0, framestyle=:none)
	wt = w0
	anim = @animate for t in 1:5:size(traces)[2]
		plot!([traces[1, t]], [traces[2, t]], st=:scatter, color=1, label="", markersize=3)
		plot!([wt[1], traces[1, t]], [wt[2], traces[2, t]], line = (:arrow, 0.8, :gray), label="")
		wt = traces[1:2, t]
	end

	gif(anim, fps=5)
end)

# ╔═╡ f979d38e-d634-4197-a774-773ea0af8193
md"##

From the bottom two corners
"

# ╔═╡ 095f95fa-75e1-488f-a1c6-afcaaad30baa
TwoColumn(let
	gr()
	w0 = [-3,-2]
	_, traces = gradient_descent(f_demo, ∇f_demo; w_init = w0)
	plt=plot(-2.:0.05:3, -2.:0.05:3, f_demo, st=:contour, color=:jet, colorbar=false, aspect_ratio=1.0, framestyle=:none)
	wt = w0
	anim = @animate for t in 1:5:size(traces)[2]
		plot!([traces[1, t]], [traces[2, t]], st=:scatter, color=1, label="", markersize=3)
		plot!([wt[1], traces[1, t]], [wt[2], traces[2, t]], line = (:arrow, 0.8, :gray), label="")
		wt = traces[1:2, t]
	end

	gif(anim, fps=5)
end, let
	gr()
	w0 = [3.5,-2]
	_, traces = gradient_descent(f_demo, ∇f_demo; w_init = w0)
	plt=plot(-2.:0.05:3, -2.:0.05:3, f_demo, st=:contour, color=:jet, colorbar=false, aspect_ratio=1.0, framestyle=:none)
	wt = w0
	anim = @animate for t in 1:5:size(traces)[2]
		plot!([traces[1, t]], [traces[2, t]], st=:scatter, color=1, label="", markersize=3)
		plot!([wt[1], traces[1, t]], [wt[2], traces[2, t]], line = (:arrow, 0.8, :gray), label="")
		wt = traces[1:2, t]
	end

	gif(anim, fps=5)
end)

# ╔═╡ 538a3ded-2ca0-4f0c-a931-fb7e43e5c24f
md"""

## Learning rate


The learning rate ``\gamma`` should be a _small_ constant 

* usually set by trial and error
* ``\gamma`` can also be adaptive: ``\gamma_t``

"""

# ╔═╡ c92b52dd-3c76-40a8-a897-0136b5c4e529
md"""

## Learning rate too large
"""

# ╔═╡ ecb72fef-744c-47aa-b54c-369aa7e16c9e
# let
# 	gr()
# 	plt = plot(-3.:0.05:1, -2.:0.05:1, f_demo_2, st=:contourf, color=:jet, colorbar=true, xlim=[-3, 1], ylim=[-2, 1])
# 	xs = range(-3, 1, length = 20)
# 	ys = range(-2, 1, length = 20)
# 	α = 0.25
# 	xs_, ys_ = meshgrid(range(-3, 1, length=20), range(-2, 1, length=20))
# 	∇f_d(x, y) = ∇f_demo_2(x, y) * α
# 	quiver!(xs_, ys_, quiver = ∇f_d, c=:black)
# 	plt
# end

# ╔═╡ 726545fd-fbbc-4ce9-a35d-30e7e23ff3f9
begin
	f_demo_2(w₁, w₂) = sin(w₁ + w₂) + cos(w₁)^2
	∇f_demo_2(w₁, w₂) = [cos(w₁ + w₂) - 2*cos(w₁)*sin(w₁),  cos(w₁ + w₂)]
end;

# ╔═╡ 78c47b3f-9641-4ea4-9f3d-fadaa0ae004c
begin
	f_demo_2(w) = f_demo_2(w[1], w[2]) #overload the function with vector input
	∇f_demo_2(w) = ∇f_demo_2(w[1], w[2])  #overload with vector input
end;

# ╔═╡ b8593384-18c7-4b03-99b0-02d7e3a0f8be
let
	gr()
	f_demo = f_demo_2
	∇f_demo = ∇f_demo_2
	# w0 = []
	_, traces = gradient_descent(f_demo, ∇f_demo; w_init=[0,-1], γ = 0.1)
	plt = plot(-3.:0.05:1, -2.:0.05:1, f_demo, st=:contour, color=:jet, colorbar=false, aspect_ratio=1.0, xlim=[-3, 1], ylim=[-2, 1], title="Learning rate : "* L"\gamma = 0.1")
	wt = [0, -1]
	anim = @animate for t in 1:5:size(traces)[2]
		plot!([traces[1, t]], [traces[2, t]], st=:scatter, color=1, label="", markersize=3)
		plot!([wt[1], traces[1, t]], [wt[2], traces[2, t]], line = (:arrow, 0.8, :gray), label="")
		wt = traces[1:2, t]
	end

	gif(anim, fps=5)
end

# ╔═╡ b81074f7-2bf7-4c4e-9f8a-efe426fdf6bd
let
	gr()
	f_demo = f_demo_2
	∇f_demo = ∇f_demo_2
	# w0 = []
	_, traces = gradient_descent(f_demo, ∇f_demo; w_init=[0,-1], γ = 1)
	plt = plot(-3.:0.05:1, -2.:0.05:1, f_demo, st=:contour, color=:jet, colorbar=false, aspect_ratio=1.0, xlim=[-3, 1], ylim=[-2, 1], title="Learning rate too large: "* L"\gamma = 1")
	wt = [0, -1]
	anim = @animate for t in 1:5:size(traces)[2]
		plot!([traces[1, t]], [traces[2, t]], st=:scatter, color=1, label="", markersize=3)
		plot!([wt[1], traces[1, t]], [wt[2], traces[2, t]], line = (:arrow, 0.5, :gray), label="")
		wt = traces[1:2, t]
	end

	gif(anim, fps=5)
end

# ╔═╡ c9baffcf-c1cc-42d6-96b5-1cc9a8801113
md"""



## Gradient check

Gradient derivation is **error**-prone
* either manually derived via chain rule
* or implementation error


We test software carefully; we should also test our gradients carefully!

> How can we check our hand-derived gradients are correct?

## Gradient check

By gradient's definition

```math
\large
\frac{\partial f(\mathbf{x})}{\partial x_i} \approx \frac{f(\mathbf{x}+ \epsilon \cdot \mathbf{e}_i) - f(\mathbf{x})}{\epsilon}
```

* ``\mathbf{e}_i`` is the i-th standard basis vector

  * as an example, for ``i=1``
```math
\mathbf{x} + \epsilon\cdot  \mathbf{e}_1 =\begin{bmatrix}x_1  \\ x_2 \\\vdots \\ x_m \end{bmatrix} + \epsilon \begin{bmatrix}1  \\ 0 \\\vdots \\ 0 \end{bmatrix}= \begin{bmatrix}x_1 + \epsilon \\ x_2 \\\vdots \\ x_m \end{bmatrix}
```

* *idiot-proof* and can be computed for all ``f`` 
* very useful to check your manual derivation! 
  * gradient check is like testing your program

"""

# ╔═╡ 40af31cc-1bc6-436c-ae6b-d13e2300779a
md"## Demonstration"

# ╔═╡ 5063f1be-5f18-43f6-b428-a0464e8fb339
md"
Take univariate function as an example


```math
\large
\frac{d f({x})}{d x} \approx \frac{f({x}+ \epsilon) - f({x})}{\epsilon}
```
"

# ╔═╡ c0acfd8e-c152-4991-80a9-65d69c3bda69
md"""

```python
## python
def df(x, f, eps)
	df = f(x+eps) - f(x)
	df/eps
```
"""

# ╔═╡ fcd96cd0-c733-4b76-9ea5-384f3253f730
md"in `Julia`:"

# ╔═╡ 1ae4f029-cb95-4897-8029-903767babf7b
function df(x; f, ϵ)  
	(f(x+ϵ) - f(x))/ϵ
end

# ╔═╡ a676ed26-1420-478f-a25c-fe99ab94c0a5
md"""The small constant ``ϵ`` is"""

# ╔═╡ 39521df0-70c2-4e59-bcb2-792f0ba97c68
@bind ϵ_ Slider(1e-7:1e-7:.5, default=0.5, show_value=true)

# ╔═╡ 4f176ca1-0326-4576-ae18-af2e1b697655
md"As an example, 


```math
\large
f(x)= \sin(x),\; f'(x) = \cos(x)
```
"

# ╔═╡ 6e25724e-79f7-4050-b3ff-fc51bfc852b5
let
	gr()
	x₀ = 0.0
	Δx = ϵ_
	xs = -1.2π : 0.1: 1.2π
	f, ∇f = sin, cos
	# anim = @animate for Δx in π:-0.1:0.0
	# Δx = 1.3
	plot(xs, sin, label=L"\sin(x)", ylim = [-1.5, 1.5], xlabel=L"x", lw=2, legend=:outerbottom, framestyle=:semi, title="Derivative at "*L"x=0", legendfontsize=10)
		df = f(x₀ + Δx)-f(x₀)
		k = Δx == 0 ? ∇f(x₀) : df/Δx
		b = f(x₀) - k * x₀ 
		# the approximating linear function with Δx 
		plot!(xs, (x) -> k*x+b, label="", lw=2)
		# the location where the derivative is defined
		scatter!([x₀], [f(x₀)], ms=3, label=L"x_0,\; \sin(x_0)")
		scatter!([x₀+Δx], [f(x₀+Δx)], ms=3, label=L"x_0+Δx,\; \sin(x_0+Δx)")
		plot!([x₀, x₀+Δx], [f(x₀), f(x₀)], lc=:gray, label="")
		plot!([x₀+Δx, x₀+Δx], [f(x₀), f(x₀+Δx)], lc=:gray, label="")
		font_size = Δx < 0.8 ? 12 : 14
		annotate!(x₀+Δx, 0.5 *(f(x₀) + f(x₀+Δx)), text(L"Δf", font_size, :top, rotation = 90))
		annotate!(0.5*(x₀+x₀+Δx), 0, text(L"Δx", font_size,:top))
		annotate!(-.6, 1, text(L"\frac{Δf}{Δx}=%$(round(k, digits=6))", 15,:top))
end

# ╔═╡ 0be36e2d-817b-4951-8a3d-4ab44ce761ce
md"
The difference:
"

# ╔═╡ 94899a98-1565-4129-a264-1c4e1855982b
let
	𝒻(x) = sin(x)
	∇𝒻(x) = cos(x)
	x₁ = 1.0
	df(x₁; f=𝒻, ϵ = ϵ_) - ∇𝒻(x₁)
end

# ╔═╡ 1643d5a5-8925-45b7-a4ff-a39165207ec1
md"""

## Multi-variate gradient


The following code snippet calculate the gradient via central method


$$\large \nabla f(\mathbf{w})=\text{grad} f(\mathbf{w}) = \begin{bmatrix}\frac{\partial f(\mathbf{w})}{\partial w_1}\\ \frac{\partial f(\mathbf{w})}{\partial w_2}\\ \vdots \\ \frac{\partial f(\mathbf{w})}{\partial w_m}\end{bmatrix} \approx \frac{1}{2\epsilon} \begin{bmatrix}f(\mathbf{w} +\epsilon\mathbf{e}_1) - f(\mathbf{w} -\epsilon\mathbf{e}_1)\\ f(\mathbf{w} +\epsilon\mathbf{e}_2) - f(\mathbf{w} -\epsilon\mathbf{e}_2)\\ \vdots \\ f(\mathbf{w} +\epsilon\mathbf{e}_m) - f(\mathbf{w} -\epsilon\mathbf{e}_m)\end{bmatrix}$$
"""

# ╔═╡ 6eec47df-4b17-48d4-a2f6-8eec8df29a4b
aside(tip(md"
In practice, we use an alternative but more accurate method called central difference:

```math
\frac{f(x+\epsilon) - f(x-\epsilon)}{2\epsilon}
```
"))

# ╔═╡ 8a5a3a3d-d341-48e9-bcba-a16094f048c4
md"""

```python
def gradient(f, initial, eps=1e-6):
  initial = np.array(initial, dtype=float)
  n = len(initial)
  output = np.zeros(n)
  for i in range(n):
    ei = np.zeros(n)
    ei[i] = 1
    f1 = f(initial + eps * ei)
    f2 = f(initial - eps * ei)
    output[i] = (f1-f2)/(2*eps)
  output = output.reshape(n,1)
  return output

```

"""

# ╔═╡ 466d0d89-bf16-40f9-8eb2-2dfe70a8b227
md"""

# Case study: linear regression 

## Recap

Linear regression: the objective is to minimise the loss: **mean** of the squared error

```math
\large
l(\mathbf{w})= \frac{1}{2n}\sum_{i=1}^n (y^{(i)} - \mathbf{w}^\top\mathbf{x}^{(i)})^2 = \frac{1}{2n}(\mathbf{y} -\mathbf{Xw})^\top(\mathbf{y} -\mathbf{Xw})
```


The gradient is


```math
\large
\nabla l(\mathbf{w}) =  -\frac{1}{n}\sum_{i=1}^n \underbrace{(y^{(i)} - \mathbf{w}^\top\mathbf{x}^{(i)})}_{\text{prediction diff: } e^{(i)}} \mathbf{x}^{(i)} = -\frac{1}{n}\mathbf{X}^\top (\mathbf{y} - \mathbf{Xw}) 
```


For this prorblem, we have the exact solution

```math
\hat{\mathbf{w}} =  (\mathbf{X}^\top \mathbf{X} )^{-1} \mathbf{X}^\top \mathbf{y}
```

* but ``\mathbf{X}^\top\mathbf{X}`` can be very expensive to compute and invert
* we can also use **gradient descent**
"""

# ╔═╡ 588735c8-d19c-4656-90e5-b2e530972caa
md"""

## Recap: gradient descent


```math
\LARGE
\mathbf{w}_{new} = \mathbf{w}_{old} - \underbrace{\colorbox{lightgreen}{$\gamma$}}_{{\color{green}\small \rm learning\; rate}} \nabla_{\mathbf{w}} f(\mathbf{w}_{old})
```


What gradient descent needs:
* ``\nabla f(\mathbf{w})``, the gradient function
* ``\colorbox{lightgreen}{$\gamma$}``, a step size called _learning rate_ *e.g.* 0.01




"""

# ╔═╡ e95f09c8-9364-4cf2-91a7-4e2dcbcd1053
md"""

## Gradient descent


Let's see the insights of how *gradient descent* works


* consider sample ``i`` only, its gradient is
```math
\large
\begin{align}
\nabla l^{(i)}(\mathbf{w}_{old}) &=  -\underbrace{(y^{(i)} - \mathbf{w}_{old}^\top\mathbf{x}^{(i)})}_{\text{prediction diff: } e^{(i)}} \mathbf{x}^{(i)} \\
&= - e^{(i)}\cdot \mathbf{x}^{(i)}

\end{align}
```

* gradient descent applies the **negative** gradient direction, which becomes 

```math
\mathbf{w}_{new} \leftarrow \mathbf{w}_{old} -\gamma  (-e^{(i)}\cdot \mathbf{x}^{(i)})
``` 

```math
\large
\boxed{\mathbf{w}_{new} \leftarrow \mathbf{w}_{old} + \gamma\cdot e^{(i)} \mathbf{x}^{(i)}\;\; \# \texttt{gradient step}}
```
* where ``e^{(i)} = y - \mathbf{w}_{old}^\top\mathbf{x}^{(i)}`` is the prediction error
"""


# ╔═╡ 9ccef028-d31d-45d6-a37f-77cd31f772dc
aside(tip(md"``\large e`` here stands for prediction **error** or difference."))

# ╔═╡ 1c5c58b6-8567-4ce0-af7a-fa84f4879d39
md"""

## Interpretation 

```math
\large
\boxed{\mathbf{w}_{new} \leftarrow \mathbf{w}_{old} + \gamma\cdot e\, \mathbf{x}\;\; \# \texttt{gradient step}}
```


* where ``e = y - \mathbf{w}_{old}^\top\mathbf{x}`` is the prediction error

After the update, the **new prediction** becomes

```math
\large
\colorbox{orange}{$
\hat{h}_{new}=\mathbf{w}_{new}^\top \mathbf{x} = \mathbf{w}_{old}^\top \mathbf{x} + \gamma\cdot e\cdot \underbrace{{\mathbf{x}}^\top \mathbf{x}}_{\geq 0}$}
```

\

when error ``e \approx 0``, the current model is perfect,  
  * _GD_ make little change

\

when error ``e > 0``, the current model under predicts ``(\mathbf{w}_{old}^\top\mathbf{x} <y )``
  * GD increases the prediction by a little
\

when error ``e < 0``, the current model over predicts  as ``( \mathbf{w}_{old}^\top\mathbf{x} > y)``
  * GD decreases the prediction by a little


"""

# ╔═╡ d94a06dc-77fb-4e27-a313-6589b5641519
md"""

## Demonstration

"""

# ╔═╡ 0130e026-cf29-45d7-9124-283b4ae5689e
md"""

Dataset: `X_train`, `y_train`

* ``\mathbf{x} \in \mathbb{R}^2``: 2 input features 
* 50 observations

"""

# ╔═╡ 8d9edfd5-6f97-4c4e-9464-3841078cc7c4
linear_reg_normal_eq(X, y) = X \ y;

# ╔═╡ 4b5ec74e-b89c-4bbf-9ac1-fa7d5b709f57
function loss(w, X, y) 
	error = X * w - y
	.5 * dot(error, error)/length(y)
end;

# ╔═╡ 4788ccd2-62a9-423f-95f8-57507ccdd6e3
function ∇linear_reg(w; X, y) 
	-X' * (y - X* w) / length(y)
end;

# ╔═╡ 1cd4daa9-4683-4b01-a4ed-7c55d769c861
md"""
## Gradient descent animation

The animation below shows how gradient descent evolves over the iterations

* the starting plane is horizontal as ``\mathbf{w}_0 = \mathbf{0}``
* it makes progress to fit the data quickly
"""

# ╔═╡ 730b641f-00bc-4b27-a832-166d1e2d75c9
md"""


## 


**Gradient descent** converges to almost the **same** result as normal equation's solution
"""

# ╔═╡ 61263a4c-f756-439e-a2bf-b7d7bc6d5b66
md"""

## A different loss?


Auto-differentiation package becomes very useful for customised loss


```math
\text{loss}_p(\mathbf{w}) = \frac{1}{n}\sum_{i=1}^n |y^{(i)} - \mathbf{w}^{\top} \mathbf{x}^{(i)}|^p 
```

* for example train a regression with ``p=4``
* one only needs to define the loss and leave the gradient derivation to auto-diff

Note that the optimisation cannot be solved analytically anymore
* but we can apply gradient descent as usual
"""

# ╔═╡ 60706166-a939-4758-b92c-27471dbade38
md"""

# Learning rate tuning



## One size fits all 

**One learning rate** is used for all parameters


```math
\begin{bmatrix}
w_0\\
w_1\\
\vdots\\

w_m
\end{bmatrix} = \begin{bmatrix}
w_0\\
w_1\\
\vdots\\

w_m
\end{bmatrix} - \colorbox{orange}{$\LARGE \gamma$} \begin{bmatrix}
\frac{\partial f}{\partial w_0}(\mathbf{w})\\
\frac{\partial f}{\partial w_1}(\mathbf{w})\\
\vdots\\

\frac{\partial f}{\partial w_m}(\mathbf{w})
\end{bmatrix}
```


* different parameter might scale differently ``\Rightarrow`` one learning rate may **not** fit all 
"""

# ╔═╡ 8974a2f8-fb66-4069-814a-a39fb50ba154
md"""
## An example

The loss surface of MSE 
"""

# ╔═╡ bd6e2825-a53a-4d2d-842f-9ef77d7eea11
html"""<center><img src="https://leo.host.cs.st-andrews.ac.uk/figs/CS5914/loss1.png" width = "500"/></center>""" 

# ╔═╡ 56c1e854-8b01-4777-b94b-d150689b81a8
md"""
##

A random **initial start point** 

* the uni-variate loss curves look very differently

"""

# ╔═╡ 29e3d19b-34ac-48b5-930f-736f4488d855
html"""<center><img src="https://leo.host.cs.st-andrews.ac.uk/figs/CS5914/loss2.png" width = "500"/></center>""" 

# ╔═╡ 7a54d37e-198c-42fa-bd5e-f4b5abe13d9e
html"""<center><img src="https://leo.host.cs.st-andrews.ac.uk/figs/CS5914/loss3.png" width = "500"/></center>""" 

# ╔═╡ 9d83f116-79ed-4dd5-93d2-cc59f7127b3a
md"""
##


One learning rate **does NOT** fit all _well_
"""

# ╔═╡ d9067223-2bf7-4406-ab4d-adf2c28e3d72
html"""<center><img src="https://leo.host.cs.st-andrews.ac.uk/figs/CS5914/loss4.png" width = "500"/></center>""" 

# ╔═╡ 6ef8a367-223e-44a2-9aa2-9a5dcfc2d9fc
md"[source](https://github.com/dvgodoy/PyTorchStepByStep)"

# ╔═╡ 42e7bc1a-7380-47f6-aa2d-c49cdaff0b08
md"""

## Solution: standardising input
$$\Large
\begin{align}
{\mu_i} = \frac{1}{N}\sum_{i=1}^N{x_i}
\\
\Large
\sigma_i = \sqrt{\frac{1}{N}\sum_{i=1}^N{(x_i - \mu_i)^2}}
\\
\Large
\text{scaled } x_i=\frac{x_i-\mu_i}{\sigma_i}\end{align}$$
"""

# ╔═╡ fe721301-1e77-4c89-8592-17d8c88776a7
md"""

## After the transform


**The middle one**: scale the input further away from _unit variances_
"""

# ╔═╡ fbf9a928-fdc7-4f38-9b93-4bf74c20ad5b
html"""<center><img src="https://leo.host.cs.st-andrews.ac.uk/figs/CS5914/loss5.png" width = "800"/></center>""" 

# ╔═╡ d294a9ce-b70e-48e4-aefc-a3e80cb7d003
begin
	function loss_p(w, X, y; p = 4)
		error = abs.(y - X*w)
		sum(error .^ p)
	end
end;

# ╔═╡ b6e7802f-3ce2-4663-b6a7-432d5295547d
md"""

# Appendix
"""

# ╔═╡ b6fca076-0639-490a-b77c-7a0155c25f39
meshgrid(x, y) = (repeat(x, outer=length(y)), repeat(y, inner=length(x))) # helper function to create a quiver grid.

# ╔═╡ e47db9dd-81ce-4ce0-a555-85d93e706bf6
let
	gr()
	α = 0.4
	plot(-2:0.05:3, -2:0.05:3, f_demo, st=:contourf, color=:jet, alpha=0.3, colorbar=false, aspect_ratio=1.0, xlim=[-2, 3], ylim=[-2, 3])
	xs_, ys_ = meshgrid(range(-2, 3, length=20), range(-2, 3, length=20))
	∇f_d(x, y) = ∇f_demo(x, y) * α
	quiver!(xs_, ys_, quiver = ∇f_d, c=:red)
end

# ╔═╡ d96be66d-d960-4402-872e-14161144a7df
md"""

### Simulate data for the linear regression example
"""

# ╔═╡ 48e8aa1f-9a19-456a-bb6d-cdbfe0f5403b
begin
	Random.seed!(123)
	num_features = 2
	num_data = 50
	true_w = rand(num_features+1) * 3
	# simulate the design matrix or input features
	X_train = [ones(num_data) rand(num_data, num_features)]
	# generate the noisy observations
	y_train = X_train * true_w + randn(num_data)
end;

# ╔═╡ 43be7d18-4417-4683-aae8-c35ffc9fb55f
let
	plotly()
	# plot(X_train[:,2], y_train, st=:scatter, label="Observations")
	scatter(X_train[:, 2], X_train[:,3], y_train, markersize=1.5, label="observations", title="Linear regression toy dataset", xlabel="x₁", ylabel="x₂", zlabel="y")
	# surface!(0:0.5:1, 0:0.5:1.0, (x1, x2) -> dot([1, x1, x2], true_w),  colorbar=false, xlabel="x₁", ylabel="x₂", zlabel="y")
end

# ╔═╡ 3e229fb0-62a3-443f-b864-848a0da18711
X_train, y_train;

# ╔═╡ 5155785f-0cda-43de-9501-db8f2d03fcb2
w_normal_eq = linear_reg_normal_eq(X_train, y_train);

# ╔═╡ 47bf5f18-23c0-4780-815d-0fed28f2bc57
w_normal_eq

# ╔═╡ 09c731d4-3b14-49c4-a609-4ab046513078
ws_history, losses=let
	∇l(x) = ∇linear_reg(x; X= X_train, y=y_train)
	max_iters = 2000
	losses = []
	# random starting point
	w₀ = zeros(num_features+1)
	push!(losses, loss(w₀, X_train, y_train))
	ws_history = zeros(num_features+1, max_iters+1)
	ws_history[:, 1] = w₀
	γ = 0.1
	for i in 1:max_iters
		w₀ = w₀ - γ * ∇l(w₀)
		push!(losses, loss(w₀, X_train, y_train)) # book keeping; optional
		ws_history[:, i+1] = w₀ # book keeping; optional
	end
	ws_history, losses
end;

# ╔═╡ 8cca1036-a57f-48e0-826c-c81c4fe19085
anim=let
	gr()
	anim = @animate for i in 1:20
		# plot(1:10)
		scatter(X_train[:, 2], X_train[:,3], y_train, markersize=1.5, label="")
		w0 = ws_history[:, i]
		surface!(0:0.5:1, 0:0.5:1.0, (x1, x2) -> dot([1, x1, x2], w0), c=:jet,  colorbar=false, xlabel=L"x_1", ylabel=L"x_2", zlabel=L"y", alpha=0.5, title="Iteration "*string(i))
	end
end;

# ╔═╡ ea4a2ce4-5b20-4ad3-a39f-51b54be5f3d3
gif(anim, fps=3)

# ╔═╡ fccd6463-ed05-4a7d-89e6-c87981d08248
ws_history[:, end] # gradient descent result;

# ╔═╡ 56e93e4b-6963-4f86-993c-5f9bc3ec3199
ws_history[:, end] ≈ w_normal_eq # should be the same

# ╔═╡ 7d5a7f03-74b0-42f8-a5c8-19e077db26bc
md"""

## Explain gradient descent

Let's consider univariate function first

A differentiable function ``f(x)`` can be approximated locally at ``x_0`` by a **linear function**


```math

f(x) \approx f(x_0) + f'(x_0) ( x- x_0)
```


* slope: ``b_1= f'(x_0)``
* intercept: ``b_0 = f(x_0) - x_0 f'(x_0)``


The idea can be generalised to multivariate function


```math

f(\mathbf{x}) \approx f(\mathbf{x}_0) + \nabla f(\mathbf{x}_0)^\top ( \mathbf{x}- \mathbf{x}_0)
```


* derivative ``\rightarrow`` gradient

""";

# ╔═╡ 7c6e1fd4-aa87-45c0-9c49-6dab2de90043
linear_approx_f(x; f, ∇f, x₀) = f(x₀) + dot(∇f(x₀), (x-x₀));

# ╔═╡ e54c9f2d-ae25-46f1-8fcc-a75184a3610b
begin
	A = Matrix(I, 2, 2)
	f(x) = dot(x, A, x)
	∇f(x) = 2* A* x
end;

# ╔═╡ dc7bbdc2-d3df-42a8-865b-74c42096cb57
md"Expansion location:";

# ╔═╡ ed7f77ac-0cc5-4141-b88b-c5a86749ddd6
x₀ = [-5, 5];

# ╔═╡ e1219a00-b1cd-4aba-a3ab-cfec9380fcf3
let
	plotly()
	plot(-15:1.0:15, -15:1.0:15, (x1, x2) -> f([x1, x2]), st=:surface, xlabel="x₁", ylabel="x₂", zlabel="f",  alpha=0.8, framestyle=:zerolines, ratio=1, colorbar=false)
	plot!(-15:1:15, -15:1:15, (x1, x2) -> linear_approx_f([x1, x2]; f=f, ∇f= ∇f, x₀), st=:surface)
end;

# ╔═╡ 55b788aa-e846-421e-b214-ef99e24fd97e
md"""

## Explain gradient descent



```math

f(\mathbf{x}) \approx f(\mathbf{x}_0) + \nabla f(\mathbf{x}_0)^\top ( \mathbf{x}- \mathbf{x}_0)
```


* the linear approximation is accurate when ``\mathbf{x}`` is close to ``\mathbf{x}_0``

**Gradient descent**
* at each iteration, find a local linear approximation
* and follow the (opposite) direction of the hyperplane 
* small learning rate: the approximation is correct only locally
""";

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
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
StatsBase = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"

[compat]
HypertextLiteral = "~0.9.5"
LaTeXStrings = "~1.3.1"
Latexify = "~0.15.17"
LogExpFunctions = "~0.3.18"
Plots = "~1.39.0"
PlutoTeachingTools = "~0.2.5"
PlutoUI = "~0.7.48"
StatsBase = "~0.33.21"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.9.4"
manifest_format = "2.0"
project_hash = "c84a167bea7fab832574c652d463de982d0e76a2"

[[deps.AbstractPlutoDingetjes]]
deps = ["Pkg"]
git-tree-sha1 = "793501dcd3fa7ce8d375a2c878dca2296232686e"
uuid = "6e696c72-6542-2067-7265-42206c756150"
version = "1.2.2"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"
version = "1.1.1"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"

[[deps.BitFlags]]
git-tree-sha1 = "2dc09997850d68179b69dafb58ae806167a32b1b"
uuid = "d1d4a3ce-64b1-5f1a-9ba4-7e7e69966f35"
version = "0.1.8"

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

[[deps.CodeTracking]]
deps = ["InteractiveUtils", "UUIDs"]
git-tree-sha1 = "c0216e792f518b39b22212127d4a84dc31e4e386"
uuid = "da1fd8a2-8d9e-5ec2-8556-3022fb5608a2"
version = "1.3.5"

[[deps.CodecZlib]]
deps = ["TranscodingStreams", "Zlib_jll"]
git-tree-sha1 = "cd67fc487743b2f0fd4380d4cbd3a24660d0eec8"
uuid = "944b1d66-785c-5afd-91f1-9de20f533193"
version = "0.7.3"

[[deps.ColorSchemes]]
deps = ["ColorTypes", "ColorVectorSpace", "Colors", "FixedPointNumbers", "PrecompileTools", "Random"]
git-tree-sha1 = "67c1f244b991cad9b0aa4b7540fb758c2488b129"
uuid = "35d6a980-a343-548e-a6ea-1d62b119f2f4"
version = "3.24.0"

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

    [deps.ColorVectorSpace.extensions]
    SpecialFunctionsExt = "SpecialFunctions"

    [deps.ColorVectorSpace.weakdeps]
    SpecialFunctions = "276daf66-3868-5448-9aa4-cd146d93841b"

[[deps.Colors]]
deps = ["ColorTypes", "FixedPointNumbers", "Reexport"]
git-tree-sha1 = "fc08e5930ee9a4e03f84bfb5211cb54e7769758a"
uuid = "5ae59095-9a9b-59fe-a467-6f913c188581"
version = "0.12.10"

[[deps.Compat]]
deps = ["UUIDs"]
git-tree-sha1 = "8a62af3e248a8c4bad6b32cbbe663ae02275e32c"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "4.10.0"
weakdeps = ["Dates", "LinearAlgebra"]

    [deps.Compat.extensions]
    CompatLinearAlgebraExt = "LinearAlgebra"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"
version = "1.0.5+0"

[[deps.ConcurrentUtilities]]
deps = ["Serialization", "Sockets"]
git-tree-sha1 = "8cfa272e8bdedfa88b6aefbbca7c19f1befac519"
uuid = "f0e56b4a-5159-44fe-b623-3e5288b988bb"
version = "2.3.0"

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

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"

[[deps.DelimitedFiles]]
deps = ["Mmap"]
git-tree-sha1 = "9e2f36d3c96a820c678f2f1f1782582fcf685bae"
uuid = "8bb1440f-4735-579b-a4ab-409b98df4dab"
version = "1.9.1"

[[deps.Distributed]]
deps = ["Random", "Serialization", "Sockets"]
uuid = "8ba89e20-285c-5b6f-9357-94700520ee1b"

[[deps.DocStringExtensions]]
deps = ["LibGit2"]
git-tree-sha1 = "2fb1e02f2b635d0845df5d7c167fec4dd739b00d"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.9.3"

[[deps.Downloads]]
deps = ["ArgTools", "FileWatching", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"
version = "1.6.0"

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
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "JLLWrappers", "LAME_jll", "Libdl", "Ogg_jll", "OpenSSL_jll", "Opus_jll", "PCRE2_jll", "Zlib_jll", "libaom_jll", "libass_jll", "libfdk_aac_jll", "libvorbis_jll", "x264_jll", "x265_jll"]
git-tree-sha1 = "466d45dc38e15794ec7d5d63ec03d776a9aff36e"
uuid = "b22a6f82-2f65-5046-a5b2-351ab43fb4e5"
version = "4.4.4+1"

[[deps.FileWatching]]
uuid = "7b1f6079-737a-58dc-b8bc-7a2ca5c1b5ee"

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
git-tree-sha1 = "27442171f28c952804dede8ff72828a96f2bfc1f"
uuid = "28b8d3ca-fb5f-59d9-8090-bfdbd6d07a71"
version = "0.72.10"

[[deps.GR_jll]]
deps = ["Artifacts", "Bzip2_jll", "Cairo_jll", "FFMPEG_jll", "Fontconfig_jll", "FreeType2_jll", "GLFW_jll", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Libtiff_jll", "Pixman_jll", "Qt6Base_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "025d171a2847f616becc0f84c8dc62fe18f0f6dd"
uuid = "d2c73de3-f751-5644-a686-071e5b155ba9"
version = "0.72.10+0"

[[deps.Gettext_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Libiconv_jll", "Pkg", "XML2_jll"]
git-tree-sha1 = "9b02998aba7bf074d14de89f9d37ca24a1a0b046"
uuid = "78b55507-aeef-58d4-861c-77aaff3498b1"
version = "0.21.0+0"

[[deps.Glib_jll]]
deps = ["Artifacts", "Gettext_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Libiconv_jll", "Libmount_jll", "PCRE2_jll", "Zlib_jll"]
git-tree-sha1 = "e94c92c7bf4819685eb80186d51c43e71d4afa17"
uuid = "7746bdde-850d-59dc-9ae8-88ece973131d"
version = "2.76.5+0"

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
git-tree-sha1 = "5eab648309e2e060198b45820af1a37182de3cce"
uuid = "cd3eb016-35fb-5094-929b-558a96fad6f3"
version = "1.10.0"

[[deps.HarfBuzz_jll]]
deps = ["Artifacts", "Cairo_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "Graphite2_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Pkg"]
git-tree-sha1 = "129acf094d168394e80ee1dc4bc06ec835e510a3"
uuid = "2e76f6c2-a576-52d4-95c1-20adfe4de566"
version = "2.8.1+1"

[[deps.Hyperscript]]
deps = ["Test"]
git-tree-sha1 = "8d511d5b81240fc8e6802386302675bdf47737b9"
uuid = "47d2ed2b-36de-50cf-bf87-49c2cf4b8b91"
version = "0.0.4"

[[deps.HypertextLiteral]]
deps = ["Tricks"]
git-tree-sha1 = "7134810b1afce04bbc1045ca1985fbe81ce17653"
uuid = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"
version = "0.9.5"

[[deps.IOCapture]]
deps = ["Logging", "Random"]
git-tree-sha1 = "d75853a0bdbfb1ac815478bacd89cd27b550ace6"
uuid = "b5f81e59-6552-4d32-b1f0-c071b021bf89"
version = "0.2.3"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"

[[deps.IrrationalConstants]]
git-tree-sha1 = "630b497eafcc20001bba38a4651b327dcfc491d2"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.2.2"

[[deps.JLFzf]]
deps = ["Pipe", "REPL", "Random", "fzf_jll"]
git-tree-sha1 = "9fb0b890adab1c0a4a475d4210d51f228bfc250d"
uuid = "1019f520-868f-41f5-a6de-eb00f4b6a39c"
version = "0.1.6"

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
git-tree-sha1 = "0592b1810613d1c95eeebcd22dc11fba186c2a57"
uuid = "aa1ae85d-cabe-5617-a682-6adf51b2e16a"
version = "0.9.26"

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
git-tree-sha1 = "50901ebc375ed41dbf8058da26f9de442febbbec"
uuid = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
version = "1.3.1"

[[deps.Latexify]]
deps = ["Formatting", "InteractiveUtils", "LaTeXStrings", "MacroTools", "Markdown", "OrderedCollections", "Printf", "Requires"]
git-tree-sha1 = "8c57307b5d9bb3be1ff2da469063628631d4d51e"
uuid = "23fbe1c1-3f47-55db-b15f-69d7ec21a316"
version = "0.15.21"

    [deps.Latexify.extensions]
    DataFramesExt = "DataFrames"
    DiffEqBiologicalExt = "DiffEqBiological"
    ParameterizedFunctionsExt = "DiffEqBase"
    SymEngineExt = "SymEngine"

    [deps.Latexify.weakdeps]
    DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
    DiffEqBase = "2b5f629d-d688-5b77-993f-72d75c75574e"
    DiffEqBiological = "eb300fae-53e8-50a0-950c-e21f52c2b7e0"
    SymEngine = "123dc426-2d89-5057-bbad-38513e3affd8"

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
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "f9557a255370125b405568f9767d6d195822a175"
uuid = "94ce4f54-9a6c-5748-9c1c-f9c7231a4531"
version = "1.17.0+0"

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
git-tree-sha1 = "c1dd6d7978c12545b4179fb6153b9250c96b0075"
uuid = "e6f89c97-d47a-5376-807f-9c37f3926c36"
version = "1.0.3"

[[deps.LoweredCodeUtils]]
deps = ["JuliaInterpreter"]
git-tree-sha1 = "60168780555f3e663c536500aa790b6368adc02a"
uuid = "6f1432cf-f94c-5a45-995e-cdbf5db27b0b"
version = "2.3.0"

[[deps.MIMEs]]
git-tree-sha1 = "65f28ad4b594aebe22157d6fac869786a255b7eb"
uuid = "6c6e2e6c-3030-632d-7369-2d6c69616d65"
version = "0.1.4"

[[deps.MacroTools]]
deps = ["Markdown", "Random"]
git-tree-sha1 = "9ee1618cbf5240e6d4e0371d6f24065083f60c48"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.11"

[[deps.Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"

[[deps.MbedTLS]]
deps = ["Dates", "MbedTLS_jll", "MozillaCACerts_jll", "NetworkOptions", "Random", "Sockets"]
git-tree-sha1 = "c067a280ddc25f196b5e7df3877c6b226d390aaf"
uuid = "739be429-bea8-5141-9913-cc70e7f3736d"
version = "1.1.9"

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

[[deps.NaNMath]]
deps = ["OpenLibm_jll"]
git-tree-sha1 = "0877504529a3e5c3343c6f8b4c0381e57e4387e4"
uuid = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
version = "1.0.2"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"
version = "1.2.0"

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
git-tree-sha1 = "cc6e1927ac521b659af340e0ca45828a3ffc748f"
uuid = "458c3c95-2e84-50aa-8efc-19380b2a3a95"
version = "3.0.12+0"

[[deps.Opus_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "51a08fb14ec28da2ec7a927c4337e4332c2a4720"
uuid = "91d4177d-7536-5919-b921-800302f37372"
version = "1.3.2+0"

[[deps.OrderedCollections]]
git-tree-sha1 = "dfdf5519f235516220579f949664f1bf44e741c5"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.6.3"

[[deps.PCRE2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "efcefdf7-47ab-520b-bdef-62a2eaa19f15"
version = "10.42.0+0"

[[deps.Parsers]]
deps = ["Dates", "PrecompileTools", "UUIDs"]
git-tree-sha1 = "a935806434c9d4c506ba941871b327b96d41f2bf"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.8.0"

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
git-tree-sha1 = "ccee59c6e48e6f2edf8a5b64dc817b6729f99eb5"
uuid = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
version = "1.39.0"

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
git-tree-sha1 = "bd7c69c7f7173097e7b5e1be07cee2b8b7447f51"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.54"

[[deps.PrecompileTools]]
deps = ["Preferences"]
git-tree-sha1 = "03b4c25b43cb84cee5c90aa9b5ea0a78fd848d2f"
uuid = "aea7be01-6a6a-4083-8856-8a6e6704d82a"
version = "1.2.0"

[[deps.Preferences]]
deps = ["TOML"]
git-tree-sha1 = "00805cd429dcb4870060ff49ef443486c262e38e"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.4.1"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"

[[deps.Qt6Base_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Fontconfig_jll", "Glib_jll", "JLLWrappers", "Libdl", "Libglvnd_jll", "OpenSSL_jll", "Vulkan_Loader_jll", "Xorg_libSM_jll", "Xorg_libXext_jll", "Xorg_libXrender_jll", "Xorg_libxcb_jll", "Xorg_xcb_util_cursor_jll", "Xorg_xcb_util_image_jll", "Xorg_xcb_util_keysyms_jll", "Xorg_xcb_util_renderutil_jll", "Xorg_xcb_util_wm_jll", "Zlib_jll", "libinput_jll", "xkbcommon_jll"]
git-tree-sha1 = "37b7bb7aabf9a085e0044307e1717436117f2b3b"
uuid = "c0090381-4147-56d7-9ebc-da0b1113ec56"
version = "6.5.3+1"

[[deps.REPL]]
deps = ["InteractiveUtils", "Markdown", "Sockets", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"

[[deps.Random]]
deps = ["SHA", "Serialization"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

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
git-tree-sha1 = "ffdaf70d81cf6ff22c2b6e733c900c3321cab864"
uuid = "05181044-ff0b-4ac5-8273-598c1e38db00"
version = "1.0.1"

[[deps.Requires]]
deps = ["UUIDs"]
git-tree-sha1 = "838a3a4188e2ded87a4f9f184b4b0d78a1e91cb7"
uuid = "ae029012-a4dd-5104-9daa-d747884805df"
version = "1.3.0"

[[deps.Revise]]
deps = ["CodeTracking", "Distributed", "FileWatching", "JuliaInterpreter", "LibGit2", "LoweredCodeUtils", "OrderedCollections", "Pkg", "REPL", "Requires", "UUIDs", "Unicode"]
git-tree-sha1 = "a38e7d70267283888bc83911626961f0b8d5966f"
uuid = "295af30f-e4ad-537b-8983-00126c2a3abe"
version = "3.5.9"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"
version = "0.7.0"

[[deps.Scratch]]
deps = ["Dates"]
git-tree-sha1 = "3bac05bc7e74a75fd9cba4295cde4045d9fe2386"
uuid = "6c6a2e73-6563-6170-7368-637461726353"
version = "1.2.1"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"

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
git-tree-sha1 = "5165dfb9fd131cf0c6957a3a7605dede376e7b63"
uuid = "a2af1166-a08f-5f64-846c-94a0d3cef48c"
version = "1.2.0"

[[deps.SparseArrays]]
deps = ["Libdl", "LinearAlgebra", "Random", "Serialization", "SuiteSparse_jll"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

[[deps.Statistics]]
deps = ["LinearAlgebra", "SparseArrays"]
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
version = "1.9.0"

[[deps.StatsAPI]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "1ff449ad350c9c4cbc756624d6f8a8c3ef56d3ed"
uuid = "82ae8749-77ed-4fe6-ae5f-f523153014b0"
version = "1.7.0"

[[deps.StatsBase]]
deps = ["DataAPI", "DataStructures", "LinearAlgebra", "LogExpFunctions", "Missings", "Printf", "Random", "SortingAlgorithms", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "d1bf48bfcc554a3761a133fe3a9bb01488e06916"
uuid = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
version = "0.33.21"

[[deps.SuiteSparse_jll]]
deps = ["Artifacts", "Libdl", "Pkg", "libblastrampoline_jll"]
uuid = "bea87d4a-7f5b-5778-9afe-8cc45184846c"
version = "5.10.1+6"

[[deps.TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"
version = "1.0.3"

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
git-tree-sha1 = "1fbeaaca45801b4ba17c251dd8603ef24801dd84"
uuid = "3bb67fe8-82b1-5028-8e26-92a6c54297fa"
version = "0.10.2"
weakdeps = ["Random", "Test"]

    [deps.TranscodingStreams.extensions]
    TestExt = ["Test", "Random"]

[[deps.Tricks]]
git-tree-sha1 = "eae1bb484cd63b36999ee58be2de6c178105112f"
uuid = "410a4b4d-49e4-4fbc-ab6d-cb71b17b3775"
version = "0.1.8"

[[deps.URIs]]
git-tree-sha1 = "67db6cc7b3821e19ebe75791a9dd19c9b1188f2b"
uuid = "5c2747f8-b7ea-4ff2-ba2e-563bfd36b1d4"
version = "1.5.1"

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
git-tree-sha1 = "242982d62ff0d1671e9029b52743062739255c7e"
uuid = "1986cc42-f94f-5a68-af5c-568840ba703d"
version = "1.18.0"

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

[[deps.Vulkan_Loader_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Wayland_jll", "Xorg_libX11_jll", "Xorg_libXrandr_jll", "xkbcommon_jll"]
git-tree-sha1 = "2f0486047a07670caad3a81a075d2e518acc5c59"
uuid = "a44049a8-05dd-5a78-86c9-5fde0876e88c"
version = "1.3.243+0"

[[deps.Wayland_jll]]
deps = ["Artifacts", "EpollShim_jll", "Expat_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Pkg", "XML2_jll"]
git-tree-sha1 = "7558e29847e99bc3f04d6569e82d0f5c54460703"
uuid = "a2964d1f-97da-50d4-b82a-358c7fce9d89"
version = "1.21.0+1"

[[deps.Wayland_protocols_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4528479aa01ee1b3b4cd0e6faef0e04cf16466da"
uuid = "2381bf8a-dfd0-557d-9999-79630e7b1b91"
version = "1.25.0+0"

[[deps.XML2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libiconv_jll", "Zlib_jll"]
git-tree-sha1 = "da69178aacc095066bad1f69d2f59a60a1dd8ad1"
uuid = "02c8fc9c-b97f-50b9-bbe4-9be30ff0a78a"
version = "2.12.0+0"

[[deps.XSLT_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libgcrypt_jll", "Libgpg_error_jll", "Libiconv_jll", "Pkg", "XML2_jll", "Zlib_jll"]
git-tree-sha1 = "91844873c4085240b95e795f692c4cec4d805f8a"
uuid = "aed1982a-8fda-507f-9586-7b0439959a61"
version = "1.1.34+0"

[[deps.XZ_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "522b8414d40c4cbbab8dee346ac3a09f9768f25d"
uuid = "ffd25f8a-64ca-5728-b0f7-c24cf3aae800"
version = "5.4.5+0"

[[deps.Xorg_libICE_jll]]
deps = ["Libdl", "Pkg"]
git-tree-sha1 = "e5becd4411063bdcac16be8b66fc2f9f6f1e8fe5"
uuid = "f67eecfb-183a-506d-b269-f58e52b52d7c"
version = "1.0.10+1"

[[deps.Xorg_libSM_jll]]
deps = ["Libdl", "Pkg", "Xorg_libICE_jll"]
git-tree-sha1 = "4a9d9e4c180e1e8119b5ffc224a7b59d3a7f7e18"
uuid = "c834827a-8449-5923-a945-d239c165b7dd"
version = "1.2.3+0"

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

[[deps.Xorg_xcb_util_cursor_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_xcb_util_image_jll", "Xorg_xcb_util_jll", "Xorg_xcb_util_renderutil_jll"]
git-tree-sha1 = "04341cb870f29dcd5e39055f895c39d016e18ccd"
uuid = "e920d4aa-a673-5f3a-b3d7-f755a4d47c43"
version = "0.1.4+0"

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

[[deps.eudev_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "gperf_jll"]
git-tree-sha1 = "431b678a28ebb559d224c0b6b6d01afce87c51ba"
uuid = "35ca27e7-8b34-5b7f-bca9-bdc33f59eb06"
version = "3.2.9+0"

[[deps.fzf_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "47cf33e62e138b920039e8ff9f9841aafe1b733e"
uuid = "214eeab7-80f7-51ab-84ad-2988db7cef09"
version = "0.35.1+0"

[[deps.gperf_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "3516a5630f741c9eecb3720b1ec9d8edc3ecc033"
uuid = "1a1c6b14-54f6-533d-8383-74cd7377aa70"
version = "3.1.1+0"

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

[[deps.libevdev_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "141fe65dc3efabb0b1d5ba74e91f6ad26f84cc22"
uuid = "2db6ffa8-e38f-5e21-84af-90c45d0032cc"
version = "1.11.0+0"

[[deps.libfdk_aac_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "daacc84a041563f965be61859a36e17c4e4fcd55"
uuid = "f638f0a6-7fb0-5443-88ba-1cc74229b280"
version = "2.0.2+0"

[[deps.libinput_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "eudev_jll", "libevdev_jll", "mtdev_jll"]
git-tree-sha1 = "ad50e5b90f222cfe78aa3d5183a20a12de1322ce"
uuid = "36db933b-70db-51c0-b978-0f229ee0e533"
version = "1.18.0+0"

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

[[deps.mtdev_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "814e154bdb7be91d78b6802843f76b6ece642f11"
uuid = "009596ad-96f7-51b1-9f1b-5ce2d5e8a71e"
version = "1.1.6+0"

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
git-tree-sha1 = "9c304562909ab2bab0262639bd4f444d7bc2be37"
uuid = "d8fb68d0-12a3-5cfd-a85a-d49703b185fd"
version = "1.4.1+1"
"""

# ╔═╡ Cell order:
# ╟─9ca39f56-5eea-11ed-03e1-81eb0fa2a37d
# ╟─f9736ecc-053f-444a-9ef5-cdbe85795fce
# ╟─8d096119-f6db-4f62-a091-6f00372468ec
# ╟─dcacb71b-bf45-4b36-a2d2-477839f52411
# ╟─d7a55322-0d9f-44e8-a2c6-4f0cead25f9d
# ╟─5bb178a2-f119-405f-a65b-ec6d59a112e0
# ╟─799ead95-2f68-4bcb-aae8-dd8b0b00560c
# ╟─afb99b70-a418-4379-969b-355fbcfe8f14
# ╟─ee92f14a-784a-407c-8955-9ea1dd9cf942
# ╟─c49a6965-727e-419b-b66e-57dc61415edf
# ╟─e83d5763-bc02-4d88-a8d0-f9fb38314ae8
# ╟─828ceffb-1423-4e13-a92e-9d85f9f3f4b3
# ╟─38a119e0-517f-492a-a675-0f318c6d8696
# ╟─768c34c9-d7a9-4bd9-abee-59a9a176ffb7
# ╟─852be981-a741-42b2-b681-6b8fbc09a41c
# ╟─d02ec857-04aa-4128-a172-e49955bbb062
# ╟─f95f299a-3ad9-47f0-8b21-f7ee0e278cc1
# ╟─520c22ab-7fa9-49ef-bb13-f50461f1a928
# ╟─5921c8dc-25d6-4acb-9f60-1bcf8644ee12
# ╟─39486178-559b-4b87-915e-b77475de02a5
# ╟─f34c38d3-3ed1-4d59-8e23-6eb2f029de4f
# ╟─540564bf-74db-43c8-8c1f-8dc471dc8ec8
# ╟─74977378-9c43-4638-8be9-cbf91c82c7ac
# ╟─35050b2d-4f25-4a96-88db-c8d193bb0a35
# ╟─0d1dc5cf-36f1-4f28-befc-c96a02dc97a0
# ╟─0e74a142-70d1-4a36-bf7e-48411cc7c12b
# ╟─7e21b526-da08-464b-8efc-f39f295a7836
# ╟─b8156163-2a1f-48c6-85fd-9ecd15d1fb76
# ╟─22e8e869-175e-45a3-83d0-652aa346a78f
# ╟─f477c82e-f4d6-4918-85d1-19ce21e90b5b
# ╟─6c8ae8d2-a924-48f8-9ec7-beb0c9a7c08a
# ╟─22296f1a-3ed6-44ab-b9dd-47c58801afa6
# ╟─af78bc1d-4586-4c4e-a12f-c34b1ed9fbac
# ╟─1437a46a-3882-4719-ad49-94d7d065ead0
# ╟─9c3d80bb-20a4-4c81-8ce8-8d611cf24a6e
# ╟─e47db9dd-81ce-4ce0-a555-85d93e706bf6
# ╟─b6fd9d29-b5ac-4940-90d8-fcf0d8063003
# ╟─a6bd512f-0c74-4fff-a894-f40526edc84b
# ╟─0c3f1ca9-d021-454a-a6db-eef0e92dc2bf
# ╟─241065bb-1733-497f-8797-8d29ec41546e
# ╟─2c16ffab-bb3b-44b1-9b64-5708085358e7
# ╟─4c515d77-632c-4aa3-af4a-ab04abddf133
# ╟─91c4a88a-dff9-491d-b213-3ed480ca304e
# ╟─20f57c9d-3543-4cb6-8844-30b30e3b08ec
# ╟─f4a1f7db-0bcb-45b6-be9d-1c57dd6e2b99
# ╟─a5bfaa1e-9d40-4f2e-b0d5-73ab9bbceddf
# ╟─46dd39e0-95bf-4079-a982-4f93b16c3a5e
# ╟─90ea67c0-6f15-4fdb-830d-c7629fae0fcd
# ╟─8d878e1a-6191-42f2-a632-6d0b44f7d702
# ╟─5d52d818-21b2-4e8f-8efa-486274b68e57
# ╟─b732f3da-7be0-4470-9a5b-a389c1d1c166
# ╟─22e3f0e5-6a3c-439e-8160-978b24dc0a7d
# ╟─f15c498f-a4af-4cd5-a2db-2ea42516fb0f
# ╟─f979d38e-d634-4197-a774-773ea0af8193
# ╟─095f95fa-75e1-488f-a1c6-afcaaad30baa
# ╟─538a3ded-2ca0-4f0c-a931-fb7e43e5c24f
# ╟─b8593384-18c7-4b03-99b0-02d7e3a0f8be
# ╟─c92b52dd-3c76-40a8-a897-0136b5c4e529
# ╟─b81074f7-2bf7-4c4e-9f8a-efe426fdf6bd
# ╟─ecb72fef-744c-47aa-b54c-369aa7e16c9e
# ╟─726545fd-fbbc-4ce9-a35d-30e7e23ff3f9
# ╟─78c47b3f-9641-4ea4-9f3d-fadaa0ae004c
# ╟─c9baffcf-c1cc-42d6-96b5-1cc9a8801113
# ╟─40af31cc-1bc6-436c-ae6b-d13e2300779a
# ╟─5063f1be-5f18-43f6-b428-a0464e8fb339
# ╟─c0acfd8e-c152-4991-80a9-65d69c3bda69
# ╟─fcd96cd0-c733-4b76-9ea5-384f3253f730
# ╟─1ae4f029-cb95-4897-8029-903767babf7b
# ╟─a676ed26-1420-478f-a25c-fe99ab94c0a5
# ╟─39521df0-70c2-4e59-bcb2-792f0ba97c68
# ╟─4f176ca1-0326-4576-ae18-af2e1b697655
# ╟─6e25724e-79f7-4050-b3ff-fc51bfc852b5
# ╟─0be36e2d-817b-4951-8a3d-4ab44ce761ce
# ╟─94899a98-1565-4129-a264-1c4e1855982b
# ╟─1643d5a5-8925-45b7-a4ff-a39165207ec1
# ╟─6eec47df-4b17-48d4-a2f6-8eec8df29a4b
# ╟─8a5a3a3d-d341-48e9-bcba-a16094f048c4
# ╟─466d0d89-bf16-40f9-8eb2-2dfe70a8b227
# ╟─588735c8-d19c-4656-90e5-b2e530972caa
# ╟─e95f09c8-9364-4cf2-91a7-4e2dcbcd1053
# ╟─9ccef028-d31d-45d6-a37f-77cd31f772dc
# ╟─1c5c58b6-8567-4ce0-af7a-fa84f4879d39
# ╟─d94a06dc-77fb-4e27-a313-6589b5641519
# ╟─0130e026-cf29-45d7-9124-283b4ae5689e
# ╟─43be7d18-4417-4683-aae8-c35ffc9fb55f
# ╟─3e229fb0-62a3-443f-b864-848a0da18711
# ╟─8d9edfd5-6f97-4c4e-9464-3841078cc7c4
# ╟─5155785f-0cda-43de-9501-db8f2d03fcb2
# ╠═4b5ec74e-b89c-4bbf-9ac1-fa7d5b709f57
# ╠═4788ccd2-62a9-423f-95f8-57507ccdd6e3
# ╟─1cd4daa9-4683-4b01-a4ed-7c55d769c861
# ╟─8cca1036-a57f-48e0-826c-c81c4fe19085
# ╟─ea4a2ce4-5b20-4ad3-a39f-51b54be5f3d3
# ╟─730b641f-00bc-4b27-a832-166d1e2d75c9
# ╠═fccd6463-ed05-4a7d-89e6-c87981d08248
# ╠═47bf5f18-23c0-4780-815d-0fed28f2bc57
# ╠═56e93e4b-6963-4f86-993c-5f9bc3ec3199
# ╟─09c731d4-3b14-49c4-a609-4ab046513078
# ╟─61263a4c-f756-439e-a2bf-b7d7bc6d5b66
# ╟─60706166-a939-4758-b92c-27471dbade38
# ╟─8974a2f8-fb66-4069-814a-a39fb50ba154
# ╟─bd6e2825-a53a-4d2d-842f-9ef77d7eea11
# ╟─56c1e854-8b01-4777-b94b-d150689b81a8
# ╟─29e3d19b-34ac-48b5-930f-736f4488d855
# ╟─7a54d37e-198c-42fa-bd5e-f4b5abe13d9e
# ╟─9d83f116-79ed-4dd5-93d2-cc59f7127b3a
# ╟─d9067223-2bf7-4406-ab4d-adf2c28e3d72
# ╟─6ef8a367-223e-44a2-9aa2-9a5dcfc2d9fc
# ╟─42e7bc1a-7380-47f6-aa2d-c49cdaff0b08
# ╟─fe721301-1e77-4c89-8592-17d8c88776a7
# ╟─fbf9a928-fdc7-4f38-9b93-4bf74c20ad5b
# ╟─d294a9ce-b70e-48e4-aefc-a3e80cb7d003
# ╟─b6e7802f-3ce2-4663-b6a7-432d5295547d
# ╟─b6fca076-0639-490a-b77c-7a0155c25f39
# ╟─d96be66d-d960-4402-872e-14161144a7df
# ╠═48e8aa1f-9a19-456a-bb6d-cdbfe0f5403b
# ╟─7d5a7f03-74b0-42f8-a5c8-19e077db26bc
# ╟─7c6e1fd4-aa87-45c0-9c49-6dab2de90043
# ╟─e54c9f2d-ae25-46f1-8fcc-a75184a3610b
# ╟─dc7bbdc2-d3df-42a8-865b-74c42096cb57
# ╟─ed7f77ac-0cc5-4141-b88b-c5a86749ddd6
# ╟─e1219a00-b1cd-4aba-a3ab-cfec9380fcf3
# ╟─55b788aa-e846-421e-b214-ef99e24fd97e
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
